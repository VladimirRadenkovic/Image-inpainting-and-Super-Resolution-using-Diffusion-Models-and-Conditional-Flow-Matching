from __future__ import annotations
import torch
from typing import Sequence, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
# from torch_geometric.typing import PairTensor
from pytorch3d.transforms import axis_angle_to_matrix

from src.utils.pypdb_utils import get_coordinates

class BondLengths(Enum):

    """
    Enum for storing backbone bond lengths, taken from:

    Engh, R.A., & Huber, R. (2006).
    Structure quality and target parameters. International Tables for Crystallography.
    """

    N_CA = 1.459
    CA_C = 1.525
    C_N = 1.336



class BondAngles(Enum):
    """
    Enum for storing backbone bond angles, taken from:

    Engh, R.A., & Huber, R. (2006).
    Structure quality and target parameters. International Tables for Crystallography.
    """

    N_CA_C = 1.937
    CA_C_N = 2.046
    C_N_CA = 2.124


@dataclass(repr=False)#slots=True, 
class OrientationFrames:

    """
    Class for storing orientation frames for a set of residues. Specifically a single orientation frame
    consists of a (3, 3) rotation matrix specifying the orientation of the residue and a (3,) translation vector
    specifying the location of the alpha carbon.

    The class can store more than one rotation/translation,
    i.e. it can store a tensor of rotations of shape (N, 3, 3)
    and a tensor of translations of shape (N, 3).
    Note that the first dimension of the two tensors must be the same size.
    """

    rotations: torch.Tensor = field(repr=False)
    translations: torch.Tensor = field(repr=False)
    batch: Optional[torch.Tensor] = field(default=None, repr=False)
    ptr: Optional[torch.Tensor] = field(default=None, repr=False)

    def __post_init__(self):
        if self.rotations.shape[0] != self.translations.shape[0]:
            raise ValueError("Rotations and translations shapes do not match")

    def __repr__(self) -> str:
        return "{}({}={})".format(
            self.__class__.__name__, "num_frames", self.num_residues
        )

    @property
    def has_batch(self) -> bool:
        """
        Boolean property indicating whether the OrientationFrames have a batch assignment vector attribute.
        """
        return self.batch is not None

    @classmethod
    def from_three_points(
        cls,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        ptr: Optional[torch.Tensor] = None,
    ) -> OrientationFrames:
        """
        Get rotations and translations from three sets of 3-D points via Gram-Schmidt process.
        In proteins, these three points are typically N, CA, and C coordinates.

        :param x1: Tensor of shape (..., 3)
        :param x2: Tensor of shape (..., 3)
        :param x3: Tensor of shape (..., 3)
        :param batch: Batch assignment vector of length N.
        :param ptr: Batch pointer vector of length equal to the number of batches plus 1.
        :return: OrientationFrames object containing the relevant rotations/translations.
        """

        v1 = x3 - x2
        v2 = x1 - x2

        return cls.from_two_vectors(v1, v2, x2, batch, ptr)

    @classmethod
    def from_two_vectors(
        cls,
        v1: torch.Tensor,
        v2: torch.Tensor,
        translations: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        ptr: Optional[torch.Tensor] = None,
    ) -> OrientationFrames:
        """
        Get rotations and translations from three two 3-D vectors via Gram-Schmidt process.
        Vectors in `v1` are taken as the first component of the orthogonal basis, then the component of `v2`
        orthogonal to `v1`, and finally the cross product of `v1` and the orthogonalised `v2`. In
        this case the translations must be provided as well.

        In proteins, these two vectors are typically N-CA, and C-CA bond vectors,
        and the translations are the CA coordinates.

        :param v1: Tensor of shape (N, 3)
        :param v2: Tensor of shape (N, 3)
        :param translations: Tensor of translations of shape (N, 3).
        :param batch: Batch assignment vector of length N.
        :param ptr: Batch pointer vector of length equal to the number of batches plus 1.
        :return: OrientationFrames object containing the relevant rotations/translations.
        """

        e1 = v1 / torch.linalg.norm(v1, dim=-1).unsqueeze(-1)
        u2 = v2 - e1 * (torch.sum(e1 * v2, dim=-1).unsqueeze(-1))
        e2 = u2 / torch.linalg.norm(u2, dim=-1).unsqueeze(-1)
        e3 = torch.cross(e1, e2, dim=-1)

        rotations = torch.stack([e1, e2, e3], dim=-2).transpose(-2, -1)
        rotations = torch.nan_to_num(rotations)

        return cls(rotations, translations, batch, ptr)

    def to(self, device: torch.device) -> OrientationFrames:
        """
        Returns a new OrientationFrames instance moved to the specified device.
        """

        rotations = self.rotations.to(device)
        translations = self.translations.to(device)

        if self.has_batch:
            batch = self.batch.to(device)
            ptr = self.ptr.to(device)
        else:
            batch = None
            ptr = None

        return self.__class__(rotations, translations, batch, ptr)

    def inverse(self):
        """Gets the inverse of the frame(s)."""
        return OrientationFrames(
            self.rotations.transpose(-2, -1),
            -torch.matmul(self.translations.unsqueeze(-2), self.rotations).squeeze(-2),
        )

    def get_trisector(self, normalise: bool = True) -> torch.Tensor:
        """
        Returns the unit-length trisector of the three basis vectors defining
        the rotation(s) of the orientation frame(s).
        """
        trisector = torch.sum(self.rotations, dim=-1)

        if normalise is True:
            return torch.nn.functional.normalize(trisector, dim=-1)
        else:
            return trisector

    def apply(self, points: torch.Tensor) -> torch.Tensor:
        """
        Applies the rotations/translations to a set of N 3-dimensional vectors `points`.
        """

        if len(points.shape) < len(self.rotations.shape):
            points = points.unsqueeze(-2)

        rotated = torch.matmul(points, self.rotations.transpose(-2, -1))

        if len(rotated.shape) > 2:
            translation = self.translations.unsqueeze(-2)
        else:
            translation = self.translations

        return rotated + translation

    def apply_inverse(self, points: torch.Tensor) -> torch.Tensor:
        """
        Applies the inverse rotations/translations to a set of N 3-dimensional vectors `points`. Points
        should have shape (N, 3).
        """

        inv = self.inverse()
        return inv.apply(points)

    def compose(self, *frames: OrientationFrames) -> OrientationFrames:
        """
        Composes a sequence of orientation frames to the current instance of frames
        by applying rotations and translations sequentially.
        """

        t = self.translations.clone()
        R = self.rotations.clone()

        for frame in frames:
            R = frame.rotations @ R
            t = (t @ frame.rotations.transpose(-2, -1)) + frame.translations

        return OrientationFrames(R, t)

    @property
    def num_residues(self) -> int:
        return self.rotations.shape[-3]

    def __len__(self) -> int:
        return self.rotations.shape[0]

    def __getitem__(self, idx: Union[int, slice, Sequence]) -> OrientationFrames:
        """
        Indexes elements in the set of orientation frames, i.e. along the -3 dimension for rotations
        and the -2 dimension for translations.
        """

        return OrientationFrames(
            self.rotations[..., idx, :, :].reshape(
                (-1, self.rotations.shape[-2], self.rotations.shape[-1])
            ),
            self.translations[..., idx, :].reshape((-1, self.translations.shape[-1])),
        )

    def __setitem__(self, idx: Union[int, slice, Sequence], values) -> None:
        """
        Sets an item via an index and a tuple of (rotation, translation) to be set at the provided index.

        Usage: `self[idx] = (rotation, translation)`
        """

        new_rotation, new_translation = values
        self.rotations[..., idx, :, :] = new_rotation
        self.translations[..., idx, :] = new_translation

    def to_atom_coords(self):
        """
        Converts the rotations and translations for each residue into
        backbone atomic coordinates for each residue. Returns a 3-tuple
        of tensors (N coords, CA coords, C coords).

        Assumes `self.translations` are the alpha carbon coordinates,
        the first column vector of each rotation in `self.rotations` is the direction
        of the C-CA bond, the next column vector is the component of the N-CA bond
        orthogonal to the C-CA bond, and the third column vector is the cross product.
        """
        C_coords = (
            self.rotations[..., :, 0] * BondLengths.CA_C.value
        ) + self.translations
        CA_coords = self.translations

        # get the N-CA bond by rotating the second column vector to get the desired
        # bond angle
        N_bond_rotation = axis_angle_to_matrix(
            self.rotations[..., :, -1] * (BondAngles.N_CA_C.value - (np.pi / 2))
        )
        N_bonds = torch.matmul(
            N_bond_rotation,
            self.rotations[..., :, 1:2] * BondLengths.N_CA.value,
        ).squeeze(-1)
        N_coords = N_bonds + self.translations

        return N_coords, CA_coords, C_coords

    @classmethod
    def combine(
        cls,
        orientation_frames_sequence: Sequence[OrientationFrames],
        add_batch: bool = True,
    ) -> OrientationFrames:
        """
        Concatenates the rotations/translations from a sequence of orientation frames along the first dimension,
        returning a new instance of OrientationFrames.
        """

        concat_rotations = torch.cat(
            [frames.rotations for frames in orientation_frames_sequence], dim=0
        )
        concat_translations = torch.cat(
            [frames.translations for frames in orientation_frames_sequence], dim=0
        )

        # if batch vectors are present, concatenate them
        if all([frames.has_batch for frames in orientation_frames_sequence]):
            batch = torch.cat(
                [frames.batch for frames in orientation_frames_sequence], dim=-1
            )
            ptr = torch.cat(
                [frames.ptr for frames in orientation_frames_sequence], dim=-1
            )
        elif add_batch is True:
            struct_lengths = torch.as_tensor(
                [len(frames) for frames in orientation_frames_sequence],
                device=concat_rotations.device,
            )
            batch = torch.repeat_interleave(
                torch.arange(
                    len(orientation_frames_sequence), device=concat_rotations.device
                ),
                repeats=struct_lengths,
            )
            ptr = torch.nn.functional.pad(torch.cumsum(struct_lengths, dim=-1), (1, 0))
        else:
            batch = None
            ptr = None

    
if __name__ == "__main__":
    #coord are 2-D torch.Tensor objects containing 3-D vectors in the last dimension
    #for some arbitrary number of residues in the first dimension
    N_coord, CA_coord, C_coord = get_coordinates('3I40')
    frames = OrientationFrames.from_three_points(N_coord, CA_coord, C_coord)
    print(frames.rotations.shape)
    print(frames.translations.shape)