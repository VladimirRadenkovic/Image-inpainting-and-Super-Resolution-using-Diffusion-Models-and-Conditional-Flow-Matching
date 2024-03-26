from typing import Callable, Literal, Union

import einops
import numpy as np
import pandas as pd
import torch
import torch_geometric
from loguru import logger
from torch_geometric.data.datapipes import functional_transform

try:
    import torch_cluster
except ImportError:
    logger.warning("torch_cluster not installed. Graph construction transforms will not work.")


__all__ = ["AddLabel", "AddNMA", "OneHotEncode", "KnnGraph", "RadiusGraph"]


# ---- Transforms ----
# NOTE: Disabled functional transform decorators bc it is not pickleable,
#   which breaks the multiprocessing datapipe. This will likely be resolved
#   in a future version of PyTorch Geometric.


# @functional_transform("add_label")
class AddLabel:
    """
    Add label to PyG data object from a label map.

    Args:
        label_map (dict): Dictionary mapping IDs to labels.
        shuffle_labels (bool, optional): Whether to shuffle the labels. Defaults to False.
        seed (int, optional): Seed for the RNG. Defaults to None.

    Returns:
        A PyG compatible transform function. That can be used on torch_geometric.data.Data objects.
    """

    def __init__(self, label_map: dict, shuffle_labels: bool = False, seed: int = None):
        self.shuffle_labels = shuffle_labels
        self.label_map = label_map
        self.seed = seed

    def _get_rng(self):
        if not hasattr(self, "_rng"):
            logger.info(f"Creating new label shuffling RNG with seed {self.seed}")
            self._rng = np.random.default_rng(seed=self.seed)
        return self._rng

    def _get_shuffled_labels(self, reshuffle: bool = False) -> dict:
        if reshuffle or not hasattr(self, "_shuffled_labels"):
            rng = self._get_rng()
            logger.info("Shuffling labels")
            permutation = rng.permutation(list(self.label_map.values()))
            self._shuffled_labels = {k: v for k, v in zip(self.label_map.keys(), permutation)}
        return self._shuffled_labels

    @classmethod
    def from_df(cls, df: pd.DataFrame, id_col: str = "id", label_col: str = "label", **kwargs):
        label_map = {row[id_col]: row[label_col] for _, row in df.iterrows()}
        return cls(label_map=label_map, **kwargs)

    def __call__(self, x: "torch_geometric.data.Data") -> "torch_geometric.data.Data":
        if x.id in self.label_map:
            if self.shuffle_labels:
                x.label = torch.as_tensor(self._get_shuffled_labels()[x.id], dtype=torch.long)
            else:
                x.label = torch.as_tensor(self.label_map[x.id], dtype=torch.long)
        else:
            logger.warning(f"ID {x.id} not in label map")
        return x


# @functional_transform("add_nma")
class AddNMA:
    """
    Add normal mode eigenvalues and eigenvectors to PyG data object.

    Args:
        model_type (Literal["gnm", "anm"]): Type of normal mode analysis model. Either "gnm" or "anm".
        nma_dataset (OnDiskDataset, optional): Dataset containing a pre-computed normal mode analysis models. Defaults to None.
        n_modes (int, optional): Number of modes to add. Defaults to 10.
        nma_model (Callable[[torch.Tensor], Union["springcraft.GNM", "springcraft.ANM"]], optional): Function that takes a
            coordinate tensor and returns a normal mode analysis model. Currently only supports CA based representations. Defaults to None.
        device (str, optional): Device to use for the normal mode analysis model. Defaults to "cpu".
    """

    def __init__(
        self,
        model_type: Literal["gnm", "anm"],
        nma_dataset: "OnDiskDataset" = None,
        n_modes: int = 10,
        nma_model: Callable[[torch.Tensor], Union["springcraft.GNM", "springcraft.ANM"]] = None,
        device: str = "cpu",
    ):
        self.nma_dataset = nma_dataset
        self.n_modes = n_modes
        self.nma_model = nma_model
        self.device = device

        # Expected number of null modes and coordinates
        self.n_null_modes = 1 if model_type == "gnm" else 6
        self.n_coords = 1 if model_type == "gnm" else 3

    def _on_the_fly_nma(self, x: "torch_geometric.data.Data"):
        # NOTE: Currently only coarse-grained CA models supported
        # TODO: Make this work for tabulated force-fields which depend on the residue types as well
        # backbone_ca = x.coords[:, 1, :]  # [seqlen, 3]
        backbone_ca = x.pos  # [seqlen, 3]
        nma_model = self.nma_model(backbone_ca)

        if self.device == "cpu":
            # NOTE: springcraft returns the eigenvectors in the shape [n_modes, seqlen * n_coords]
            #       e.g. [[mode1_res1_x, mode1_res1_y, mode1_res1_z, mode1_res2_x, ...],]]
            evals, evecs = nma_model.eigen()
            evals = torch.from_numpy(evals)
            evecs = torch.from_numpy(evecs)
        else:
            x.to(self.device)
            hessian = torch.from_numpy(nma_model.hessian, device=self.device)
            evals, evecs = torch.linalg.eigh(hessian)

        return evals, evecs

    def _subset_evals(self, evals: torch.Tensor, remove_null_modes: bool) -> torch.Tensor:
        """
        Subset the eigenvalues to the number of modes requested, ignoring the null modes.

        Args:
            evals (torch.Tensor): A tensor of eigenvalues of shape [n_total_modes = seqlen * self.n_coords]
            remove_null_modes (bool): Whether to remove the null modes. Usually True for both,
              on-the-fly NMA and NMA data from the dataset.

        Returns:
            out_evals (torch.Tensor): A tensor of eigenvalues of shape [self.n_modes].
                Zero padded if the number of modes requested is less than the number of available modes.
        """
        n_modes_total = len(evals)

        if remove_null_modes:
            n_modes_to_extract = min(self.n_modes, n_modes_total - self.n_null_modes)
            _selection = slice(self.n_null_modes, self.n_null_modes + n_modes_to_extract)
        else:
            n_modes_to_extract = min(self.n_modes, n_modes_total)
            _selection = slice(0, n_modes_to_extract)

        # NOTE: Zero padding is required to make the output shape consistent if
        #       the number of modes requested is less than the number of modes present
        out_evals = torch.zeros(self.n_modes, dtype=torch.float32)  # [self.n_modes]
        out_evals = evals[_selection]  # [self.n_modes]
        out_evals = out_evals.unsqueeze(0)  # [1, self.n_modes] (ordered low > high)

        return out_evals

    def _subset_evecs(self, evecs: torch.Tensor, remove_null_modes: bool) -> torch.Tensor:
        """
        Subset the eigenvectors to the number of modes requested, ignoring the null modes.

        Args:
            evecs (torch.Tensor): A tensor of eigenvectors of shape [n_modes_saved, seqlen, n_coords]
                For data from on-the-fly NMA,
                    n_modes_saved = seqlen * n_coords

                For data from the database, the null modes have already been filtered out.
                The number of `n_modes_svaed` is determined by the number of nodes saved in the
                database `n_saved` (typically 25) and the number of null modes:
                    n_modes_saved = min(n_saved, (seqlen * n_coords) - n_null_modes_present)
            remove_null_modes (bool): Whether to remove the null modes. Usually True for on-the-fly NMA, but
                False for NMA data from the database, because the null modes have already been removed.

        Returns:
            out_evecs (torch.Tensor): A tensor of eigenvectors of shape [self.n_modes, seqlen, n_coords].
        """
        n_modes_saved, seqlen, n_coords = evecs.shape
        assert n_coords == self.n_coords

        if remove_null_modes:
            n_modes_to_extract = min(self.n_modes, n_modes_saved - self.n_null_modes)
            _selection = slice(self.n_null_modes, self.n_null_modes + n_modes_to_extract)
        else:
            n_modes_to_extract = min(self.n_modes, n_modes_saved)
            _selection = slice(0, n_modes_to_extract)

        # NOTE: Zero padding is required to make the output shape consistent if
        #       the number of modes requested is less than the number of modes present

        out_evecs = torch.zeros(
            self.n_modes, seqlen, n_coords, dtype=torch.float32
        )  # [self.n_modes, seqlen, coords]
        out_evecs = evecs[_selection]  # [self.n_modes, seqlen, coords]

        out_evecs = einops.rearrange(
            out_evecs,
            "modes seqlen coords -> seqlen modes coords",
            modes=self.n_modes,
            seqlen=seqlen,
            coords=n_coords,
        )

        return out_evecs

    def __call__(self, x: "torch_geometric.data.Data") -> "torch_geometric.data.Data":
        if self.nma_dataset is not None and x.id in self.nma_dataset:
            nma_data = self.nma_dataset[x.id]

            x.eig_vals = self._subset_evals(
                torch.from_numpy(nma_data["evals"]), remove_null_modes=True
            )  # [1, self.n_modes]
            x.eig_vecs = self._subset_evecs(
                torch.from_numpy(nma_data["evecs"]), remove_null_modes=False
            )  # [seqlen, self.n_modes, coords]
        else:
            evals, evecs = self._on_the_fly_nma(
                x
            )  # [n_modes], [n_modes, seqlen*n_coords] (n_modes = seqlen * n_coords)

            x.eig_vals = self._subset_evals(evals, remove_null_modes=True)  # [1, self.n_modes]

            seqlen = x.pos.shape[0]
            n_modes_total = seqlen * self.n_coords
            x.eig_vecs = self._subset_evecs(
                evecs.view(n_modes_total, seqlen, self.n_coords), remove_null_modes=True
            )  # [seqlen, self.n_modes, coords]
        return x


# @functional_transform("to_one_hot")
class OneHotEncode:
    def __init__(self, feature: str, num_classes: int):
        self.feature = feature
        self.num_classes = num_classes

    def __call__(self, x: "torch_geometric.data.Data") -> "torch_geometric.data.Data":
        x[self.feature] = torch.functional.F.one_hot(
            x[self.feature], num_classes=self.num_classes
        ).float()
        return x


class KnnGraph:
    def __init__(
        self,
        k: int,
        *,
        loop: bool = False,
        flow: str = "source_to_target",
        cosine: bool = False,
        num_workers: int = 1,
        edge_name: str = "edge_index",
        pos_feature: str = "pos",
        remove_duplicate_edges: bool = True,
    ):
        self.k = k
        self.edge_name = edge_name
        self.pos_feature = pos_feature
        self.knn_graph_kwargs = {
            "loop": loop,
            "flow": flow,
            "cosine": cosine,
            "num_workers": num_workers,
        }
        self.remove_duplicate_edges = remove_duplicate_edges

    def __call__(self, x: "torch_geometric.data.Data") -> "torch_geometric.data.Data":
        edge_index = torch_cluster.knn_graph(
            x[self.pos_feature], k=self.k, batch=x.batch, **self.knn_graph_kwargs
        )

        if self.edge_name in x and self.remove_duplicate_edges:
            logger.info("Found existing edge_index, adding new edges without duplicates")
            # Add the new edges to the existing ones (remove duplicates)
            edge_index = torch_geometric.utils.coalesce(
                torch.cat([x[self.edge_name], edge_index], dim=1)
            )
        x[self.edge_name] = edge_index

        return x


class RadiusGraph:
    def __init__(
        self,
        r: float,
        *,
        loop: bool = False,
        max_num_neighbors: int = 32,
        flow: str = "source_to_target",
        num_workers: int = 1,
        edge_name: str = "edge_index",
        pos_feature: str = "pos",
        remove_duplicate_edges: bool = True,
    ):
        self.r = r
        self.max_num_neighbors = max_num_neighbors
        self.edge_name = edge_name
        self.pos_feature = pos_feature
        self.radius_graph_kwargs = {
            "loop": loop,
            "max_num_neighbors": max_num_neighbors,
            "flow": flow,
            "num_workers": num_workers,
        }
        self.remove_duplicate_edges = remove_duplicate_edges

    def __call__(self, x: "torch_geometric.data.Data") -> "torch_geometric.data.Data":
        edge_index = torch_cluster.radius_graph(
            x[self.pos_feature], r=self.r, batch=x.batch, **self.radius_graph_kwargs
        )

        if self.edge_name in x and self.remove_duplicate_edges:
            logger.info("Found existing edge_index, adding new edges without duplicates")
            # Add the new edges to the existing ones (remove duplicates)
            edge_index = torch_geometric.utils.coalesce(
                torch.cat([x[self.edge_name], edge_index], dim=1)
            )
        x[self.edge_name] = edge_index

        return x


# @functional_transform("save_graph_to_disk")
class SaveGraphToDisk:
    def __init__(self, filepath_fn: Callable[[torch_geometric.data.Data], str]):
        self.filepath_fn = filepath_fn

    def __call__(self, x: "torch_geometric.data.Data") -> "torch_geometric.data.Data":
        path = self.filepath_fn(x)
        torch.save(x, path)
        return x