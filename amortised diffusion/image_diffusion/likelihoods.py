from typing import Type

import dataclasses
import torch
import numpy as np
import torch.nn.functional as F

from .plotting_utils import to_imshow


class Likelihood:

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        samples = []
        for i in range(len(x)):
            samples.append(self._sample(x[i:i+1]))
        return torch.concatenate(samples, dim=0)

    def _sample(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        samples = []
        for k in range(x.shape[0]):
            e = self._sample(x[[k]])
            samples.append(e)
        return torch.cat(samples, dim=0)

    def none_like(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def plot_condition(self, x, y, ax):
        raise NotImplementedError


class Painting(Likelihood):

    @classmethod
    def from_configdict(cls, config):
        return cls(patch_size=config['patch_size'], pad_value=config['pad_value'])

    def __init__(self, patch_size: int, pad_value: float):
        self.pad_value = pad_value
        self.patch_size = patch_size

    def get_random_patch(self, image_size):
        # don't sample to close to border.
        h = torch.randint(5, image_size - self.patch_size - 5, size=())
        w = torch.randint(5, image_size - self.patch_size - 5, size=())
        return h, w

    def none_like(self, x):
        return torch.ones_like(x) * self.pad_value

    def loss(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        x: [N, C, H, W]
        condition: [N, C, H, W]
        """
        x = torch.where(condition == self.pad_value, 0.0, x)
        condition = torch.where(condition == self.pad_value, 0.0, condition)
        loss = torch.sum((x - condition)**2, dim=(1, 2, 3))  # [N,]
        return loss

    def plot_condition(self, x, y, ax):
        condition = torch.where(y == self.pad_value, torch.nan, y)
        x0_true = torch.where(y == self.pad_value, x, torch.nan)
        ax.imshow(to_imshow(condition))
        ax.imshow(to_imshow(x0_true), alpha=0.1)
        return ax

class InPainting(Painting):
    """Condition is image with a missing patch."""

    def _sample(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [N, C, H, W]
        """
        image_size = images.shape[-1]
        h, w = self.get_random_patch(image_size)
        condition = images.detach().clone()
        slice_ = np.s_[:, :, h : h + self.patch_size, w : w + self.patch_size]
        condition[slice_] = self.pad_value
        return condition


class OutPainting(Painting):
    """
    The condition only contain a patch and everything else is masked out.
    """

    def _sample(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [N, C, H, W]
        """
        image_size = images.shape[-1]
        h, w = self.get_random_patch(image_size)
        slice_ = np.s_[:, :, h : h + self.patch_size, w : w + self.patch_size]
        condition = torch.ones_like(images) * self.pad_value
        condition[slice_] = images[slice_].detach().clone()
        return condition


class HyperResolution(Likelihood):

    @classmethod
    def from_configdict(cls, config):
        target_height = config['target_height']
        target_width = config['target_width']
        return cls(target_height, target_width)

    def __init__(self, target_height: int, target_width: int):
        self.target_height = target_height
        self.target_width = target_width

    def _sample(self, images: torch.Tensor) -> torch.Tensor:
        """
        Downscale the images to the target height and width using bilinear interpolation.
        """
        target_size = (self.target_height, self.target_width)
        downscaled_images = F.interpolate(images, size=target_size, mode='bilinear', align_corners=False)
        upscaled_images = F.interpolate(downscaled_images, (images.shape[2], images.shape[3]), mode="bilinear")
        return upscaled_images

    def none_like(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return a tensor of zeros with the same shape as the input tensor.
        """
        return torch.zeros_like(x)

    def loss(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss between the upscaled image and the original image.
        """
        upscaled = F.interpolate(condition, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return F.mse_loss(upscaled, x)

    def plot_condition(self, x, y, ax):
        """
        Visualize the condition.
        """
        ax.imshow(to_imshow(y))
        return ax


# TODO(Vincent): use register as with datasets
def get_likelihood(type_: str) -> Type[Likelihood]:
    if type_.lower() == "inpainting":
        return InPainting
    elif type_.lower() == "outpainting":
        return OutPainting
    elif type_.lower() == "hyperresolution":
        return HyperResolution
    else:
        raise NotImplementedError(f"Unknown conditioning {type_}")

