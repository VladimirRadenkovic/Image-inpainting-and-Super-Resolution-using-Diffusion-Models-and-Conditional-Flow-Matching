import math
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from PIL import Image
from glob import glob
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, Flowers102, CelebA
from torchvision.datasets import VisionDataset
from typing import Callable, Optional

__DATASET__ = {}

DatasetFn = Callable[[str, bool], VisionDataset]


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return T.functional.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )



def register_dataset(name: str):
    name = name.lower()

    def wrapper(fn: DatasetFn):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = fn
        return fn

    return wrapper


def get_dataset(name: str) -> DatasetFn:
    name = name.lower()
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name]


@register_dataset("mnist")
def mnist(root: str, train: bool) -> VisionDataset:
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5),
        ]
    )
    return MNIST(root, train=train, transform=transform, download=True)


@register_dataset("flowers")
def _flowers102(root, train) -> VisionDataset:
    image_size = 64
    transform = T.Compose(
        [
            T.Lambda(lambda img: F.center_crop(img, min(*img._size))),
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),  # normalize to [-1, 1]
        ]
    )
    dataset = Flowers102(root=root, split="train" if train else "test", transform=transform, download=True)
    return dataset


@register_dataset("celeba")
def _celeba(root, train) -> VisionDataset:

    # preprocess following https://github.com/ermongroup/ddim
    cx, cy = 89, 121
    x1, x2, y1, y2 = cy - 64, cy + 64, cx - 64, cx + 64
    image_size = 64
    transform = T.Compose(
        [
            Crop(x1, x2, y1, y2),
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),  # normalize to [-1, 1]
        ]
    )
    dataset = CelebA(root=root, split="train" if train else "test", transform=transform, download=True)
    return dataset
