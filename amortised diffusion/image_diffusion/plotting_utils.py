import numpy as np
import matplotlib.pyplot as plt


def plot_image_and_patch(image, patch, slice_, ax=None):
    """
    image: [C, H, W]
    patch: [C, h, w]
    """
    assert image.shape[0] == 1
    if ax is None:
        fig, ax = plt.subplots()
    ax.axis("off")
    ax.matshow(image[0], cmap="gray_r")
    patch_with_mask = np.ones_like(image) * np.nan
    patch_with_mask[slice_] = patch
    ax.imshow(patch_with_mask[0], cmap="Reds", alpha=0.8)


def plot_image_grid(images, patches=None, slice_=None):
    """
    images: [N, 1, C, H]
    """
    n = int(len(images) ** 0.5)
    fig, axes = plt.subplots(n, n, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        if patches is None:
            ax.imshow(images[i][0].cpu(), cmap="gray_r")
        elif patches is not None and slice_ is not None:
            plot_image_and_patch(images[i].cpu(), patches[i].cpu(), slice_, ax)
        ax.axis("off")
    # plt.tight_layout()
    return fig, axes


def to_imshow(x):
    """
    x: [C, H, W] or [H, W, C]
    detects the order by checking if C is 1 or 3.
    Also normalizes the output to be in range [0, 1]
    ignoring nan values.

    outputs [H, W, C]
    """

    def _maybe_transpose(x):
        # x: [C, H, W] or [H, W, C]
        if x.shape[0] in [1, 3]:
            x = np.transpose(x, [1, 2, 0])
        return x

    if hasattr(x, "cpu"):
        x = x.cpu().numpy()

    x = _maybe_transpose(x)
    x -= np.nanmin(x)
    x /= np.nanmax(x)
    return x
