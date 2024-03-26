import pathlib
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection
from matplotlib.text import Text
from mpl_toolkits.mplot3d import Axes3D

__all__ = ["plot_pointcloud", "save_trajectory_as_gif"]


def plot_pointcloud(x: "torch.Tensor | np.ndarray",chain=True, motif_inds=None, **kwargs) -> None:
    """Plot a point cloud.

    Args:
        x (torch.Tensor): Point cloud of shape (n_points, n_dims)
    """
    dim = x.shape[1]
    if dim == 2:
        plt.scatter(x[:, 0], x[:, 1], cmap="viridis", c=np.arange(x.shape[0]), **kwargs)
        plt.axis("equal")
    elif dim == 3:
        # Plot 3D point cloud
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        c=np.arange(x.shape[0])
        if motif_inds is not None:
            c= ['r' if i in motif_inds else 'b' for i in range(x.shape[0])]
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], cmap="viridis", c=c, **kwargs)
        if chain: #plot chain
            ax.plot(x[:, 0], x[:, 1], x[:, 2], color="k", alpha=0.5)

    else:
        raise ValueError(f"Invalid point dimension: {dim}")
    # plt.show()
    # return fig

def plot_2_pointclouds(x: "torch.Tensor | np.ndarray", y: "torch.Tensor | np.ndarray", chain=True, **kwargs) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c='red', **kwargs)
    ax.scatter(y[:, 0], y[:, 1], y[:, 2], c='gray', **kwargs)
    if chain: #plot chain
        ax.plot(x[:, 0], x[:, 1], x[:, 2], color="k", alpha=0.5)
        ax.plot(y[:, 0], y[:, 1], y[:, 2], color="k", alpha=0.5)


def _update_scatter_plot(
    i: int, traj: np.ndarray, scatter_plot: Union[PathCollection, Axes3D], time_text: Text
) -> None:
    """Update scatter plot and time label in-place.

    Args:
        i (int): Current frame index
        traj (np.ndarray): Trajectory of shape (n_frames, n_points, n_dims)
        scatter_plot (Union[PathCollection, Axes3D]): Scatter plot (2D or 3D based on dimensions)
        time_text (Text): Time label
    """
    if traj.shape[-1] == 2:  # For 2D scatter plot
        scatter_plot.set_offsets(traj[i, ...])
    else:  # For 3D scatter plot
        scatter_plot._offsets3d = (traj[i, :, 0], traj[i, :, 1], traj[i, :, 2])

    # Update time label
    time_text.set_text(f"t = {i}")


def save_trajectory_as_gif(
    trajectory: Union[np.ndarray, "torch.Tensor"],
    filename: Union[str, pathlib.Path],
    *,
    fps: int = 10,
    t_skip: int = 1,
    dpi: int = 30,
    xlim: Tuple[int, int] = None,
    ylim: Tuple[int, int] = None,
    zlim: Tuple[int, int] = None,
    figsize: Tuple[int, int] = (8, 8),
    **scatter_kwargs,
) -> FuncAnimation:
    """Plots a trajectory of a 2D or 3D diffusion process as an animated GIF.

    Args:
        trajectory (Union[np.ndarray, torch.Tensor]): Trajectory of shape
            (n_frames, n_points, n_dims) with n_dims=2 for (x, y) coordinates or n_dims=3 for (x, y, z).
            If `torch.Tensor`, it must be on the CPU.
        filename (str): Filename of the GIF. Must end with .gif.
        fps (int, optional): Frames per second. Defaults to 10.
        t_skip (int, optional): Number of frames to skip between each frame in the
            GIF. Defaults to 1.
        dpi (int, optional): Dots per inch. Defaults to 30.
        xlim (Tuple[int, int], optional): x-axis limits. Defaults to None. If None,
            the limits are set to the minimum and maximum x-coordinates in the
            trajectory.
        ylim (Tuple[int, int], optional): y-axis limits. Defaults to None. If None,
            the limits are set to the minimum and maximum y-coordinates in the
            trajectory.
        zlim (Tuple[int, int], optional): z-axis limits. Used only for 3D plots. Defaults to None. If None,
            the limits are set to the minimum and maximum z-coordinates in the
            trajectory.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (8, 8).
        **scatter_kwargs: Additional keyword arguments passed to `matplotlib.pyplot.scatter` or `ax.scatter3D`.

    Returns:
        FuncAnimation: Animation object (also saved to disk as a GIF file at `filename`).
    """

    traj = np.asarray(trajectory)
    filename = pathlib.Path(filename)

    assert filename.suffix == ".gif", "Filename must end with .gif"
    assert traj.ndim == 3, "Trajectory must be of shape (n_frames, n_points, n_dims)"
    assert traj.shape[-1] in (
        2,
        3,
    ), "Trajectory must be of shape (n_frames, n_points, n_dims) with n_dims=2 or 3"

    n_frames, n_points, n_dims = traj.shape

    xlim = xlim if xlim is not None else (traj[:, :, 0].min(), traj[:, :, 0].max())
    ylim = ylim if ylim is not None else (traj[:, :, 1].min(), traj[:, :, 1].max())

    fig = plt.figure(figsize=figsize)

    if n_dims == 2:  # For 2D scatter plot
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        scatter_plot = plt.scatter(x=traj[0, :, 0], y=traj[0, :, 1], **scatter_kwargs)
        time_text = plt.text(
            0.05,
            0.95,
            f"t = {0}",
            horizontalalignment="left",
            verticalalignment="top",
            transform=plt.gca().transAxes,
        )
    else:  # For 3D scatter plot
        zlim = zlim if zlim is not None else (traj[:, :, 2].min(), traj[:, :, 2].max())
        ax = plt.axes(projection="3d")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        scatter_plot = ax.scatter3D(traj[0, :, 0], traj[0, :, 1], traj[0, :, 2], **scatter_kwargs)
        time_text = ax.text2D(0.05, 0.95, f"t = {0}", transform=ax.transAxes)

    animation = FuncAnimation(
        fig,
        _update_scatter_plot,
        frames=range(0, n_frames, t_skip),
        fargs=(traj, scatter_plot, time_text),
        interval=1000.0 / fps,
    )

    animation.save(filename, writer="imagemagick", dpi=dpi)
    return animation



if __name__ == "__main__":
    import os

    # Generate a simple 2D random walk and save as a gif
    n_frames, n_points, n_dims = 100, 10, 2
    trajectory_2d = np.cumsum(np.random.randn(n_frames, n_points, n_dims), axis=0)
    save_trajectory_as_gif(trajectory_2d, "2d_random_walk.gif", fps=10, dpi=30)
    plot_pointcloud(trajectory_2d[0])

    # Generate a simple 3D random walk and save as a gif
    n_frames, n_points, n_dims = 100, 10, 3
    trajectory_3d = np.cumsum(np.random.randn(n_frames, n_points, n_dims), axis=0)
    save_trajectory_as_gif(trajectory_3d, "3d_random_walk.gif", fps=10, dpi=30)
    plot_pointcloud(trajectory_3d[0])

    # Remove the generated gifs
    os.remove("2d_random_walk.gif")
    os.remove("3d_random_walk.gif")

