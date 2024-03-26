from typing import Sequence, Union

import einops
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


def inflate_like(
    x: torch.Tensor, target: torch.Tensor, map_dims: Union[int, Sequence[int]]
) -> torch.Tensor:
    """
    Inflates the input tensor x to the shape of the target tensor by creating a view of x with
    the same dimensions of the target tensor.
    This is useful for broadcasting tensors with different shapes.

    Args:
        x (torch.Tensor): Input tensor to inflate
        target (torch.Tensor): Target tensor to match dimensions of, for broadcasting
        map_dims (Union[int, Sequence[int]]): dimension(s) of target tensor to match with x
            For example, if x.shape = (3, 4) and target.shape = (2, 3, 4, 5), then
            map_dims = [1, 2] will inflate x to shape (1, 3, 4, 1) to match the target.

    Returns:
        torch.Tensor: Inflated tensor (a contiguous view of x)
    """
    target_shape = [1] * len(target.shape)
    map_dims = [map_dims] if isinstance(map_dims, int) else map_dims
    for i, dim in enumerate(map_dims):
        assert (
            target.shape[dim] == x.shape[i]
        ), f"Dimension {dim} of target shape {target.shape} does not match dimension {i} of input shape {x.shape}"
        target_shape[dim] = x.shape[i]
    return x.view(target_shape)


def inflate_batch_array(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
    axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape. This is useful for broadcasting.
    """
    _remaining_dims = len(target.shape) - 1
    _remaining_dims = " ".join(["1"] * _remaining_dims)  # looks like e.g. "1 1 1"
    return einops.rearrange(x, f"b -> b {_remaining_dims}")  # [batch_size, ...]

def sum_consecutive(tensor):
    """
    Calculates the sums of 10 consecutive entries in a PyTorch tensor.

    Args:
    tensor (torch.Tensor): A 1D PyTorch tensor of length N.

    Returns:
    torch.Tensor: A 1D PyTorch tensor of length N-9 containing the sums of 10 consecutive entries.
    """

    window_size = 10
    n_windows = tensor.size()[0] - window_size + 1

    # Create a sliding window view of the tensor
    strides = (tensor.stride(0),) * 2
    window_view = tensor.as_strided(size=(n_windows, window_size), stride=strides)

    # Sum the values along the second dimension (axis 1)
    return window_view.sum(dim=1)

SCALE_FACTOR = 15. # Transform the coordinates from Angstroms to nanometers

def positions_to_graph(x, position_scale_factor: float = SCALE_FACTOR):
    # Convert the sample to a torch Tensor
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(torch.float32) 

    x = x / SCALE_FACTOR
    # Center the coordinates
    x = x - x.mean(dim=0)

    N = x.shape[0]
    # Create the fully connected adjacency matrix
    adj_matrix = torch.ones((N, N)) - torch.eye(N)

    # Convert the adjacency matrix to sparse edge indices
    edge_index = dense_to_sparse(adj_matrix)[0]

    # Add a chain order index to the node features
    chain_order = torch.arange(N)

    # Create a torch_geometric.data.Data object
    data = Data(pos=x, edge_index=edge_index, order=chain_order)
    return data