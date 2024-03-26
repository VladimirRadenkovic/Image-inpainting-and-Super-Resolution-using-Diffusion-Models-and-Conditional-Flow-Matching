import torch
import torch.nn.functional as F
from torch_geometric.nn.pool import radius
from typing import Tuple

def sequential_distances(x, batch_info):
    """
    Calculate distances between consecutive nodes within the same graph.
    
    :param x: Node positions
    :param batch_info: Batch information indicating which nodes belong to which graph
    :return: Distances between consecutive nodes
    """
    x_shifted = torch.cat([x[1:], x[:1]], dim=0)
    batch_shifted = torch.cat([batch_info[1:], batch_info[:1]])
    diffs = x_shifted - x
    dists = torch.norm(diffs[:-1], dim=-1)
    same_graph = batch_info[:-1] == batch_shifted[:-1]
    return dists[same_graph]

def get_spatial_indices(x: torch.Tensor, batch_info: torch.Tensor, radius_val: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find indices of neighbors within a specified radius for each graph.
    
    :param x: Node positions
    :param batch_info: Batch information indicating which nodes belong to which graph
    :param radius_val: Radius within which neighbors are identified
    :return: Row and col indices representing neighbors within the specified radius
    """
    return radius(x, x, radius_val, batch_x=batch_info, batch_y=batch_info)


def calculate_distances(x: torch.Tensor, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
    """
    Calculate distances between nodes specified by the given indices.
    
    :param x: Node positions
    :param row: Row indices
    :param col: Col indices
    :return: Distances between nodes specified by the indices
    """
    diffs = x[row] - x[col]
    return torch.norm(diffs, dim=-1)
