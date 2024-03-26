from __future__ import annotations

import functools

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_mean

from src.models.positional_embeddings import SinusoidalEncoding
from src.models.denoiser import Denoiser
from src.models.gvp import GVP, LayerNorm
from src.utils.gvp_gnn_utils import *
from src.models.gvp_conv_layer import GVPConv, GVPConvLayer

def _normalize(tensor, dim=-1):
    """
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    """
    return torch.nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0.0, D_max=6.0, D_count=16, device="cpu"):
    """
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    D_max and D_min and used to compute sigma of the RBF kernel.

    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF


def _edge_features(coords, edge_index, D_max=4.5, num_rbf=16, device="cuda"):
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), D_max=D_max, D_count=num_rbf, device=device)

    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))

    return edge_s, edge_v


class GVPDenoiser(Denoiser):
    """
    GVP-GNN model from https://arxiv.org/abs/2106.03843. Some modifications to simplify the code
    were added.
    Current implementation uses a single message-passing iteration with many GVP units in the message function.
        Args:
            n_steps: number of diffusion steps
            max_protein_length: maximum protein length used for positional encoding
            n_in_node_feats: (scalar, vector) input node features channels
            n_lookup_feats (int): number of features in the embedding lookup for nodes
            n_in_edge_feats: (scalar, vector) input edge features
            n_h_node_feats: (scalar, vector) hidden node features
            n_h_edge_feats: (scalar, vector) hidden edge features
            n_conv_layers: (int) number of GVPConv layers
            n_msg_layers (int): number of GVP layers used withing GVPConv as a message function
            sin_temp_enc (bool): whether to use sinusoidal temporal encoding for time steps
            device: device to use
    """

    def __init__(
        self,
        n_steps: int = 10,
        max_protein_length: int = 112,
        n_in_node_feats: tuple[int, int] = (1,1),  # (scalar, vector)
        n_lookup_feats: int = 16,
        n_in_edge_feats: tuple[int, int] = (16,1),  # (scalar, vector)
        n_h_node_feats: tuple[int, int] = (256, 64),  # (scalar, vector)
        n_h_edge_feats: tuple[int, int] = (256, 64),  # (scalar, vector)
        n_conv_layers: int = 5,
        n_msg_layers: int = 3,
        sin_temp_enc: bool = False,
        device: str = "cuda",
    ):
        super().__init__()

        self.trainer_device = device # this is used in the trainer
        self.sin_temp_enc = sin_temp_enc

        activations = (F.relu, None)
        # activations = (F.leaky_relu, None)

        # self.W_e = nn.Sequential(
        # LayerNorm(n_in_edge_feats),
        #     GVP(
        #         n_in_edge_feats,  # (s, V) of edge embeddings - distances & direction
        #         n_h_edge_feats,
        #         activations=(None, None),
        #         vector_gate=True,
        #     ),
        # )

        # Embedding lookup for initial node features
        self.emb_in = SinusoidalEncoding(embed_dim=n_lookup_feats, max_steps=max_protein_length)


        # Embedding for time steps
        if self.sin_temp_enc == True:
            self.emb_time = SinusoidalEncoding(embed_dim=n_lookup_feats, max_steps=n_steps)
            self.requires_normalized_times = False
        else:
            n_lookup_feats += 1  # increase by one bc we concat time
            self.requires_normalized_times = True

        self.W_e = nn.Sequential(  # HERE WAS LAYER NORM WHICH I KICKED OUT, COMPARE WITH ABOVE
            GVP(
                n_in_edge_feats,  # (s, V) of edge embeddings - distances & direction
                n_h_edge_feats,
                activations=(None, None),
                vector_gate=True,
            ),
        )

        self.W_v = nn.Sequential(
            # I THINK IT IS APPROPRIATE TO REMOVE THIS LAYER NORM AS WELL
            # LayerNorm(n_in_node_feats),  # (s, V) of node embeddings - node types & None
            GVP(
                (n_lookup_feats, n_in_node_feats[1]),
                n_h_node_feats,
                activations=(None, None),
                vector_gate=True,
            ),
        )

        # self.conv = GVPConv(
        #     in_dims=n_h_node_feats,
        #     out_dims=n_h_node_feats,
        #     edge_dims=n_h_edge_feats,
        #     n_layers=n_msg_layers,
        #     aggr="mean",
        #     activations=activations,
        #     vector_gate=True,
        # )

        self.convs = nn.ModuleList(
            GVPConv(
                in_dims=n_h_node_feats,
                out_dims=n_h_node_feats,
                edge_dims=n_h_edge_feats,
                n_layers=n_msg_layers,
                activations=activations,
                aggr="mean",
                vector_gate=True,
            )
            for _ in range(n_conv_layers)
        )

        ns, _ = n_h_node_feats

        # W_OUT DEVIATES FROM WHAT THEY DO IN THE ORIGINAL CODE.
        # MOST IMPORTANTLY I DO NOT USE DENSE LAYER ON THE OUTPUT SCALAR FEATURE
        # ALSO I OUTPUT ONE VECTOR FEATURE, THIS IS GOING TO BE OUR NOISE PREDICTION
        self.W_out = nn.Sequential(
            LayerNorm(n_h_node_feats),
            GVP(n_h_node_feats, (ns, 1), activations=activations, vector_gate=True),
        )

        self.to(device)

    def forward(
        self,
        batch: torch_geometric.data.Data | torch_geometric.data.batch.Batch,
        steps: torch.tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the GVP-GNN.

        Args:
            batch (torch_geometric.data.Data): batch of graphs
            steps (torch.tensor): not normalised time instances

        Returns:
            torch.Tensor: equivariant noise prediction
        """

        pos_3d = batch.pos

        edge_attr_s, edge_attr_v = _edge_features(
            pos_3d, batch.edge_index, D_max=6, num_rbf=16, device=self.trainer_device
        )

        pos_vector_channel = einops.repeat(pos_3d, "n d -> n v d", v=1).to(torch.float)

        x_s = self.emb_in(batch.node_order)  # (n,) -> (n, d)

        if self.sin_temp_enc == True:
            temp = self.emb_time(steps)[batch.batch]
            x_s = x_s + temp # not sure how this is gonna work if we use both sin enc for positions and time, but for me we might never want to use sin temp enc
        else:
            x_s = torch.cat((x_s, steps[batch.batch].unsqueeze(1)), dim=1)

        h_V = (x_s, pos_vector_channel)
        h_E = (edge_attr_s, edge_attr_v)
        h_V = self.W_v(h_V)  # IN THE ORIGNAL CODE THEY INITIALLY PROJECT LIKE THIS, before GVPConv
        h_E = self.W_e(h_E)

        for conv in self.convs:
            dh = conv(h_V, batch.edge_index, h_E)
            h_V = tuple_sum(h_V, dh)

        out = self.W_out(h_V)
        out_node_v = out[1]  # we output the vector channel
        out_node_v = out_node_v.reshape(batch.num_nodes, 3)

        if batch.pos.shape[1] == 2:
            out_node_v = out_node_v[:, 0:2]

        out_mean = scatter_mean(out_node_v, batch.batch, dim=0)

        return out_node_v - out_mean[batch.batch]

    def to(self, device: str):
        self.device = device
        return super().to(device)


class GVPDenoiserV2(Denoiser):
    """
    GVP-GNN model from https://arxiv.org/abs/2106.03843. Modifed version of my first GVP Denoiser. DO NOT INCLUDE SELF LOOPS in the edge index
    Current implementation uses a single message-passing iteration with many GVP units in the message function.
        Args:
            n_steps: number of diffusion steps
            max_protein_length: maximum protein length used for positional encoding
            n_in_node_feats: (scalar, vector) input node features channels
            n_lookup_feats (int): number of features in the embedding lookup for nodes
            n_in_edge_feats: (scalar, vector) input edge features
            n_h_node_feats: (scalar, vector) hidden node features
            n_h_edge_feats: (scalar, vector) hidden edge features
            n_conv_layers: (int) number of GVPConv layers
            n_msg_layers (int): number of GVP layers used withing GVPConv as a message function
            n_ff_layers (int): number of feed forward layers in the GVPConvLayer to update node embedding using aggregated msgs
            drop_rate (float): dropout rate in 
            sin_temp_enc (bool): whether to use sinusoidal temporal encoding for time steps
            device: device to use
    """

    def __init__(
        self,
        n_steps: int = 10,
        max_protein_length: int = 112,
        n_in_node_feats: tuple[int, int] = (1,1),  # (scalar, vector)
        n_lookup_feats: int = 16,
        n_in_edge_feats: tuple[int, int] = (16,1),  # (scalar, vector)
        n_h_node_feats: tuple[int, int] = (256, 64),  # (scalar, vector)
        n_h_edge_feats: tuple[int, int] = (256, 64),  # (scalar, vector)
        n_conv_layers: int = 5,
        n_msg_layers: int = 3,
        n_ff_layers: int = 1,
        drop_rate: float = 0,
        sin_temp_enc: bool = False,
        device: str = "cuda",
    ):
        super().__init__()

        self.trainer_device = device # this is used in the trainer
        self.sin_temp_enc = sin_temp_enc

        activations = (F.relu, None)

        # Embedding lookup for initial node features
        self.emb_in = SinusoidalEncoding(embed_dim=n_lookup_feats, max_steps=max_protein_length)


        # Embedding for time steps
        if self.sin_temp_enc == True:
            self.emb_time = SinusoidalEncoding(embed_dim=n_lookup_feats, max_steps=n_steps)
            self.requires_normalized_times = False
        else:
            n_lookup_feats += 1  # increase by one bc we concat time
            self.requires_normalized_times = True

        self.W_e = nn.Sequential(  # HERE WAS LAYER NORM WHICH I KICKED OUT, COMPARE WITH ABOVE
            GVP(
                n_in_edge_feats,  # (s, V) of edge embeddings - distances & direction
                n_h_edge_feats,
                activations=(None, None),
                vector_gate=True,
            ), LayerNorm(n_h_edge_feats)
        )

        self.W_v = nn.Sequential(
            # I THINK IT IS APPROPRIATE TO REMOVE THIS LAYER NORM AS WELL
            # LayerNorm(n_in_node_feats),  # (s, V) of node embeddings - node types & None
            GVP(
                (n_lookup_feats, n_in_node_feats[1]),
                n_h_node_feats,
                activations=(None, None),
                vector_gate=True,
            )
        )

        self.convs = nn.ModuleList(
            GVPConvLayer(
                node_dims=n_h_node_feats,
                edge_dims=n_h_edge_feats,
                n_message=n_msg_layers,
                n_feedforward=n_ff_layers,
                activations=activations,
                drop_rate=drop_rate,
                vector_gate=True,
            )
            for _ in range(n_conv_layers)
        )

        ns, _ = n_h_node_feats

        # I DO NOT USE DENSE LAYER ON THE OUTPUT SCALAR FEATURE
        # ALSO I OUTPUT ONE VECTOR FEATURE, THIS IS GOING TO BE OUR NOISE PREDICTION
        self.W_out = nn.Sequential(
            LayerNorm(n_h_node_feats),
            GVP(n_h_node_feats, (ns, 1), activations=activations, vector_gate=True),
        )

        self.to(device)

    def forward(
        self,
        batch: torch_geometric.data.Data | torch_geometric.data.batch.Batch,
        steps: torch.tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the GVP-GNN.

        Args:
            batch (torch_geometric.data.Data): batch of graphs
            steps (torch.tensor): not normalised time instances

        Returns:
            torch.Tensor: equivariant noise prediction
        """

        pos_3d = batch.pos

        edge_attr_s, edge_attr_v = _edge_features(
            pos_3d, batch.edge_index, D_max=6, num_rbf=16, device=self.trainer_device
        )

        pos_vector_channel = einops.repeat(pos_3d, "n d -> n v d", v=1).to(torch.float)

        x_s = self.emb_in(batch.node_order)  # (n,) -> (n, d)

        if self.sin_temp_enc == True:
            temp = self.emb_time(steps)[batch.batch]
            x_s = x_s + temp # not sure how this is gonna work if we use both sin enc for positions and time, but for me we might never want to use sin temp enc
        else:
            x_s = torch.cat((x_s, steps[batch.batch].unsqueeze(1)), dim=1)

        h_V = (x_s, pos_vector_channel)
        h_E = (edge_attr_s, edge_attr_v)
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for conv in self.convs:
            h_V = conv(h_V, batch.edge_index, h_E)

        out = self.W_out(h_V)
        out_node_v = out[1]  # we output the vector channel
        out_node_v = out_node_v.reshape(batch.num_nodes, 3)

        out_mean = scatter_mean(out_node_v, batch.batch, dim=0)

        return out_node_v - out_mean[batch.batch]

    def to(self, device: str):
        self.device = device
        return super().to(device)