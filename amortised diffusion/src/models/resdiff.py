import torch
from pytorch_lightning.core import LightningModule
from torch.nn import functional as F
import matplotlib.pyplot as plt
from typing import *

from src.evaluation.visualize import plot_pointcloud
from torch_geometric.data import Data, Batch

from src.utils.torch_utils import inflate_batch_array
from src.utils.distances import sequential_distances, get_spatial_indices, calculate_distances
import src.utils.geometry as geometry
from src.models.gvp_gnn import GVPDenoiserV2, GVPDenoiser
from src.diffusion.sde_diffusion import HoogeboomGraphSDE

from src.utils.log_utils import get_logger
logger = get_logger(__name__)

class ResDiff(LightningModule):
    def __init__(self, diffuser=None, denoiser=None, conditional_training=False, aux_loss=False):
        super().__init__()

        if diffuser is None:
            diffuser = HoogeboomGraphSDE()
        if denoiser is None:
            denoiser = GVPDenoiserV2()
        self.diffuser = diffuser
        self.model = denoiser
        self.conditional_training = conditional_training
        self.aux_loss = aux_loss
    
    def forward(self, x,t,mask=None,motif=None):
        return self.model(x,t,mask,motif)


    def training_step(self, batch, batch_idx): 
        x = batch
        #conditional training
        if self.conditional_training:
            mask = self.get_mask(x)
            motif = self.get_motif(x, mask)

        #noising
        noised_x = x.clone()
        t = self.sample_timesteps(x)#.to(x.pos.device)  
        noised_x, noise = self.diffuser.noising(noised_x, t)

        #forward pass 
        if self.conditional_training:
            noise_hat = self.model.forward(noised_x, t, mask, motif)
        else:
            noise_hat = self.model.forward(noised_x, t)

        denoised_x = self.diffuser.denoising(noised_x, noise_hat, t)
        dsm_loss, bb_loss, dist_loss = self.loss_fn(t, x, noised_x, noise, noise_hat, denoised_x)
        loss = dsm_loss + bb_loss + dist_loss
        if self.conditional_training:
            motif_loss = self.motif_loss_fn(denoised_x, mask, motif)
            self.log('motif_loss', loss)
            loss = loss + motif_loss
        #logging
        self.log('train_loss', loss, batch_size=x.num_graphs)
        self.log('dsm_loss', dsm_loss)
        self.log('bb_loss', bb_loss)
        self.log('dist_loss', dist_loss)
        self.log('x_mean', x.pos.mean())
        self.log('noised_x_mean', noised_x.pos.mean())
        self.log('noise_mean', noise.mean())
        self.log('noise_hat_mean', noise_hat.mean())
        # self.plot_sample(x, noised_x, noise, noise_hat, denoised_x)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        #conditional training
        if self.conditional_training:
            mask = self.get_mask(x)
            motif = self.get_motif(x, mask)

        #noising
        noised_x = x.clone()
        t = self.sample_timesteps(x)#.to(x.pos.device)  
        noised_x, noise = self.diffuser.noising(noised_x, t)

        #forward pass 
        if self.conditional_training:
            noise_hat = self.model.forward(noised_x, t, mask, motif)
        else:
            noise_hat = self.model.forward(noised_x, t)

        denoised_x = self.diffuser.denoising(noised_x, noise_hat, t)
        dsm_loss, bb_loss, dist_loss = self.loss_fn(t, x, noised_x, noise, noise_hat, denoised_x)
        loss = dsm_loss + bb_loss + dist_loss
        self.log('val_loss', loss, batch_size=x.num_graphs)
        # self.plot_sample(x, noised_x, noise, noise_hat, denoised_x)
        return loss
    
    def loss_fn(self, t, x, noised_x, true_noise, noise_hat, denoised_x):
        dsm_loss = F.mse_loss(true_noise, noise_hat)
        bb_loss, dist_loss = self.auxiliary_loss_fn(t, x, denoised_x)
        return dsm_loss, bb_loss, dist_loss
    
    def motif_loss_fn(self, denoised_x, mask, motif):
        """Calculate the motif loss, which is the MSE between the denoised node positions and the motif."""
        motif_sample = denoised_x.pos[mask]
        rot_mat, trans_vec = geometry.differentiable_kabsch(motif, motif_sample)
        motif_sample_aligned = geometry.rototranslate(motif_sample, rot_mat, trans_vec)
        motif_loss = F.mse_loss(motif_sample_aligned, motif)
        return motif_loss

    def auxiliary_loss_fn(
            self, t: torch.Tensor, x: torch.Tensor, denoised_x: torch.Tensor,
            cutoff: float = 125, weight: float = 0.25, option: str = "sequential", radius: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the auxiliary loss, which includes the backbone loss and the distogram loss.
        
        :param t: Time information
        :param x: Original node positions
        :param denoised_x: Denoised node positions
        :param cutoff: Time cutoff for backbone loss calculation, auxiliary loss is only calculated in later part of diffusion. Default: 0.25 for t=[0,1] and 125 for t=[0,500]
        :param weight: Weight for the losses
        :param option: Option for distogram loss calculation ('sequential' or 'spatial')
        :param radius: Radius within which distances are calculated for spatial distogram loss
        :return: Weighted backbone loss and distogram loss
        """
        # Mask nodes based on time cutoff
        loss_mask = t <= cutoff
        node_mask = torch.isin(x.batch, torch.arange(len(t), device=t.device)[loss_mask])
        if node_mask.sum() == 0: #if no graph fulfills the cutoff, return 0 for auxiliary losses
            return 0,0
        # Extract node positions and batch information
        x_selected = x.pos[node_mask]
        denoised_x_selected = denoised_x.pos[node_mask]
        node_batch_info = x.batch[node_mask]

        # Compute backbone loss
        backbone_loss = F.mse_loss(x_selected, denoised_x_selected)
        #normalisation from FrameDiff
        # backbone_loss = backbone_loss / (4*x_selected.shape[0])

        # Compute distogram loss based on chosen option
        assert option in ["sequential", "spatial"], "Invalid option. Choose either 'sequential' or 'spatial'"
        
        if option == "sequential":
            distances_x = sequential_distances(x_selected, node_batch_info)
            distances_denoised_x = sequential_distances(denoised_x_selected, node_batch_info)
        else:
            assert radius is not None, "Radius must be specified for spatial option"
            row, col = get_spatial_indices(x_selected, node_batch_info, radius)
            distances_x = calculate_distances(x_selected, row, col)
            distances_denoised_x = calculate_distances(denoised_x_selected, row, col)
        
        distogram_loss = F.mse_loss(distances_x, distances_denoised_x)
        #normalisation from FrameDiff
        # distogram_loss = distogram_loss / (x_selected.shape[0] - distances_x.shape[0])

        return backbone_loss * weight, distogram_loss * weight

    
    # def preprocess_batch(self, batch, coordinate_scale=15):
    #     #center batch positions at 0
    #     batch.pos = batch.pos - batch.pos.mean(dim=0)
    #     #scale batch positions down by factor 30
    #     batch.pos = batch.pos / coordinate_scale
    #     return batch
    
    def sample_timesteps(self, x, a=1e-3, b=1-(1e-3)):
        t = (a-b) * torch.rand((len(x.protein_id),), device=x.pos.device) + b
        return t
    
    def sample_integer_timesteps(self, x, N=500):
        t = torch.randint(0, N, (len(x.protein_id),), device=x.pos.device)
        return t
    
    def _predict_noise(self, x: Batch, steps: torch.LongTensor) -> Batch:
        return super()._predict_noise(x, steps)
    
    def apply_noise(
        self,
        *,
        batch: Batch,
        feature: str,
        eps: torch.Tensor,
        steps: Sequence[int],
    ) -> torch.Tensor:
        """
        Adds noise `eps` to the given `feature` of `batch` at specified timesteps `steps`.
        It is assumed that `batch` is graph-batched, i.e. that the values for each graph are
        concatenated along the first dimension. The number of nodes for each graph in the batch
        is given by `batch_num_nodes`.
  
          Args:
            batch (Batch): Graph-batched data object. Must contain the attributes:
                - `feature` (torch.Tensor): Feature tensor of shape [sum(batch_num_nodes), ...].
                - `batch` (torch.Tensor): Batch vector of shape [sum(batch_num_nodes)].
                - `ptr` (torch.Tensor): Pointer vector of shape [batch_size + 1].
                - (optional) `mask` (torch.Tensor): Mask vector of shape [sum(batch_num_nodes)].
            feature (str): Feature to add noise to. Only `pos` is supported for now.
            eps (torch.Tensor): Noise tensor of shape [sum(batch_num_nodes), ...].
            steps (Sequence[int]): Timesteps at which to add noise.

        Returns:
            torch.Tensor: Noisy feature tensor of the same shape as `batch[feature]`.
        """
        assert feature == "pos", "Only `pos` noise is supported for now"

        x = batch[feature]
        assert x.shape[0] == eps.shape[0]
        assert len(steps) == batch.num_graphs  # batch_size

        abar = self.schedule.alpha_bar(steps)  # [batch_size]

        # Expand alpha_bar to match shape of x.pos (sum_i^batch_size(num_nodes[i])
        abar = abar[batch.batch]  # [sum(batch_num_nodes)]
        abar = inflate_batch_array(abar, x)  # [sum(batch_num_nodes), ...]

        # Add noise to x
        x_noised = abar.sqrt() * x + (1.0 - abar).sqrt() * eps  # [sum(batch_num_nodes), ...]

        # Optionally mask out nodes that should not receive noise
        if "mask" in batch:
            x_noised[~batch.mask] = x[~batch.mask]

        return x_noised  # [sum(batch_num_nodes), ...]
    
    def plot_sample(self, x, noised_x, noise, noise_hat, denoised_x):
        first_graph_indices = (x.batch == 0).nonzero().squeeze()
        x_sample = x.pos[first_graph_indices].clone()
        noised_x_sample = noised_x.pos[first_graph_indices].clone()
        noise_sample = noise[first_graph_indices].clone()
        noise_hat_sample = noise_hat[first_graph_indices].clone()
        denoised_x_sample = denoised_x.pos[first_graph_indices].clone()

        plt.figure()
        plot_pointcloud(x_sample.cpu())
        plt.savefig("original.png")
        plt.close()

        plt.figure()
        plot_pointcloud(noised_x_sample.detach().cpu())
        plt.savefig("noised.png")
        plt.close()

        plt.figure()
        plot_pointcloud(noise_sample.cpu())
        plt.savefig("noise.png")
        plt.close()

        plt.figure()
        plot_pointcloud(noise_hat_sample.detach().cpu())
        plt.savefig("noise_hat.png")
        plt.close()

        plt.figure()
        plot_pointcloud(denoised_x_sample.detach().cpu())
        plt.savefig("denoised_x.png")
        plt.close()


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


def get_mask(x: Batch, min_len: int, max_len: int) -> torch.Tensor:
    """
    Given a Pytorch Geometric Batch, this function samples a mask length randomly between min_len and max_len,
    then selects a range of nodes randomly with this length for each graph in the batch, 
    and returns a boolean tensor indicating selected node ranges for each graph.

    Parameters:
        x (Batch): Pytorch Geometric Batch.
        min_len (int): Minimum length of the mask.
        max_len (int): Maximum length of the mask.

    Returns:
        masks (torch.Tensor): Boolean tensor indicating selected node ranges for each graph.
    """
    batch_sizes = x.batch.bincount()
    num_graphs = x.batch.unique().size(0)
    
    mask_lengths = torch.min(torch.randint(min_len, max_len + 1, (num_graphs,)), batch_sizes)
    max_start_indices = batch_sizes - mask_lengths
    start_indices = torch.floor(torch.rand((num_graphs,)) * (max_start_indices + 1)).int()

    upper_bound_indices = start_indices + mask_lengths
    
    cumsum_sizes = torch.cat([torch.tensor([0]), batch_sizes.cumsum(0)[:-1]])
    arange_global = torch.arange(x.num_nodes)
    
    start_bounds = cumsum_sizes[x.batch] + start_indices[x.batch]
    end_bounds = cumsum_sizes[x.batch] + upper_bound_indices[x.batch]
    masks = (arange_global >= start_bounds) & (arange_global < end_bounds)
    
    return masks


def get_motif(x: Batch, masks: torch.Tensor) -> torch.Tensor:
    """
    Given a Pytorch Geometric Batch and boolean masks, this function returns
    the indices of the nodes corresponding to the masks for each graph in the batch.

    Parameters:
        x (Batch): Pytorch Geometric Batch.
        masks (torch.Tensor): Boolean tensor indicating selected node ranges for each graph.

    Returns:
        motif_positions (torch.Tensor): Indices of the nodes corresponding to the masks for each graph.
    """
    
    batch_sizes = x.batch.bincount().tolist()
    motif_positions = []
    
    mask_splits = torch.split(masks, batch_sizes)
    pos_splits = torch.split(x.pos, batch_sizes)
    
    for graph_mask, graph_pos in zip(mask_splits, pos_splits):
        motif_positions.append(graph_pos[graph_mask])
    
    return motif_positions