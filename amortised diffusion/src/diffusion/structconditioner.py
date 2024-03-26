from typing import List, Sequence, Union

import torch
import torch_scatter
import torch_geometric
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dense_to_sparse
import numpy as np
from tqdm.autonotebook import tqdm
import einops
from typing import List, Sequence, Union
from src.utils.torch_utils import inflate_batch_array
import src.utils.geometry as geometry
from src.diffusion.conditioner import Conditioner

class Structconditioner(Conditioner):
    def __init__(self, 
                 alphas,
                 alpha_cumprods,
                 predict_noise_func,
                 mode="motif", 
                 loss_norm = "l2",
                 device = "cuda",
                 n_steps = 250):
        super().__init__()
        assert loss_norm in ['l1', 'l2'], 'loss norm should be l1 or l2'
        if loss_norm == 'l1':
            self.loss_func = torch.nn.functional.l1_loss
        if loss_norm == 'l2':
            self.loss_func = torch.nn.functional.mse_loss
        self.device = device
        self.n_steps = len(alphas)
        self.alphas = alphas
        self.alpha_cumprods = alpha_cumprods
        self.predict_noise_func = predict_noise_func

        if mode == 'motif':
            self.calc_condition = self.apply_cond_motif_method

    def set_monitor(self, num_graphs):
        self.monitor_total = [[] for i in range(num_graphs)] # measure of success

    # def _start_conditioning(self, step):

    #     if not hasattr(self, 'cond_start_step'):   
    #         self.cond_start_step = self.n_steps*self.cond_frac

    #     if step < self.cond_start_step:
    #         return True
    #     else:
    #         return False 

    def set_condition(self, motif_pos: torch.tensor, motif_inds: List[int], cond_frac: float, bayesian_gs: float = None, denoiser = None):
        """Set the condition of the diffusion process.
        Args:
            motif_pos (torch.Tensor): the position of the motif
            motif_inds (List[int]): the indices of the atoms that belong to the motif e.g. [15,16,20,31]
            cond_frac (float): fraction of the last timesteps when we apply the condition
            bayesian_gs (float): guidance scale for the bayesian method. If None, then in the apply_condition it will be replaced by dynamic_gs recalculated at each time step
        """
        self.motif_pos = motif_pos
        self.motif_inds = torch.tensor([motif_inds], device=self.device)
        self.cond_frac = cond_frac
        self.gs = bayesian_gs
        self.denoiser = denoiser
     

    def _total_denoise(self, x, batch_indx, pred_noise, steps):
        """Totally denoise the batch by applying the inverse of the noise prediction. 
        THIS IS NOT THE SAMPLING PROCEDURE.
        Args:
            batch_indx (torch.Tensor): same as batch_corrected.batch
            pred_noise (torch.Tensor): Noise tensor of shape [sum(batch_num_nodes), ...].
            steps (torch.LongTensor): Timesteps at which to denoise.
        Returns:
            torch.Tensor: Denoised feature tensor of the same shape as `batch.pos`."""

        abar = self.alpha_cumprods[steps]  #[1]
        abar = abar.expand(len(batch_indx))  # [batch_num_nodes]
        # Expand alpha_bar to match shape of x.pos (sum_i^batch_size(num_nodes[i])
        abar = inflate_batch_array(abar, x)  # [batch_num_nodes, ...]
        # Add noise to x
        x_denoised = ( x - (1.0 - abar).sqrt() * pred_noise ) / abar.sqrt()  # [sum(batch_num_nodes), ...]
        return x_denoised
        
    
    def apply_cond_motif_method(self, batch_corrected: Batch, step: int, t: torch.tensor):
        """
        This method is used to condition the predictions of the model on a given structural motif.
        Args:
            batch_corrected: the batch of graphs with the correct coordinates
            pred: the predicted coordinates
            step: the current step of the training
        Returns:
            pred_cond: the conditioned predictions
        """
        a = self.alphas[step]
        abar = self.alpha_cumprods[step]

        with torch.enable_grad():
            # pos_ = self._initialize_positions(batch_corrected)
            batch_grad = self.initialize_batch_grad(batch_corrected)

            # new_pred = self.denoiser(batch_corrected, t)
            new_pred = self.predict_noise_func(batch_grad, t)
            pos_denoised = self._denoise_positions(batch_grad.pos, batch_corrected, new_pred, step)
            
            batch_loss = self._compute_batch_loss(batch_corrected, pos_denoised, self.motif_pos, self.motif_inds)

            # if self._start_conditioning(step) == False or self.switched_on == False: # either too early to condition, or we do not condition at all
            #     return torch.zeros_like(pos_denoised, device=self.device)
            
            # self._perform_backward_pass(batch_loss, step)
            batch_loss.backward()
            
            pos_grad = - batch_grad.pos.grad.clone()

        gs = self._get_condition_scale(batch_corrected.num_graphs, a, gs_time_scaling=True)

        pred_cond = torch.einsum("bij, b -> bij", pos_grad.reshape(batch_corrected.num_graphs, -1, 3), gs)
        pred_cond = pred_cond.reshape(batch_corrected.pos.shape[0], 3)
        pred_cond *= (1 - a)

        return pred_cond
    
    def _initialize_positions(self, batch_corrected):
        return torch.nn.Parameter(batch_corrected.pos.data.clone())

    def _denoise_positions(self, pos, batch_corrected, pred, step):
        return self._total_denoise(pos, batch_corrected.batch, pred, step)
    
    def _get_condition_scale(self, num_graphs, a, gs_time_scaling=True):
        if gs_time_scaling == True:
            gs = self.gs*(a.expand(num_graphs)) # here added correction
        else:
            gs = self.gs.expand(num_graphs)

        return gs

    def _compute_batch_loss(self, batch_corrected, pos_denoised, motif_pos, motif_inds):
        # This function might be different from before, depending on how motif affects the batch loss
        batch_loss = 0
        num_graphs = batch_corrected.num_graphs
        loss_list = []
        for i in range(num_graphs):
            # extract motif from denoised positions
            # Boolean mask to get the indices of nodes belonging to the current graph
            node_indices_in_current_graph = (batch_corrected.batch == i).nonzero().squeeze()
            # Find global indices corresponding to the motif_inds
            global_motif_indices = node_indices_in_current_graph[motif_inds]
            motif_pos_sample = pos_denoised[global_motif_indices] # (1, n_motif_nodes, 3)
            motif_pos_sample = motif_pos_sample.squeeze(0) # (n_motif_nodes, 3)

            #align denoised motif to reference motif via Kabsch algorithm
            rot_mat, trans_vec = geometry.differentiable_kabsch(motif_pos, motif_pos_sample)
            if rot_mat is None: #if alignment fails in that step, go to next step
                self.monitor_total[i].append(self.monitor_total[i][-1])
                continue
            aligned_motif_pos_sample = geometry.rototranslate(motif_pos_sample, rot_mat, trans_vec)
            graph_loss = self.loss_func(aligned_motif_pos_sample, motif_pos)
            graph_loss = graph_loss / len(motif_inds) # normalize by number of motif nodes
            self.monitor_total[i].append(graph_loss.item()) # record the loss for each graph
            loss_list.append(graph_loss)

        batch_loss = torch.stack(loss_list, dim=0)
            
        return batch_loss
    
    def record_results(self, batch_corrected):

        batch_corrected.monitor_total_structure = self.monitor_total
        batch_corrected.motif_pos_sample = []
        for i in range(batch_corrected.num_graphs):
            node_indices_in_current_graph = (batch_corrected.batch == i).nonzero().squeeze()
            # Find global indices corresponding to the motif_inds
            global_motif_indices = node_indices_in_current_graph[self.motif_inds]
            motif_pos_sample = batch_corrected.pos[global_motif_indices]
            batch_corrected.motif_pos_sample.append(motif_pos_sample)

        batch_corrected.motif_pos_sample = torch.stack(batch_corrected.motif_pos_sample)
        batch_corrected.motif_pos = self.motif_pos
        batch_corrected.motif_inds = self.motif_inds

        return batch_corrected