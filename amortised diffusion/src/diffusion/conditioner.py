from abc import ABC, abstractmethod
import torch
import torch_geometric
from torch_geometric.data import Data

class Conditioner(ABC):
    """
    Abstract class Conditioner that defines the blueprint for a conditioner with
    several methods.
    """
    @abstractmethod
    def set_condition(self, *args, **kwargs):
        """
        Method to set the condition.
        """
        pass
    # @abstractmethod
    # def _start_conditioning(self, *args, **kwargs):
    #     """
    #     Method to start the condition.
    #     """
    #     pass

    @abstractmethod
    def _compute_batch_loss(self, *args, **kwargs):
        """
        Method to compute the batch loss.
        """
        pass
    @abstractmethod
    def _denoise_positions(self, *args, **kwargs):
        """
        Method to denoise positions.
        """
        pass
    @abstractmethod
    def record_results(self, *args, **kwargs):
        """
        Method to record the results.
        """
        pass
    @abstractmethod
    def set_monitor(self, *args, **kwargs):
        """
        Method to set the monitor.
        """
        pass

    def _perform_backward_pass(self, batch_loss, step):
        try:
            batch_loss.backward()
        except RuntimeError as e:
            print(f"Error occurred during backward pass at step {step}. Error message: {str(e)}")

    def _start_conditioning(self, step):
        if not hasattr(self, 'cond_start_step'):   
            self.cond_start_step = self.n_steps*self.cond_frac

        if step < self.cond_start_step:
            return True
        else:
            return False 
        
    def _get_condition_scale(self, num_graphs, a):
        if self.gs_time_scaling == True:
            gs = self.gs*(a.expand(num_graphs)) # here added correction
        else:
            if hasattr(self, "const_batch_gs") == False:
                self.const_batch_gs = self.gs.expand(num_graphs)
            gs = self.const_batch_gs # self.gs.expand((batch_corrected.num_graphs,)) # expand the constant guidance scale to the batch size

        return gs
    
    def initialize_batch_grad(self, batch_corrected):
        batch_grad = Data(batch=batch_corrected.batch.clone().detach(), pos=batch_corrected.pos.data.clone().detach().requires_grad_(True),
                          edge_index=batch_corrected.edge_index.clone().detach(), node_order = batch_corrected.node_order.clone().detach(), num_graphs=1) 
        return batch_grad