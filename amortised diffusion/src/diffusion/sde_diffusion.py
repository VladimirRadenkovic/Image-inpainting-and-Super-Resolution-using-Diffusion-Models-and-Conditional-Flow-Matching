"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs.
From: https://github.com/yang-song/score_sde_pytorch/blob/main/sde_lib.py
"""
import abc
import numpy as np
from loguru import logger
from functools import partial, cached_property, lru_cache
from typing import *
# import torch_scatter
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse
# from torch_geometric.data.batch import DataBatch

from src.utils.torch_utils import inflate_batch_array, positions_to_graph
from src.diffusion.covariance_utils import Rfunc
from src.evaluation.plot_proteins import quick_vis
from src.evaluation.visualize import plot_pointcloud
from src.constants import HYDRA_CONFIG_NAME, HYDRA_CONFIG_PATH, PROJECT_PATH, DATA_PATH

from src.utils.log_utils import get_logger
logger = get_logger(__name__)

# NOTE: The SDE essentially corresponds to the combination of (schedule, noise_type) 
#  in the `ddpm` picture. 
class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, N):
    """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.N = N

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  @abc.abstractmethod
  def prior_logp(self, z):
    """Compute log-density of the prior distribution.

    Useful for computing the log-likelihood via probability flow ODE.

    Args:
      z: latent code
    Returns:
      log probability density
    """
    pass

  def discretize(self, x, t):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.

    Args:
      x: a torch tensor
      t: a torch float representing the time step (from 0 to `self.T`)

    Returns:
      f, G
    """
    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
    return f, G

  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        score = score_fn(x, t)
        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t)
        rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()


class VPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
    return logps

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    f = torch.sqrt(alpha)[:, None, None, None] * x - x
    G = sqrt_beta
    return f, G
  
class HoogeboomGraphSDE(SDE):
  def __init__(self, *, s=1e-5, clip_value=0.25, exponent=2, N=250):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    super().__init__(N)
    self.s = s
    self.N = N
    self.clip_value = clip_value
    self.exponent = exponent
    self.alphas_cumprod_func = lambda t: (1.0 - t**self.exponent) ** 2 * (1-2*self.s) + self.s
    self.alphas_cumprod = torch.tensor([self.alphas_cumprod_func(t) for t in torch.linspace(0, 1, self.N)], device=device)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
    log_alpha_cumprod_func = lambda t: torch.log(self.alphas_cumprod_func(t))
    self.beta_func = lambda t: -torch.vmap(torch.func.grad(log_alpha_cumprod_func))(t)
    self.alpha_func = lambda t: 1.0 - self.beta_func(t)
    t = torch.linspace(0, 1, self.N, device=device)
    self.discrete_betas = torch.clamp(self.beta_func(t)/self.N, max=self.clip_value)
    self.alphas = 1. - self.discrete_betas

  def marginal_prob(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parameters to determine the marginal distribution of the SDE, $p_{0t}(x(t)|x(0))$.

    NOTE: 
      For a VP SDE, the marginal distribution is a Gaussian of the following form:
      $p_{0t}(x(t)|x(0)) = \mathcal{N}(x(t); \mu_t x(0), \sigma_t^2 I)$

      This function returns the parameters $\mu_t$ and $\sigma_t$.

      Used for the forward noising process.
    
    Args:
      t (torch.Tensor): time (scalar or vector).

    Returns:
      mean_scale (torch.Tensor): scaling factor $\mu_t$ for the mean of $p_t(x)$
      std_scale (torch.Tensor): scaling factor $\sigma_t$ for the standard deviation of $p_t(x)$
    """
    # t: [batch_size]
    #!Check if this is correct
    # alpha_bar = self.alphas_cumprod_func(t) # [batch_size]
    alpha_bar = self.alphas_cumprod_func(t) # [batch_size]
    mean = torch.sqrt(alpha_bar)  # [batch_size]
    std = torch.sqrt(1. - alpha_bar)  # [batch_size]
  
    return mean, std

  @staticmethod
  def _match_to_graph_batch(quantity: torch.Tensor, batch, feature: str = "pos"):
    """Match the shape of a quantity to the batch shape of a batch of graphs."""
    # quantity: [batch_size]
    # batch[feature]: [n_nodes_batch, ...]
    assert quantity.shape[0] == batch.num_graphs
    quantity = quantity[batch.batch]  # [n_nodes_batch]
    quantity = inflate_batch_array(quantity, batch[feature])  # [n_nodes_batch, 1, ... 1]
    return quantity  # [n_nodes_batch, 1, ... 1]
  
  @property
  def T(self):
    return 1

  def sde(self, x: Data, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # t: [batch_size]
    # x.pos: [n_nodes_batch, n_nodes, n_dim]

    beta_t = self.beta_func(t)  # [batch_size]
    # For graph case:
    beta_t = self._match_to_graph_batch(beta_t, x.pos)  # [n_nodes_batch, 1]

    drift = -0.5 * beta_t * x.pos  # [n_nodes_batch, n_nodes, n_dim]
    diffusion = torch.sqrt(self.beta_t)  # [n_nodes_batch, n_nodes, n_dim]
    # TODO: Think about whether to make these scalar: drift_scale, diffusion_scale
    return drift, diffusion

  @torch.no_grad()
  def sample_blob(self, num_samples, num_atoms, device):
      """Sample fully noised blob to start the reverse diffusion. Subtracts COM."""
      
      graphs = []
      
      for _ in range(num_samples):
          node_order = torch.arange(num_atoms, dtype=torch.long, device=device)
          init_pos = torch.randn(num_atoms, 3, device=device)
          centered_pos = init_pos - init_pos.mean(dim=0, keepdim=True)
          edge_index = torch.ones(num_atoms, num_atoms, device=device)  # fully connected graph
          edge_index = edge_index.fill_diagonal_(0)  # remove self loops on diagonal
          edge_index = dense_to_sparse(edge_index)[0]

          new_graph = Data(edge_index=edge_index, node_order=node_order)
          new_graph.pos = centered_pos
          graphs.append(new_graph)

      batch = Batch.from_data_list(graphs)
      return batch

  def prior_sampling(self, shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
      """Sample from the prior distribution, $p_0(x)$."""
      assert len(shape) == 2, "Shape must be (n_samples, n_nodes)"
      n_samples, n_nodes = shape
      n_dim = 3

      # Sample white noise
      x = torch.randn(n_samples, n_nodes, n_dim, device=device)
      # Centering
      x = x - x.mean(dim=1, keepdim=True)  # [n_samples, n_nodes, n_dim]

      # Transform to DataBatch
      batch = Batch.from_data_list([positions_to_graph(g, position_scale_factor=1.) for g in x])
      batch.id = torch.arange(n_samples, device=device)

      return batch

  @torch.no_grad()
  def reverse_diffusion_sampling(self, batch: Data, score_model, conditioner=None, save_trajectory=True) -> torch.Tensor:
    """Reverse diffusion sampling.
    Sample from the distribution $p_T(x)$ by sampling iteratively from the
    conditional distributions $p_{t-1}(x_{t-1}|x_t)$.

    Args:
      x (DataBatch): batch of graphs with node positions in `x.pos`
      score_model (nn.Module): model to compute the score function
      save_trajectory (bool): whether to save the trajectory of the reverse diffusion

    Returns:
      x (DataBatch): batch of graphs with node positions in `x.pos`
    """
    trajectory = [batch.pos.clone()] if save_trajectory else None
    # Reverse diffusion sampling
    for t in tqdm(reversed(range(self.N))):
      a = self.alphas[t]  # [1]
      abar = self.alphas_cumprod[t]  # [1]
      step = t
      #normalise the time step
      t = torch.tensor([t/self.N], device=batch.pos.device)#[1]
      #inflate N to batch size
      t = t.repeat(batch.num_graphs) #[batch_size]
      # Predict noise
      noise_hat = score_model(batch, t)  # [n_nodes_batch, n_dim]
      #expand a and abar to n_nodes_batch size
      a = a.unsqueeze(0).expand(batch.num_nodes, -1) # [n_nodes_batch]
      abar = abar.unsqueeze(0).expand(batch.num_nodes, -1) # [n_nodes_batch]
      # Sample white noise
      z = self.noise_like(batch)  # [n_nodes_batch, n_dim]

      #apply conditioner
      if conditioner and step < 125:
        t_cond = torch.tensor([step/self.N], device=batch.pos.device, requires_grad=True)#[1]
        cond_update = conditioner.apply_cond_motif_method(batch, step, t_cond)
        batch.pos += cond_update

      # Update
      if step > 2: #no noise for last X steps
        batch.pos = (1/torch.sqrt(a)) * (batch.pos - ((1-a)/(torch.sqrt(1-abar))) * noise_hat) + torch.sqrt(1-a) * z  # [n_nodes_batch, n_dim]
      else:
        batch.pos = (1/torch.sqrt(a)) * (batch.pos - ((1-a)/(torch.sqrt(1-abar))) * noise_hat)

      if save_trajectory:
        trajectory.append(batch.pos.clone()) 
    
    if conditioner:
      batch = conditioner.record_results(batch)
    if save_trajectory:
            return torch.stack(trajectory, dim=0), batch
    
    return batch

  def discretize(self, x, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """DDPM discretization.
    Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.

    Args:
      x (DataBatch): batch of graphs with node positions in `x.pos`
      t (torch.Tensor): time (scalar or vector)

    Returns:
      f (torch.Tensor): drift term
      G (torch.Tensor): diffusion term
    """
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    # For graph case:
    f = self._match_to_graph_batch(alpha.sqrt(), x) * x.pos - x.pos  # [n_nodes_batch, n_dim]
    G = sqrt_beta
    return f, G

  def prior_logp(self, z: torch.Tensor):
    raise NotImplementedError()

  # def noise_like(self, *, batch: Batch, feature: str) -> torch.Tensor:
  #       # Samples Gaussian noise and subtract center of mass
  #       assert feature == "pos", "Only `pos` noise is supported for now"

  #       noise = torch.randn_like(batch[feature])  # [nodes ,coords]

  #       # Subtract center of mass for each graph in the batch
  #       noise_mean = torch_scatter.scatter_mean(noise, batch.batch, dim=0)  # [graphs, coords]s
  #       noise = noise - noise_mean[batch.batch]  # [nodes, coords]

  #       return noise  # [nodes, coords]

  def noise_like(self, batch) -> torch.Tensor:

    # For graph batching case:
    noise = torch.zeros_like(batch.pos)  # [n_nodes_batch, n_dim]
    for graph, (ptr0, ptr1) in enumerate(zip(batch.ptr[:-1], batch.ptr[1:])):
      n_nodes = ptr1 - ptr0
      # White noise
      z = torch.randn(n_nodes, noise.shape[1])  # [n_nodes, n_dim]
      # Subtracting center of mass
      z -= z.mean(dim=0, keepdim=True)  # [n_nodes, n_dim]
      noise[ptr0:ptr1] = z
      
    return noise  # [n_nodes_batch, n_dim]

  def noising(self, batch, t: torch.Tensor):
    mean_scale, std_scale = self.marginal_prob(t)  # [batch_size], [batch_size]

    # For graph case:
    mean_scale = self._match_to_graph_batch(mean_scale, batch, "pos")  # [n_nodes_batch, 1]
    std_scale = self._match_to_graph_batch(std_scale, batch, "pos")  # [n_nodes_batch, 1]
    eps = self.noise_like(batch)

    batch.pos = mean_scale * batch.pos + std_scale * eps  # [n_nodes_batch, n_dim]
    return batch, eps

  def denoising(self, batch, eps, t: torch.Tensor):
    mean_scale, std_scale = self.marginal_prob(t)  # [batch_size], [batch_size]

    # For graph case:
    mean_scale = self._match_to_graph_batch(mean_scale, batch, "pos")  # [n_nodes_batch, 1]
    std_scale = self._match_to_graph_batch(std_scale, batch, "pos")  # [n_nodes_batch, 1]

    batch.pos =  (batch.pos - std_scale * eps) / mean_scale  # [n_nodes_batch, n_dim]
    return batch

  # def reverse(self, score_fn, probability_flow=False):
  #   """Create the reverse-time SDE/ODE.

  #   Args:
  #     score_fn: A time-dependent score-based model that takes x and t and returns the score.
  #     probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
  #   """
  #   N = self.N
  #   T = self.T
  #   sde_fn = self.sde
  #   discretize_fn = self.discretize

  #   # Build the class for reverse-time SDE.
  #   class RSDE(self.__class__):
  #     def __init__(self):
  #       self.N = N
  #       self.probability_flow = probability_flow

  #     @property
  #     def T(self):
  #       return T

  #     def sde(self, x, t):
  #       """Create the drift and diffusion functions for the reverse SDE/ODE."""
  #       drift, diffusion = sde_fn(x, t)
  #       score = score_fn(x, t)
  #       drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
  #       # Set the diffusion function to zero for ODEs.
  #       diffusion = 0. if self.probability_flow else diffusion
  #       return drift, diffusion

  #     def discretize(self, x, t):
  #       """Create discretized iteration rules for the reverse diffusion sampler."""
  #       f, G = discretize_fn(x, t)
  #       rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
  #       rev_G = torch.zeros_like(G) if self.probability_flow else G
  #       return rev_f, rev_G

  #   return RSDE()
  

  
class VPGraphSDE(VPSDE):
  def __init__(self, *, beta_min=0.1, beta_max=20, N=1000):
    super().__init__(beta_min, beta_max, N)

  def marginal_prob(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parameters to determine the marginal distribution of the SDE, $p_{0t}(x(t)|x(0))$.

    NOTE: 
      For a VP SDE, the marginal distribution is a Gaussian of the following form:
      $p_{0t}(x(t)|x(0)) = \mathcal{N}(x(t); \mu_t x(0), \sigma_t^2 I)$

      This function returns the parameters $\mu_t$ and $\sigma_t$.

      Used for the forward noising process.
    
    Args:
      t (torch.Tensor): time (scalar or vector).

    Returns:
      mean_scale (torch.Tensor): scaling factor $\mu_t$ for the mean of $p_t(x)$
      std_scale (torch.Tensor): scaling factor $\sigma_t$ for the standard deviation of $p_t(x)$
    """
    # t: [batch_size]
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff)  # [batch_size]
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))  # [batch_size]
    return mean, std

  @staticmethod
  def _match_to_graph_batch(quantity: torch.Tensor, batch, feature: str = "pos"):
    """Match the shape of a quantity to the batch shape of a batch of graphs."""
    # quantity: [batch_size]
    # batch[feature]: [n_nodes_batch, ...]
    assert quantity.shape[0] == batch.num_graphs
    quantity = quantity[batch.batch]  # [n_nodes_batch]
    quantity = inflate_batch_array(quantity, batch[feature])  # [n_nodes_batch, 1, ... 1]
    return quantity  # [n_nodes_batch, 1, ... 1]

  def sde(self, x: Data, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # t: [batch_size]
    # x.pos: [n_nodes_batch, n_nodes, n_dim]

    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)  # [batch_size]
    # For graph case:
    beta_t = self._match_to_graph_batch(beta_t, x.pos)  # [n_nodes_batch, 1]

    drift = -0.5 * beta_t * x.pos  # [n_nodes_batch, n_nodes, n_dim]
    diffusion = torch.sqrt(self.beta_t)  # [n_nodes_batch, n_nodes, n_dim]
    # TODO: Think about whether to make these scalar: drift_scale, diffusion_scale
    return drift, diffusion

  def prior_sampling(self, shape: Tuple[int, int]) -> torch.Tensor:
    """Sample from the prior distribution, $p_0(x)$."""
    assert len(shape) == 3, "Shape must be (n_samples, n_nodes, n_dim)"
    n_samples, n_nodes, n_dim = shape

    # Sample white noise
    z = torch.randn(n_samples, n_nodes, n_dim)

    # Centering
    x = x - x.mean(dim=1, keepdim=True)  # [n_samples, n_nodes, n_dim]

    # Transform to DataBatch
    batch = Batch.from_data_list([positions_to_graph(g, position_scale_factor=1.) for g in x])
    batch.id = torch.arange(n_samples)

    return batch

  def discretize(self, x, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """DDPM discretization.
    Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.

    Args:
      x (DataBatch): batch of graphs with node positions in `x.pos`
      t (torch.Tensor): time (scalar or vector)

    Returns:
      f (torch.Tensor): drift term
      G (torch.Tensor): diffusion term
    """
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    # For graph case:
    f = self._match_to_graph_batch(alpha.sqrt(), x) * x.pos - x.pos  # [n_nodes_batch, n_dim]
    G = sqrt_beta
    return f, G

  def prior_logp(self, z: torch.Tensor):
    raise NotImplementedError()

  # def noise_like(self, *, batch: Batch, feature: str) -> torch.Tensor:
  #       # Samples Gaussian noise and subtract center of mass
  #       assert feature == "pos", "Only `pos` noise is supported for now"

  #       noise = torch.randn_like(batch[feature])  # [nodes ,coords]

  #       # Subtract center of mass for each graph in the batch
  #       noise_mean = torch_scatter.scatter_mean(noise, batch.batch, dim=0)  # [graphs, coords]s
  #       noise = noise - noise_mean[batch.batch]  # [nodes, coords]

  #       return noise  # [nodes, coords]

  def noise_like(self, batch) -> torch.Tensor:

    # For graph batching case:
    noise = torch.zeros_like(batch.pos)  # [n_nodes_batch, n_dim]
    for graph, (ptr0, ptr1) in enumerate(zip(batch.ptr[:-1], batch.ptr[1:])):
      n_nodes = ptr1 - ptr0
      # White noise
      z = torch.randn(n_nodes, noise.shape[1])  # [n_nodes, n_dim]
      # Subtracting center of mass
      z -= z.mean(dim=0, keepdim=True)  # [n_nodes, n_dim]
      noise[ptr0:ptr1] = z
      
    return noise  # [n_nodes_batch, n_dim]

  def noising(self, batch, t: torch.Tensor):
    mean_scale, std_scale = self.marginal_prob(t)  # [batch_size], [batch_size]

    # For graph case:
    mean_scale = self._match_to_graph_batch(mean_scale, batch, "pos")  # [n_nodes_batch, 1]
    std_scale = self._match_to_graph_batch(std_scale, batch, "pos")  # [n_nodes_batch, 1]
    eps = self.noise_like(batch)

    batch.pos = mean_scale * batch.pos + std_scale * eps  # [n_nodes_batch, n_dim]
    return batch, eps
  
  def denoising(self, batch, eps, t: torch.Tensor):
    mean_scale, std_scale = self.marginal_prob(t)  # [batch_size], [batch_size]

    # For graph case:
    mean_scale = self._match_to_graph_batch(mean_scale, batch, "pos")  # [n_nodes_batch, 1]
    std_scale = self._match_to_graph_batch(std_scale, batch, "pos")  # [n_nodes_batch, 1]

    batch.pos =  (batch.pos - std_scale * eps) / mean_scale  # [n_nodes_batch, n_dim]
    return batch
  


  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        score = score_fn(x, t)
        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t)
        rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()
  