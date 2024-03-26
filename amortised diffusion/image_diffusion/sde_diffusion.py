from typing import Callable
import abc

import torch.nn.functional as F
from torch import nn
import torch
from functorch import vmap, grad



Network = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


bm = 0.1
bd = 20


def int_b(t):
    """Integral b(t) for Variance preserving noise schedule."""
    return bm * t + (bd - bm) * t**2 / 2


def beta(t):
    """TODO: implement as d int_b / dt"""
    return bm + (bd - bm) * t


def unsqueeze_like(x, *objs):
    """Append additional axes to each obj in objs for each extra axis in x.

    Example: x of shape (bs,n,c) and s,t both of shape (bs,),
    sp,tp = unsqueeze_like(x,s,t) has sp and tp of shape (bs,1,1)

    Args:
      x: ndarray with shape that to unsqueeze like
      *objs: ndarrays to unsqueeze to that shape

    Returns:
      unsqueeze_objs: unsqueezed versions of *objs
    """
    if len(objs) != 1:
        return [unsqueeze_like(x, obj) for obj in objs]
    elif hasattr(objs[0], "shape") and len(objs[0].shape):  # broadcast to x shape
        return objs[0][(Ellipsis,) + len(x.shape[1:]) * (None,)]
    else:
        return objs[0]  # if it is a scalar, it already broadcasts to x shape


class VPSDE(abc.ABC):
    tmin = 1e-4
    tmax = 1.0

    def scale(self, t):
        """
        p(x(t) | x(0)) = N(x(t) | s(t) x(0), \sigma(t)^2 I)
        """
        return torch.exp(-int_b(t) / 2)

    def sigma(self, t):
        """
        p(x(t) | x(0)) = N(x(t) | s(t) x(0), \sigma(t)^2 I)
        """
        return torch.sqrt(1 - torch.exp(-int_b(t)))

    def drift(self, x, t):
        """dx_t = drift(t) dt + g(t) dW"""
        a = unsqueeze_like(x, -0.5 * beta(t))
        return a * x

    def diffusion(self, t):
        return torch.sqrt(beta(t))

    def backward_drift(self, score_fn, x, t):
        g = unsqueeze_like(x, self.diffusion(t))
        return self.drift(x, t) - g**2 * score_fn(x, t)

    def backward_diffusion(self, t):
        return self.diffusion(t)
    

    def backward_dynamics(self, score_fn, x, t):
        """Prob flow"""
        g = unsqueeze_like(x, self.diffusion(t))
        return self.drift(x, t) - 0.5 * g**2 * score_fn(x, t)

    def noise_score(self, xt, x0, t):
        r"""`\nabla\logp(xt)`."""
        scale, sigma = unsqueeze_like(x0, self.scale(t), self.sigma(t))
        return (scale * x0 - xt) / sigma**2

    def noise_input(self, x0, t):
        """Sample xt | x0 ~ N(s(t) x0, sigma^2 I)."""
        scale, sigma = unsqueeze_like(x0, self.scale(t), self.sigma(t))
        return scale * x0 + sigma * torch.randn_like(x0)

    def denoise_input(self, score_fn, xt, t):
        scale, sigma = unsqueeze_like(xt, self.scale(t), self.sigma(t))
        return (xt + sigma**2 * score_fn(xt, t)) / scale


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDPM(nn.Module):
    """Time-discretized version of VP SDE.

    Given a `score_fn` trained for a VP SDE diffusion model,
    we can derive a model to predict `x0` as

    def model_x0(x, i):
      t = 1.0 * i / ddpm.Ns   # [0, 1]
      x0 = vp_sde.denoise_input(score_fn, x, t)
      # apply clipping to x0 if wanted.
      return x0

    vica versa

    def score_fn(x, t):
      i = (t * ddpm.Ns)  # [0, ..., Ns-]
      x0 = model_x0(x, i)
      return vp_sde.noise_score(xt, x0, t)
    """

    def __init__(self, Ns: int):
        super().__init__()
        self.Ns = Ns
        self.tmin = 0.00001
        #self.tmin = 0.
        self.tmax = 1.0
        self.ts = torch.linspace(self.tmin, self.tmax, Ns, dtype=torch.float32)
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        betas = beta(self.ts) / Ns
        alphas = register_buffer("alphas", 1.0 - betas)
        alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        register_buffer("recip_sqrt_m1_alphas_cumprod", 1.0 / torch.sqrt(1 - alphas_cumprod))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer(
            "posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - alphas_cumprod)
        )

    def diffusion(self, t):
        # beta_t = self.beta_func(t)
        # beta_t = self.betas[t]
        beta_t = beta(t)# / self.Ns
        return torch.sqrt(beta_t)
        
    #def backward_diffusion(self, t):
        #return self.diffusion(t)
    
    def backward_diffusion(self, t):
        t = self.ts[t.cpu()]
        return self.diffusion(t)
        
    def drift(self, x, t):
        # beta_t = self.beta_func(t)
        # beta_t = self.betas[t]
        beta_t = beta(t)# / self.Ns
        return  -0.5 * unsqueeze_like(beta_t, x) * x
    
    #def backward_drift(self, score_fn, x, t):
        #g = unsqueeze_like(x, self.diffusion(t))
        #return self.drift(x, t) - g**2 * score_fn(x, t)
    
    def backward_drift(self, score_fn, x, noise, t):
        t = self.ts[t.cpu()].to(noise.device)
        g = unsqueeze_like(x, self.diffusion(t))
        return self.drift(x, t) - g**2 * score_fn(noise, t)
    """
    def backward_drift(self, score, x, noise, t):
        t = self.ts[t.cpu()].to(noise.device)
        g = unsqueeze_like(x, self.diffusion(t))
        return self.drift(x, t) - g**2 * score
    """
    def score_from_noise(self, noise, t):
        sigma = torch.sqrt(1 - torch.exp(-int_b(t)))
        score = -noise/sigma.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        inv_sigma = 1. / sigma

    # Check for NaNs in 1/sigma
        if torch.isnan(inv_sigma).any() or torch.isnan(score).any():
            print('--------------------')
            print("NaN detected in 1/sigma")
            print('---------------------')
        return score
    
    def score_from_x0(self, x_0, i):
        return (
            - extract(self.recip_sqrt_m1_alphas_cumprod, i, x_0.shape) * x_0
        )
    
        
    def predict_start_from_noise(self, x_i, i, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, i, x_i.shape) * x_i
            - extract(self.sqrt_recipm1_alphas_cumprod, i, x_i.shape) * noise
        )

    def q_posterior(self, x0, x_i, i):
        posterior_mean = (
            extract(self.posterior_mean_coef1, i, x_i.shape) * x0
            + extract(self.posterior_mean_coef2, i, x_i.shape) * x_i
        )
        posterior_variance = extract(self.posterior_variance, i, x_i.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, i, x_i.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x_start, x, i):
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x0=x_start, x_i=x, i=i)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def q_sample(self, x_start, i):
        noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, i, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, i, x_start.shape) * noise
        ), noise
