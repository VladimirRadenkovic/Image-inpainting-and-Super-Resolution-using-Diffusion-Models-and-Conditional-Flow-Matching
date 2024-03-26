import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from functools import lru_cache
class ContinuousNoiseSchedule(torch.nn.Module, ABC):
    """Base class for continuous noise schedules."""

    # @lru_cache(maxsize=1000)
    # def alpha(self, t):
    #     """Returns the value of alpha at time t, where t is in [0, T]."""
    #     return self.alpha_t(t)
    
    @lru_cache(maxsize=1000)
    def alpha_bar(self, t):
        """Returns the value of alpha_bar at time t, where t is in [0, T]."""
        return self.alpha_bar_t(t)
    
    @lru_cache(maxsize=1000)
    def beta(self, t):
        """Returns the value of beta at time t, where t is in [0, T]."""
        beta = self.beta_t(t)
        return beta
    
    def alpha(self, t):
        """Returns the value of alpha at time t, where t is in [0, T]."""
        return 1 - self.beta(t)
    
    def from_alpha_bar(self, alpha_bar_t):
        """Constructs continuous beta noise schedule from alpha_bar_t."""
        log_alpha_bar_t = lambda t: torch.log(alpha_bar_t(t))
        beta_t = lambda t: -torch.vmap(torch.func.grad(log_alpha_bar_t))(t)
        return beta_t
    
    def discretise_beta(self, n_steps=1000, clamp_max=0.25):
        """Discretises the beta schedule into n_steps."""
        t = torch.linspace(0, 1, n_steps)
        return torch.clamp(self.beta(t)/n_steps, max=clamp_max)
    
    def discretise_alpha(self, n_steps=1000, clamp_max=0.25):
        """Discretises the alpha schedule into n_steps."""
        return 1-self.discretise_beta(n_steps, clamp_max)
    
    def plot(self):
        """Plots the beta, alpha, alpha_bar and sqrt(alpha_bar) schedules."""
        n_steps = 1000
        t = torch.linspace(0, 1, n_steps)
        plt.plot(t, self.discretise_beta(n_steps), label=r"$\beta(t)$", marker=".")
        plt.plot(t, self.discretise_alpha(n_steps), label=r"$\alpha_t = 1 - \beta_t$", marker=".")
        plt.plot(t, 
            self.discretise_alpha(n_steps).sqrt(),
            label=r"$\sqrt{\alpha_t}$",
            color="C1",
            alpha=0.2,
            linestyle="--",
        )
        plt.plot(t, self.alpha_bar(t), label=r"$\bar{\alpha}(t)$", marker=".")
        plt.plot(
            t,
            self.alpha_bar(t).sqrt(),
            label=r"$\sqrt{\bar{\alpha}_t}$",
            color="C0",
            alpha=0.2,
            linestyle="--",
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(linestyle="--", alpha=0.1)
        plt.xlabel(r"Step $t$")


class LinearSchedule(ContinuousNoiseSchedule):
    def __init__(self, beta_0: float = 1e-4, beta_T: float = 2e-2):
        self.beta_0 = beta_0
        self.beta_T = beta_T
        self.alpha_bar_t = lambda t: torch.exp(-(beta_0*t + 0.5*(t**2)*(beta_T-beta_0)))
        self.beta_t = self.from_alpha_bar(self.alpha_bar_t)
    
    def beta_hardcoded(self, t): #for debugging
        return self.beta_0 + (self.beta_T - self.beta_0) * t

    def alpha(self, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_T - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff)
        return mean ** 2
    
class HoogeboomSchedule(ContinuousNoiseSchedule):
    """Noise schedule from Hoogeboom et al. https://arxiv.org/pdf/2203.17003.pdf, appendix B
    https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/fce07d701a2d2340f3522df588832c2c0f7e044a/equivariant_diffusion/en_diffusion.py#L38
    """

    def __init__(
        self,
        *,
        s: float = 1e-4,
        exponent: float = 3,
    ):
        self.s = s
        self.exponent = exponent
        self.alpha_bar_t = lambda t: (1.0 - t**self.exponent) ** 2 * (1-2*self.s) + self.s
        self.beta_t = self.from_alpha_bar(self.alpha_bar_t)
