import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt




class DiscreteNoiseSchedule(nn.Module, ABC):
    def __init__(self, *, n_steps: int, validate: bool = True):
        super().__init__()
        self.n_steps = n_steps
        self.setup_schedule()
        if validate:
            self.check_schedule_validity()

    @property
    def device(self):
        return next(self.buffers()).device

    @abstractmethod
    def setup_schedule(self) -> torch.Tensor:
        raise NotImplemented

    def beta(self, step: int):
        # NOTE: beta(step) is the forward process variance going from
        # step-1 to step
        return torch.atleast_1d(self.betas[step])

    def alpha(self, step: int):
        return torch.atleast_1d(self.alphas[step])

    def alpha_bar(self, step: int):
        return torch.atleast_1d(self.alpha_bars[step])

    def check_schedule_validity(self):
        assert (self.betas > 0).all() and (self.betas <= 1).all(), "beta should be in (0, 1]"
        assert (self.alphas >= 0).all() and (self.alphas <= 1).all(), "alpha should be in [0, 1]"
        assert (self.alpha_bars >= 0).all() and (
            self.alpha_bars <= 1
        ).all(), "alpha_bar should be in [0, 1]"

    def __len__(self):
        return self.n_steps

    def __repr__(self):
        return f"{self.__class__.__name__}(n_steps={self.n_steps}, beta_start={self.betas[0]:.3f}, beta_end={self.betas[-1]:.3f})"

    def plot(self):
        plt.plot(self.betas, label=r"$\beta_t$", marker=".")
        plt.plot(self.alphas, label=r"$\alpha_t = 1 - \beta_t$", marker=".")
        plt.plot(
            self.alphas.sqrt(),
            label=r"$\sqrt{\alpha_t}$",
            color="C1",
            alpha=0.2,
            linestyle="--",
        )
        plt.plot(self.alpha_bars, label=r"$\bar{\alpha}_t = \prod_s^t~\alpha_s$", marker=".")
        plt.plot(
            self.alpha_bars.sqrt(),
            label=r"$\sqrt{\bar{\alpha}_t}$",
            color="C2",
            alpha=0.2,
            linestyle="--",
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(linestyle="--", alpha=0.1)
        plt.xlabel(r"Step $t$")


class LinearSchedule(DiscreteNoiseSchedule):
    def __init__(self, *, n_steps: int = 300, beta_start: float = 1e-4, beta_end: float = 2e-2):
        """Linear schedule from https://arxiv.org/pdf/2006.11239.pdf, appendix B"""
        self.beta_start = beta_start
        self.beta_end = beta_end
        super().__init__(n_steps=n_steps)

    def setup_schedule(self) -> None:
        self.register_buffer("betas", torch.linspace(self.beta_start, self.beta_end, self.n_steps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_bars", self.alphas.cumprod(dim=0))


class CosineSchedule(DiscreteNoiseSchedule):
    """Cosine schedule from https://arxiv.org/pdf/2102.09672.pdf, eq 17"""

    def __init__(
        self,
        *,
        n_steps: int = 300,
        offset: float = 0.008,
        max_beta: float = 0.999,
        min_beta: float = 1e-4,
    ):
        self.offset = offset
        self.max_beta = max_beta
        self.min_beta = min_beta
        super().__init__(n_steps=n_steps)

    def setup_schedule(self) -> None:
        t = torch.arange(self.n_steps + 1)
        t_max = t[-1]

        _angle = ((t / t_max) + self.offset) / (1.0 + self.offset)
        _f = torch.cos(_angle * torch.pi * 0.5) ** 2
        alphas_cumprod = _f / _f[0]

        self.register_buffer("alpha_bars", alphas_cumprod[1:])

        betas = torch.zeros(self.n_steps)
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.register_buffer("betas", torch.clip(betas, min=self.min_beta, max=self.max_beta))
        self.register_buffer("alphas", 1.0 - self.betas)


class QuadraticBetaSchedule(DiscreteNoiseSchedule):
    """Quadratic schedule as in https://huggingface.co/blog/annotated-diffusion"""

    def __init__(self, *, n_steps: int = 300, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.beta_start = beta_start
        self.beta_end = beta_end
        super().__init__(n_steps=n_steps)

    def setup_schedule(self) -> None:
        self.register_buffer(
            "betas", torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.n_steps) ** 2
        )
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_bars", self.alphas.cumprod(dim=0))


class SigmoidBetaSchedule(DiscreteNoiseSchedule):
    """Sigmoid schedule as in https://huggingface.co/blog/annotated-diffusion"""

    def __init__(self, *, n_steps: int = 300, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.beta_start = beta_start
        self.beta_end = beta_end
        super().__init__(n_steps=n_steps)

    def setup_schedule(self) -> None:
        self.register_buffer(
            "betas",
            torch.sigmoid(torch.linspace(-6, 6, self.n_steps)) * (self.beta_end - self.beta_start)
            + self.beta_start,
        )
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_bars", self.alphas.cumprod(dim=0))


class HoogeboomSchedule(DiscreteNoiseSchedule):
    """Noise schedule from Hoogeboom et al. https://arxiv.org/pdf/2203.17003.pdf, appendix B
    https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/fce07d701a2d2340f3522df588832c2c0f7e044a/equivariant_diffusion/en_diffusion.py#L38
    """

    def __init__(
        self,
        *,
        n_steps: int = 1000,
        s: float = 1e-4,
        clip_value: float = 0.001,
        exponent: float = 3,
    ):
        self.s = s
        self.n_steps = n_steps
        self.clip_value = clip_value
        self.exponent = exponent

        super().__init__(n_steps=n_steps, validate=False)

    @staticmethod
    def clip_noise_schedule(alpha_bars, clip_value: float = 0.001):
        alpha_bars = torch.cat([torch.ones(1), alpha_bars], axis=0)
        alphas = alpha_bars[1:] / alpha_bars[:-1]
        alphas = torch.clip(alphas, min=clip_value, max=1.0)

        return alphas, alphas.cumprod(dim=0)

    def setup_schedule(self) -> None:
        t = torch.linspace(0, self.n_steps, self.n_steps)
        alpha_bars = (1.0 - torch.pow(t / (self.n_steps), self.exponent)) ** 2

        alpha_bars = alpha_bars * (1.0 - 2 * self.s) + self.s
        alphas, alpha_bars = self.clip_noise_schedule(alpha_bars, clip_value=self.clip_value)

        self.register_buffer("betas", 1.0 - alphas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)