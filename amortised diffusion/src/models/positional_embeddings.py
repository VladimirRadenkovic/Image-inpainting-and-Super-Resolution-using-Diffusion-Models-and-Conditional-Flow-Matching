from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class TemporalEncoding(nn.Module):
    def __init__(self, *, embed_dim: int, max_steps: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_steps = max_steps
        self.setup_encoding()
        self.check_validity()

    @abstractmethod
    def setup_encoding(self) -> None:
        ...

    def forward(self, timesteps: torch.LongTensor) -> torch.Tensor:
        return self.pos_encoding[timesteps]

    def check_validity(self):
        assert self.pos_encoding.ndim == 2, "Positional encoding should be 2D"
        assert (
            self.pos_encoding.shape[0] == self.max_steps
        ), "Positional encoding should have `max_steps` rows"
        assert (
            self.pos_encoding.shape[1] == self.embed_dim
        ), "Positional encoding should have `embed_dim` columns"
        assert (self.pos_encoding >= -1).all() and (
            self.pos_encoding <= 1
        ).all(), "Positional encoding should be in [-1, 1]"

    def __repr__(self):
        return f"{self.__class__.__name__}(embed_dim={self.embed_dim}, max_steps={self.max_steps})"

    def plot(self):
        for embed_dim in range(self.pos_encoding.shape[1]):
            plt.plot(self.pos_encoding[:, embed_dim], label=f"dim {embed_dim}")


class SinusoidalEncoding(TemporalEncoding):
    """Sinusoidal positional encoding from the original Transformer paper
    https://arxiv.org/pdf/1706.03762.pdf (section 3.5)"""

    def setup_encoding(self) -> None:
        # Compute the positional encodings once in log space.
        pos_encoding = torch.zeros(self.max_steps, self.embed_dim)
        position = torch.arange(0, self.max_steps).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2) * -(np.log(self.max_steps) / self.embed_dim)
        )
        pos_encoding[:, 0::2] = torch.sin(position * div_term)  # even terms
        pos_encoding[:, 1::2] = torch.cos(position * div_term)  # odd terms
        # make positional encoding unlearnable
        self.register_buffer("pos_encoding", pos_encoding)  # [max_steps, embed_dim]


class FairSeqSinusoidalEncoding(TemporalEncoding):
    """seq2seq positional encoding from fairseq, which is a bit different from the original
    Transformer paper, https://arxiv.org/pdf/1706.03762.pdf (section 3.5)"""

    def setup_encoding(self) -> None:
        # Compute the positional encodings once in log space.
        half_dim = self.embed_dim // 2
        position = torch.arange(0, self.max_steps).float()
        pos_encoding = torch.exp(
            torch.arange(0, half_dim) * -(np.log(self.max_steps) / (half_dim - 1))
        )  # [half_dim]
        pos_encoding = position.unsqueeze(1) * pos_encoding.unsqueeze(0)  # [max_steps, half_dim]
        pos_encoding = torch.cat(
            [torch.sin(pos_encoding), torch.cos(pos_encoding)], dim=1
        )  # [max_steps, 2*half_dim]
        # Concat sin and cos terms along embedding dimension
        if self.embed_dim % 2 == 1:
            pos_encoding = torch.cat(
                [pos_encoding, torch.zeros(self.max_steps, 1)], dim=1
            )  # [max_steps, embed_dim]
        # make positional encoding unlearnable
        self.register_buffer("pos_encoding", pos_encoding)  # [max_steps, embed_dim]


class GaussianFourierProjection(TemporalEncoding):
    """Gaussian Fourier embeddings for noise levels from
    https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, *, embed_dim: int, max_steps: int, scale: float = 1.0):
        self.scale = scale
        super().__init__(embed_dim=embed_dim, max_steps=max_steps)

    def setup_encoding(self) -> None:
        self.weights = torch.randn(self.embed_dim // 2) * self.scale
        position = torch.arange(0, self.max_steps).float()
        pos_encoding = position.unsqueeze(1) * self.weights.unsqueeze(0) * 2 * np.pi
        pos_encoding = torch.cat([torch.sin(pos_encoding), torch.cos(pos_encoding)], dim=1)
        # Concat sin and cos terms along embedding dimension
        if self.embed_dim % 2 == 1:
            pos_encoding = torch.cat(
                [pos_encoding, torch.zeros(self.max_steps, 1)], dim=1
            )  # [max_steps, embed_dim]
        # make positional encoding unlearnable
        self.register_buffer("pos_encoding", pos_encoding)  # [max_steps, embed_dim]