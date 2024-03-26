from abc import ABC, abstractmethod

import torch
from torch import nn

from src.utils.torch_utils import inflate_batch_array


class Denoiser(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, ts, timesteps, mask):
        raise NotImplemented


#  TODO: ENGNN denoiser


class MLPDenoiser(Denoiser):
    def __init__(
        self,
        num_layers=5,
        emb_dim=128,
        in_dim=1,
        pos_dim=2,
    ):
        super().__init__()

        self.dummy_emb = nn.Linear(pos_dim + 1, emb_dim)

        # Stack of simple linear layers
        self.linears = nn.ModuleList()
        for layer in range(num_layers - 2):
            self.linears.append(nn.Linear(emb_dim, emb_dim))
            self.linears.append(nn.LeakyReLU())

        self.linears.append(nn.Linear(emb_dim, 2))

        self.emb_dim = emb_dim
        self.requires_normalized_times = True

    def forward(self, cur_pos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        n_atoms = cur_pos.shape[-2]

        t = inflate_batch_array(t, target=cur_pos)  # [batch_size, ...] where all ... are 1
        t = t.expand(-1, n_atoms, -1)  # [batch_size, n_atoms, 1]

        pos_t = torch.cat([cur_pos, t], -1)
        pos_t = self.dummy_emb(pos_t)

        for linear in self.linears:
            pos_t = linear(pos_t)

        # Like Hoogeboom, predict noise is new positions minus old positions (this ignores our alpha bar factors)
        pred_pos_noise = pos_t - cur_pos

        # In Hoobeboom, they subtract center of gravity of the predicted noise
        pred_pos_noise = pred_pos_noise - torch.mean(pred_pos_noise, dim=-2, keepdim=True)

        return pred_pos_noise
