from typing import Mapping
from plum import dispatch
from einops import reduce
import torch.nn.functional as F

import torch

from .conditioning import Conditioning, ReconstructionGuidance, Amortized
from .likelihoods import Likelihood
from .sde_diffusion import DDPM, Network


@dispatch
def get_loss_function(network: Network, ddpm: DDPM, conditioning: Conditioning, likelihood: Likelihood):
    """Most generic loss function routine."""
    print("Traditional training")

    def eps_model(xi, i):
        return network(xi, 1.0 * i / ddpm.Ns)

    def loss(x):
        device = x.device
        b = x.shape[0]
        i = torch.randint(0, ddpm.Ns, (b,), device=device).long()
        xi, noise = ddpm.q_sample(x, i)
        noise_hat = eps_model(xi, i)
        loss = F.mse_loss(noise_hat, noise, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")
        # TODO(Vincent): Add weighting
        # loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()
    
    return loss, eps_model


@dispatch
def get_loss_function(network: Network, ddpm: DDPM, conditioning: Amortized, likelihood: Likelihood):
    """Loss for amortized training where we pass a sample from the likelihood as input to the network."""

    print("Amortized training")

    def eps_model(xi, i):
        return network(xi, 1.0 * i / ddpm.Ns)

    def loss(x):

        if torch.rand(()) < conditioning.p_cond:
            condition = likelihood.sample(x)
        else:
            condition = likelihood.none_like(x)

        device = x.device
        b = x.shape[0]
        i = torch.randint(0, ddpm.Ns, (b,), device=device).long()
        xi, noise = ddpm.q_sample(x, i)
        xi_condition = torch.concat((xi, condition), axis=-3)  # concat across channels
        noise_hat = eps_model(xi_condition, i)
        loss = F.mse_loss(noise_hat, noise, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")
        return loss.mean()
    
    return loss, eps_model

