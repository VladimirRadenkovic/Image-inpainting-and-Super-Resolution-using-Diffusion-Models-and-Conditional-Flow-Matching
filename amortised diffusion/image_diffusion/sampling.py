import math
from plum import dispatch, Union
import numpy as np

import torch
from torch.func import vmap, grad

from .sde_diffusion import Network, DDPM, extract
from .conditioning import ReconstructionGuidance, Amortized, Conditioning, Replacement
from .likelihoods import Likelihood


def process_x0(img):
    return torch.clip(img, -1, 1)


@dispatch
def _get_x0_model(eps_model: Network, ddpm: DDPM, conditioning: Union[ReconstructionGuidance, Replacement], likelihood: Likelihood):
    # print("X0 Reconstruction Guidance sampling")

    def x0_model(xi, i, cond=None):
        assert cond is None
        noise_hat = eps_model(xi, i)
        x0_hat = ddpm.predict_start_from_noise(xi, i, noise_hat)
        return process_x0(x0_hat)
    
    return x0_model


@dispatch
def _get_x0_model(eps_model: Network, ddpm: DDPM, conditioning: Amortized, likelihood: Likelihood):
    print("X0 amortized sampling")

    def x0_model(xi, i, cond=None):

        if cond is None:
            cond = likelihood.none_like(xi)

        xi_condition = torch.concat((xi, cond), axis=-3)  # concat across channels
        noise_hat = eps_model(xi_condition, i)
        x0_hat = ddpm.predict_start_from_noise(xi, i, noise_hat)
        return process_x0(x0_hat)
    
    return x0_model


# Prior sampling
# ----------------------

def get_prior_sample_fn(eps_model: Network, ddpm: DDPM, conditioning: Conditioning, likelihood: Likelihood):
    print("Prior sampling")

    @torch.no_grad()
    def sample(xT):
        x0_model = _get_x0_model(eps_model, ddpm, conditioning, likelihood)
        xi = xT
        xs = [xi]

        def step(xi, i):
            b = xi.shape[0]
            batched_times = torch.full((b,), i, device=xi.device, dtype=torch.long)
            x0_pred = x0_model(xi, batched_times)
            model_mean, variance, model_log_variance, x_start = ddpm.p_mean_variance(x0_pred, x=xi, i=batched_times)
            noise = torch.randn_like(xi) if i > 0 else 0.0  # no noise if t == 0
            scale = (0.5 * model_log_variance).exp()
            pred_img = model_mean + scale * noise
            return pred_img, x_start

        for i in reversed(range(ddpm.Ns)):
            xi, _ = step(xi, i)
            xs.append(xi)

        return process_x0(xi)
    
    return sample

# Conditional sampling
# ----------------------

@dispatch
def get_conditional_sample_fn(eps_model: Network, ddpm: DDPM, conditioning: Amortized, likelihood: Likelihood):
    print("Amortized conditional sampling")

    @torch.no_grad()
    def sample(xT, condition):
        x0_model = _get_x0_model(eps_model, ddpm, conditioning, likelihood)
        xi = xT
        xs = [xi]

        def step(xi, i):
            b = xi.shape[0]
            batched_times = torch.full((b,), i, device=xi.device, dtype=torch.long)
            x0_pred = x0_model(xi, batched_times, condition)  # we now pass condition to noise model
            model_mean, variance, model_log_variance, x_start = ddpm.p_mean_variance(x0_pred, x=xi, i=batched_times)
            noise = torch.randn_like(xi) if i > 0 else 0.0  # no noise if t == 0
            scale = (0.5 * model_log_variance).exp()
            pred_img = model_mean + scale * noise
            return pred_img, x_start
        
        def em_step(xi,i):
            b = xi.shape[0]
            batched_times = torch.full((b,), i, device=xi.device, dtype=torch.long)
            xi_condition = torch.concat((xi, condition), axis=-3) 
            noise_hat = eps_model(xi_condition, batched_times)
            score_fn = ddpm.score_from_noise
            drift = ddpm.backward_drift(score_fn, xi, noise_hat, batched_times)
            diffusion = ddpm.backward_diffusion(batched_times).to(xi.device)
            dt = 1/ddpm.Ns
            z = torch.randn_like(xi, device = xi.device)
            x = xi - dt*drift + diffusion.unsqueeze(1).unsqueeze(2).unsqueeze(3)*z*np.sqrt(dt)
            return x, xi

        def corrector_step(xi, i):
            b = xi.shape[0]
            batched_times = torch.full((b,), i, device=xi.device, dtype=torch.long)
            score = ddpm.score_from_x0(x0_model(xi, batched_times), batched_times)
            dt = (ddpm.tmax - ddpm.tmin) / ddpm.Ns
            drift = 0.5 * dt * conditioning.delta * score
            noise = math.sqrt(dt * conditioning.delta) * torch.randn_like(xi)
            xi += drift + noise
            return xi

        for i in reversed(range(ddpm.Ns)):
            xi, _ = step(xi, i)

            for _ in range(conditioning.n_corrector):
                xi = corrector_step(xi, i)

            xs.append(xi)

        return process_x0(xi)
    
    return sample


@dispatch
def get_conditional_sample_fn(eps_model: Network, ddpm: DDPM, conditioning: ReconstructionGuidance, likelihood: Likelihood):
    print("Reconstruction guidance conditional sampling")

    @torch.no_grad()
    def sample(xT, condition):
        x0_model = _get_x0_model(eps_model, ddpm, conditioning, likelihood)
        xi = xT
        xs = [xi]
        
        def step(xi, i):
            b = xi.shape[0]
            batched_times = torch.full((b,), i, device=xi.device, dtype=torch.long)

            # conditioning
            x_update = 0.0
            if i < int(ddpm.Ns * conditioning.start_fraction):

                def constraint(xi, i, y):
                    xi = xi.unsqueeze(0)  # [1, C, H, W]
                    i = i.unsqueeze(0)  # [1,]
                    y = y.unsqueeze(0)  # [1, C, H, W]
                    x0 = x0_model(xi, i)  # [1, C, H, W]
                    loss = likelihood.loss(x0, y)  # [1]
                    return loss.squeeze(0) # []
            
                xi_ = xi.detach().clone().requires_grad_()
                x_grad = vmap(grad(constraint, argnums=(0)))(xi_, batched_times, condition)  # [N, ...]

                # Computing scaling as in protein motif-scaffolding
                alpha_i = ddpm.alphas[i]
                scale = conditioning.gamma * alpha_i * (1 - alpha_i)

                # conditional update
                x_update = - scale * x_grad
                if conditioning.update_rule == "before":
                    xi += x_update

            # Computing x0_pred after updating xi using the conditional score
            # seems not to match with the math, where the score is computed w.r.t. xi
            # and not xi + x_update
            x0_pred = x0_model(xi, batched_times)
            model_mean, variance, model_log_variance, x_start = ddpm.p_mean_variance(x0_pred, x=xi, i=batched_times)
            noise = torch.randn_like(xi) if i > 0 else 0.0  # no noise if t == 0
            scale = (0.5 * model_log_variance).exp()
            pred_img = model_mean + scale * noise

            if conditioning.update_rule == "after":
                pred_img += x_update

            return pred_img, x_start
        
        def corrector_step(xi, i):
            b = xi.shape[0]
            batched_times = torch.full((b,), i, device=xi.device, dtype=torch.long)
            score = ddpm.score_from_x0(x0_model(xi, batched_times), batched_times)
            dt = (ddpm.tmax - ddpm.tmin) / ddpm.Ns
            drift = 0.5 * dt * conditioning.delta * score
            noise = math.sqrt(dt * conditioning.delta) * torch.randn_like(xi)
            xi += drift + noise
            return xi

        for i in reversed(range(ddpm.Ns)):
            xi, _ = step(xi, i)
            for _ in range(conditioning.n_corrector):
                xi = corrector_step(xi, i)
            xs.append(xi)

        return process_x0(xi)
    
    return sample


@dispatch
def get_conditional_sample_fn(eps_model: Network, ddpm: DDPM, conditioning: Replacement, likelihood: Likelihood):
    print("Replacement conditional sampling")

    @torch.no_grad()
    def sample(xT, condition):
        # print('condition == likelihood.pad_value', (condition == likelihood.pad_value).sum().item(), np.prod(condition.size()), (condition == likelihood.pad_value).sum().item() / np.prod(condition.size()))
        x0_model = _get_x0_model(eps_model, ddpm, conditioning, likelihood)
        xi = xT
        xs = [xi]

        def predictor_step(xi, i):
            b = xi.shape[0]
            batched_times = torch.full((b,), i, device=xi.device, dtype=torch.long)

            # replacement
            if i < int(ddpm.Ns * conditioning.start_fraction):
                # noise condition
                if conditioning.noise:
                    noised_condition, _ = ddpm.q_sample(condition, batched_times)
                else:
                    noised_condition = condition
                # concatenate noise condition with current state
                xi = torch.where(condition == likelihood.pad_value, xi, noised_condition)

            x0_pred = x0_model(xi, batched_times)
            model_mean, variance, model_log_variance, x_start = ddpm.p_mean_variance(x0_pred, x=xi, i=batched_times)
            noise = torch.randn_like(xi) if i > 0 else 0.0  # no noise if t == 0
            scale = (0.5 * model_log_variance).exp()
            pred_img = model_mean + scale * noise
            return pred_img, x_start

        def corrector_step(xi, i):
            b = xi.shape[0]
            batched_times = torch.full((b,), i, device=xi.device, dtype=torch.long)

            score = ddpm.score_from_x0(x0_model(xi, batched_times), batched_times)
            dt = (ddpm.tmax - ddpm.tmin) / ddpm.Ns
            drift = 0.5 * dt * conditioning.delta * score
            noise = math.sqrt(dt * conditioning.delta) * torch.randn_like(xi)
            xi += drift + noise
            return xi

        for i in reversed(range(ddpm.Ns)):
            xi, _ = predictor_step(xi, i)
            for _ in range(conditioning.n_corrector):
                xi = corrector_step(xi, i)
            xs.append(xi)

        return process_x0(xi)
    
    return sample
