import copy

import torch
import torchdiffeq
from torchdyn.core import NeuralODE
import numpy as np
# from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



def downsample_images(images, target_size):
    """
    Downsample a batch of images to a specified size.

    :param images: A batch of images in the shape [batch_size, channels, height, width].
    :param target_size: A tuple (target_height, target_width) for the output size.
    :return: Downsampled images.
    """
    downsampled_images = F.interpolate(images, size=target_size, mode='bilinear', align_corners=False)
    #upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
    return downsampled_images


def generate_samples(model, parallel, savedir, step, net_="normal"):
    """Save 64 generated images (8 x 8) for sanity check along training.

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    """
    model.eval()

    model_ = model

    node_ = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(64, 1, 28, 28, device=device),
            t_span=torch.linspace(0, 1, 100, device=device),
        )
        traj = traj[-1, :].view([-1, 1, 28, 28]).clip(-1, 1)
        traj = traj / 2 + 0.5
    save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)

    model.train()


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x
"""
def generate_samples_eval(model, test_images, batch_size = 8, step=0, net_="normal"):
    model.eval()
    with torch.no_grad():
        def ode_func(t, x):
            nonlocal nfe
            nfe +=1
            return (model.forward(x[0], t, low_res=x[1]), x[1])
        low_res_size = (7, 7)
        low_res = downsample_images(test_images, low_res_size).to(device)
        x_0 = torch.randn(32, 1, 28, 28, device=device)
        nfe = 0
        traj = torchdiffeq.odeint(
                ode_func,
                (x_0.to(device),low_res),
                torch.linspace(0, 1, 2, device=device),
                atol=1e-4,
                rtol=1e-4,
                method="dopri5",
            )
    model.train()
    return traj, low_res, nfe
"""
"""
def generate_samples_eval(model, test_images, batch_size = 8, step=0, net_="normal"):
    model.eval()
    def ode_func(t, x, args):
            nonlocal nfe
            nfe +=1
            x_t = model.forward(x[:,0].unsqueeze(1), t, con=x[:,1].unsqueeze(1))
            return torch.cat((x_t, x[:,1].unsqueeze(1)), dim=1)
    node_ = NeuralODE(ode_func, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        con = downsample_images(test_images).to(device)
        x_0 = torch.randn(batch_size, 1, 28, 28, device=device)
        nfe = 0

        traj = node_.trajectory(
            torch.cat((x_0.to(device), con), dim=1),
            t_span=torch.linspace(0, 1, 1000, device=device),
        )
        traj = traj[-1, :][:,0,:,:].view([-1, 1, 28, 28]).clip(-1, 1)

    model.train()
    return traj, con, nfe
    """

def generate_samples_eval(model, test_images, batch_size = 8, step=0, net_="normal"):
    model.eval()
    def ode_func(t, x, args):
            nonlocal nfe
            nfe +=1
            low_res_size = (7, 7)
            low_res = downsample_images(x, low_res_size).to(device)
            x_t = model.forward(x, t, low_res=low_res)
            return x_t
    node_ = NeuralODE(ode_func, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        low_res_size = (7, 7)
        low_res = downsample_images(test_images, low_res_size).to(device)
        x_0 = torch.randn(batch_size, 1, 28, 28, device=device)
        nfe = 0

        traj = node_.trajectory(
           x_0.to(device),
            t_span=torch.linspace(0, 1, 1000, device=device),
        )
        traj = traj[-1, :][:,0,:,:].view([-1, 1, 28, 28]).clip(-1, 1)
        #traj = traj[0][-1, :].view([-1, 3, 64, 64]).clip(-1, 1)
    model.train()
    return traj, low_res, nfe
"""

def generate_samples_eval(model, test_images, batch_size = 8, step=0, net_="normal"):
    model.eval()
    with torch.no_grad():
        def ode_func(t, x):
            nonlocal nfe
            nfe +=1
            return (model.forward(x[0], t, low_res=x[1]), x[1])
        low_res_size = (7, 7)
        low_res = downsample_images(test_images, low_res_size).to(device)
        x_0 = torch.randn(batch_size, 1, 28, 28, device=device)
        nfe = 0
        traj = torchdiffeq.odeint(
                ode_func,
                (x_0.to(device),low_res),
                torch.linspace(0, 1, 2, device=device),
                atol=1e-4,
                rtol=1e-4,
                method="dopri5",
            )
        traj = traj[0][-1, :][:,0,:,:].view([-1, 1, 28, 28]).clip(-1, 1)
    model.train()
    return traj, low_res, nfe

"""