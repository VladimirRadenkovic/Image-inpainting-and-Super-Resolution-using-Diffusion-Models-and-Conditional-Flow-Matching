import copy

import torch
import torchdiffeq
from torchdyn.core import NeuralODE
import numpy as np
# from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



def get_random_patch(image_size=28, patch_size=14):
        # don't sample to close to border.
    h = torch.randint(5, image_size - patch_size - 5, size=())
    w = torch.randint(5, image_size - patch_size - 5, size=())
    return h, w


def _sample(images, pad_value=2, patch_size=14):
        """
        images: [N, C, H, W]
        """
        image_size = images.shape[-1]
        pad_value = -2
        patch_size = 14
        h, w = get_random_patch(image_size, patch_size)
        condition = images.detach().clone()
        slice_ = np.s_[:, :, h : h + patch_size, w : w + patch_size]
        condition[slice_] = pad_value
        return condition

def sample(x):
    samples = []
    for k in range(x.shape[0]):
        e = _sample(x[[k]])
        samples.append(e)
    return torch.cat(samples, dim=0)



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

def generate_samples_eval(model, test_images, savedir, batch_size = 8, step=0, net_="normal"):
    model.eval()
    
    # Concatenate x with con
    #x_0 = torch.cat((x_0, con), dim=1)
    with torch.no_grad():
        def ode_func(t, x):
            return (model.forward(x[0], t, con=x[1]), x[1])
        
        con = sample(test_images).to(device)
        x_0 = torch.randn(batch_size, 1, 28, 28, device=device)
        traj = torchdiffeq.odeint(
                ode_func,
                (x_0.to(device), con),
                torch.linspace(0, 1, 2, device=device),
                atol=1e-4,
                rtol=1e-4,
                method="dopri5",
            )
        traj = traj[0][-1, :][:,0,:,:].view([-1, 1, 28, 28]).clip(-1, 1)
        #traj = traj / 2 + 0.5
    """
    with torch.no_grad():
        traj = torchdiffeq.odeint(
                model.forward,
                x_0.to(device),
                torch.linspace(0, 1, 2, device=device),
                atol=1e-4,
                rtol=1e-4,
                method="dopri5",
            )
        traj0 = traj[-1, :][:,0,:,:].view([-1, 1, 28, 28]).clip(-1, 1)
        traj0 = traj / 2 + 0.5
     """
        
        #grid1 = make_grid(
            #traj[-1, :][:,0,:,:].view([-1, 1, 28, 28]).clip(-1, 1), value_range=(-1, 1), padding=0, nrow=1)
        #img1 = ToPILImage()(grid1)
        #plt.figure()
        #plt.imshow(img1)
        #plt.show()
        #plt.savefig('results/otcfm/mnist/images/traj0.png')
        
    #save_image(traj0, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)
    model.train()
    return traj, con