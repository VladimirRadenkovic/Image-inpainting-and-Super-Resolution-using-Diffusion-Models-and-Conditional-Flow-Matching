import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tqdm
import numpy as np
import torch
import torch.nn.functional as F

import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from functorch import vmap, grad, make_functional
import matplotlib.pyplot as plt

def plot_image_and_patch(image, patch, slice_, ax=None):
    """
    image: [C, H, W]
    patch: [C, h, w]
    """
    assert image.shape[0] == 1
    if ax is None:
        fig, ax = plt.subplots()
    ax.axis('off')
    ax.matshow(image[0], cmap="gray_r")
    patch_with_mask = np.ones_like(image) * np.nan
    patch_with_mask[slice_] = patch
    ax.imshow(patch_with_mask[0], cmap='Reds', alpha=.8)


def plot_image_grid(images, patches=None, slice_=None):
    """
    images: [N, 1, C, H]
    """
    n = int(len(images) ** .5)
    fig, axes = plt.subplots(n, n, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        if patches is None:
            ax.imshow(images[i][0].cpu(), cmap='gray_r')
        elif patches is not None and slice_ is not None:
            plot_image_and_patch(images[i].cpu(), patches[i].cpu(), slice_, ax)
        ax.axis('off')
    # plt.tight_layout()
    return fig, axes

# %%

# Define a transform to convert images to tensor and normalize them to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST dataset for training
batch_size = 64
train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# %%

# Plot data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch = next(iter(train_loader))
images = batch[0].to(device)

plot_image_grid(images.cpu())

# %%
# Initialize network
#from image_diffusion.sampling import euler_maruyama_integrate_loop
from image_diffusion.sde_diffusion import unsqueeze_like
@torch.inference_mode()
def euler_maruyama_integrate_loop(drift, diffusion, x0, ts):
    """Integrate diffusion SDE with euler maruyama integrator."""

    t12s = torch.stack([ts[:-1], ts[1:]], -1)

    x = x0
    traj = [x]
    for j, (t1, t2) in enumerate(t12s):
        i = len(ts) - j - 1
        print(i, t1.item(), t2.item())
        t1_ = t1 * torch.ones((len(x),), device=x.device)
        f = drift(x, t1_)
        if i > 1: # NOTE: do not add noise in the last step
            noise = torch.randn_like(x0)
        else:
            noise = torch.zeros_like(x0)
        g = unsqueeze_like(noise, diffusion(t1))
        x = x + (t2 - t1) * f + g * noise * torch.sqrt(torch.abs(t1 - t2))
        if x.isnan().any():
            raise ValueError(f"NaN encountered in trajectory during {t1} -> {t2}")
        traj.append(x)

    return x, torch.stack(traj)

from image_diffusion.sde_diffusion import VPSDE, DDPM
diffusion = DDPM(100)
N = diffusion.Ns
x, xs = euler_maruyama_integrate_loop(
    drift=diffusion.drift,
    diffusion=diffusion.diffusion,
    x0=images,
    ts=torch.linspace(diffusion.tmin, diffusion.tmax, N, device=device)
)

indices = [0, 10, 25, 50, 99]
fig, axes = plt.subplots(1, len(indices), figsize=(10, 3))
for i, ax in enumerate(axes):
    axes[i].imshow(xs[indices[i], 0, 0].cpu())
    axes[i].axis('off')
# %%
from image_diffusion.unet import create_model
model = create_model(
    image_size=28,
    in_channels=1,
    out_channels=1,
    #num_classes=None,  # no class conditioning
    num_channels=32,
    num_res_blocks=1,
    channel_mult="1, 2, 2",
    resblock_updown=True,
    model_path=os.getcwd() + "/model_weights.pth"
    # model_path=os.getcwd() + "/weights/unet_mnist_score.pth"
).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params / 1e6:.2f}M")
# %%
from image_diffusion.loss_functions import get_loss_function
#from image_diffusion.sde_diffusion import get_loss_fn, get_score_fn, DDPM, get_ddpm_loss_fn
from image_diffusion.likelihoods import get_likelihood
from image_diffusion.conditioning import get_conditioning

# score_fn = get_score_fn(model, diffusion)
# loss_fn = get_loss_fn(score_fn, diffusion)
# loss_fn(batch[0].to(device))
likelihood = get_likelihood("inpainting")(patch_size=14, pad_value=-2)
conditioning = get_conditioning("reconstruction_guidance")(gamma=10.0, start_fraction=1.0,
                                                           update_rule="before",
                                                           n_corrector = 0,
                                                           delta = 0.1)
loss_fn, eps_model = get_loss_function(model, diffusion, conditioning, likelihood)

def train(num_epochs=25, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    num_batches = len(train_loader)
    for e in range(num_epochs):
        pbar = tqdm.tqdm(total=num_batches)
        for i, (X, _) in enumerate(train_loader):
            X = X.to(device)
            opt.zero_grad()
            loss_value = loss_fn(X)
            loss_value.backward()
            opt.step()
            losses.append(loss_value.item())
            pbar.update(1)
        print(f"Epoch {e} - {np.mean(losses[-100:]):.2f}")

    torch.save(model.state_dict(), 'model_weights.pth')
    return losses

# losses = train()
# plt.plot(losses)
# plt.savefig("loss.png")
# exit(0)

# %%
from image_diffusion.sde_diffusion import extract

score_fn = diffusion.get_score_fn(eps_model, conditioning)

ts = torch.linspace(diffusion.tmax, diffusion.tmin, 500).to(device)
# Reverse solve Probability Flow
x, xs = euler_maruyama_integrate_loop(
    drift=lambda x, t: diffusion.backward_drift(score_fn, x, t),
    diffusion=diffusion.backward_diffusion,
    # diffusion=lambda *args: 0.0,
    x0=torch.randn_like(images),
    ts=ts
)
# x, xs = heun_integrate_loop(
#     dynamics=lambda x, t: diffusion.backward_dynamics(score_fn, x, t),
#     x0=torch.randn_like(images),
#     ts=ts
# )

# %%
plot_image_grid(x.cpu());
# %%

from image_diffusion.sde_diffusion import unsqueeze_like

# @torch.inference_mode()
def conditioned_scores(
        xt: torch.Tensor, t, y, slice_, alpha=10, overwrite_observed_score: bool = True,
    ):
    """
    xt: [N, C, H, W]
    t: [N]
    y: [N, C, h, w], in our case these will be patches
    slice_: the following must hold
        xt[:, slice_].shape == y.shape
    """
    def constraint(x, t, y):
        x = x.unsqueeze(0)  # [1, ...]
        t = t.unsqueeze(0)  # [1,]
        x0 = diffusion.denoise_input(score_fn, x, t)[0]  # [C, H, W]
        sliced = x0[slice_]  # [C, h, w]
        return torch.sum((sliced - y)**2)  # []
    
    mean_scale = diffusion.scale(t)  # [N,]
    sigma = diffusion.sigma(t)  # [N,]
    # sigma += 1e-6
    scale = alpha * mean_scale**2 / sigma ** 2  # [N,]
    scale = unsqueeze_like(xt, scale)  # [N, ...]

    xt_ = xt.detach().clone().requires_grad_()
    score_observed = vmap(grad(constraint, argnums=(0)))(xt_, t, y)  # [N, ...]
    score = score_fn(xt, t) - scale * score_observed  # [N, ...]

    if overwrite_observed_score:
        # replace score at observed indices by 'true' score,
        # which can be calculated as we have access to the true
        # values at t=0.
        yt = xt[(np.s_[:], *slice_)]
        observed_score = diffusion.noise_score(yt, y, t)  # [N, C, h, w]
        score[(np.s_[:], *slice_)] = observed_score
    return score


IMAGE_WIDTH = IMAGE_HEIGHT = 28
PATCH_HEIGHT = PATCH_WIDTH = 15

def get_random_patch_slices(H=IMAGE_HEIGHT, W=IMAGE_WIDTH, h=PATCH_HEIGHT, w=PATCH_WIDTH):
    """
    Get random slice object to extract a patch of size [h, w]
    from an image assuming CHW ordering.
    """
    h_start = np.random.randint(0, H - h + 1)
    w_start = np.random.randint(0, W - w + 1)
    return np.s_[:, h_start:h_start+h, w_start:w_start+w]


patch_slice = get_random_patch_slices()
patches = images[(np.s_[:], *patch_slice)]
plot_image_grid(images, patches, patch_slice);

def conditional_sample(alpha, overwrite_observed_score):
    cond_score_fn = lambda x, t: conditioned_scores(x, t, patches, patch_slice, alpha=alpha, overwrite_observed_score=overwrite_observed_score)
    """
    x, _ = euler_maruyama_integrate_loop(
        drift=lambda x, t: diffusion.backward_drift(cond_score_fn, x, t),
        diffusion=diffusion.backward_diffusion,
        # diffusion=lambda *args: 0.0,
        x0=torch.randn_like(images),
        ts=ts
    )
    return x
    """
    return 0


# %%[markdown]
# ## Overwriting the observed score

# In the conditional score we replace the score at the observed indices
# with the 'true' score.

# %%
for alpha in [0.0, 0.1, 1, 10]:
    samples = conditional_sample(alpha, overwrite_observed_score=True)
    fig, _ = plot_image_grid(samples, patches, patch_slice)
    fig.suptitle(f"$\\alpha={alpha}$");
# %%

# ## 'Pure' reconstruction guidance

# following Finzi et al.

# %%
for alpha in [0.0, 0.1, 1, 10]:
    samples = conditional_sample(alpha, overwrite_observed_score=False)
    fig, _ = plot_image_grid(samples, patches, patch_slice)
    fig.suptitle(f"$\\alpha={alpha}$");