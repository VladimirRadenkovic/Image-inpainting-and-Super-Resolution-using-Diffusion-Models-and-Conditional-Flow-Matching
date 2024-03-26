# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import os
import json
import pathlib

import torch
from absl import app, flags
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import trange
from utils_mnist_hy2 import ema, generate_samples, infiniteloop, generate_samples_eval
#from ema_pytorch import EMA
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import lpips
from torch.utils.data import DataLoader
from data import get_dataset
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper, InPaintModelWrapper, SuperResModelWrapper
from actions import PeriodicCallback
from writers import LocalWriter
from data import get_dataset

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 32, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 0.001, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 10, help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 32, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    2000,
    help="frequency of saving checkpoints, 0 to disable during training",
)

def get_experiment_name(config):
    # now = datetime.datetime.now().strftime("%b%d_%H%M%S")
    experiment_name = "_".join([
        config.dataset.name,
        config.conditioning.name,
        config.likelihood.name,
    ])
    return experiment_name


def get_experiment_dir(experiment_name: str):
    """
    Creates and returns directory for an experiment.

    Creates a different subdirectory, named v0, v1, ..., for experiments
    with the same name.
    """
    cwd = os.getcwd()
    path = pathlib.Path(cwd, 'experiments', experiment_name)
    path.mkdir(parents=True, exist_ok=True)    
    return path

def to_img(x: torch.Tensor) -> torch.Tensor:
    """From x in [-1, 1] to [0, 1]"""
    x = torch.clamp(x, -1, 1)
    x = (x + 1.) / 2.
    return x

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def plot_condition(x, y, ax):
        ax.imshow(to_imshow(y.squeeze(0)))
        return ax


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup

def to_imshow(x):
    """
    x: [C, H, W] or [H, W, C]
    detects the order by checking if C is 1 or 3.
    Also normalizes the output to be in range [0, 1]
    ignoring nan values.

    outputs [H, W, C]
    """

    def _maybe_transpose(x):
        # x: [C, H, W] or [H, W, C]
        if x.shape[0] in [1, 3]:
            x = np.transpose(x, [1, 2, 0])
        return x

    if hasattr(x, "cpu"):
        x = x.cpu().numpy()

    x = _maybe_transpose(x)
    x -= np.nanmin(x)
    x /= np.nanmax(x)
    return x



    """
    x0, y = sample_conditional()
    fig, axes = plt.subplots(len(x0), 2, figsize=(4, 2*len(x0)))
    for i in range(len(x0)):
        axes[i, 0].imshow(to_imshow(x0[i]))
        likelihood.plot_condition(test_images[i], y[i], axes[i, 1])
        axes[i, 0].axis('off'); axes[i, 1].axis('off');
    fig.tight_layout()
    writer.write_figures(step, dict(condition=fig))
    """




import numpy as np

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



def calculate_psnr(img1, img2):
    # Convert tensors to NumPy arrays
    print('---------------------')
    print(img1.shape)
    print('---------------------')
    print(img2.shape)
    print('---------------------')
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    # Calculate PSNR
    return psnr(img1_np, img2_np, data_range=img2_np.max() - img2_np.min())

def calculate_ssim(img1, img2):
    # Convert tensors to NumPy arrays
    img1_np = img1.detach().cpu().numpy().squeeze(0)
    img2_np = img2.detach().cpu().numpy().squeeze(0)
    # Calculate SSIM
    return ssim(img1_np, img2_np, multichannel=True, data_range=img2_np.max() - img2_np.min())

def eval(net_model, test_dataset):
    lpips_loss_fn = lpips.LPIPS(net='vgg').cuda()
    metrics = {"mse": [], "lpips": [], "psnr": [], "ssim": [], "nfe": []}
    torch.manual_seed(0)
    test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=16)
    test_loader = iter(test_dataloader)
    n = 3
    for batch in tqdm(range(n), desc="Computing test set."):
        batch = next(test_loader)[0].cuda()
        x1, test_condition, nfe = generate_samples_eval(net_model, batch, batch_size = 32, net_="net_model")
        #x1 = x1[0]
        metrics["mse"].append(torch.mean((x1 - batch)**2, dim=(1, 2, 3)))
        metrics["lpips"].append(lpips_loss_fn(x1, batch).squeeze(dim=(1,2,3)))
        metrics["nfe"].append(nfe)

        for i in range(batch.size(0)):
            metrics["psnr"].append(calculate_psnr(x1[i], batch[i]))
            metrics["ssim"].append(calculate_ssim(x1[i], batch[i]))
        
    metrics = {k: torch.tensor(v, dtype=torch.float) if k in ['psnr', 'ssim', 'nfe'] else torch.concat(v, dim=0) for k, v in metrics.items()}
    results = {
        **{f"{k}_mean": torch.mean(v).item() for k, v in metrics.items()},
        **{f"{k}_median": torch.median(v).item() for k, v in metrics.items()},
        **{f"{k}_std": torch.std(v).item() for k, v in metrics.items()},
        }
    return results


            

def fit(net_model, ema_model, flow_model, num_epochs, train_dataloader, test_dataset, optimizer, device, savedir, callbacks,
        lr_scheduler=None):
    step = 0
    net_model.train()
    results_per_step = []
    num_steps = len(train_dataloader) * num_epochs
    every_steps = num_steps // 20
    for epoch in range(num_epochs):
        metrics_epoch = []
        pbar = tqdm(total=len(train_dataloader), desc=f"Epoch: {epoch} / {num_epochs}")
        for batch in train_dataloader:
            step += 1
            x1, _ = batch
            x1 = x1.to(device)
            optimizer.zero_grad()
            x0 = torch.randn_like(x1)
            low_res_size = (7, 7)
            low_res = downsample_images(x1, low_res_size).to(device)
            t, xt, ut = flow_model.sample_location_and_conditional_flow(x0, x1)
            #xt = torch.cat((xt,con), dim=1)
            #vt = net_model(t, xt)
            vt = net_model(xt, t, low_res=low_res)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
            optimizer.step()
            #lr_scheduler.step()
            ema(net_model, ema_model, FLAGS.ema_decay)
            #print(f"epoch: {epoch}, steps: {step}, loss: {loss.item():.4}", end="\r")
            lr = optimizer.param_groups[0]['lr']
            metrics_epoch.append(
                {'loss': loss.item(), 'lr': lr})
                

            if step % every_steps == 0:
                    evaluation_results = eval(net_model, test_dataset)
                    results_per_step.append({
                    'step': step,
                    'evaluation_results': evaluation_results
                })
                    
            for action in callbacks:
                action(step, t=None, metrics=metrics_epoch[-1], epoch=epoch)

            if step % 2 == 0:
                pbar.set_postfix({"loss": metrics_epoch[-1]['loss']})

            
            

            pbar.update(1)
        loss_mean_epoch = np.mean([m['loss'] for m in metrics_epoch])
        print(f"Average loss over epoch: {loss_mean_epoch:.2f}")
        #generate_samples(net_model, FLAGS.parallel, savedir, step, net_="normal")
        #generate_samples(ema_model, FLAGS.parallel, savedir, step, net_="ema")
    with open('results/otcfm/mnist/results_per_epoch.json', 'w') as file:
                json.dump(results_per_step, file, indent=4)

def train(argv):
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )

    # DATASETS/DATALOADER
    train_dataset = datasets.MNIST(
    "../data",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    # Data
    train_dataset = get_dataset('mnist')("./data", train=True)
    #train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, num_workers=16, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=16, shuffle=True)
    test_dataset = get_dataset('mnist')("./data", train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=8, num_workers=16)
    test_images = next(iter(test_dataloader))[0].to(device)


    num_epochs = 10
    total_steps = len(train_loader) * num_epochs
    # MODELS
    """
    net_model = UNetModelWrapper(dim=(2, 28, 28), num_channels=32,
                                 num_res_blocks=1,
                                 num_classes=None,
                                 class_cond=True,
                                 ).to(device)
    """
    net_model = SuperResModelWrapper(
        dim=(1, 28, 28), num_channels=32, 
        num_res_blocks=1, 
        num_classes=None, 
        class_cond=True,
        ).to(device)
    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters())
    #sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    
    print("num params:", sum([p.numel() for p in net_model.parameters()]))
    savedir = FLAGS.output_dir + FLAGS.model + "/mnist/"
    os.makedirs(savedir, exist_ok=True)
    writer = LocalWriter(str(savedir), flush_every_n=100)


    def compute_test_metrics(step, **kwargs):
        x1, _, nfe = generate_samples_eval(net_model, test_images, step=step, net_='net_model')
        mse = F.mse_loss(x1, test_images)
        metrics = {'test_conditional_mse': mse.item()}
        writer.write_scalars(step, metrics)
        return metrics

    def plots(step, **kwargs):
        #ema_network.eval()
        # conditional
        x1, cond_image, nfe = generate_samples_eval(net_model, test_images, step=step, net_="net_model")
        #1 = x1[0]
        fig, axes = plt.subplots(len(x1), 2, figsize=(4, 2*len(x1)))
        mse = F.mse_loss(x1, test_images)
        metrics = {'test_conditional_mse': mse.item()}
        writer.write_scalars(step, metrics)
        for i in range(len(x1)):
            axes[i, 0].imshow(to_imshow(x1[i]))
            plot_condition(test_images[i], cond_image[i], axes[i, 1])
            axes[i, 0].axis('off'); axes[i, 1].axis('off');
        fig.tight_layout()
        writer.write_figures(step, dict(condition=fig))

    #################################
    #            OT-CFM
    #################################

    sigma = 0.0
    if FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {FLAGS.model}, must be one of ['otcfm', 'icfm', 'fm', 'si']"
        )

    
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    #FM = TargetConditionalFlowMatcher(sigma=sigma)

    callbacks = [
            PeriodicCallback(every_steps=10, callback_fn=lambda step,**kwargs: writer.write_scalars(step, kwargs['metrics'])),
            #PeriodicCallback(every_steps=total_steps//20, callback_fn=compute_test_metrics),
            PeriodicCallback(every_steps=total_steps//20, callback_fn=plots),
        ]
    

    sigma = 0.0
   #model = net_model
    #model = UNetModelWrapper(
        #dim=(2, 28, 28), num_channels=32, num_res_blocks=1, num_classes=None, class_cond=True,
    #).to(device)
   #optimizer = torch.optim.Adam(model.parameters())
    optim = torch.optim.Adam(net_model.parameters(),lr=2e-4)
    optimizer = optim
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    # Users can try target FM by changing the above line by
    # FM = TargetConditionalFlowMatcher(sigma=sigma)
    fit(net_model, ema_model, FM, num_epochs, train_loader, test_dataset, optim, device, savedir, callbacks,
        lr_scheduler=None)
    """
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            con = sample(data[0]).to(device)
            #con = data[0].to(device)
            optimizer.zero_grad()
            x1 = data[0].to(device)
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            xt = torch.cat((xt,con), dim=1)
            vt = net_model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            optimizer.step()
            print(f"epoch: {epoch}, steps: {i}, loss: {loss.item():.4}", end="\r")
    #scheduler.step(loss)
    """
    eval = True
    if eval:
        lpips_loss_fn = lpips.LPIPS(net='vgg').cuda()
        torch.manual_seed(0)
        test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=16, shuffle=True,)
        print("test_loader", len(test_dataloader), len(test_dataloader.dataset))
        metrics = {"mse": [], "lpips": [], "nfe": []}
        test_loader = iter(test_dataloader)
        n = 2
        path = os.path.join(savedir, "generated")
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True) 

        path_groundtruth = os.path.join(savedir, "generated_groundtruth")
        path_groundtruth = Path(path_groundtruth)
        path_groundtruth.mkdir(exist_ok=True, parents=True)
        idx = 0

        for batch in tqdm(range(n), desc="Computing test set."):
            batch = next(test_loader)[0].cuda()
            x1, test_condition, nfe = generate_samples_eval(net_model, batch, batch_size = 32, net_="net_model")
            #x1 = x1[0]
            test_condition_plt = test_condition.cpu()
            for im_true, im_sample, cond in zip(batch, x1, test_condition_plt):
                save_image(to_img(im_sample), str(path / f"image_{str(idx).zfill(3)}.jpg"))
                save_image(to_img(cond), str(path_groundtruth / f"image_gt_{str(idx).zfill(3)}.jpg"))
                save_image(to_img(im_true), str(path_groundtruth / f"image_gt2_{str(idx).zfill(3)}.jpg"))
                idx += 1
            metrics["mse"].append(torch.mean((x1 - batch)**2, dim=(1, 2, 3)))
            metrics["lpips"].append(lpips_loss_fn(x1, batch).squeeze(dim=(1,2,3)))
            metrics["nfe"].append(nfe)
        
        metrics = {k: torch.tensor(v, dtype=torch.float) if k == "nfe" else torch.concat(v, dim=0) for k, v in metrics.items()}
        #metrics = {k: torch.concatenate(v, dim=0) for k, v in metrics.items()}
        results = {
            **{f"{k}_mean": torch.mean(v).item() for k, v in metrics.items()},
            **{f"{k}_median": torch.median(v).item() for k, v in metrics.items()},
            **{f"{k}_std": torch.std(v).item() for k, v in metrics.items()},
        }
        path = os.path.join(savedir, "results.json")
        with open(path, 'w') as f:
            json.dump(results, f)
        
        print(results)


if __name__ == "__main__":
    app.run(train)
