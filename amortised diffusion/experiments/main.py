"""
Script for training and evaluating conditional diffusion models.

Usage:

```bash
python main.py --config=config.py:<dataset>,<likelihood>,<conditioning>
        [--config.training.batch_size=16]
```

where
    - dataset can be `mnist`, `flowers`.
    - likelihood can be `inpaiting`, `outpainting`.
    - conditioning can be `amortized`, `reconstruction_guidance`.
"""
# TODO(Vincent): import torch; import tensorboardX breaks.  Importing
# tensorboardX first doesn't which is why we import it before everything else.
#import tensorboardX

import os
import json
import pathlib
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from absl import app, flags
from ml_collections import config_flags

from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torchmetrics.image.fid import FrechetInceptionDistance
from ema_pytorch import EMA
import lpips


from image_diffusion.sde_diffusion import DDPM
from image_diffusion.unet import create_model
from image_diffusion.sampling import get_prior_sample_fn, get_conditional_sample_fn
from image_diffusion.plotting_utils import to_imshow
from image_diffusion.data import get_dataset
from image_diffusion.utils import get_latest_version
#from image_diffusion.trainer import Trainer
from image_diffusion.trainer2 import Trainer
from image_diffusion.writers import LocalWriter, TensorBoardWriter, MultiWriter
from image_diffusion.likelihoods import get_likelihood
from image_diffusion.conditioning import get_conditioning
from image_diffusion.loss_functions import get_loss_function
from image_diffusion.actions import PeriodicCallback
from image_diffusion.checkpoint_manager import CheckpointManager


config_flags.DEFINE_config_file(
    'config', 'config.py:mnist,inpainting,amortized',
    'The config file.', lock_config=False)


flags.DEFINE_string(
    "mode", default="all", help="Options: train, eval, all"
)

flags.DEFINE_string(
    "save_dir", default="logs", help="Directory to store experiment artifacts."
)

FLAGS = flags.FLAGS


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
    path = pathlib.Path(cwd, FLAGS.save_dir, experiment_name)
    version = get_latest_version(path) + 1  # increment version number
    path = path / f"version_{str(version).zfill(2)}"
    path.mkdir(parents=True, exist_ok=True)    
    return path


def to_img(x: torch.Tensor) -> torch.Tensor:
    """From x in [-1, 1] to [0, 1]"""
    x = torch.clamp(x, -1, 1)
    x = (x + 1.) / 2.
    return x


def main(_):
    config = FLAGS.config
    print(config)
    experiment_name = get_experiment_name(config)
    print(config.dataset.image_size)
    print(experiment_name)

    pathlib.Path(os.path.join(os.getcwd(), FLAGS.save_dir, experiment_name)).mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)

    # Data
    train_dataset = get_dataset(config.dataset.name)("./data", train=True)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, num_workers=16, shuffle=True)
    test_dataset = get_dataset(config.dataset.name)("./data", train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=8, num_workers=16)
    test_images = next(iter(test_dataloader))[0].to(device)

    # Neural net
    in_channels = config.dataset.num_channels
    if config.conditioning.name == "amortized":
        in_channels = 2 * in_channels  # allow to concat condition
    network = create_model(
        image_size=config.dataset.image_size,
        in_channels=in_channels,
        out_channels=config.dataset.num_channels,
        **config.network
    )
    network = network.to(device)
    print("num params:", sum([p.numel() for p in network.parameters()]))
    ema_network = EMA(network, beta=0.995, update_every=10)
    image_size = (config.dataset.num_channels, config.dataset.image_size, config.dataset.image_size)

    likelihood = get_likelihood(config.likelihood.name).from_configdict(config.likelihood)
    conditioning = get_conditioning(config.conditioning.name).from_configdict(config.conditioning)

    ddpm = DDPM(config.diffusion.num_steps).to(device)
    loss_fn, _ = get_loss_function(network, ddpm, conditioning, likelihood)
    eps_model_ema = lambda xi, i: ema_network(xi, 1.0 * i / ddpm.Ns)
    prior_sample_fn = get_prior_sample_fn(eps_model_ema, ddpm, conditioning, likelihood)
    cond_sample_fn = get_conditional_sample_fn(eps_model_ema, ddpm, conditioning, likelihood)

    test_condition = likelihood.sample(test_images)

    def compute_test_metrics(step, **kwargs):
        print('--------------')
        print('COMPUTE TEST METRICS')
        print('--------------')
        xT = torch.randn_like(test_images)
        x0 = cond_sample_fn(xT, test_condition)
        mse = F.mse_loss(x0, test_images)
        metrics = {'test_conditional_mse': mse.item()}
        writer.write_scalars(step, metrics)
        return metrics

    def sample_prior():
        ema_network.eval()
        xT = torch.randn_like(test_images)
        x0 = prior_sample_fn(xT)
        return x0
    
    def sample_conditional(step):
        ema_network.eval()
        xT = torch.randn_like(test_images)
        x0 = cond_sample_fn(xT, test_condition)
        mse = F.mse_loss(x0, test_images)
        metrics = {'test_conditional_mse': mse.item()}
        writer.write_scalars(step, metrics)
        return x0, test_condition

    def plots(step, **kwargs):
        print('--------------')
        print('PLOTS')
        print('--------------')
        ema_network.eval()
        # prior
        prior_samples = sample_prior()   # [N, C, H, W]
        prior_img = make_grid(prior_samples, nrow=5, pad_value=1, normalize=True).cpu().numpy()  # (3, H, W) in range [0, 1]
        writer.write_images(step, dict(prior=prior_img))
        # conditional
        x0, y = sample_conditional(step)
        fig, axes = plt.subplots(len(x0), 2, figsize=(4, 2*len(x0)))
        for i in range(len(x0)):
            axes[i, 0].imshow(to_imshow(x0[i]))
            likelihood.plot_condition(test_images[i], y[i], axes[i, 1])
            axes[i, 0].axis('off'); axes[i, 1].axis('off');
        fig.tight_layout()
        writer.write_figures(step, dict(condition=fig))

    experiment_dir = get_experiment_dir(experiment_name)
    print(experiment_dir)
    num_steps = len(train_loader) * config.training.num_epochs

    print("MODE:", FLAGS.mode)

    if FLAGS.mode in ["train", "all"]:
        local_writer = LocalWriter(str(experiment_dir), flush_every_n=100)
        tb_writer = TensorBoardWriter(str(experiment_dir / "tensorboard"))
        writer = MultiWriter([tb_writer, local_writer])
    elif FLAGS.mode == "eval":
        # in eval mode only write locally.
        writer = LocalWriter(str(experiment_dir), flush_every_n=100)
    else:
        raise ValueError(f"Unknown mode: {FLAGS.mode}. Options: train, eval, all.")

    writer.log_hparams(config.to_dict())

    if FLAGS.mode in ["train", "all"]:
        def save_checkpoint_callback(step, **kwargs):
            path = experiment_dir / "checkpoints"
            path.mkdir(exist_ok=True, parents=True)
            state = {
                "step": step,
                "ema": ema_network.state_dict(),
                "network": network.state_dict(),
            }
            CheckpointManager(state, str(path), "checkpoint").save(step)

        callbacks = [
            PeriodicCallback(every_steps=num_steps//10, callback_fn=save_checkpoint_callback),
            PeriodicCallback(every_steps=1, callback_fn=lambda *_, **__: ema_network.update()),
            PeriodicCallback(every_steps=10, callback_fn=lambda step,**kwargs: writer.write_scalars(step, kwargs['metrics'])),
            #PeriodicCallback(every_steps=num_steps//10, callback_fn=compute_test_metrics),
            PeriodicCallback(every_steps=num_steps//10, callback_fn=plots),
        ]

        # optimizer and lr scheduler
        warmup_period = config.training.warmup_steps
        optimizer = torch.optim.Adam(network.parameters(), lr=config.training.lr_end_warmup)

        if config.training.lr_schedule == "warmup_cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=num_steps, last_epoch=-1, eta_min=config.training.lr_final)
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda step: step/warmup_period)
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup_scheduler, lr_scheduler], milestones=[warmup_period]
            )
        elif config.training.lr_schedule == "constant":
            lr_scheduler = None

        num_epochs = config.training.num_epochs
        trainer = Trainer(
            loss_fn, train_loader, test_dataset, likelihood, cond_sample_fn, num_epochs=num_epochs, callbacks=callbacks
        )
        trainer.fit(optimizer, device, lr_scheduler=lr_scheduler)
        save_checkpoint_callback(num_steps)
    

    if FLAGS.mode in ["eval", "all"]:
        # plots(0)
        lpips_loss_fn = lpips.LPIPS(net='vgg').cuda()

        train_loader = DataLoader(train_dataset, batch_size=config.testing.batch_size, num_workers=16, shuffle=True)
        print("train_loader", len(train_loader), len(train_loader.dataset))
        torch.manual_seed(config.testing.seed)
        test_dataloader = DataLoader(test_dataset, batch_size=config.testing.batch_size, num_workers=16, shuffle=True,)
        print("test_loader", len(test_dataloader), len(test_dataloader.dataset))

        if config.testing.fid:
            train_loader = DataLoader(train_dataset, batch_size=config.testing.batch_size, num_workers=16, shuffle=True)
            fid = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False).cuda()
            # compute train statistics, can take a while...
            for batch in tqdm(train_loader, desc="Computing stats training set."):
                batch = batch[0].cuda()
                fid.update(to_img(batch), real=True)
        else:
            fid = None

        metrics = {"mse": [], "lpips": []}
        test_loader = iter(test_dataloader)
        n = config.testing.num_test // config.testing.batch_size
        path = (experiment_dir / "generated")
        path.mkdir(exist_ok=True, parents=True)
        path_groundtruth = (experiment_dir / "generated_groundtruth")
        path_groundtruth.mkdir(exist_ok=True, parents=True)
        idx = 0
        for batch in tqdm(range(n), desc="Computing test set."):
            batch = next(test_loader)[0].cuda()
            test_condition = likelihood.sample(batch)
            if config.likelihood.name == 'hyperresolution': 
                test_condition_plt = test_condition.cpu()    
            else:
                print('---------------------')
                print(config.likelihood)
                print('---------------------')
                test_condition_plt = torch.where(test_condition == likelihood.pad_value, 1.0, test_condition).cpu() 
            xT = torch.randn_like(batch)
            x0 = cond_sample_fn(xT, test_condition)

            if config.testing.fid:
                fid.update(to_img(x0), real=False)

            for im_true, im_sample, cond in zip(batch, x0, test_condition_plt):
                save_image(to_img(im_sample), str(path / f"image_{str(idx).zfill(3)}.jpg"))
                save_image(to_img(cond), str(path_groundtruth / f"image_gt_{str(idx).zfill(3)}.jpg"))
                save_image(to_img(im_true), str(path_groundtruth / f"image_gt2_{str(idx).zfill(3)}.jpg"))
                idx += 1

            metrics["mse"].append(torch.mean((x0 - batch)**2, dim=(1, 2, 3)))
            metrics["lpips"].append(lpips_loss_fn(x0, batch).squeeze(dim=(1,2,3)))
        
        metrics = {k: torch.concatenate(v, dim=0) for k, v in metrics.items()}
        results = {
            **{f"{k}_mean": torch.mean(v).item() for k, v in metrics.items()},
            **{f"{k}_median": torch.median(v).item() for k, v in metrics.items()},
            **{f"{k}_std": torch.std(v).item() for k, v in metrics.items()},
            "fid": fid.compute().item() if fid else None
        }
        with open(str(experiment_dir / "results.json"), 'w') as f:
            json.dump(results, f)
        
        print(results)


if __name__ == "__main__":
    app.run(main)