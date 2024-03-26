from __future__ import annotations
from typing import Callable, List, Protocol

import numpy as np
import tqdm
import torch

from torch.utils.data import DataLoader
from .actions import PeriodicCallback
import lpips
import json
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2):
    # Convert tensors to NumPy arrays
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    # Calculate PSNR
    return psnr(img1_np, img2_np, data_range=img2_np.max() - img2_np.min())

def calculate_ssim(img1, img2):
    # Convert tensors to NumPy arrays
    #img1_np = img1.detach().cpu().numpy().squeeze(0)
    #img2_np = img2.detach().cpu().numpy().squeeze(0)
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    # Calculate SSIM
    #return ssim(img1_np, img2_np, multichannel=True, data_range=img2_np.max() - img2_np.min())
    return ssim(img1_np, img2_np, multichannel=True, channel_axis=0, data_range=img2_np.max() - img2_np.min())


class Trainer:
    def __init__(
        self,
        loss_fn: Callable,
        train_dataloader: DataLoader,
        test_dataset,
        likelihood,
        cond_sample_fn,
        num_epochs: int,
        callbacks: List[PeriodicCallback] | None = None,
    ):
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.test_dataset = test_dataset
        self.num_epochs = num_epochs
        self.callbacks = [] if callbacks is None else callbacks
        self.likelihood = likelihood
        self.cond_sample_fn = cond_sample_fn
    def fit(self, optimizer, device, lr_scheduler=None):
        step = 0
        results_per_epoch = []
        num_steps = len(self.train_dataloader) * self.num_epochs
        every_steps = num_steps // 10
        for epoch in range(self.num_epochs):
            metrics_epoch = []
            pbar = tqdm.tqdm(total=len(self.train_dataloader), desc=f"Epoch: {epoch} / {self.num_epochs}")
            for batch in self.train_dataloader:
                step += 1
                images, _ = batch
                images = images.to(device)
                optimizer.zero_grad()
                loss_value = self.loss_fn(images)
                loss_value.backward()
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                lr = optimizer.param_groups[0]['lr']
                metrics_epoch.append(
                    {'loss': loss_value.item(), 'lr': lr})

                if step % every_steps == 0:
                    evaluation_results = self.eval()
                    results_per_epoch.append({
                    'step': step,
                    'evaluation_results': evaluation_results
                })

                for action in self.callbacks:
                    action(step, t=None, metrics=metrics_epoch[-1], epoch=epoch)

                if step % 10 == 0:
                    pbar.set_postfix({"loss": metrics_epoch[-1]['loss']})

                pbar.update(1)

            loss_mean_epoch = np.mean([m['loss'] for m in metrics_epoch])
            print(f"Average loss over epoch: {loss_mean_epoch:.2f}")
            #evaluation_results = self.eval()
            #print(f"Evaluation results for Epoch {epoch}: {evaluation_results}")

            # Append the results for the current epoch
            #results_per_epoch.append({
            #    'epoch': epoch,
            #    'evaluation_results': evaluation_results
            #})

            # Save results to a file after each epoch
            with open('logs/results_per_epoch.json', 'w') as file:
                json.dump(results_per_epoch, file, indent=4)
        
    def eval(self):
        lpips_loss_fn = lpips.LPIPS(net='vgg').cuda()
        metrics = {"mse": [], "lpips": [], "psnr": [], "ssim": []}
        torch.manual_seed(0)
        test_dataloader = DataLoader(self.test_dataset, batch_size=8, num_workers=16)
        test_loader = iter(test_dataloader)
        n = 3
        for batch in tqdm.tqdm(range(n), desc="Computing test set."):
            batch = next(test_loader)[0].cuda()
            test_condition = self.likelihood.sample(batch)
            xT = torch.randn_like(batch)
            x0 = self.cond_sample_fn(xT, test_condition)

            metrics["mse"].append(torch.mean((x0 - batch)**2, dim=(1, 2, 3)))
            metrics["lpips"].append(lpips_loss_fn(x0, batch).squeeze(dim=(1,2,3)))

            for i in range(batch.size(0)):
                metrics["psnr"].append(calculate_psnr(x0[i], batch[i]))
                metrics["ssim"].append(calculate_ssim(x0[i], batch[i]))
        
        metrics = {k: torch.tensor(v) if k in ['psnr', 'ssim'] else torch.concat(v, dim=0) for k, v in metrics.items()}
        results = {
            **{f"{k}_mean": torch.mean(v).item() for k, v in metrics.items()},
            **{f"{k}_median": torch.median(v).item() for k, v in metrics.items()},
            **{f"{k}_std": torch.std(v).item() for k, v in metrics.items()},
            }
        return results
            