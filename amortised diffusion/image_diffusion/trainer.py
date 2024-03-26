from __future__ import annotations
from typing import Callable, List, Protocol

import numpy as np
import tqdm
import torch

from torch.utils.data import DataLoader
from .actions import PeriodicCallback


class Trainer:
    def __init__(
        self,
        loss_fn: Callable,
        train_dataloader: DataLoader,
        num_epochs: int,
        callbacks: List[PeriodicCallback] | None = None,
    ):
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.num_epochs = num_epochs
        self.callbacks = [] if callbacks is None else callbacks

    def fit(self, optimizer, device, lr_scheduler=None):
        step = 0
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

                for action in self.callbacks:
                    action(step, t=None, metrics=metrics_epoch[-1], epoch=epoch)

                if step % 10 == 0:
                    pbar.set_postfix({"loss": metrics_epoch[-1]['loss']})

                pbar.update(1)

            loss_mean_epoch = np.mean([m['loss'] for m in metrics_epoch])
            print(f"Average loss over epoch: {loss_mean_epoch:.2f}")
