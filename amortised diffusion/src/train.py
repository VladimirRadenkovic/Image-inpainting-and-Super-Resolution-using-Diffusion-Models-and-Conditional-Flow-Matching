import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from torch_geometric.transforms import Compose
from torch_geometric.data.lightning import LightningDataset
from tqdm.autonotebook import tqdm

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
import wandb

from src.constants import HYDRA_CONFIG_NAME, HYDRA_CONFIG_PATH, PROJECT_PATH, DATA_PATH
from src.models.resdiff import ResDiff
from src.models.gvp_gnn import GVPDenoiser
from src.utils.data_utils import get_cath_data, get_scope_data
from src.utils.callbacks import instantiate_callbacks
from src.utils.log_utils import get_logger
import warnings

logger = get_logger(__name__)

@hydra.main(
    config_path=str(HYDRA_CONFIG_PATH),
    config_name=HYDRA_CONFIG_NAME,
    version_base="1.3",)
def train(cfg: DictConfig) -> None:
    logger.info("Run config: \n %s" % OmegaConf.to_yaml(cfg, resolve=True))
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    # Set training seed
    pl.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision('high')

    vp_diffuser = hydra.utils.instantiate(cfg.diffusion.vp_diffuser)
    hoog_diffuser = hydra.utils.instantiate(cfg.diffusion.hoog_diffuser)
    logger.info("Diffuser instantiated")

    #Set up GVP-GNN
    denoiser_model = hydra.utils.instantiate(cfg.model)
    logger.info("Model instantiated")

    #set up lightning module
    res_diff = ResDiff(hoog_diffuser, denoiser_model, 
                       conditional_training=False,
                       aux_loss=False)
    logger.info("Lightning module instantiated")

    # This LighningDataset is in fact a Datamodule (PyG's way of doing things)
    train_data, val_data = get_scope_data()
    # train_data, val_data = get_cath_data()

    #take only the first batch (32 graphs) for now
    # train_data_subset = train_data[0,10,20]

    #set run directory in DATA_PATH based on cfg.name
    run_dir = DATA_PATH / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    #set up logger
    
    if cfg.name == "test":
        wandb_logger = None
    elif cfg.load_ckpt:
        wandb_logger = WandbLogger(project=cfg.logger.project, entity=cfg.logger.entity, save_dir=run_dir,
                                    name=cfg.name, log_model="all", checkpoint_name="scope_seqaux", resume=True)
    else:
        wandb_logger = WandbLogger(project=cfg.logger.project, entity=cfg.logger.entity, save_dir=run_dir,
                                    name=cfg.name, log_model="all", checkpoint_name="scope_seqaux")
    # wandb_logger.watch(denoiser_model, log="all", log_freq=5)

    # Set up callbacks
    # callbacks = instantiate_callbacks(cfg.callbacks)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=run_dir,
    filename="{cfg.name}{epoch}-{val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=3,
    auto_insert_metric_name=True,
    )

    # next_every_50_dir = os.path.join(wandb_logger.save_dir,"every_50")
    # every_50_callback = pl.callbacks.ModelCheckpoint(
    #     dirpath=next_every_50_dir,
    #     filename="{epoch}-{val_loss:.4f}",
    #     auto_insert_metric_name=True,
    #     every_n_epochs=50
    # )

    #set up datamodule
    datamodule = LightningDataset(
        train_dataset=train_data,
        val_dataset=val_data,
        batch_size = cfg.dataset.loader.batch_size,
        num_workers = cfg.dataset.loader.num_workers
    )
    logger.info("Datamodule instantiated")

    #set up trainer
    trainer = pl.Trainer(
        default_root_dir=run_dir,
        # accelerator=cfg.trainer.accelerator,
        # devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        logger=wandb_logger,
        log_every_n_steps=100,
        callbacks=[checkpoint_callback],
        plugins=[SLURMEnvironment()]
        )
    #run training
    if cfg.load_ckpt:
        checkpoint_reference = f"{cfg.logger.entity}/{cfg.logger.project}/{cfg.logger.run_id}:latest"
        # download checkpoint locally (if not already cached)
        run = wandb.init(project="protein-diffusion")
        artifact = run.use_artifact(checkpoint_reference, type="model")
        artifact_dir = artifact.download()
        # load checkpoint
        res_diff_ckpt = ResDiff.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
        trainer.fit(res_diff_ckpt, datamodule)
    else:
        trainer.fit(res_diff, datamodule)

if __name__ == "__main__":
    train()