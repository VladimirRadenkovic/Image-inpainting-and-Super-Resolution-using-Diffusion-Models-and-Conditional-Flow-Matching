import os
from pathlib import Path
import hydra
# from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch_geometric.transforms import Compose
from torch_geometric.data.lightning import LightningDataset
from tqdm.autonotebook import tqdm
from typing import *

import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
import wandb
import seaborn as sns

from src.constants import HYDRA_CONFIG_NAME, HYDRA_CONFIG_PATH, PROJECT_PATH, DATA_PATH
from src.utils.log_utils import get_logger
from src.utils.data_utils import get_cath_data
from src.utils.callbacks import instantiate_callbacks
from src.models.resdiff import ResDiff
from src.evaluation.visualize import save_trajectory_as_gif, plot_pointcloud, plot_2_pointclouds
from src.evaluation.plot_proteins import quick_vis
from src.evaluation.plotstyle import matplotlib_defaults
from src.diffusion.structconditioner import Structconditioner
from src.utils.pypdb_utils import get_motif_coordinates
import src.utils.geometry as geometry

logger = get_logger(__name__)
palette = sns.color_palette("Dark2")
matplotlib_defaults(autoupdate=True)


@hydra.main(
    config_path=str(HYDRA_CONFIG_PATH),
    config_name=HYDRA_CONFIG_NAME,
    version_base="1.3",)
def sample(cfg: DictConfig) -> None:
    train_data, val_data = get_cath_data()


    # checkpoint_reference = f"{cfg.logger.entity}/{cfg.logger.project}/{cfg.logger.run_id}:best"
    # # download checkpoint locally (if not already cached)
    # run = wandb.init(project="protein-diffusion")
    # artifact = run.use_artifact(checkpoint_reference, type="model")
    # artifact_dir = artifact.download()
    # artifact_dir = "/home/ked48/rds/hpc-work/protein-diffusion/artifacts/newgvp_j"
    artifact_dir = DATA_PATH / "models"
    # load checkpoint
    res_diff_ckpt = ResDiff.load_from_checkpoint(Path(artifact_dir) / "auxseq_epoch=185-val_loss=0.0924.ckpt")
    batch_size = 1
    protein_length = 50
    device = "cuda"
    run_dir = DATA_PATH / "exp_data/6e6r/gs1500_nonoise3steps"
    run_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = run_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    #create txt file in this dir with some info about the run
    with open(run_dir / "info.txt", "w") as f:
        f.write(f"New denoiser, but no noise in the last 3 steps to improve chain distances and sample quality. Lengths of proteins same as first 100 from validation set.")

    plot_dir = run_dir / "sample_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    loss_dir = run_dir / "cond_loss_samples"
    loss_dir.mkdir(parents=True, exist_ok=True)
    # sample from model
    # motif_pos = get_motif_coordinates(cfg.sampling.pdb_id, cfg.sampling.motif_start, cfg.sampling.motif_end, cfg.sampling.chain_id)
    # motif_inds = cfg.sampling.motif_inds
    motif_pos = np.load(DATA_PATH / "motif_coordinates/rfdiff/6E6R_motif_coordinates.npy")
    motif_inds = np.load(DATA_PATH / "motif_coordinates/rfdiff/6E6R_motif_indices.npy")
    #convert motif pos to torch tensor
    motif_pos = torch.tensor(motif_pos, dtype=torch.float32, device=device)
    #convert motif inds to list
    motif_inds = list(motif_inds)

    #unconditional sampling
    for i in range(0,100):
        # graph_ind = np.random.randint(0, len(val_data))
        protein_length = val_data[i].num_nodes
        logger.info(f"Sampling protein of length {protein_length}")
        #Place motif at the center of the protein via motif inds
        motif_inds = place_indices_block_within_bounds(motif_inds, protein_length)
        logger.info(f"Motif inds: {motif_inds}")
        struct_conditioner = Structconditioner(res_diff_ckpt.diffuser.alphas, res_diff_ckpt.diffuser.alphas_cumprod, res_diff_ckpt.model.forward)
        struct_conditioner.set_condition(motif_pos=motif_pos, motif_inds=motif_inds, cond_frac=0, bayesian_gs=1500, denoiser=res_diff_ckpt.model)
        struct_conditioner.set_monitor(batch_size)

        x_T = res_diff_ckpt.diffuser.sample_blob(batch_size, protein_length, device) #protein length is 70
        trajectory, x_0 = res_diff_ckpt.diffuser.reverse_diffusion_sampling(
            x_T, res_diff_ckpt.model, conditioner = struct_conditioner, save_trajectory=True)
        if i % 20 == 0:
            save_graphs(struct_conditioner, x_0, x_T, i, trajectory, plot_dir)
        torch.save(x_0, sample_dir / f"{i}.pt")
        np.save(loss_dir / f"condloss_{i}.npy", struct_conditioner.monitor_total[0])


    logger.info("Sampling done. To evaluate, execute `python src/evaluation/eval_pipeline.py` with the appropriate sample and training dataset directory:")
    logger.info(f"python src/evaluation/evaluation_pipeline.py --sample_dir {run_dir} --training_dataset <path to training dataset, set by default>")
    #conditional sampling
    # for i in range(100):
    #     x_T = res_diff_ckpt.diffuser.sample_blob(batch_size, protein_length, device) #protein length is 50
    #     trajectory, x_0 = res_diff_ckpt.diffuser.reverse_diffusion_sampling(
    #         x_T, res_diff_ckpt.model, conditioner = struct_conditioner, save_trajectory=True)
    #     torch.save(x_0, cond_sample_dir / f"{i}.pt")
    #     if i % 20 == 0:
    #         save_trajectory_as_gif(trajectory.cpu(), plot_dir / f"{i}.gif")
    #         save_graphs(struct_conditioner, x_0, x_T, i, trajectory, cond_sample_dir, cfg.sampling.motif_inds, motif_pos.clone())


def save_graphs(conditioner, x_0, x_T, i, trajectory, plot_dir):
    x_0_pos = x_0.pos.detach().cpu().numpy()
    x_T_pos = x_T.pos.detach().cpu().numpy()
    #first sample plot
    x_sample = x_0_pos[:50]
    plt.figure()
    plot_pointcloud(x_sample, motif_inds=x_0.motif_inds[0])
    plt.savefig(plot_dir / f"x_0_{i}.png")
    #motif plot
    motif_pos_sample = torch.tensor(x_0_pos[x_0.motif_inds[0].cpu()], device=x_0.motif_pos.device)
    rot_mat, trans_vec = geometry.differentiable_kabsch(x_0.motif_pos, motif_pos_sample)
    motif_pos_aligned = geometry.rototranslate(motif_pos_sample, rot_mat, trans_vec)
    motif_pos_aligned= motif_pos_aligned.detach().cpu().numpy()
    motif_pos = x_0.motif_pos.detach().cpu().numpy()
    motif_pos_sample = motif_pos_sample.detach().cpu().numpy()
    plot_2_pointclouds(motif_pos_aligned, motif_pos)
    plt.savefig(plot_dir / f"x_0_{i}_motif_chain.png")
    plot_2_pointclouds(motif_pos_aligned, motif_pos, chain=False)
    plt.savefig(plot_dir / f"x_0_{i}_motif.png")
    plot_2_pointclouds(motif_pos_sample, motif_pos, chain=False)
    plt.savefig(plot_dir / f"x_0_{i}_motif_unaligned.png")
    #cond loss plot
    plt.figure()
    plt.plot(conditioner.monitor_total[0]*(15**2))
    plt.savefig(plot_dir / f"cond_loss_{i}.png")
    # #scaled plot
    x_sample = x_sample*15
    plot_pointcloud(x_sample)
    plt.savefig(plot_dir / f"x_0_{i}_scaled.png")
    plt.close()
    quick_vis(x_sample, save_path=plot_dir / f"ss_x_0_{i}_scaled.png")
    c = np.arange(trajectory.shape[1])
    # save_trajectory_as_gif(trajectory.cpu().detach().numpy(), f"{plot_dir}/{i}_R_GIF.gif", c=c,dpi=60, xlim=(-3, 3), ylim=(-3, 3), zlim=(-3, 3))
    np.save(plot_dir / f"x_0_{i}.npy", x_0_pos)
    np.save(plot_dir / f"x_T_{i}.npy", x_T_pos)
    np.save(plot_dir / f"motif_pos{i}", motif_pos)
    np.save(plot_dir / f"motif_pos_sample{i}", motif_pos_aligned)
    np.save(plot_dir / f"cond_loss{i}", conditioner.monitor_total)

def place_indices_block_within_bounds(indices, vec_length):
    mid = vec_length // 2
    indices_length = max(indices) - min(indices) + 1
    if indices_length > vec_length:
        raise ValueError('The indices block is larger than the vector.')
    # Calculate the start and end points of the adjusted indices block
    start = mid - indices_length // 2
    end = start + indices_length
    # Calculate the offset to adjust the indices
    offset = start - min(indices)
    adjusted_indices = [index + offset for index in indices]
    return adjusted_indices








if __name__ == "__main__":
    sample() 