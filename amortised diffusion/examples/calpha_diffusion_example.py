import hydra
from omegaconf import OmegaConf, DictConfig
import numpy as np
import torch
import os

import src.constants as constants
from src.configs import config
from src.data.datasets import protein_dataset
from src.constants import DATA_PATH
from src.diffusion import r3_diffuser, so3_diffuser, se3_diffuser
from src.plots import visualise

data_path = DATA_PATH / "pdb_domain_ca_coords_v2023-04-25.npz"

# Load hydra config from yaml filses and command line arguments.
@hydra.main(
    config_path=str(constants.HYDRA_CONFIG_PATH),
    config_name="diffusion_demo",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    dataset = protein_dataset.NpzDataset(data_path)
    # cfg = config.validate_config(cfg)
    print(cfg)
    
    # train_model(cfg)
    # Load the model
    # r3_diffuser = hydra.utils.instantiate(cfg.diffusion.r3)
    # print(r3_diffuser)

    r3_diff = hydra.utils.instantiate(cfg.se3_diffuser.r3_diffuser)
    so3_diff = hydra.utils.instantiate(cfg.se3_diffuser.so3_diffuser)

    se3_diff = se3_diffuser.SE3Diffuser(diffuse_rot = True, 
                                            so3_diffuser = so3_diff, 
                                            diffuse_trans = True, 
                                            r3_diffuser = r3_diff)

    diffused_point_clouds = []
    scores = []

    for t in np.linspace(0.1, 1, 10):
        diffused_point_cloud, score_t = r3_diff.forward_marginal(
            dataset[0], t)
        diffused_point_clouds.append(diffused_point_cloud)
        scores.append(score_t)
    diffused_point_clouds_np = np.stack(diffused_point_clouds)
    scores_np = np.stack(scores)



    #get location of this file
    cwd = os.getcwd()
    # Save the diffused point clouds 
    np.savez(
        cwd + "/output/diffused_point_clouds.npz",
        diffused_point_clouds=diffused_point_clouds_np,
        scores=scores_np,
    )

    
    loaded = np.load(cwd + "/output/diffused_point_clouds.npz")
    point_clouds = loaded['diffused_point_clouds']
    scores = loaded['scores']

    print(point_clouds.shape)
    print(scores.shape)

    # Visualise the diffused point clouds via iterating
    #over the point clouds and scores
    visualise.save_3d_trajectory_as_gif(point_clouds, cwd + "/output/diffusion.gif", fps=2, dpi=60)






if __name__ == "__main__":
    main()
