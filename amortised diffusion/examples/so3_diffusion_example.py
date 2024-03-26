import hydra
from omegaconf import OmegaConf
import numpy as np

from src.diffusion import so3_utils as so3
from src.diffusion import so3_diffuser as so3_diff

@hydra.main(version_base=None, config_path="/home/ked48/rds/hpc-work/protein-diffusion/configs", config_name="diffusion_demo")
def instantiate_diffuser(cfg):
    config_dict = OmegaConf.to_container(cfg)
    print(config_dict)
    # Load the model
    diffuser = hydra.utils.instantiate(cfg.diffuser)
    print(diffuser)
    print(diffuser.discrete_sigma.shape)
    print(diffuser.discrete_sigma[0:10])
    print(diffuser.diffusion_coef(t=0.5))

    dt = 0.01
    n_steps = 1000
    n_samples = 10

     # Sample a forward diffusion process
    forward_samples = []
    for i in range(n_samples):
        forward_samples.append(diffuser.sample(t=1, n_samples=n_steps))
    print(forward_samples[0].shape)

    # Print the mean and variance of the forward and reverse samples
    print("Forward mean: ", np.mean(forward_samples))
    print("Forward variance: ", np.var(forward_samples))



if __name__ == "__main__":
    instantiate_diffuser()



