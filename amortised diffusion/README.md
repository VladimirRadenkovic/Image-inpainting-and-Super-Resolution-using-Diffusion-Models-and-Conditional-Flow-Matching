## Environment
### 1. Conda environment
Create a conda environment for the project via
```bash
make venv # will create a cpu environment
# NOTE: This will simply call
#  conda env create --prefix=./venv -f requirements/env.yml

# For a gpu environment call
#  make name=venv_gpu sys=gpu venv
#  conda activate ./venv_gpu

# For a Mac m1 environment call
#  make name=venv sys=m1 venv
#  conda activate ./venv

# To activate the environment, use:
conda activate ./venv
```

After creating and activating the environment, install the project as a package via
```bash
pip install -e .
```

### 2. Evaluation Pipeline

The backbone part of the evaluation pipeline should work after the local pip install mentioned above.

To use the self-consistency part of the evaluation pipeline, you need to install both our wrapper for [ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187#:~:text=ProteinMPNN%20solves%20sequence%20design%20problems,rescues%20previously%20failed%20designs%20made) as well as a local version of [ColabFold](https://www.nature.com/articles/s41592-022-01488-1).

To install the ProteinMPNN wrapper, go into the `protein_diffusion` project directory and execute the following commands (from [here](https://github.com/Croydon-Brixton/proteinmpnn_wrapper/tree/main)):

```bash
git clone --recurse-submodules https://github.com/Croydon-Brixton/proteinmpnn_wrapper.git
cd proteinmpnn_wrapper
pip install .
```

To install a local version of ColabFold, follow the instructions for your operating system on [this webpage](https://github.com/YoshitakaMo/localcolabfold).

### 3. Environment variables

Set environemnt variables for you system in a `.env` file at the project directory
(same as this readme.)

## Repo Overview

- The main training loop is located under `src/train.py` and can be submitted to a SLURM cluster via `src/train.sh`.
- The overall generative model is located in `src/models/resdiff.py` and has two objects as main attributes:
    - The diffuser providing the noise schedule, the methods for denoising etc, located in `src/diffusion/sde_diffusion.py`. This diffuser is adapted for graph based data and makes heavy use of PyTorch Geometric due to that.
    - The denoiser which in our case predicts the added noise (noise formulation from Ho et al). In our case this takes the form of a GNN and is located in `src/models/gvp_gnn.py`.
- The diffuser can take a conditioner (ABC in `src/diffusion/conditioner.py`); in the specific case of the motif-scaffolding problem we use the structure conditioner (`src/diffusion/structconditioner.py`).


## Experiment logging with wandb

To log to wandb, you will first need to log in. To do so, simply install wandb via pip
with `pip install wandb` and call `wandb login` from the commandline.

If you are already logged in and need to relogin for some reason, use `wandb login --relogin`.

## Training a model with pytorch lightning and logging on wandb

To run a model simply use

```
python src/train.py run_name=<YOUR_RUN_NAME>
```

To use parameters that are different from the default parameters in `src/configs/config.yaml`
you can simply provide them in the command line call. For example:

```
python src/train.py run_name=<YOUR_RUN_NAME> epochs=100
```
To configure extra things such as logging, use
```
# LOGGING
# For running at DEBUG logging level:
#  (c.f. https://hydra.cc/docs/tutorials/basic/running_your_app/logging/ )
## Activating debug log level only for loggers named `__main__` and `hydra`
python src/train.py 'hydra.verbose=[__main__, hydra]'
## Activating debug log level for all loggers
python src/train.py hydra.verbose=true

# PRINTING CONFIG ONLY
## Print only the job config, then return without running
python src/train.py --cfg job

# GET HELP
python src/train.py --help
```
