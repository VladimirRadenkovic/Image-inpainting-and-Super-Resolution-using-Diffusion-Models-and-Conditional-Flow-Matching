#!/bin/bash 
#! Name of the job: 
#SBATCH -J sample-kd
#SBATCH -o sample.out # File to which STDOUT will be written 
#SBATCH -e sample.err # File to which STDERR will be written 
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'): 
#SBATCH --account LIO-SL3-GPU
#! How many whole nodes should be allocated? 
#SBATCH --nodes=1 
#! How many (MPI) tasks will there be in total? 
#! Note probably this should not exceed the total number of GPUs in use. 
#SBATCH --ntasks=1 
#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1). 
#! Note that the job submission script will enforce no more than 32 cpus per GPU. 
#SBATCH --gres=gpu:1 
#! How much wallclock time will be required? 
#SBATCH --time=12:00:00 
#! What types of email messages do you wish to receive? 
#SBATCH --mail-type=begin        # send email when job begins 
#SBATCH --mail-type=end 
#SBATCH --mail-user=...
#! Partition: 
#SBATCH -p ampere
#!90 seconds before training ends
#SBATCH --signal=SIGUSR1@90
#SBATCH --requeue
 
. /etc/profile.d/modules.sh                # Leave this line (enables the module command) 
module purge                               # Removes all modules still loaded 
module load rhel8/default-amp              # REQUIRED - loads the basic environment 
module unload miniconda/3 
module load cuda/11.8
module list 

source /home/ked48/.bashrc 
 
cd /home/ked48/rds/hpc-work/protein-diffusion
mamba activate ./venv_gpu

cd /home/ked48/rds/hpc-work/protein-diffusion/src
/rds/user/ked48/hpc-work/protein-diffusion/venv_gpu/bin/python /home/ked48/rds/hpc-work/protein-diffusion/src/sample.py