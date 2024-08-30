#!/bin/bash
#SBATCH --time=23:15:00  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu 4G
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=1gpu_bas

pwd; hostname; date

nvidia-smi

# Some setup for download the dataset on master node before moving offline:
# huggingface-cli login

echo 'Loading arrow'
module load arrow

echo 'SLURM_TMPDIR'
echo $SLURM_TMPDIR

export NCCL_BLOCKING_WAIT=1

export HF_HOME=/project/def-bovy/walml/cache/huggingface
export HF_DATASETS_CACHE=/project/def-bovy/walml/cache/huggingface/datasets

export HF_LOCAL_DATASETS_CACHE=$SLURM_TMPDIR/cache/huggingface/datasets
export GZ_EVO_MANUAL_DOWNLOAD_LOC='/project/def-bovy/walml/tmp/gz-evo'

# no internet on worker nodes
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export WANDB__SERVICE_WAIT="300"


# SEED=1  # don't change this when dividing up dataset

PYTHON=/project/def-bovy/walml/envs/gz-evo/bin/python
source /project/def-bovy/walml/envs/gz-evo/bin/activate
REPO_DIR=/project/def-bovy/walml/repos/gz-evo

# echo 'Running classification baseline'
# srun $PYTHON $REPO_DIR/baseline/classification_baseline.py 

echo 'Running regression baseline'
srun $PYTHON $REPO_DIR/baseline/regression_baseline.py 
