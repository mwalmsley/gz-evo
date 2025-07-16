#!/bin/bash
#SBATCH --time=300:00:00                                # Time limit hrs:min:sec
#SBATCH --constraint=A100 
#SBATCH --mem=80G  # high mem node is more reliable
#SBATCH --job-name=baseln
#SBATCH --output=%x.%A.out
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --exclude=compute-0-103
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks 1



pwd; hostname; date

nvidia-smi

# torch device should have same order as nvidia-smi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "CUDA_DEVICE_ORDER set to $CUDA_DEVICE_ORDER"

export HYDRA_FULL_ERROR=1
export TORCH_NCCL_BLOCKING_WAIT=1
# export NCCL_DEBUG=INFO

export WANDB_DIR=/share/nas2/walml/wandb
export WANDB_ARTIFACT_DIR=/share/nas2/walml/wandb/artifacts

# modular cache for torch.compile
# https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html
export TORCHINDUCTOR_CACHE_DIR="/share/nas2/walml/cache/torchinductor"  

# export HF_DATASETS_IN_MEMORY_MAX_SIZE=50000000000  # Use to ignore HF_DATASETS_CACHE and keep in mem if possible.
# seems to make dataset.filter() very slow, perhaps it turns off the column selection
# avoid for classification baseline

export HF_HOME="/share/nas2/walml/cache/huggingface" # hub downloads
# export HF_DATASETS_CACHE="/share/nas2/walml/cache/huggingface/datasets" # load prepared dataset on nas
export HF_DATASETS_CACHE='/state/partition1/walml/cache/huggingface/datasets'  # load on node

# export HF_LOCAL_DATASETS_CACHE='/state/partition1/walml/cache/huggingface/datasets' # evo manual only
# export GZ_EVO_MANUAL_DOWNLOAD_LOC='/share/nas2/walml/tmp/gz-evo'  # evo manual only
# scripts read seed from SEED, default is 42
# SEED=$RANDOM
# echo Using seed $SEED

PYTHON="/share/nas2/walml/miniconda3/envs/zoobot39_cu118_dev/bin/python"
REPO_DIR="/share/nas2/walml/repos/gz-evo"

echo SLURM_NTASKS $SLURM_NTASKS 
echo SLURM_NTASKS_PER_NODE $SLURM_NTASKS_PER_NODE
export SLURM_NTASKS_PER_NODE=$SLURM_NTASKS # this isn't set correctly by old galahad slurm, it sets NTASKS_PER_NODE not SLURM_NTASKS_PER_NODE
echo SLURM_NTASKS_PER_NODE now $SLURM_NTASKS_PER_NODE

# ----

# echo 'Running classification baseline'
# srun $PYTHON $REPO_DIR/gz_evo/core/classification/train.py 

echo 'Running multinomial baseline'
srun $PYTHON $REPO_DIR/gz_evo/core/multinomial/train.py 

# publish to hub
# echo 'Publishing encoders to hub'
# $PYTHON $REPO_DIR/gz_evo/encoder_to_hub.py
