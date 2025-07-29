#!/bin/bash
#SBATCH --time=300:00:00                                # Time limit hrs:min:sec
#SBATCH --constraint=A100 
#SBATCH --mem=80G  # high mem node is more reliable
#SBATCH --exclude=compute-0-103
#SBATCH --job-name=core_1gpu
#SBATCH --output=%x.%A.out
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks 1
#SBATCH --exclusive

GPUS=1


pwd; hostname; date

nvidia-smi

export HYDRA_FULL_ERROR=1

export WANDB_DIR=/share/nas2/walml/wandb
export WANDB_ARTIFACT_DIR=/share/nas2/walml/wandb/artifacts


# export HF_HOME="/share/nas2/walml/cache/huggingface" # hub downloads including models
export HF_HOME="/share/nas2/walml/cache/huggingface_tmp"

# export HF_DATASETS_CACHE="/share/nas2/walml/cache/huggingface/datasets" # load prepared dataset on nas
export HF_DATASETS_CACHE='/state/partition1/walml/cache/huggingface/datasets'  # load on node
rm -rf $HF_DATASETS_CACHE  # clear cache to avoid loading old datasets

# export HF_LOCAL_DATASETS_CACHE='/state/partition1/walml/cache/huggingface/datasets' # evo manual only
# export GZ_EVO_MANUAL_DOWNLOAD_LOC='/share/nas2/walml/tmp/gz-evo'  # evo manual only
# scripts read seed from SEED, default is 42
# SEED=$RANDOM
# echo Using seed $SEED


PYTHON="/share/nas2/walml/miniconda3/envs/zoobot39_cu118_dev/bin/python"
REPO_DIR="/share/nas2/walml/repos/gz-evo"

export SLURM_NTASKS_PER_NODE=$GPUS # this isn't set correctly by old galahad slurm, it sets NTASKS_PER_NODE not SLURM_NTASKS_PER_NODE
echo SLURM_NTASKS_PER_NODE now $SLURM_NTASKS_PER_NODE

# ----

# echo 'Running classification baseline'
# srun $PYTHON $REPO_DIR/gz_evo/core/classification/train.py 

echo 'Running multinomial baseline'
srun $PYTHON $REPO_DIR/gz_evo/core/multinomial/train.py 

echo 'Exiting'