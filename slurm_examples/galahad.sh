#!/bin/bash
#SBATCH --time=71:00:00                                # Time limit hrs:min:sec
#SBATCH --constraint=A100 
#SBATCH --ntasks 1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=baseln
#SBATCH --output=%x.%A.out
#SBATCH --exclusive  

pwd; hostname; date

nvidia-smi

export HYDRA_FULL_ERROR=1

export WANDB_DIR=/share/nas2/walml/wandb
export WANDB_ARTIFACT_DIR=/share/nas2/walml/wandb/artifacts

export HF_HOME="/share/nas2/walml/cache/huggingface"
export HF_DATASETS_CACHE="/share/nas2/walml/cache/huggingface/datasets"

export HF_LOCAL_DATASETS_CACHE='/state/partition1/walml/cache/huggingface/datasets'
export GZ_EVO_MANUAL_DOWNLOAD_LOC='/share/nas2/walml/tmp/gz-evo'

# SEED=$RANDOM
# echo Using seed $SEED

PYTHON="/share/nas2/walml/miniconda3/envs/zoobot39_cu118_dev/bin/python"
REPO_DIR="/share/nas2/walml/repos/gz-evo"

# echo 'Running classification baseline'
# $PYTHON $REPO_DIR/baseline/classification_baseline.py 

echo 'Running regression baseline'
$PYTHON $REPO_DIR/baseline/regression_baseline.py 

# TODO consider adding imagenet normalisation, I do wonder if it might be silently hurting performance
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/transforms_factory.py#L12
