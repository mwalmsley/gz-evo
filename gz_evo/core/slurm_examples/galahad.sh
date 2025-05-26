#!/bin/bash
#SBATCH --time=300:00:00                                # Time limit hrs:min:sec
#SBATCH --constraint=A100 
#SBATCH --ntasks 1
#SBATCH --mem=60G  # no need for high mem
#SBATCH --cpus-per-task=10
#SBATCH --job-name=baseln
#SBATCH --output=%x.%A.out
#SBATCH --exclusive  

pwd; hostname; date

nvidia-smi

export HYDRA_FULL_ERROR=1

export WANDB_DIR=/share/nas2/walml/wandb
export WANDB_ARTIFACT_DIR=/share/nas2/walml/wandb/artifacts

# export HF_DATASETS_IN_MEMORY_MAX_SIZE=50000000000  # Use to ignore HF_DATASETS_CACHE and keep in mem if possible.
# seems to make dataset.filter() very slow, perhaps it turns off the column selection

export HF_HOME="/share/nas2/walml/cache/huggingface" # hub downloads
export HF_DATASETS_CACHE="/share/nas2/walml/cache/huggingface/datasets" # load prepared dataset on nas
# export HF_DATASETS_CACHE='/state/partition1/walml/cache/huggingface/datasets'  # load on node

export HF_LOCAL_DATASETS_CACHE='/state/partition1/walml/cache/huggingface/datasets' # evo manual only
export GZ_EVO_MANUAL_DOWNLOAD_LOC='/share/nas2/walml/tmp/gz-evo'  # evo manual only
# scripts read seed from SEED, default is 42
# SEED=$RANDOM
# echo Using seed $SEED

PYTHON="/share/nas2/walml/miniconda3/envs/zoobot39_cu118_dev/bin/python"
REPO_DIR="/share/nas2/walml/repos/gz-evo"

echo 'Running classification baseline'
$PYTHON $REPO_DIR/gz_evo/core/classification/train.py 

# echo 'Running multinomial baseline'
# $PYTHON $REPO_DIR/gz_evo/core/multinomial/train.py 

# publish to hub
# echo 'Publishing encoders to hub'
# $PYTHON $REPO_DIR/gz_evo/encoder_to_hub.py
