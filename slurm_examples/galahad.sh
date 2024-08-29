#!/bin/bash
#SBATCH --time=71:00:00                                # Time limit hrs:min:sec
#SBATCH --constraint=A100 
#SBATCH --mem=30G
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=baseln
#SBATCH --output=%x.%A.out

pwd; hostname; date

nvidia-smi

export HF_HOME="/share/nas2/walml/cache/huggingface"
export HF_DATASETS_CACHE="/share/nas2/walml/cache/huggingface/datasets"
export HYDRA_FULL_ERROR=1

# SEED=$RANDOM
# echo Using seed $SEED

PYTHON="/share/nas2/walml/miniconda3/envs/zoobot39_cu118_dev/bin/python"
REPO_DIR="/share/nas2/walml/repos/gz-evo"

# echo 'Running classification baseline'
# $PYTHON $REPO_DIR/baseline/classification_baseline.py 

echo 'Running regression baseline'
$PYTHON $REPO_DIR/baseline/regression_baseline.py 
