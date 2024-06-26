#!/bin/bash
#SBATCH --time=00:15:00  
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

# mkdir $SLURM_TMPDIR/cache

export NCCL_BLOCKING_WAIT=1

export HF_HOME=/project/def-bovy/walml/cache/huggingface
export HF_DATASETS_CACHE=/project/def-bovy/walml/cache/huggingface/datasets
# no internet on worker nodes
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

# SEED=1  # don't change this when dividing up dataset

PYTHON=/home/walml/envs/zoobot39_dev/bin/python
source ~/envs/zoobot39_dev/bin/activate
REPO_DIR=/project/def-bovy/walml/repos/gz-evo

echo 'Running minimal baseline'
srun $PYTHON $REPO_DIR/baseline/minimal_baseline.py 
