#!/bin/bash
#SBATCH --time=300:00:00                                # Time limit hrs:min:sec
#SBATCH --constraint=A100 
#SBATCH --mem=30G  # no need for high mem
#SBATCH --job-name=baseln
#SBATCH --output=%x.%A.out
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --exclude=compute-0-18
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1

pwd; hostname; date

nvidia-smi 

PYTHON="/share/nas2/walml/miniconda3/envs/zoobot39_cu118_dev/bin/python"
REPO_DIR="/share/nas2/walml/repos/gz-evo"

echo SLURM_NTASKS $SLURM_NTASKS 
echo SLURM_NTASKS_PER_NODE $SLURM_NTASKS_PER_NODE
export SLURM_NTASKS_PER_NODE=$SLURM_NTASKS # this isn't set correctly by old galahad slurm, it sets NTASKS_PER_NODE not SLURM_NTASKS_PER_NODE
echo SLURM_NTASKS_PER_NODE now $SLURM_NTASKS_PER_NODE

# ----

srun $PYTHON $REPO_DIR/gz_evo/core/slurm_examples/cuda_debug.py 
