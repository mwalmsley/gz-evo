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

pwd; hostname; date

nvidia-smi 

# torch device should have same order as nvidia-smi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "CUDA_DEVICE_ORDER set to $CUDA_DEVICE_ORDER"

PYTHON="/share/nas2/walml/miniconda3/envs/zoobot39_cu118_dev/bin/python"
REPO_DIR="/share/nas2/walml/repos/gz-evo"

echo SLURM_NTASKS $SLURM_NTASKS 
echo SLURM_NTASKS_PER_NODE $SLURM_NTASKS_PER_NODE
export SLURM_NTASKS_PER_NODE=$SLURM_NTASKS # this isn't set correctly by old galahad slurm, it sets NTASKS_PER_NODE not SLURM_NTASKS_PER_NODE
echo SLURM_NTASKS_PER_NODE now $SLURM_NTASKS_PER_NODE

# -w for specific node
# -h for no header
# -t R for running jobs only
# -o for output format, %A for job ID
# wc -l for line count i.e. number of jobs 
# CURRENT_NODE=$(hostname)
# echo "Current node: $CURRENT_NODE"
# NUM_CURRENT_JOBS=squeue -w $CURRENT_NODE -t R -o %A | wc -l
# echo "1 + Number of running jobs on $CURRENT_NODE: $NUM_CURRENT_JOBS"

# export CUDA_VISIBLE_DEVICES=((NUM_CURRENT_JOBS - 1))
# export CUDA_VISIBLE_DEVICES=$NUM_CURRENT_JOBS
# echo "CUDA_VISIBLE_DEVICES set to $NUM_CURRENT_JOBS"

# ----

srun $PYTHON $REPO_DIR/gz_evo/core/slurm_examples/cuda_debug.py 
