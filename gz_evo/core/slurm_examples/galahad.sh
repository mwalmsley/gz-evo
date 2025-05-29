#!/bin/bash
#SBATCH --time=300:00:00                                # Time limit hrs:min:sec
#SBATCH --constraint=A100 
#SBATCH --mem=60G  # no need for high mem
#SBATCH --cpus-per-task=10
#SBATCH --job-name=baseln
#SBATCH --output=%x.%A.out
#SBATCH --exclusive  
#SBATCH --ntasks-per-node=2

#### SBATCH --ntasks 1

pwd; hostname; date

nvidia-smi

# NCCL socket varies by node, needs some hacking here to set correctly

# Extract the number from SLURMD_NODENAME (e.g., compute-0-99 -> 99)
node_number=$(echo $SLURMD_NODENAME | grep -o -E '[0-9]+$')

# Check the range of the number and set NCCL_SOCKET_IFNAME accordingly
if (( 0 <= $node_number && $node_number < 100 )); then
    export NCCL_SOCKET_IFNAME='em1'
elif (( 100 <= $node_number && $node_number < 200 )); then
    export NCCL_SOCKET_IFNAME='eno1'
else
    echo "SLURMD_NODENAME is out of expected range."
fi


export HYDRA_FULL_ERROR=1

export WANDB_DIR=/share/nas2/walml/wandb
export WANDB_ARTIFACT_DIR=/share/nas2/walml/wandb/artifacts

# modular cache for torch.compile
# https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html
export TORCHINDUCTOR_CACHE_DIR="/share/nas2/walml/cache/torchinductor"  

# export HF_DATASETS_IN_MEMORY_MAX_SIZE=50000000000  # Use to ignore HF_DATASETS_CACHE and keep in mem if possible.
# seems to make dataset.filter() very slow, perhaps it turns off the column selection

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

echo 'Running classification baseline'
$PYTHON $REPO_DIR/gz_evo/core/classification/train.py 

# echo 'Running multinomial baseline'
# $PYTHON $REPO_DIR/gz_evo/core/multinomial/train.py 

# publish to hub
# echo 'Publishing encoders to hub'
# $PYTHON $REPO_DIR/gz_evo/encoder_to_hub.py
