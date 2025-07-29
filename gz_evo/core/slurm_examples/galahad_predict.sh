#!/bin/bash
#SBATCH --time=300:00:00                                # Time limit hrs:min:sec
#SBATCH --constraint=A100 
#SBATCH --mem=80G  # high mem node is more reliable
#SBATCH --exclude=compute-0-103
#SBATCH --job-name=preds
#SBATCH --output=%x.%A.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks 1
#SBATCH --exclusive

# GPUS=1

pwd; hostname; date

nvidia-smi


PYTHON="/share/nas2/walml/miniconda3/envs/zoobot39_cu118_dev/bin/python"
REPO_DIR="/share/nas2/walml/repos/gz-evo"

echo 'Running multinomial baseline'
$PYTHON $REPO_DIR/gz_evo/core/multinomial/test.py

# publish to hub
# echo 'Publishing encoders to hub'
# $PYTHON $REPO_DIR/gz_evo/encoder_to_hub.py

