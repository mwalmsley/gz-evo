#!/bin/bash
#SBATCH --constraint=A100
#SBATCH --time=10-23
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive

pwd; hostname; date

nvidia-smi

# https://docs.wandb.ai/guides/track/environment-variables/
export WANDB_DIR=/share/nas2/walml/wandb
export WANDB_DATA_DIR=/share/nas2/walml/wandb/data
export WANDB_CONFIG_DIR=/share/nas2/walml/wandb/config
export WANDB_CACHE_DIR=/share/nas2/walml/wandb/cache
export WANDB_ARTIFACTS=/share/nas2/walml/code/wandb/artifacts

# https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables
# export HF_HOME=/share/nas2/walml/huggingface
# export HF_HUB_CACHE=/share/nas2/walml/huggingface/hub
# export HF_DATASETS_CACHE=/share/nas2/walml/huggingface/datasets

# store models and datasets on the remote machine
export HF_HOME=/state/partition1/walml/cache/huggingface

# always look here for token, regardless of HF_HOME
export HF_TOKEN_PATH=/share/nas2/walml/cache/huggingface/token

# cache datasets in memory
export HF_DATASETS_IN_MEMORY_MAX_SIZE=21474836480  # 20GB

export NCCL_BLOCKING_WAIT=1

GPUS=1
NUM_NODES=1

REPO_DIR=/share/nas2/walml/repos/gz-evo

PYTHON=/share/nas2/walml/miniconda3/envs/zoobot39_cu118_dev/bin/python

echo HF_HOME $HF_HOME
echo HF_TOKEN_PATH $HF_TOKEN_PATH
# echo $HF_HUB_CACHE
# echo $HF_DATASETS_CACHE

LEARNER="convnext_nano"
ENCODER_HUB_PATH="hf_hub:mwalmsley/baseline-encoder-regression-convnext_nano"
# ENCODER_HUB_PATH="hf_hub:timm/convnext_nano.in12k"  # override for imagenet timm weights

# LEARNER="convnext_base"
# ENCODER_HUB_PATH="hf_hub:mwalmsley/baseline-encoder-regression-convnext_base"

# LEARNER="maxvit_tiny_rw_224"  # only regression, not evo prepared
# ENCODER_HUB_PATH="hf_hub:mwalmsley/baseline-encoder-regression-maxvit_tiny"

# LEARNER="maxvit_rmlp_small_rw_224"
# ENCODER_HUB_PATH="hf_hub:mwalmsley/baseline-encoder-regression-maxvit_small"  # not yet prepared

# LEARNER="resnet50"
# ENCODER_HUB_PATH="hf_hub:mwalmsley/baseline-encoder-regression-resnet50"
# ENCODER_HUB_PATH="hf_hub:timm/resnet50.a1_in1k"

# gap = global average pooling, no gap = attention pooling
# so400m_patch14 = shape optimized ('getting vits in shape'), 400M variant, 14x14 patches
# vit_base_patch16 = 16x16 patches, 86M parameters (from original paper, also the smallest!)
LEARNER="vit_so400m_siglip"
# ENCODER_HUB_PATH="hf_hub:timm/vit_so400m_patch14_siglip_gap_224.v2_webli"  # <- this one for imagenet standard
# ENCODER_HUB_PATH="hf_hub:timm/vit_so400m_patch14_siglip_224.v2_webli" 
# ENCODER_HUB_PATH="hf_hub:timm/vit_base_patch16_siglip_224.v2_webli" 
# ENCODER_HUB_PATH="hf_hub:timm/vit_base_patch16_siglip_gap_224.v2_webli" 
# and my new finetuned version
ENCODER_HUB_PATH="hf_hub:mwalmsley/baseline-encoder-regression-vit_so400m_siglip_ft"

# LEARNER='maxvit_base'
# ENCODER_HUB_PATH="hf_hub:mwalmsley/baseline-encoder-regression-maxvit_base"

# LEARNER='tf_efficientnetv2_m'
# ENCODER_HUB_PATH="hf_hub:mwalmsley/baseline-encoder-regression-tf_efficientnetv2_m"

# LEARNER='tf_efficientnetv2_l'
# ENCODER_HUB_PATH="hf_hub:mwalmsley/baseline-encoder-regression-tf_efficientnetv2_l"

# currently, all models are trained on 224x224 images, but this is now easy to change with the new augmentation transforms

# LEARNER="convnext_nano_gz_euclid"
# ENCODER_HUB_PATH="local:/share/nas2/walml/gz-evo/results/downstream/dnb_debug/jiruf12f/checkpoints/12.ckpt"

# new MAE encoders
# LEARNER="vit_small_custom"
# ENCODER_HUB_PATH="local_hybrid:/share/nas2/walml/repos/zoobot-foundation/results/pretrain/gimg2gls/checkpoints/last.ckpt"
# runs without nan (though not well)

# LEARNER="vit_so400m_siglip"
# this was 22 epoch train on default gz evo images with a slightly high learning rate. Patch size 14.
# dies with nan immediately, even without any training
# ENCODER_HUB_PATH="local_hybrid:/share/nas2/walml/repos/zoobot-foundation/results/pretrain/pegxszsz/checkpoints/last.ckpt"



# LEARNER="convnext_pico" # not on HF
# LEARNER="convnext_base"
# LEARNER="maxvit_rmlp_small_rw_224"
# LEARNER="resnet50"
# TODO for learner in..."jwst"
# for DATASET in "decals10" "is-lsb" "which-lsb"  "gz_rings"
# for DATASET in "euclid_strong_lens_expert_judges" "is-lsb" "which-lsb" "gz_euclid" "gz_rings" 

# DIVISOR=1

for DATASET in "euclid_strong_lens_expert_judges" "gz_euclid"
# for DATASET in "euclid_strong_lens_expert_judges" "gz_euclid" "which-lsb"
# for DATASET in "euclid_strong_lens_expert_judges" "is-lsb" "which-lsb" "gz_euclid" "gz_rings"  
# for DATASET in "euclid_strong_lens_expert_judges" "is-lsb"  # these smaller datasets need extra runs

do

    for DIVISOR in 1 #2 4 8 16 32 64

    do
        echo "Finetuning ${ENCODER_HUB_PATH} on ${DATASET} dataset"

        $PYTHON $REPO_DIR/gz_evo/downstream/finetune.py \
        +learner=$LEARNER \
        ++learner.encoder_hub_path=$ENCODER_HUB_PATH \
        ++learner.normalize=False \
        ++dataset=${DATASET} \
        +hardware=galahad \
        ++wandb=True \
        ++debug=False \
        ++pretrained=True \
        ++divisor=$DIVISOR \
        ++seed=$RANDOM  

    done

done

    # 
        # ++learner.learning_rate=0.00001 \

    # ++learner.learning_rate=0.00001 \

    # ++learner.stochastic_depth=0.4

    #         ++pretrained=imagenet

    

    # DATASET="euclid_strong_lens_expert_judges"
# for LEARNER in "convnext_nano" "efficientnet_b0"
# for LEARNER in "convnext_nano" 