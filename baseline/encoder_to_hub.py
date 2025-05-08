import logging

import glob
import timm

from baseline_models import ClassificationBaseline, RegressionBaseline


def publish_encoder_to_hf_hub(architecture_name, checkpoint_dir, model_lightning_class):
    logging.info(f"Publishing {architecture_name} encoder to Hugging Face Hub")
    checkpoints = list(glob.glob(checkpoint_dir + '/checkpoints/*.ckpt'))
    checkpoints.sort()
    assert checkpoints, checkpoint_dir
    checkpoint_loc = checkpoints[-1]
    timm_encoder = model_lightning_class.load_from_checkpoint(checkpoint_loc).encoder

    if isinstance(model_lightning_class, ClassificationBaseline):
        encoder_training = 'classification'
    elif isinstance(model_lightning_class, RegressionBaseline):
        encoder_training = 'regression'
    else:
        raise ValueError(f"Unknown model_lightning_class: {model_lightning_class}")

    timm.models.push_to_hf_hub(timm_encoder, f'mwalmsley/baseline-encoder-{encoder_training}-{architecture_name}')


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    # utility script to load trained baseline checkpoints, extract the timm encoder, and push to Hugging Face Hub
    # finetune script can then expect any timm encoder and pull from the hub
    # so other users can make their own timm encoders separately, and finetuning will work



    results_dir = '/share/nas2/walml/repos/gz-evo/results/baselines/classification/'
    # copy-paste from classification_baseline.py
    # optionally can change architecture name to anything else, it's just for our benefit
    for _, architecture_name, checkpoint_dir in [
        # ('gz_evo', 'tf_efficientnetv2_s',  results_dir + 'tf_efficientnetv2_s_534895718'),
        # ('gz_evo', 'maxvit_base',  results_dir + 'maxvit_base_534895718'),
        # ('gz_evo', 'convnext_base',  results_dir + 'convnext_base_534895718'),
        # ('gz_evo', 'convnext_base',  results_dir + 'convnext_base_finetune_494155588'),
        # ('gz_evo', 'convnext_nano',  results_dir + 'convnext_nano_534895718'),  # not great, might need redo. Redone and still not great.
        # ('gz_evo', 'convnext_nano',  results_dir + 'convnext_nano_finetune_494155588'),  # failed
        # ('gz_evo', 'maxvit_tiny',  results_dir + 'maxvit_tiny_rw_224_534895718'),
        ('gz_evo', 'resnet50',  results_dir + 'resnet50_534895718'),
        # ('gz_evo', 'convnextv2_base.fcmae_ft_in22k_in1k',  results_dir + 'convnextv2_base.fcmae_ft_in22k_in1k_534895718'), 
    ]:
        
        publish_encoder_to_hf_hub(architecture_name, checkpoint_dir, ClassificationBaseline)

    # repeat for regression TODO

