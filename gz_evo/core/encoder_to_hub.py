import logging

import glob
import timm

import baseline_models


def publish_encoder_to_hf_hub(architecture_name, checkpoint_dir, baseline_type):
    logging.info(f"Publishing {architecture_name} encoder to Hugging Face Hub")
    checkpoints = list(glob.glob(checkpoint_dir + '/checkpoints/*.ckpt'))
    checkpoints.sort()
    assert checkpoints, checkpoint_dir
    checkpoint_loc = checkpoints[-1]

    if baseline_type == 'classification':
        model_lightning_class = baseline_models.ClassificationBaseline
    elif baseline_type == 'regression':
        model_lightning_class = baseline_models.RegressionBaseline
    else:
        raise ValueError(f"Unknown baseline version: {baseline_type}")
    
    timm_encoder = model_lightning_class.load_from_checkpoint(checkpoint_loc).encoder

    timm.models.push_to_hf_hub(timm_encoder, f'mwalmsley/baseline-encoder-{baseline_type}-{architecture_name}')


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    # utility script to load trained baseline checkpoints, extract the timm encoder, and push to Hugging Face Hub
    # finetune script can then expect any timm encoder and pull from the hub
    # so other users can make their own timm encoders separately, and finetuning will work



    # results_dir = '/share/nas2/walml/repos/gz-evo/results/baselines/classification/'
    # # copy-paste from classification_baseline.py
    # # optionally can change architecture name to anything else, it's just for our benefit
    # for _, architecture_name, checkpoint_dir in [
    #     ('gz_evo', 'tf_efficientnetv2_s',  results_dir + 'tf_efficientnetv2_s_534895718'),
    #     ('gz_evo', 'maxvit_base',  results_dir + 'maxvit_base_534895718'),
    #     ('gz_evo', 'convnext_base',  results_dir + 'convnext_base_534895718'),
    #     ('gz_evo', 'convnext_base_finetuned',  results_dir + 'convnext_base_finetune_494155588'),
    #     ('gz_evo', 'convnext_nano',  results_dir + 'convnext_nano_534895718'),  # not great, might need redo. Redone and still not great.
    #     # ('gz_evo', 'convnext_nano',  results_dir + 'convnext_nano_finetune_494155588'),  # failed
    #     ('gz_evo', 'maxvit_tiny',  results_dir + 'maxvit_tiny_rw_224_534895718'),
    #     ('gz_evo', 'resnet50',  results_dir + 'resnet50_534895718'),
    #     ('gz_evo', 'convnextv2_base.fcmae_ft_in22k_in1k',  results_dir + 'convnextv2_base.fcmae_ft_in22k_in1k_534895718'), 
    # ]:
    #     publish_encoder_to_hf_hub(architecture_name, checkpoint_dir, 'classification')
    # logging.info("Published all classification encoders to Hugging Face Hub")

    # repeat for regression
    results_dir = '/share/nas2/walml/repos/gz-evo/results/baselines/regression/'
    for _, architecture_name, checkpoint_dir in [
        # ('gz_evo', 'resnet50',  results_dir + 'resnet50_534895718_1746547649'),
        # ('gz_evo', 'convnext_nano',  results_dir + 'convnext_nano_534895718_1746542691'),
        # ('gz_evo', 'convnext_base',  results_dir + 'convnext_base_534895718_1746547550'),
        # ('gz_evo', 'tf_efficientnetv2_s',  results_dir + 'tf_efficientnetv2_s_534895718_1746547782'),
        # ('gz_evo', 'maxvit_tiny',  results_dir + 'maxvit_tiny_534895718_1746547757'),
        # ('gz_evo', 'convnext_large', results_dir + 'convnext_large_534895718_1746548055'),
        # ('gz_evo', 'maxvit_base', results_dir + 'maxvit_base_534895718_1746561752'),
        # ('gz_evo', 'maxvit_large', results_dir + 'maxvit_large_534895718_1746561915'),  # skip, not yet ready
        # ('gz_evo', 'tf_efficientnetv2_l', results_dir + 'tf_efficientnetv2_l_534895718_1746653208'),
        # ('gz_evo', 'tf_efficientnetv2_m', results_dir + 'tf_efficientnetv2_m_534895718_1746653116'),
        ('gz_evo', 'vit_so400m_siglip_ft', results_dir + 'vit_so400m_siglip_finetune_534895718_1746914089'),  # finetuned with lr_decay=0.5
        # TODO vit_so400m siglip no ft
    ]:
        publish_encoder_to_hf_hub(architecture_name, checkpoint_dir, 'regression')
    logging.info("Published all regression encoders to Hugging Face Hub")


# regression/convnext_large_534895718_1746548055/checkpoints/
# regression/maxvit_base_534895718_1746561752/checkpoints/
# regression/maxvit_large_534895718_1746561915/checkpoints/
# regression/tf_efficientnetv2_l_534895718_1746653208/
# regression/tf_efficientnetv2_l_534895718_1746653208/checkpoints/
# regression/tf_efficientnetv2_m_534895718_1746653116/
# regression/tf_efficientnetv2_m_534895718_1746653116/checkpoints/
