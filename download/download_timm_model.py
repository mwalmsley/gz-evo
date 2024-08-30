
import os


if __name__ == '__main__':

    os.environ["HF_HOME"] = "/project/def-bovy/walml/cache/huggingface"

    # import huggingface_hub
    import timm

    for model in [
        # 'resnet50.a1_in1k',
        # 'resnet50.a1_in1k',
        # 'resnet50_clip.openai',
        # 'resnet101.a1h_in1k',
        'resnet50',
        'resnet50',
        'resnet50_clip.openai',
        'resnet101',

        'convnext_atto', 
        'convnext_pico', 
        'convnext_nano',
        "convnextv2_nano.fcmae",
        "convnextv2_nano.fcmae_ft_in22k_in1k",
        'convnext_tiny',
        'convnext_small', 
        'convnext_base',
        # 'convnext_large',
        # 'convnext_xlarge',

        # 'laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg'

        'convnextv2_nano.fcmae',
        'convnextv2_base.fcmae',
        'convnextv2_nano.fcmae_ft_in22k_in1k',
        'convnextv2_base.fcmae_ft_in22k_in1k',
        'convnext_base.clip_laion2b_augreg_ft_in12k',

        'efficientnet_b0',
        'tf_efficientnetv2_s',
        # 'tf_efficientnetv2_l',
        # 'tf_efficientnetv2_xl'

        'maxvit_tiny_rw_224',
        'maxvit_rmlp_small_rw_224', 
        'maxvit_rmlp_base_rw_224',
        # 'maxvit_large_tf_224',
        # 'maxvit_xlarge_tf_224'

        # not pretrained :(
        # 'vit_base_patch16_clip_224.openai',
        # 'vit_base_patch32_clip_224.openai',

        'vit_base_patch16_224.augreg2_in21k_ft_in1k',

        # 'vit_base_patch16_clip_224.laion2b_ft_in12k',
        # 'vit_base_patch32_clip_224.laion2b_ft_in12k',
        
        'convnext_base.clip_laion2b_augreg_ft_in12k',

        'vit_medium_patch32_clip_224.tinyclip_laion400m'

    ]:
        print(model)
        timm.create_model(model, pretrained=True)
        # huggingface_hub.snapshot_download('timm/' + model, repo_type='model', cache_dir='/project/def-bovy/walml/cache/huggingface')