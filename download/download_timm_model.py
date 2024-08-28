
import os


if __name__ == '__main__':

    os.environ["HF_HOME"] = "/project/def-bovy/walml/cache/huggingface"

    import timm

    for model in [
        'resnet18',
        'resnet50',
        'resnet101',

        'convnext_pico', 
        'convnext_nano',
        'convnext_tiny',
        'convnext_small', 
        'convnext_base',
        'convnext_large',
        # 'convnext_xlarge',

        'convnextv2_nano.fcmae',
        'convnextv2_nano.fcmae_ft_in22k_in1k',
        'convnextv2_base.fcmae_ft_in22k_in1k',
        'convnext_base.clip_laion2b_augreg_ft_in12k_in1',

        'efficientnet_b0',
        'tf_efficientnetv2_s',
        'tf_efficientnetv2_l',
        # 'tf_efficientnetv2_xl'

        'maxvit_tiny_rw_224',
        'maxvit_rmlp_small_rw_224', 
        'maxvit_rmlp_base_rw_224',
        'maxvit_large_tf_224',
        # 'maxvit_xlarge_tf_224'

    ]:
        model = timm.create_model(model, pretrained=True)