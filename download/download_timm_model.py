
import os


if __name__ == '__main__':

    os.environ["HF_HOME"] = "/project/def-bovy/walml/cache/huggingface"

    import timm

    for model in [
        'resnet18',
        'resnet50',
        'convnext_nano',
        'convnext_base',
    ]:
        model = timm.create_model(model, pretrained=True)