import logging

from galaxy_datasets import transforms


def get_finetuning_transforms(cfg):
    """
    Get the transforms for finetuning.
    """

    if cfg.learner.normalize:
        logging.info("Using normalization loaded from timm")
        import timm
        data_config = timm.data.resolve_model_data_config(cfg.learner.encoder_hub_path)
        # e.g. {'input_size': [3, 224, 224], 'interpolation': 'bicubic', 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'crop_pct': 0.9, 'crop_mode': 'center'}
        normalize = {'mean': data_config['mean'], 'std': data_config['std']}
    else:
        normalize = False

    default_transform_cfg = transforms.default_view_config()
    minimal_transform_cfg = transforms.minimal_view_config()
    default_transform_cfg.normalize = normalize
    minimal_transform_cfg.normalize = normalize

    # update the default transform config based on model chosen via yaml
    default_transform_cfg.pil_to_tensor=True  # HF uses my own loading transform to allow for iterabledataset coversion, not set_format(torch)
    default_transform_cfg.erase_iterations=0  # for simplicity
    default_transform_cfg.output_size=cfg.learner.input_size
    default_transform_cfg.greyscale=cfg.learner.channels==1

    # same for minimal
    minimal_transform_cfg.pil_to_tensor=True
    minimal_transform_cfg.output_size=cfg.learner.input_size
    minimal_transform_cfg.greyscale=cfg.learner.channels==1

    logging.info(f"Train transform: {default_transform_cfg}")
    logging.info(f"Test transform: {minimal_transform_cfg}")

    train_transform = transforms.GalaxyViewTransform(default_transform_cfg)
    test_transform = transforms.GalaxyViewTransform(default_transform_cfg)

    return train_transform, test_transform

