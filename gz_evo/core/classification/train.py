import os
import logging
import time

import numpy as np
import pytorch_lightning as pl
import datasets
import torch

from galaxy_datasets.transforms import get_galaxy_transform, default_view_config, minimal_view_config
from galaxy_datasets.pytorch import galaxy_datamodule, dataset_utils

from gz_evo.core import  baseline_models, baseline_datamodules, baseline_training


def main():

    # architecture_name = 'resnet50'

    # architecture_name = 'convnext_atto'
    # architecture_name = 'convnext_pico'
    architecture_name = 'convnext_nano'
    # architecture_name = 'convnext_base'
    # architecture_name = 'convnext_large'
    # architecture_name = 'convnextv2_base.fcmae_ft_in22k_in1k'
    # architecture_name = 'convnext_base.clip_laion2b_augreg_ft_in12k'

    # architecture_name = 'efficientnet_b0'

    # architecture_name = 'tf_efficientnetv2_s'
    # architecture_name = 'tf_efficientnetv2_m'
    # architecture_name = 'tf_efficientnetv2_l'

    # architecture_name = 'maxvit_tiny'
    # architecture_name = 'maxvit_small'
    # architecture_name = 'maxvit_base'
    # architecture_name = 'maxvit_large'

    # architecture_name = 'vit_so400m_siglip'
    
    # with lr decay
    # architecture_name = 'convnext_nano_finetune'  
    # architecture_name = 'convnext_base_finetune'
    # architecture_name = 'vit_so400m_siglip_finetune'

    dataset_name='gz_evo'
    # dataset_name='gz_hubble'
    # dataset_name='gz2'
    save_dir = f"results/baselines/classification/{architecture_name}_{np.random.randint(1e9)}_{int(time.time())}" # type: ignore

    cfg = baseline_training.get_config(architecture_name, dataset_name, save_dir)
    
    datamodule = set_up_task_data(cfg)

    lightning_model = get_lightning_model(cfg)

    baseline_training.run_training(cfg, lightning_model, datamodule)


def set_up_task_data(cfg):
    """
    Create datamodule for classification task.

    Args:
        cfg (omegaconf): configuration object

    Returns:
        DataModule: with dataloaders yielding batches of images and labels
    """

    dataset_dict: datasets.DatasetDict = baseline_training.get_dataset_dict(cfg) # type: ignore

    # before we do anything heavy, split between ranks to distribute the work
    dataset_dict = dataset_utils.distribute_dataset_with_lightning(dataset_dict)

    # naively, only train on examples with labels, from all telescopes
    logging.info(f'{dataset_dict["train"].num_rows} training examples before filtering')
    dataset_dict = dataset_dict.filter(
        has_labels,
        input_columns='summary',  # important to specify, for speed
        # load_from_cache_file=False
        num_proc=cfg.num_workers
    )
    # filter will create an index mapping and need flatten_indices later for speed
    logging.info(f'{dataset_dict["train"].num_rows} training examples after filtering')
    # logging.info(dataset_dict)
    logging.info(f'{dataset_dict["train"][0]["summary"]} is an example summary')
    logging.info(f'{dataset_dict["train"][1]["summary"]} is another example summary')

    # we need to flatten after the filter
    # do validation split here, which flattens anyway after shuffling
    # this avoids flattening *again* with add_column below
    dataset_dict = dataset_utils.add_validation_split(dataset_dict=dataset_dict, seed=seed, num_workers=cfg.num_workers)
    # also flatten test indices post-filter, where we can control num_proc
    logging.info('Flattening indices for test set')
    dataset_dict['test'] = dataset_dict['test'].flatten_indices(num_proc=cfg.num_workers)

    # map to calculate labels
    def summary_to_label(summary):
        return baseline_datamodules.LABEL_ORDER_DICT[summary]
    # add_column includes a flatten_indices call internally, but no num_proc option, so flatten first
    for split in dataset_dict:
        # operating on a single column seems much quicker than mapping the whole dataset
        dataset_dict[split] = dataset_dict[split].add_column('label', [summary_to_label(x) for x in dataset_dict[split]['summary']])


    train_transform_config = default_view_config()
    test_transform_config = minimal_view_config()
    train_transform_config.random_affine['scale'] = (1.0, 1.4)
    train_transform_config.erase_iterations = 0  # disable masking small patches for now
    # train_transform_config = fast_view_config()
    # test_transform_config = fast_view_config()

    train_transform = get_galaxy_transform(train_transform_config)
    test_transform = get_galaxy_transform(test_transform_config)

    datamodule = galaxy_datamodule.HuggingFaceDataModule(
        dataset_dict=dataset_dict,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        iterable=cfg.iterable,  # if True, will use IterableDataset, else use MapDataset
        prefetch_factor=cfg.prefetch_factor,
        seed=seed
    )

    # batch = next(iter(datamodule.test_dataloader()))
    # images, _ = batch
    # wandb_logger.experiment.log(
    #     {
    #         "test_examples": [
    #             wandb.Image(image) for image in images[:5]
    #         ]  # , caption=f'label: {label}') for image, label in zip(images, labels)]
    #     }
    # )

    # check dynamic range
    # logging.info('Images: {} to {}, dtype={}'.format(images.min(), images.max(), images.dtype))

    return datamodule


def get_lightning_model(cfg):
    lightning_model = baseline_models.ClassificationBaseline(
        architecture_name=cfg.architecture_name,
        channels=cfg.channels,
        timm_kwargs={
            'drop_path_rate': cfg.drop_path_rate, 
            'pretrained': cfg.pretrained
        },
        head_kwargs={
            'dropout_rate': cfg.dropout_rate,
            'num_classes': len(baseline_datamodules.LABEL_ORDER_DICT.keys())
        },
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    if cfg.compile_encoder:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True  # fall back to eager mode on errors
        logging.info('Compiling model')
        lightning_model = torch.compile(lightning_model, mode="default")
    
    return lightning_model


def has_labels(summary):
    assert isinstance(summary, str), f"summary is not a string, but a {type(summary), {summary}}"
    return summary != ''


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting classification baseline")

    seed: int = os.environ.get('SEED', 42)  # type: ignore
    logging.info(f"Using seed: {seed}")
    # seed = 41  # maxvit small has nan problem
    pl.seed_everything(seed)

    main()
