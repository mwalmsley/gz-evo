import logging
import omegaconf
import os

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
import datasets

# from galaxy_datasets.pytorch import galaxy_datamodule

# from galaxy_datasets import transforms
from galaxy_datasets.transforms import GalaxyViewTransform, default_view_config

# from zoobot.shared import schemas
# from zoobot.pytorch.estimators import define_model


import baseline_models  # relative import
import baseline_datamodules  # relative import
import baseline_training  # relative import

def main():
    architecture_name = 'resnet50'
    # dataset_name='gz_evo'
    # dataset_name='gz_hubble'
    dataset_name='gz2'

    cfg = baseline_training.get_config(architecture_name, dataset_name)
    datamodule = set_up_data(cfg)
    baseline_training.run_training(cfg, datamodule)


def set_up_data(cfg):


    # schema = schemas.gz_evo_v1_public_schema
    # schema = schemas.gz_hubble_ortho_schema
    # schema = schemas.cosmic_dawn_ortho_schema
    # label_cols = schema.label_cols

    dataset_dict = datasets.load_dataset(
        f"mwalmsley/{cfg.dataset_name}", 
        name=cfg.subset_name, 
        keep_in_memory=cfg.keep_in_memory, 
        cache_dir=cfg.hf_cache_dir,
        download_mode=cfg.download_mode,
    )
    
    dataset_dict.set_format("torch")
    # print(dataset_dict['train'][0]['image'])
    # print(dataset_dict['train'][0]['image'].min(), dataset_dict['train'][0]['image'].max())

    # naively, only train on examples with labels, from all telescopes
    dataset_dict = dataset_dict.filter(lambda example: example['summary'] != '')  # remove examples without labels
    # example = dataset_dict['train'][0]
    # print(example['summary'], type(example['summary']))

    transform_config = default_view_config()
    # transform_config = fast_view_config()
    transform_config.random_affine['scale'] = (1.25, 1.45)  # touch more zoom to compensate for loosing 24px crop
    transform_config.erase_iterations = 0  # disable masking small patches for now
    transform = GalaxyViewTransform(transform_config)

    # any callable that takes an HF example (row) and returns a label
    # target_transform = None  
    # load the summary column as integers

    def target_transform(example):
        example['label'] = baseline_datamodules.LABEL_ORDER_DICT[example['summary']]
        # optionally could delete the other keys besides image and id_str
        return example


    datamodule = baseline_datamodules.GenericDataModule(
        dataset_dict=dataset_dict,
        train_transform=transform,
        test_transform=transform,
        target_transform=target_transform,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
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



if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting classification baseline")

    seed = 42
    pl.seed_everything(seed)

    main()
