import logging
import  time
import os

import torch
import numpy as np
import pytorch_lightning as pl
import datasets

from galaxy_datasets.shared import label_metadata
from galaxy_datasets.transforms import GalaxyViewTransform, default_view_config, minimal_view_config, fast_view_config

from gz_evo.core import baseline_models, baseline_datamodules, baseline_training


def main():

    # architecture_name = 'convnext_atto'
    # architecture_name = 'convnext_pico'
    # architecture_name = 'convnext_nano'
    architecture_name = 'convnext_base'
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

    # architecture_name = 'resnet50'
    # architecture_name = 'resnet50_clip.openai'

    # architecture_name = 'vit_so400m_siglip'

    # with lr decay
    # architecture_name = 'vit_so400m_siglip_finetune'


    dataset_name='gz_evo'
    save_dir = f"results/baselines/regression/{architecture_name}_{np.random.randint(1e9)}_{int(time.time())}"  # type: ignore

    cfg = baseline_training.get_config(architecture_name, dataset_name, save_dir)

    # further exploration: adjust some parameters, see what happens
    # cfg.dropout_rate = 0.0  # (already done, probably, now 0.5 is the default)
   
    # (was 0.4 for paper)
    # cfg.drop_path_rate = 0. 
    cfg.drop_path_rate = 0.75 

    # larger weight decay
    # cfg.weight_decay = 0.2

    # paper: batch size 4096 - mine already is, with two devices, but can play
    # cfg.accumulate_grad_batches = 4 * 4096 / (cfg.batch_size * cfg.devices)
    # cfg.accumulate_grad_batches = 1024 / (cfg.batch_size * cfg.devices)

    # paper: There is a 20-epoch linear warmup and a cosine decaying schedule afterward.
    # https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    # T0 should be about 40 epochs
    # eta_min = 5e-7 say? 2e-5 initially

    lightning_model = get_lightning_model(cfg)
    
    datamodule = set_up_task_data(cfg)

    baseline_training.run_training(cfg, lightning_model, datamodule)


def set_up_task_data(cfg):

    dataset_dict: datasets.DatasetDict = baseline_training.get_dataset_dict(cfg) # type: ignore

    dataset_dict = baseline_datamodules.distribute_dataset_with_lightning(dataset_dict)
    # test set not distributed

    # dataset_dict = baseline_datamodules.pil_to_tensors(dataset_dict, num_workers=cfg.num_workers)
    
    dataset_dict = baseline_datamodules.add_validation_split(dataset_dict=dataset_dict, seed=seed, num_workers=cfg.num_workers)
    # no need to flatten test set, not changed

    train_transform_config = default_view_config()
    test_transform_config = minimal_view_config()
    train_transform_config.random_affine['scale'] = (1.0, 1.4)
    train_transform_config.erase_iterations = 0  # disable masking small patches for now
    # # TODO temp speed test
    # logging.warning("Using fast view config for training and testing transforms")
    # train_transform_config = fast_view_config()
    # test_transform_config = fast_view_config()

    datamodule = baseline_datamodules.GenericDataModule(
        dataset_dict=dataset_dict,
        train_transform=GalaxyViewTransform(train_transform_config).transform,
        test_transform=GalaxyViewTransform(test_transform_config).transform,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        iterable=cfg.iterable,
        prefetch_factor=cfg.prefetch_factor,
        seed=seed
    )
    return datamodule


def get_lightning_model(cfg):

    # dict like {'smooth-or-featured-gz2': ['_smooth', '_featured-or-disk', '_artifact'], ...}
    # specifying the questions and answers
    question_answer_pairs = label_metadata.gz_evo_v1_public_pairs  

    # need counts for loss and fractions for MSE metrics
    answer_keys = [q + a for q, a_list in question_answer_pairs.items() for a in a_list]
    answer_fraction_keys = [col + '_fraction' for col in answer_keys]
    # no need for totals
    label_cols = answer_keys + answer_fraction_keys

    lightning_model = baseline_models.RegressionBaseline(
        label_cols=label_cols,  # filters columns from HF within the model, not really necessary
        architecture_name=cfg.architecture_name,
        channels=cfg.channels,
        timm_kwargs={
            'drop_path_rate': cfg.drop_path_rate, 
            'pretrained': cfg.pretrained
        },
        head_kwargs={
            'dropout_rate': cfg.dropout_rate,
            'question_answer_pairs': question_answer_pairs
        },
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    if cfg.compile_encoder:
        logging.info('Compiling model')
        lightning_model = torch.compile(lightning_model, mode="default")

    return lightning_model


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting multinomial baseline")

    seed: int = os.environ.get('SEED', 42)  # type: ignore
    logging.info(f"Using seed: {seed}")
    pl.seed_everything(seed)


    main()
