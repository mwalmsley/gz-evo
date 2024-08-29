import logging

import numpy as np
import pytorch_lightning as pl
import datasets

from galaxy_datasets.shared import label_metadata
from galaxy_datasets.transforms import GalaxyViewTransform, default_view_config

import baseline_models  # relative import
import baseline_datamodules  # relative import
import baseline_training  # relative import


def main():
    # architecture_name = 'resnet50'
    architecture_name = 'convnext_nano'
    # architecture_name = 'convnext_pico'

    # dataset_name='gz_evo'
    # dataset_name='gz_hubble'
    dataset_name='gz2'
    save_dir = f"results/baselines/regression/{architecture_name}_{np.random.randint(1e9)}"  # relative

    cfg = baseline_training.get_config(architecture_name, dataset_name, save_dir)
    datamodule = set_up_task_data(cfg)

    lightning_model = get_lightning_model(cfg)

    baseline_training.run_training(cfg, lightning_model, datamodule)


def set_up_task_data(cfg):

    dataset_dict = datasets.load_dataset(
        f"mwalmsley/{cfg.dataset_name}", 
        name=cfg.subset_name, 
        keep_in_memory=cfg.keep_in_memory, 
        cache_dir=cfg.hf_cache_dir,
        download_mode=cfg.download_mode,
    )
    
    dataset_dict.set_format("torch")
    print(dataset_dict['train'][0]['image'])
    print(dataset_dict['train'][0]['image'].min(), dataset_dict['train'][0]['image'].max())

    # naively, only train on examples with labels, from all telescopes
    dataset_dict = dataset_dict.filter(lambda example: example['summary'] != '')  # remove examples without labels
    # example = dataset_dict['train'][0]
    # print(example['summary'], type(example['summary']))

    transform_config = default_view_config()
    # transform_config = fast_view_config()
    transform_config.random_affine['scale'] = (1.25, 1.45)  # touch more zoom to compensate for loosing 24px crop
    transform_config.erase_iterations = 0  # disable masking small patches for now
    transform = GalaxyViewTransform(transform_config)

    datamodule = baseline_datamodules.GenericDataModule(
        dataset_dict=dataset_dict,
        train_transform=transform,
        test_transform=transform,
        # target_transform=target_transform,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=seed
    )
    return datamodule


def get_lightning_model(cfg):

    if cfg.dataset_name == 'gz2':
        question_answer_pairs = label_metadata.gz2_ortho_pairs
    elif cfg.dataset_name == 'gz_hubble':
        question_answer_pairs = label_metadata.hubble_ortho_pairs
    elif cfg.dataset_name == 'gz_evo':
        question_answer_pairs = label_metadata.gz_evo_v1_public_pairs
    else:
        raise ValueError(f"Unknown dataset_name: {cfg.dataset_name}")

    # lazy duplicate
    question_totals_keys = [question + '_total-votes' for question in question_answer_pairs.keys()]
    answer_keys = [q + a + '_fraction' for q, a_list in question_answer_pairs.items() for a in a_list]
    label_cols = question_totals_keys + answer_keys
    # def target_transform(example):
    #     example = dict([example[k] for k in answer_keys + question_totals_keys + ['id_str', 'image'])
    #     # optionally could delete the other keys besides image and id_str
    #     return example

    lightning_model = baseline_models.RegressionBaseline(
        label_cols=label_cols,
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

    return lightning_model


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting regression baseline")

    seed = 42
    pl.seed_everything(seed)

    main()
