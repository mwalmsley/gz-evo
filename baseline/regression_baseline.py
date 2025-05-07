import logging
import  time
import glob

import omegaconf
import pandas as pd
import torch
import numpy as np
import pytorch_lightning as pl
# import datasets

from galaxy_datasets.shared import label_metadata
from galaxy_datasets.transforms import GalaxyViewTransform, default_view_config, minimal_view_config

import baseline_models  # relative import
import baseline_datamodules  # relative import
import baseline_training  # relative import


def main():

    # with just gz2 first question and no weighting, model seems to learn pretty well
    # now try gz2 and weighting
    # currently running as 7232 atto regression, possible duplicate with 7208
    # looks okay, training well still
    # continuing on to first four gz2 questions, with weighting/atto, 7261
    # horribly broken
    # now adjusted loss

    # architecture_name = 'convnext_atto'
    # architecture_name = 'convnext_pico'
    # architecture_name = 'convnext_nano'
    # architecture_name = 'convnext_base'
    # architecture_name = 'convnext_large'

    # architecture_name = 'convnextv2_base.fcmae_ft_in22k_in1k'
    # architecture_name = 'convnext_base.clip_laion2b_augreg_ft_in12k'

    # architecture_name = 'efficientnet_b0'

    # architecture_name = 'tf_efficientnetv2_s'
    
    # architecture_name = 'maxvit_tiny'
    # architecture_name = 'maxvit_small'
    # architecture_name = 'maxvit_base'
    architecture_name = 'maxvit_large'

    # architecture_name = 'resnet50'
    # architecture_name = 'resnet50_clip.openai'



    dataset_name='gz_evo'
    # dataset_name='gz_hubble'
    # dataset_name='gz2'
    save_dir = f"results/baselines/regression/{architecture_name}_{np.random.randint(1e9)}_{int(time.time())}"  # relative

    cfg = baseline_training.get_config(architecture_name, dataset_name, save_dir)

    lightning_model = get_lightning_model(cfg)
    
    datamodule = set_up_task_data(cfg)

    baseline_training.run_training(cfg, lightning_model, datamodule)


def evaluate():

    # dataset_name = 'gz2'
    # architecture_name = 'convnext_atto'
    # checkpoint_dir = '/home/walml/repos/gz-evo/results/baselines/regression/convnext_atto_534895718'
    # evaluate_single_model(checkpoint_dir, architecture_name, dataset_name)

    # debug_dir = '/home/walml/repos/gz-evo/results/baselines/regression/'
    # beluga_dir = '/project/def-bovy/walml/repos/gz-evo/results/baselines/regression/'
    results_dir = '/share/nas2/walml/repos/gz-evo/results/baselines/regression/'

    for dataset_name, architecture_name, checkpoint_dir in [
        ('gz_evo', 'resnet50',  results_dir + 'resnet50_534895718_1746547649'),  # technically still training
        # ('gz_evo', 'convnext_nano',  results_dir + 'convnext_nano_534895718_1746542691'),
        # ('gz_evo', 'convnext_base',  results_dir + 'convnext_base_534895718_1746547550'),
        # ('gz_evo', 'tf_efficientnetv2_s',  results_dir + 'tf_efficientnetv2_s_534895718_1746547782')
        ('gz_evo', 'maxvit_tiny_rw_224',  results_dir + 'maxvit_tiny_rw_224_534895718_1746547757')
    ]:

        logging.info(f"Evaluating {dataset_name} {architecture_name} {checkpoint_dir}")
        cfg = baseline_training.get_config(architecture_name, dataset_name, save_dir='foobar')

        baseline_training.evaluate_single_model(
            checkpoint_dir, cfg, model_lightning_class=baseline_models.RegressionBaseline, task_data_func=set_up_task_data
            )

    logging.info('Test predictions complete for all models. Exiting.')
    
    """
    rsync -avz walml@beluga.alliancecan.ca:"/project/def-bovy/walml/repos/gz-evo/results/baselines/regression" --exclude="*.ckpt" results/baselines
    """



# def safe_cpu_cast(x):
#     try:
#         return x.cpu()
#     except AttributeError:
#         return x

def set_up_task_data(cfg):

    dataset_dict = baseline_training.get_dataset_dict(cfg)

    # TODO for evo, filter like this but each column
    # for GZ2 filter not yet needed, masking is enough
    # dataset_dict = dataset_dict.filter(
    #     has_minimal_votes,
    #     input_columns=['smooth-or-featured-gz2_total-votes']
    # )

    # print(dataset_dict['train'].num_rows)
    # dataset_dict = dataset_dict.filter(
    #     has_minimal_votes,
    #     input_columns=['has-spiral-arms-gz2_total-votes']
    # )
    # print(dataset_dict['train'].num_rows)

    dataset_dict.set_format("torch")

    train_transform_config = default_view_config()
    test_transform_config = minimal_view_config()
    # transform_config = fast_view_config()
    train_transform_config.random_affine['scale'] = (1.0, 1.4)
    train_transform_config.erase_iterations = 0  # disable masking small patches for now


    datamodule = baseline_datamodules.GenericDataModule(
        dataset_dict=dataset_dict,
        train_transform=GalaxyViewTransform(train_transform_config),
        test_transform=GalaxyViewTransform(test_transform_config),
        # target_transform=target_transform,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=seed
    )
    return datamodule


def get_lightning_model(cfg):

    if cfg.dataset_name == 'gz2':
        # if cfg.debug:
        # only use the first question, should be easy for GZ2
        # logging.warning("Using debug question_answer_pairs, first GZ2 question only")
        # question_answer_pairs =  {'smooth-or-featured-gz2': ['_smooth', '_featured-or-disk', '_artifact']}
        # logging.warning("Using debug question_answer_pairs, four GZ2 question only")
        # question_answer_pairs =  {
        #     'smooth-or-featured-gz2': ['_smooth', '_featured-or-disk', '_artifact'],
        #     'disk-edge-on-gz2': ['_yes', '_no'],
        #     'has-spiral-arms-gz2': ['_yes', '_no'],
        #     'bar-gz2': ['_yes', '_no']
        # }

        # else:
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

    if cfg.compile_encoder:
        logging.info('Compiling model')
        lightning_model = torch.compile(lightning_model, mode="reduce-overhead")

    return lightning_model

def has_minimal_votes(gz2_votes, threshold=10):
    return gz2_votes >= threshold

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting regression baseline")

    seed = 42
    pl.seed_everything(seed)

    # main()
    evaluate()
