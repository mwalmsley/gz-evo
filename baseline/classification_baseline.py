import logging
import os

import pytorch_lightning as pl
import datasets
import numpy as np

from galaxy_datasets.transforms import GalaxyViewTransform, default_view_config, minimal_view_config

import baseline_models  # relative import
import baseline_datamodules  # relative import
import baseline_training  # relative import

def main():
    # these are all good to be run on gz evo on galahad (although filter takes ages)
    # do not raise the learning rate, it seems to break training (strangely)

    # architecture_name = 'resnet50'
    # architecture_name = 'resnet50_clip.openai'  # timm looks for "resnet50_clip" and doesn't find it, probaly as it is designed for openclip

    # architecture_name = 'convnext_atto'
    # architecture_name = 'convnext_pico'
    # architecture_name = 'convnext_nano'
    # architecture_name = 'convnext_base'
    # architecture_name = 'convnext_large'
    # architecture_name = 'convnextv2_base.fcmae_ft_in22k_in1k'
    # architecture_name = 'convnext_base.clip_laion2b_augreg_ft_in12k'

    # architecture_name = 'efficientnet_b0'

    # architecture_name = 'tf_efficientnetv2_s'
    # architecture_name = 'tf_efficientnetv2_m'
    architecture_name = 'tf_efficientnetv2_l'

    # architecture_name = 'maxvit_tiny'
    # architecture_name = 'maxvit_small'
    # architecture_name = 'maxvit_base'
    # architecture_name = 'maxvit_large'

    # architecture_name = 'convnext_nano_finetune'
    # architecture_name = 'convnext_base_finetune'

    # base evo now started as 7222, long filtering step, others waiting for this
    # filtering completed but epoch is pretty long, 1 hour
    # starting another with atto for quick check, 7263
    # and starting one with nano and a filter, 7278

    dataset_name='gz_evo'
    # dataset_name='gz_hubble'
    # dataset_name='gz2'
    save_dir = f"results/baselines/classification/{architecture_name}_{np.random.randint(1e9)}"  # relative

    cfg = baseline_training.get_config(architecture_name, dataset_name, save_dir)
    datamodule = set_up_task_data(cfg)

    lightning_model = get_lightning_model(cfg)

    baseline_training.run_training(cfg, lightning_model, datamodule)


def set_up_task_data(cfg):

    dataset_dict = baseline_training.get_dataset_dict(cfg)

    # naively, only train on examples with labels, from all telescopes
    print(dataset_dict['train'].num_rows)
    dataset_dict = dataset_dict.filter(
        has_labels,
        input_columns='summary',  # important to specify, for speed
        # load_from_cache_file=False
        num_proc=cfg.num_workers
    )
    print(dataset_dict['train'].num_rows)
    # print(dataset_dict)
    print(dataset_dict['train'][0]['summary'], 'is the example summary')
    print(dataset_dict['train'][1]['summary'], 'is the example summary')
    print(dataset_dict['train'][2]['summary'], 'is the example summary')
    print(dataset_dict['train'][3]['summary'], 'is the example summary')

    # dataset_dict = dataset_dict.flatten_indices() #num_proc=cfg.num_workers)

    dataset_dict.set_format("torch")  #  breaks flatten_indices if you do it first!

    train_transform_config = default_view_config()
    test_transform_config = minimal_view_config()
    # transform_config = fast_view_config()
    train_transform_config.random_affine['scale'] = (1.0, 1.4)
    train_transform_config.erase_iterations = 0  # disable masking small patches for now

    train_transform = GalaxyViewTransform(train_transform_config)
    test_transform = GalaxyViewTransform(test_transform_config)

    # any callable that takes an HF example (row) and returns a label
    # load the summary column as integers
    def target_transform(example):
        example['label'] = baseline_datamodules.LABEL_ORDER_DICT[example['summary']]
        # optionally could delete the other keys besides image and id_str
        return example

    datamodule = baseline_datamodules.GenericDataModule(
        dataset_dict=dataset_dict,
        train_transform=train_transform,
        test_transform=test_transform,
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



def evaluate():

    # debug_dir = '/home/walml/repos/gz-evo/results/baselines/classification/'
    # beluga_dir = '/project/def-bovy/walml/repos/gz-evo/results/baselines/classification/'
    results_dir = '/share/nas2/walml/repos/gz-evo/results/baselines/classification/'


# WIP
# classification/maxvit_large_534895718/
# classification/maxvit_large_534895718/checkpoints/
# TODO maxvit small
# TODO other efficientnets


    for dataset_name, architecture_name, checkpoint_dir in [
        # ('gz_evo', 'tf_efficientnetv2_s',  results_dir + 'tf_efficientnetv2_s_534895718'),
        # ('gz_evo', 'maxvit_base',  results_dir + 'maxvit_base_534895718'),
        # ('gz_evo', 'convnext_base',  results_dir + 'convnext_base_534895718'),
        # ('gz_evo', 'convnext_base',  results_dir + 'convnext_base_finetune_494155588'),
        # ('gz_evo', 'convnext_nano',  results_dir + 'convnext_nano_534895718'),  # not great, might need redo
        # ('gz_evo', 'convnext_nano',  results_dir + 'convnext_nano_finetune_494155588'),  # failed
        ('gz_evo', 'maxvit_tiny',  results_dir + 'maxvit_tiny_rw_224_534895718'),
        ('gz_evo', 'resnet50',  results_dir + 'resnet50_534895718'),
        ('gz_evo', 'convnextv2_base.fcmae_ft_in22k_in1k',  results_dir + 'convnextv2_base.fcmae_ft_in22k_in1k_534895718'), 
    ]:
        logging.info(f"Evaluating {dataset_name} {architecture_name} {checkpoint_dir}")
        cfg = baseline_training.get_config(architecture_name, dataset_name, save_dir='foobar') # save_dir is not used

        # overrride batch size
        # cfg.batch_size = cfg.batch_size // 2  
        # not sure why but predictions don't fit with full batch size,
        # even though neither training nor predictions are distributed
        # it seems to be that the model doesn't fit in memory when second gpu is in use, even though it is not used
        try:
            baseline_training.evaluate_single_model(
                checkpoint_dir, cfg, model_lightning_class=baseline_models.ClassificationBaseline, task_data_func=set_up_task_data
            )
        except Exception as e:
            logging.error(f"Failed to evaluate {dataset_name} {architecture_name} {checkpoint_dir}")
            logging.error(e)

    logging.info('Test predictions complete for all models. Exiting.')
    
    """
    rsync -avz walml@beluga.alliancecan.ca:"/project/def-bovy/walml/repos/gz-evo/results/baselines/classification" --exclude="*.ckpt" results/baselines
    rsync -avz -e 'ssh -A -J walml@external.jb.man.ac.uk' --exclude="*.ckpt" walml@galahad.ast.man.ac.uk:"/share/nas2/walml/repos/gz-evo/results/baselines/classification" --exclude="*.ckpt" results/baselines
    """


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
    
    return lightning_model

def has_labels(summary):
    # raise ValueError(summary)
    assert isinstance(summary, str), f"summary is not a string, but a {type(summary), {summary}}"
    return summary != ''

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting classification baseline")

    seed = 42
    # seed = 43  # for testing the new 'core finetuning' idea
    pl.seed_everything(seed)

    main()
    # evaluate()
