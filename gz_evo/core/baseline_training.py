import os
import logging
from dataclasses import asdict
import glob
import pandas as pd

import omegaconf
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch

from zoobot.pytorch.training import train_with_pytorch_lightning

from datasets import load_dataset

import baseline_configs


def get_config(architecture_name, dataset_name, save_dir, debug=False):
    # some path management, adjust to your system as needed 

    # TODO may refactor as argparse or omegaconf
    if os.path.isdir('/project/def-bovy/walml'):
        # hf_cache_dir = '/project/def-bovy/walml/cache/huggingface/datasets'
        # node_cache_dir = os.environ.get('SLURM_TMPDIR', hf_cache_dir)
        subset_name = 'default'
        accelerator="gpu"
        
        # A100
        # batch_size_key = 'A100_batch_size'
        # num_workers = 12

        # V100
        num_workers = 10
        batch_size_key = 'v100_batch_size'

    elif os.path.isdir('/share/nas2'):
        # hf_cache_dir = '/share/nas2/walml/cache/huggingface/datasets'
        # node_cache_dir = '/state/partition1/huggingface_tmp'
        subset_name = 'default'
        num_workers = 8  # of 16 per node
        batch_size_key = 'a100_batch_size'
        accelerator="gpu"

    # TODO add your own system here

    else:
        # hf_cache_dir = None
        # node_cache_dir = hf_cache_dir
        subset_name = 'tiny'  # debugging
        # subset_name = 'default'
        num_workers = 4
        debug = True
        accelerator="auto"
        batch_size_key = 'debug'

    cfg: omegaconf.DictConfig = omegaconf.OmegaConf.create(
        dict(
            architecture_name=architecture_name,
            dataset_name=dataset_name,
            subset_name=subset_name,
            # hf_cache_dir=hf_cache_dir,
            # node_cache_dir=node_cache_dir,
            save_dir=save_dir,

            # download_mode="force_redownload",
            download_mode="reuse_dataset_if_exists",
            num_workers=num_workers,  # 4 for local desktop
            compile_encoder=False,  # beluga, ModuleNotFoundError: No module named 'triton.common'
            pretrained=True, #imagenet 12k, for timm kwargs
            channels=3,

            keep_in_memory=False,
            accelerator=accelerator,
            devices=1,
            nodes=1,
            epochs=1000,
            # strategy="auto",
            precision="16-mixed",  # bf16 doesn't support lgamma for dirichlet loss
            plugins=None,
            patience=5,
            grad_clip_val=0.3,
            # sync_batchnorm=False,  # only one device
            transform_mode='default',
            debug=debug,
            batch_size_key=batch_size_key
        )
    )
    cfg.update(asdict(baseline_configs.MODEL_CONFIGS[cfg.architecture_name]))  # arch, batch_size, etc.
    if debug:
        cfg.batch_size = 32
        cfg.accumulate_grad_batches = 2
        cfg.epochs = 20
    else:
        cfg.batch_size = cfg[cfg.batch_size_key]
        cfg.accumulate_grad_batches = 2048 // cfg.batch_size
        cfg.debug = debug

    logging.info(f'using config:\n{omegaconf.OmegaConf.to_yaml(cfg)}')

    return cfg



def run_training(cfg, lightning_model, datamodule):

    torch.set_float32_matmul_precision("medium")

    wandb_logger = WandbLogger(
        name=f"{cfg.architecture_name} latest",
        project="gz-evo-baseline",
        log_model=False,
        # https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#omegaconf-to-container
        config=omegaconf.OmegaConf.to_container(cfg),
    )

    # log a few images to make sure the transforms look good
    log_images(wandb_logger, datamodule)

    checkpoint_callback, callbacks = train_with_pytorch_lightning.get_default_callbacks(
        cfg.save_dir, cfg.patience
    )

    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        log_every_n_steps=150,
        accelerator=cfg.accelerator,
        devices=cfg.devices,  # per node
        num_nodes=cfg.nodes,
        # strategy=cfg.strategy,
        precision=cfg.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=cfg.epochs,
        default_root_dir=cfg.save_dir,
        plugins=cfg.plugins,
        gradient_clip_val=cfg.grad_clip_val,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        # sync_batchnorm=cfg.sync_batchnorm,
    )

    logging.info(f'logging config for wandb:\n{omegaconf.OmegaConf.to_yaml(cfg)}')

    trainer.fit(lightning_model, datamodule)  # uses batch size of datamodule
    # can test as per the below, but note that datamodule must have a test dataset attribute as per pytorch lightning docs.
    # also be careful not to test regularly, as this breaks train/val/test conceptual separation and may cause hparam overfitting
    if datamodule.test_dataloader is not None:
        logging.info(
            f"Testing on {checkpoint_callback.best_model_path} with single GPU. Be careful not to overfit your choices to the test data..."
        )
        datamodule.setup(stage="test")
        # TODO with webdataset, no need for new trainer/datamodule (actually it breaks), but might still be needed with normal dataset?
        trainer.test(
            model=lightning_model,
            datamodule=datamodule,
            ckpt_path=checkpoint_callback.best_model_path,  # can optionally point to a specific checkpoint here e.g. "/share/nas2/walml/repos/gz-decals-classifiers/results/early_stopping_1xgpu_greyscale/checkpoints/epoch=26-step=16847.ckpt"
        )


# def manually_load_gz_evo():
#     gz_evo_manual_download_loc = os.environ['GZ_EVO_MANUAL_DOWNLOAD_LOC']
#     train_locs = glob.glob(gz_evo_manual_download_loc + '/data/train*.parquet')
#     test_locs = glob.glob(gz_evo_manual_download_loc + '/data/test*.parquet')
#     assert train_locs, f"no train files found in {gz_evo_manual_download_loc}"
#     return load_dataset(
#         path=gz_evo_manual_download_loc,
#         # data_files must be explicit paths seemingly, not just glob strings. Weird.
#         data_files={'train': train_locs, 'test': test_locs},
#         # load LOCALLY to this machine
#         cache_dir=os.environ['HF_LOCAL_DATASETS_CACHE']
#     )


def get_dataset_dict(cfg):
    # if cfg.dataset_name == 'gz_evo' and os.environ.get('GZ_EVO_MANUAL_DOWNLOAD_LOC'):
    #     logging.info('Loading gz evo from manual download')
    #     # will load to HF_LOCAL_DATASETS_CACHE
    #     dataset_dict=manually_load_gz_evo()
    # else:
    dataset_loc = f"mwalmsley/{cfg.dataset_name}"
    logging.info(f"Loading dataset from {dataset_loc}")
    print(f"Loading dataset from {dataset_loc}")
    dataset_dict = load_dataset(
        dataset_loc, 
        name=cfg.subset_name, 
        keep_in_memory=cfg.keep_in_memory, 
        # cache_dir=cfg.hf_cache_dir,  # no, just use env variables
        download_mode=cfg.download_mode,
    )
    return dataset_dict


def log_images(wandb_logger, datamodule):
    datamodule.setup()

    batch = next(iter(datamodule.train_dataloader()))
    images = batch['image']
    np_images = [image.numpy().transpose(2, 1, 0) for image in images[:5]]
    # print([x.shape for x in np_images])
    # print([(x.min(), x.max()) for x in np_images])
    # exit()
    wandb_logger.experiment.log(
        {
            "train_examples": [
                wandb.Image(image) for image in np_images
            ]  # , caption=f'label: {label}') for image, label in zip(images, labels)]
        }
    )



def evaluate_single_model(checkpoint_dir, cfg, model_lightning_class, task_data_func):

    checkpoints = list(glob.glob(checkpoint_dir + '/checkpoints/*.ckpt'))
    checkpoints.sort()
    assert checkpoints, checkpoint_dir
    checkpoint_loc = checkpoints[-1]
    model = model_lightning_class.load_from_checkpoint(checkpoint_loc)

    datamodule = task_data_func(cfg)
    datamodule.setup()  # all stages

    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,  # per node
        num_nodes=cfg.nodes,
        precision=cfg.precision,
        max_epochs=cfg.epochs,
        default_root_dir=cfg.save_dir,
        gradient_clip_val=cfg.grad_clip_val
    )


    # for name, dataloader in [('train', datamodule.train_dataloader()), ('val', datamodule.val_dataloader()), ('test', datamodule.test_dataloader())]:
    for name, dataloader in [('test', datamodule.test_dataloader())]:
        print(name)

        dfs: list = trainer.predict(model=model, dataloaders=dataloader)  # type: ignore
        # list of dfs, each is batch like {'id_str': ..., 'answer_a_fraction': ..., ...}
        # this works because the predict_step in ClassificationBaseline returns a dict of dataframes
        df = pd.concat(dfs, ignore_index=True)

        predictions_save_loc = checkpoint_dir + f'/{name}_predictions.csv'
        df.to_csv(predictions_save_loc, index=False)
