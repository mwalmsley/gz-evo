import os
import subprocess
import logging
from dataclasses import asdict
import glob

import numpy as np
import pandas as pd
import omegaconf
import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from datasets import load_dataset

from gz_evo.core import baseline_configs

def get_config(architecture_name, dataset_name, save_dir, debug=False):
    # some path management, adjust to your system as needed 

    # TODO may refactor as argparse or omegaconf
    if os.path.isdir('/project/def-bovy/walml'):
        subset_name = 'default'
        accelerator="gpu"
        
        # A100
        # batch_size_key = 'A100_batch_size'
        # num_workers = 12

        # V100
        num_workers = 10
        batch_size_key = 'v100_batch_size'
        devices = 1
        prefetch_factor = 4

    elif os.path.isdir('/share/nas2'):
        subset_name = 'default'
        # subset_name = 'tiny' 
        batch_size_key = 'a100_batch_size'
        accelerator="gpu"
        devices = int(os.environ.get('SLURM_NTASKS_PER_NODE', 1))
        prefetch_factor = 8
        num_workers = 16 // devices
        # of 24 per node on some, or 16 on others. 16 nodes are more reliable.

    elif os.path.isdir('/Users/user'):
        # macbook
        subset_name = 'tiny' 
        num_workers = 1
        debug = True
        accelerator="cpu"  # not auto, MPS still has issues
        batch_size_key = 'debug'
        devices = 1
        prefetch_factor = 2
    
    # TODO add your own system here

    else:
        logging.warning(
            "No known system detected, using defaults for local debugging. See baseline_training.py to add system"
        )
        # defaults to local debugging mode
        subset_name = 'tiny' 
        # subset_name = 'default'
        num_workers = 4
        debug = True
        accelerator="auto"
        batch_size_key = 'debug'
        devices = 1
        prefetch_factor = 2

    cfg: omegaconf.DictConfig = omegaconf.OmegaConf.create(
        dict(
            architecture_name=architecture_name,
            dataset_name=dataset_name,
            subset_name=subset_name,
            save_dir=save_dir,
            download_mode="reuse_dataset_if_exists",  # or "force_redownload",
            verification_mode="basic_checks",  # or "no_checks"
            # keep_in_memory=None,  # None: keep if it fits in HF_DATASETS_IN_MEMORY_MAX_SIZE. Override with False.
            keep_in_memory=False,
            num_workers=num_workers,
            iterable=True,  # seems to actually be slower rn
            prefetch_factor=prefetch_factor, 
            compile_encoder=False,  # doesn't work on galahad, missing gcc headers
            pretrained=True, # passed to timm
            channels=3,
            accelerator=accelerator,
            devices=devices,
            nodes=1,
            # epochs=3,
            epochs=1000,
            precision="16-mixed",  # bf16 doesn't support lgamma for dirichlet loss
            plugins=None,
            patience=5,
            grad_clip_val=0.3,
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
        # always the same effective batch size, after accumulation
        cfg.accumulate_grad_batches = 4096 // (cfg.batch_size  * cfg.devices)  # 4096 is the effective batch size, per node
        cfg.debug = debug

    logging.info(f'using config before updates:\n{omegaconf.OmegaConf.to_yaml(cfg)}')

    return cfg


def get_dataset_dict(cfg):
    # if cfg.dataset_name == 'gz_evo':
    #     # gz_evo dataset is not available on HF, so we load it manually
    #     # this is a workaround for the fact that gz_evo is not available on HF datasets hub
    #     logging.warning('Manually loading GZ Evo for speed test')
    #     return manually_load_gz_evo()
    # else:
    dataset_loc = f"mwalmsley/{cfg.dataset_name}"
    logging.info(f"Loading dataset from {dataset_loc}, subset {cfg.subset_name}")
    dataset_dict = load_dataset(
        dataset_loc, 
        name=cfg.subset_name, 
        # typically stick to defaults here
        keep_in_memory=cfg.keep_in_memory,  # None: keep if it fits in HF_DATASETS_IN_MEMORY_MAX_SIZE. Override with False.
        download_mode=cfg.download_mode,  # reuse_dataset_if_exists
        verification_mode=cfg.verification_mode,  # basic_checks
    )
    logging.info(f"Dataset loaded: {cfg.dataset_name} ({cfg.subset_name})")
    return dataset_dict


def manually_load_gz_evo():
    gz_evo_manual_download_loc = os.environ['GZ_EVO_MANUAL_DOWNLOAD_LOC']
    train_locs = glob.glob(gz_evo_manual_download_loc + '/data/train*.parquet')
    test_locs = glob.glob(gz_evo_manual_download_loc + '/data/test*.parquet')
    assert train_locs, f"no train files found in {gz_evo_manual_download_loc}"
    return load_dataset(
        path=gz_evo_manual_download_loc,
        # data_files must be explicit paths seemingly, not just glob strings. Weird.
        data_files={'train': train_locs, 'test': test_locs},
        # load LOCALLY to this machine
        cache_dir=os.environ['HF_LOCAL_DATASETS_CACHE']
    )


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
    # only do on main process
    if os.environ.get('SLURM_PROCID', '0') == '0':  # slurm env var
        log_images(wandb_logger, datamodule)

    monitor_metric = 'validation/supervised_loss' 
    checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(cfg.save_dir, 'checkpoints'),
            monitor=monitor_metric,
            save_weights_only=True,
            mode='min',
    )
    early_stopping_callback = EarlyStopping(monitor=monitor_metric, patience=cfg.patience, check_finite=True)
    callbacks = [checkpoint_callback, early_stopping_callback]

    # galahad cluster has old slurm and doesn't set correctly
    # os.environ['SLURM_NTASKS_PER_NODE'] = str(cfg.devices)  
    logging.info(f"SLURM_NTASKS_PER_NODE is {os.environ['SLURM_NTASKS_PER_NODE']}")
    logging.info(f"SLURM_NTASKS is {os.environ['SLURM_NTASKS']}")

    # delete SLURM env vars to avoid issues with pytorch lightning
    # if 'SLURM_NTASKS_PER_NODE' in os.environ:
    #     del os.environ['SLURM_NTASKS_PER_NODE']
    # if 'SLURM_NTASKS' in os.environ:
    #     del os.environ['SLURM_NTASKS']

    if cfg.devices == 1:
        devices = [get_highest_free_memory_device()]  # list of ints interpreted as device indices
    else:
        devices = cfg.devices
    logging.info(f"Using {devices} devices for training")

    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        log_every_n_steps=150 if cfg.subset_name == 'default' else 10,  # more frequent logging for tiny subset to avoid nan metrics
        accelerator=cfg.accelerator,
        devices=devices,  # single node only
        num_nodes=cfg.nodes,
        precision=cfg.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=cfg.epochs,
        default_root_dir=cfg.save_dir,
        plugins=cfg.plugins,
        gradient_clip_val=cfg.grad_clip_val,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
    )

    logging.info(f'logging config for wandb:\n{omegaconf.OmegaConf.to_yaml(cfg)}')

    trainer.fit(lightning_model, datamodule)  # uses batch size of datamodule
    # can test as per the below, but note that datamodule must have a test dataset attribute as per pytorch lightning docs.
    # also be careful not to test regularly, as this breaks train/val/test conceptual separation and may cause hparam overfitting

    # make new test trainer
    if os.environ.get('SLURM_PROCID', '0') == '0':  # only on main process
        logging.info("Training finished, now testing the best model")
    
        test_trainer = pl.Trainer(
            accelerator=cfg.accelerator,
            devices=1,
            num_nodes=1,
            precision=cfg.precision,
            logger=wandb_logger,
            default_root_dir=cfg.save_dir,
            plugins=cfg.plugins
        )

        
        if datamodule.test_dataloader is not None:
            logging.info(
                f"Testing on {checkpoint_callback.best_model_path} with single GPU. Be careful not to overfit your choices to the test data..."
            )
            datamodule.setup(stage="test")
            test_trainer.test(
                model=lightning_model,
                datamodule=datamodule,
                ckpt_path=checkpoint_callback.best_model_path,  # can optionally point to a specific checkpoint here e.g. "/share/nas2/walml/repos/gz-decals-classifiers/results/early_stopping_1xgpu_greyscale/checkpoints/epoch=26-step=16847.ckpt"
            )

    logging.info({os.environ.get('SLURM_PROCID', '0'): "Run finished"})




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


def get_available_memory_by_device():
    # uses nvidia-smi as torch.cuda.memory_allocated() seems broken, always 0
    # requires export CUDA_DEVICE_ORDER=PCI_BUS_ID to ensure nvidia-smi order = torch.cuda device order
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    memory_by_device = [int(x) for x in result.stdout.decode('utf-8').strip().split('\n')]

    logging.info("Available memory by device: {}".format(dict(zip(
        range(len(memory_by_device)), 
        memory_by_device)
    )))  # type: ignore

    return memory_by_device


def get_highest_free_memory_device() -> int:
    memory_by_device = get_available_memory_by_device()
    highest_free_mem_device = np.argmax(memory_by_device)
    logging.info(f"Highest free memory device: {highest_free_mem_device} with {memory_by_device[highest_free_mem_device]} bytes free")
    return int(highest_free_mem_device)  # int64 without the cast, for some reason (who has that many gpus lol)


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
