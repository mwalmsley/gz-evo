import logging
import omegaconf
import os

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
import datasets

from galaxy_datasets.pytorch import galaxy_datamodule

from galaxy_datasets import transforms
# from galaxy_datasets.transforms import GalaxyViewTransform, default_view_config, fast_view_config

from zoobot.shared import schemas
from zoobot.pytorch.estimators import define_model
from zoobot.pytorch.training import train_with_pytorch_lightning


def log_images(wandb_logger, datamodule):
    datamodule.setup()

    batch = next(iter(datamodule.train_dataloader()))
    images, _ = batch
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
    logging.info('Images: {} to {}, dtype={}'.format(images.min(), images.max(), images.dtype))


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting minimal baseline")

    seed = 42
    pl.seed_everything(seed)

    # some path management, adjust to your system as needed 
    # TODO may refactor as argparse or omegaconf
    if os.path.isdir('/project/def-bovy/walml'):
        hf_cache_dir = '/project/def-bovy/walml/cache/huggingface/datasets'
        node_cache_dir = os.environ.get('SLURM_TMPDIR', hf_cache_dir)
        subset_name = 'default'
        num_workers = 12
    elif os.path.isdir('/share/nas2'):
        hf_cache_dir = '/share/nas2/walml/cache/huggingface/datasets'
        node_cache_dir = '/state/partition1/huggingface_tmp'
        subset_name = 'default'
        # node_cache_dir = hf_cache_dir
        num_workers = 12
    else:
        hf_cache_dir = None
        node_cache_dir = hf_cache_dir
        subset_name = 'tiny'
        num_workers = 4

    cfg: omegaconf.DictConfig = omegaconf.OmegaConf.create(
        dict(
            dataset_name='gz_evo',
            subset_name=subset_name,
            hf_cache_dir=hf_cache_dir,
            node_cache_dir=node_cache_dir,
            save_dir="results/minimal/debug",  # relative

            # download_mode="force_redownload",
            download_mode="reuse_dataset_if_exists",
            batch_size=32,
            # batch_size=512,  # 32 for local desktop
            num_workers=num_workers,  # 4 for local desktop
            architecture_name="convnext_nano",
            drop_path_rate=0.4,  # for timm_kwargs
            channels=3,
            dropout_rate=0.5,
            learning_rate=1e-4,
            compile_encoder=False,
            weight_decay=0.05,
            accelerator="gpu",
            keep_in_memory=False,
            devices=1,
            nodes=1,
            epochs=1000,
            strategy="auto",
            precision="16-mixed",  # bf16 doesn't support lgamma for dirichlet loss
            plugins=None,
            patience=8,
        )
    )
    logging.info(f'using config:\n{omegaconf.OmegaConf.to_yaml(cfg)}')

    # -----

    wandb_logger = WandbLogger(
        name=f"{cfg.architecture_name} latest",
        project="gz-evo-baseline",
        log_model=False,
        # https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#omegaconf-to-container
        config=omegaconf.OmegaConf.to_container(cfg),
    )

    torch.set_float32_matmul_precision("medium")

    schema = schemas.gz_evo_v1_public_schema
    # schema = schemas.cosmic_dawn_ortho_schema
    label_cols = schema.label_cols

    # node_save_loc = f"{cfg.node_cache_dir}/gz_evo/{cfg.subset_name}"
    # if not os.path.exists(node_save_loc):
    #     # download and cache the dataset on the worker
    #     dataset = datasets.load_dataset("mwalmsley/gz_evo", name=cfg.subset_name, keep_in_memory=cfg.keep_in_memory, cache_dir=cfg.hf_cache_dir)
    #     dataset.save_to_disk(f"{cfg.node_cache_dir}/gz_evo/{cfg.subset_name}")
    # # load the dataset from disk on worker
    # dataset = datasets.load_from_disk(node_save_loc)

    dataset = datasets.load_dataset(
        f"mwalmsley/{cfg.dataset_name}", 
        name=cfg.subset_name, 
        keep_in_memory=cfg.keep_in_memory, 
        cache_dir=cfg.hf_cache_dir,
        download_mode=cfg.download_mode,
    )
    
    dataset.set_format("torch")
    # print(dataset['train'][0]['image'])
    # print(dataset['train'][0]['image'].max())
    # exit()

    # - these are older albumentation transforms, now deprecated for torchvision + omegaconf
    transform_to_wrap = transforms.default_transforms(
        initial_center_crop=400,
        pytorch_greyscale=False, 
        to_float=True, 
        to_tensor=True
    ) 
    # hf dataset format returens CHW tensor, albumentations expects HWC numpy
    # to_tensor=True triggers ToTensorV2, which converts back to CHW tensor
    # isn't data loading fun?
    transform = lambda x: transform_to_wrap(image=x.numpy().transpose(2, 1, 0))['image']
    
    
    # dataset.set_format("torch")
    # transform_config = default_view_config()
    # transform_config = fast_view_config()
    # transform_config.erase_iterations = 0  # disable masking small patches
    # transform = GalaxyViewTransform(transform_config)

    datamodule = galaxy_datamodule.HF_GalaxyDataModule(
        dataset=dataset,
        label_cols=label_cols,
        train_transform=transform,
        test_transform=transform,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=seed,
    )

    # log a few images to make sure the transforms look good
    log_images(wandb_logger, datamodule)

    lightning_model = define_model.ZoobotTree(
        output_dim=len(schema.label_cols),
        question_answer_pairs=schema.question_answer_pairs,
        dependencies=schema.dependencies,
        architecture_name=cfg.architecture_name,
        channels=cfg.channels,
        test_time_dropout=True,
        dropout_rate=cfg.dropout_rate,
        learning_rate=cfg.learning_rate,
        timm_kwargs={'drop_path_rate': cfg.drop_path_rate},
        compile_encoder=cfg.compile_encoder,
        weight_decay=cfg.weight_decay
    )

    checkpoint_callback, callbacks = train_with_pytorch_lightning.get_default_callbacks(
        cfg.save_dir, cfg.patience
    )

    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        log_every_n_steps=150,
        accelerator=cfg.accelerator,
        devices=cfg.devices,  # per node
        num_nodes=cfg.nodes,
        strategy=cfg.strategy,
        precision=cfg.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=cfg.epochs,
        default_root_dir=cfg.save_dir,
        plugins=cfg.plugins,
        gradient_clip_val=0.3,
    )

    trainer.fit(lightning_model, datamodule)  # uses batch size of datamodule

    best_model_path = trainer.checkpoint_callback.best_model_path

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
