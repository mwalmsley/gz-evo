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
from zoobot.pytorch.training import train_with_pytorch_lightning

import baseline_models  # relative import
import baseline_datamodules  # relative import

# def log_images(wandb_logger, datamodule):
#     datamodule.setup()

#     batch = next(iter(datamodule.train_dataloader()))
#     images, _ = batch
#     np_images = [image.numpy().transpose(2, 1, 0) for image in images[:5]]
#     # print([x.shape for x in np_images])
#     # print([(x.min(), x.max()) for x in np_images])
#     # exit()
#     wandb_logger.experiment.log(
#         {
#             "train_examples": [
#                 wandb.Image(image) for image in np_images
#             ]  # , caption=f'label: {label}') for image, label in zip(images, labels)]
#         }
#     )

#     # batch = next(iter(datamodule.test_dataloader()))
#     # images, _ = batch
#     # wandb_logger.experiment.log(
#     #     {
#     #         "test_examples": [
#     #             wandb.Image(image) for image in images[:5]
#     #         ]  # , caption=f'label: {label}') for image, label in zip(images, labels)]
#     #     }
#     # )

#     # check dynamic range
#     logging.info('Images: {} to {}, dtype={}'.format(images.min(), images.max(), images.dtype))


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting classification baseline")

    seed = 42
    pl.seed_everything(seed)

    # some path management, adjust to your system as needed 
    # TODO may refactor as argparse or omegaconf
    if os.path.isdir('/project/def-bovy/walml'):
        hf_cache_dir = '/project/def-bovy/walml/cache/huggingface/datasets'
        node_cache_dir = os.environ.get('SLURM_TMPDIR', hf_cache_dir)
        subset_name = 'default'
        # A100 on narval
        # num_workers = 12
        # batch_size = 512 
        # V100 on beluga
        num_workers = 8
        batch_size = 128  
    elif os.path.isdir('/share/nas2'):
        hf_cache_dir = '/share/nas2/walml/cache/huggingface/datasets'
        node_cache_dir = '/state/partition1/huggingface_tmp'
        subset_name = 'default'
        # node_cache_dir = hf_cache_dir
        num_workers = 12
        batch_size = 512
    else:
        hf_cache_dir = None
        node_cache_dir = hf_cache_dir
        subset_name = 'tiny'
        # subset_name = 'default'
        num_workers = 4
        batch_size = 32

    cfg: omegaconf.DictConfig = omegaconf.OmegaConf.create(
        dict(
            # dataset_name='gz_evo',
            dataset_name='gz_hubble',
            subset_name=subset_name,
            hf_cache_dir=hf_cache_dir,
            node_cache_dir=node_cache_dir,
            save_dir="results/baselines/classification/debug",  # relative

            # download_mode="force_redownload",
            download_mode="reuse_dataset_if_exists",
            batch_size=batch_size,
            num_workers=num_workers,  # 4 for local desktop
            architecture_name="convnext_nano",
            drop_path_rate=0.4,  # for timm_kwargs
            pretrained=True, #imagenet 12k, for timm kwargs
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
            accumulate_grad_batches=4,
            grad_clip_val=0.3,
            sync_batchnorm=False,  # only one device
            transform_mode='default'
            # transform_mode='albumentations'
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

    # any callable that takes an HF example (row) and returns a label
    # target_transform = None  
    # load the summary column as integers
    from baseline_datamodules import LABEL_ORDER_DICT
    def target_transform(example):
        example['label'] = LABEL_ORDER_DICT[example['summary']]
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

    # log a few images to make sure the transforms look good
    # log_images(wandb_logger, datamodule)

    lightning_model = baseline_models.ClassificationBaseline(
        architecture_name=cfg.architecture_name,
        channels=cfg.channels,
        timm_kwargs={
            'drop_path_rate': cfg.drop_path_rate, 
            'pretrained': cfg.pretrained
        },
        head_kwargs={
            'dropout_rate': cfg.dropout_rate,
            'num_classes': len(LABEL_ORDER_DICT.keys())
        },
        learning_rate=cfg.learning_rate,
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
        gradient_clip_val=cfg.grad_clip_val,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        sync_batchnorm=cfg.sync_batchnorm,
    )

    logging.info(f'logging config for wandb:\n{omegaconf.OmegaConf.to_yaml(cfg)}')


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
