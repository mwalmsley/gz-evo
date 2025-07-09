import logging
import os

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
import hydra
# from hydra.utils import instantiate
from datasets import load_dataset, DatasetDict

from galaxy_datasets.pytorch.galaxy_datamodule import HuggingFaceDataModule, CatalogDataModule
from zoobot.pytorch.training import finetune

import wandb
from gz_evo.downstream.transforms import get_finetuning_transforms
from gz_evo.downstream import finetune_utils


"""
python gz_evo/downstream/finetune.py +learner=convnext_nano +hardware=local ++dataset=gz_euclid ++debug=True
python gz_evo/downstream/finetune.py +learner=convnext_nano +hardware=local ++dataset=is-lsb ++debug=True
python gz_evo/downstream/finetune.py +learner=convnext_nano +hardware=local ++dataset=is-lsb ++wandb=True

python gz_evo/downstream/finetune.py +learner=convnext_nano +hardware=local ++dataset=euclid_strong_lens_expert_judges ++wandb=False ++debug=True
python gz_evo/downstream/finetune.py +learner=vit_tiny +hardware=local ++dataset=euclid_strong_lens_expert_judges ++wandb=False ++debug=True
python gz_evo/downstream/finetune.py +learner=vit_tiny +hardware=local ++dataset=euclid_strong_lens_expert_judges ++wandb=True ++debug=False

python gz_evo/downstream/finetune.py +learner=convnext_nano ++learner.encoder_hub_path=hf_hub:mwalmsley/baseline-encoder-regression-convnext_nano +hardware=local ++dataset=is-lsb ++debug=True

python gz_evo/downstream/finetune.py +learner=siglip +hardware=local ++dataset=gz_euclid ++debug=True ++learner.batch_size=2

python gz_evo/downstream/finetune.py +learner=convnext_nano_gz_euclid +hardware=local ++dataset=euclid_strong_lens_expert_judges ++debug=True

"""

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):

    # --
    # manual HF seems to resolve occasional auth errors - you may be able to remove 
    # import json
    # with open('secret_token.json', 'r') as f:
    #     token = json.load(f)["token"]

    # from huggingface_hub import whoami
    # user = whoami(token=token)
    # logging.info('Logged in as {}'.format(user['name']))
    # --

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    # Seed for reproducibility
    logging.info(f"Setting seed to {cfg.seed}")
    pl.seed_everything(cfg.seed)

    cfg.name = f"{cfg.learner.architecture_name}_{cfg.dataset}"

    # check if wandb and set run id
    if cfg.wandb:
        wandb.init(
            name=cfg.name,
            project=cfg.project,
            config=dict(cfg)
        )
        cfg.run_id = wandb.run.id
    else:
        import time
        current_time_formatted = time.strftime("%Y-%m-%d_%H-%M-%S")
        cfg.run_id = f'local_{cfg.dataset}_{cfg.seed}_{current_time_formatted}'

    logging.info(f"Run ID: {cfg.run_id}")

    # version is from default config, changed rarely

    # results/{version}/
    #   pretrain
        # run_id
    #   finetune
        # run_id

    results_dir = get_results_dir()
    save_dir = os.path.join(results_dir, cfg.version, cfg.run_id)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    logging.info(f"Saving to {save_dir}")
    # exit()

    if cfg.wandb:
        # pull any overridden params from wandb project init, which is cleverly overridden on wandb's side
        logging.info(f"Config pulled from wandb: {wandb.config}")
        # update cfg.finetune with wandb sweep values
        for key, value in wandb.config.items():
            if key.startswith("finetune."):
                actual_key = key.replace("finetune.", "")
                logging.info(f"Overriding {actual_key} with {value}")
                cfg.finetune[actual_key] = value
    
    # some downstream datasets need a specific finetuning setup
    apply_dataset_specific_overrides(cfg)

    model, datamodule = prepare_experiment(cfg, token=None)

    if cfg.wandb:
        # hopefully reuses wandb.init already called
        logger = loggers.WandbLogger(
            name=cfg.name,
            save_dir=save_dir,
            project=cfg.project,
            log_model=False,
            offline=False,
        )
        logger.log_hyperparams(dict(cfg))
    else:
        logging.info("warning, csv logging only, use for debugging only")
        logger = loggers.CSVLogger(name=cfg.name, save_dir=save_dir)


    datamodule.setup("fit")
    loader = datamodule.train_dataloader()
    batch = next(iter(loader))
    images = batch['image']
    logging.info((images.min(), images.max(), images.shape, images.dtype))
    # reset
    datamodule.setup("fit")

    trainer = finetune.get_trainer(
        save_dir,
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        num_nodes=cfg.hardware.num_nodes,
        strategy="auto",
        precision=cfg.hardware.precision,
        max_epochs=2 if cfg.debug else cfg.learner.max_epochs,
        enable_progress_bar=True,  # temp
        check_val_every_n_epoch=cfg.learner.check_val_every_n_epoch,
        logger=logger,
        patience=cfg.learner.patience,
        overfit_batches=1 if cfg.debug else 0,
    )
    trainer.fit(model, datamodule)
    trainer.test(datamodule=datamodule, ckpt_path="best")

    # TODO add save_predictions
    save_predictions(model, datamodule, trainer, save_dir)

    logging.info("Finetuning complete")



def prepare_experiment(cfg, token=None):
    # typing fixes
    cfg.hardware.num_workers = int(cfg.hardware.num_workers)
    # if cfg.learner.pretrained == "False":
    #     cfg.learner.pretrained = False

    encoder = get_encoder(cfg)

    # now we need to set up the model (with that encoder) and the datamodule

    # cfg.finetune already overrided with wandb if needed for sweep
    # backbone_name = f"{model.cfg.data.dataset_name}_{model.cfg.ssl.arch}_{model.cfg.aug.standard_ssl_view.output_size}_div{model.cfg.unlabelled_dataset_divisor}"
    init_args_for_all_models = dict(
        encoder=encoder,
        # backbone_name=backbone_name,
        n_blocks=cfg.learner.n_blocks,
        always_train_batchnorm=cfg.learner.always_train_batchnorm,
        lr_decay=cfg.learner.lr_decay,
        weight_decay=cfg.learner.weight_decay,
        learning_rate=cfg.learner.learning_rate,
        dropout_prob=cfg.learner.dropout_prob,  # head dropout
        # optional scheduler params
        cosine_schedule=cfg.learner.cosine_schedule,
        warmup_epochs=cfg.learner.warmup_epochs,
        max_cosine_epochs=cfg.learner.max_cosine_epochs,
        max_learning_rate_reduction_factor=cfg.learner.max_learning_rate_reduction_factor,
        # overrides
        # from_scratch=cfg.from_scratch,
        # visualize_images=cfg.visualize_images,
    )

    # gz_rings
    if cfg.dataset == "gz_rings":
        # runs but metrics are tricky
        model = finetune.FinetuneableZoobotRegressor(
            unit_interval=False, **init_args_for_all_models
        )
        dataset_dict = load_dataset("mwalmsley/gz_rings", "regression", token=token)  # type: ignore
        # target_transform = ToFloat_Transform() # should already be float
        target_transform = None

    elif cfg.dataset == "gz-rings-binary":
        model = finetune.FinetuneableZoobotClassifier(num_classes=2, **init_args_for_all_models)
        dataset_dict = load_dataset("mwalmsley/gz_rings", "classification", token=token)  # type: ignore
        # target_transform = ToFloat_Transform()
        target_transform = None # hmm, this is classification, should be int
        label_cols = ['label']

    # is-lsb
    elif cfg.dataset == "is-lsb":
        model = finetune.FinetuneableZoobotClassifier(num_classes=2, **init_args_for_all_models)
        dataset_dict = load_dataset("mwalmsley/is-lsb", token=token)  # type: ignore
        target_transform = None

    # which-lsb
    elif cfg.dataset == "which-lsb":
        model = finetune.FinetuneableZoobotClassifier(num_classes=4, **init_args_for_all_models)
        dataset_dict = load_dataset("mwalmsley/which-lsb", token=token)  # type: ignore
        target_transform = None

    # galaxy10_decals
    elif cfg.dataset == "decals10":
        model = finetune.FinetuneableZoobotClassifier(num_classes=10, **init_args_for_all_models)
        dataset_dict = load_dataset("mwalmsley/galaxy10_decals", "galaxyzoo", token=token)  # type: ignore
        target_transform = None

    # jwst
    elif cfg.dataset == "jwst":
        model = finetune.FinetuneableZoobotClassifier(num_classes=5, **init_args_for_all_models)
        dataset_dict = load_dataset("mwalmsley/jwst", token=token)  # type: ignore
        target_transform = None

    elif cfg.dataset == "euclid_strong_lens_expert_judges":
        model = finetune.FinetuneableZoobotClassifier(num_classes=2, **init_args_for_all_models)
        logging.info("Loading euclid_strong_lens_expert_judges dataset")
        dataset_dict = load_dataset("mwalmsley/euclid_strong_lens_expert_judges", "classification")  # type: ignore
        target_transform = None

    elif cfg.dataset == "gz_euclid":
        from zoobot.shared import schemas
        schema = schemas.euclid_ortho_schema
        model = finetune.FinetuneableZoobotTree(schema=schema, **init_args_for_all_models)
        dataset_dict: DatasetDict = load_dataset("mwalmsley/gz_euclid", token=token)  # type: ignore
        target_transform = None

    # TODO JWST COSMOS tree

    else:
        raise ValueError(cfg.dataset)

    if cfg.divisor > 1:
        n_samples = len(dataset_dict["train"]) // cfg.divisor
        dataset_dict["train"] = dataset_dict["train"].shuffle(seed=cfg.seed).select(range(n_samples))
    logging.info(f"Divisor applied: {cfg.divisor}. Train samples: {len(dataset_dict['train'])}")

    # Define transforms
    train_transform, test_transform = get_finetuning_transforms(cfg)

    datamodule = HuggingFaceDataModule(
        dataset_dict=dataset_dict,
        train_transform=train_transform,
        test_transform=test_transform,
        target_transform=target_transform,
        batch_size=cfg.learner.batch_size,
        num_workers=cfg.hardware.num_workers,
        prefetch_factor=cfg.hardware.prefetch_factor,
    )
    datamodule.setup()

    return model, datamodule

# to finetune new encoders replace this whatever you like
def get_encoder(cfg):

    # WIP advanced baselines
    # SSL or hybrid encoder
    if cfg.learner.encoder_hub_path.startswith('hf_hub:mwalmsley/wip-hybrid-encoder'):
        from foundation.models.base_hybrid import BaseHybridLearner
        from huggingface_hub import hf_hub_download
        repo_id = cfg.learner.encoder_hub_path.replace('hf_hub:', '')
        ckpt_path = hf_hub_download(repo_id=repo_id, filename="last.ckpt", repo_type="model")
        model = BaseHybridLearner.load_from_checkpoint(ckpt_path)
    elif cfg.learner.encoder_hub_path.startswith('local_hybrid:'):
        local_path = cfg.learner.encoder_hub_path.replace('local_hybrid:', '')
        from foundation.models.base_hybrid import BaseHybridLearner
        model = BaseHybridLearner.load_from_checkpoint(local_path)
        encoder = model.ssl.backbone # MaskedVisionTransformerTIMM, 
        # has forward() method that uses mask-supporting encode
        # just make sure global_pool is 'avg' or 'map' or 'cls' for finetuning
        
        assert encoder.vit.global_pool in ['avg', 'map', 'cls'], \
            f"Encoder global_pool must be 'avg', 'map' or 'cls', got {encoder.vit.global_pool}"

        # encoder = model.ssl.encoder  # timm.models.vision_transformer.VisionTransformer

    # supervised encoders, trained with gz_evo.core, or timm equivalents
    elif cfg.learner.encoder_hub_path.startswith('hf_hub:mwalmsley/baseline-encoder') or cfg.learner.encoder_hub_path.startswith('hf_hub:timm'):
        # possibly only has an effect when training from scratch, otherwise is already fixed
        timm_kwargs = {
            "drop_path_rate": cfg.learner.stochastic_depth,
        }

        encoder = finetune_utils.get_timm_encoder(
            name=cfg.learner.encoder_hub_path,
            channels=cfg.learner.channels,
            pretrained=not cfg.learner.from_scratch,
            timm_kwargs=timm_kwargs
        )


    elif cfg.learner.encoder_hub_path.startswith('local:'):
        # local path to a checkpoint, which we assume is previously regression-finetuned on another downstream dataset
        # this allow for chaining finetuning e.g. pretrain, finetune on A, finetune on B
        local_path = cfg.learner.encoder_hub_path.replace('local:', '')

        import timm
        blank_model = timm.create_model(
            cfg.learner.architecture_name,
            pretrained=False,
            num_classes=0,  # no head
            in_chans=cfg.learner.channels,
        )

        # load the encoder from the checkpoint
        ckpt = torch.load(local_path, map_location="cpu", weights_only=False)  # check if the path is valid
        print(ckpt['state_dict'].keys())  # debug, check the keys

        # copy encoder weights
        blank_model.load_state_dict(
            {k.replace("encoder.", ""): v for k, v in ckpt['state_dict'].items() if k.startswith("encoder.")}
        )

        encoder = blank_model


    else:
        raise ValueError(f"Unknown encoder hub path: {cfg.learner.encoder_hub_path}")

    return encoder


def apply_dataset_specific_overrides(cfg):
    # additional patience for somewhat noisy small datasets during linear finetune, to avoid accidental early stopping
    if cfg.learner.n_blocks == 0:
        if cfg.dataset == "jwst":
            logging.warning("overriding patience for JWST linear finetune, setting 400")
            cfg.learner.patience = 400
        elif cfg.dataset == "which-lsb":
            logging.warning("overriding patience for which-lsb linear finetune, setting 50")
            cfg.learner.patience = 50

    # if cfg.learner.architecture_name == "convnext_nano":
        # Normalize(mean=tensor([0.5000, 0.5000, 0.5000]), std=tensor([0.5000, 0.5000, 0.5000]))

    # save_predictions(model, datamodule, trainer, save_dir)

# TODO update prediction process with HF dataset, currently only built for catalogs
def save_predictions(
    model: pl.LightningModule,
    datamodule: HuggingFaceDataModule,
    trainer: pl.Trainer,
    save_dir,
):
    for split in ["train", "validation", "test"]:
        logging.info(f"Beginning predictions for split {split}")

        # TODO head is now struggling, take a break
        # but I just need to work out how to make predictions on a passed huggingface dataset
        # and save those predictions to a csv alongside the id_str
        # might be overcomplicating things
        predict_dataloader = datamodule.get_predict_dataloader(split=split) # dicts with {'image': ..., 'id_str': ...}

        catalog_to_predict: pd.DataFrame = datamodule.dataset_dict[split].to_pandas()[['id_str']]  # assume same ordering as dataloader

        predictions = trainer.predict(
            model, dataloaders=predict_dataloader
        )  # list with [batch, classes] tensor entries?
        predictions = torch.concat(predictions).numpy()  # [n, classes], or just [n] for regression

        # set format of predictions
        if isinstance(model, finetune.FinetuneableZoobotClassifier):
            logging.info("Detected classification model")
            class_label_cols = [f"class_{i}" for i in range(predictions.shape[-1])]
            catalog_to_predict[class_label_cols] = predictions
            logging.info(catalog_to_predict[["id_str"] + class_label_cols])
        elif isinstance(model, finetune.FinetuneableZoobotRegressor):
            logging.info("Detected regression model")
            catalog_to_predict["regression"] = predictions.squeeze()
        elif isinstance(model, finetune.FinetuneableZoobotTree):
            logging.info("Detected tree model")
            logging.info(model.schema.label_cols)
            catalog_to_predict[model.schema.label_cols] = predictions  # label cols columns, for dirichlet outputs
        else:
            raise ValueError("Unknown model type", type(model))
            
        catalog_to_predict.to_csv(os.path.join(save_dir, f"predictions_{split}.csv"), index=False)


def get_results_dir():
    if os.path.isdir('/share/nas2'):
        return '/share/nas2/walml/gz-evo/results/downstream'
    elif os.path.isdir('/home/walml'):
        return '/home/walml/repos/gz-evo/results/downstream'
    else:
        raise FileNotFoundError("Please set results_dir here")

if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    main()