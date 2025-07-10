# gz-evo

Baselines for Galaxy Zoo Evo.

## Overview

GZ Evo is a dataset of 1M labelled galaxy images available from HuggingFace. It includes ~850k images from four telescopes with general labels ("Core") and five "Downstream" subsets with labels of specific scientific interest (~150k images in total).

GZ Evo is available from HuggingFace, and you might like to just get cracking with your own code:

```python
import datasets

# for Core, or "is-lsb", "gz_euclid" etc for Downstream datasets
ds = datasets.load_dataset("mwalmsley/gz_evo")
ds.set_format("torch")  # your choice of framework

# TODO clever stuff
```

 If helpful, here, we provide baselines for:

- Training on Core datasets (typically as a pretext task)
- Finetuning on Downstream datasets

For training on Core, we include minimal self-contained examples under `gz_evo/core`. These use PyTorch Lightning to train your choice of `timm` encoder using a standard training recipe.

For finetuning on Downstream, we use the [Zoobot](github.com/mwalmsley/zoobot) package to adapt your choice of `timm` encoder. `Zoobot` is often used by astronomers for galaxy image tasks. It is effectively a wrapper for Pytorch Lightning and (for data loading and augmentations) [galaxy-datasets](github.com/mwalmsley/galaxy-datasets).

## Quick Start

```bash
pip install -r requirements.txt
pip install "zoobot[pytorch-cu121]" --extra-index-url https://download.pytorch.org/whl/cu121
```

and then run training on Core datasets e.g.

```bash
python gz_evo/classification/train.py 
```

or finetune on Downstream datasets e.g.

```bash
python gz_evo/downstream/finetune.py 
```

## Training on Core dataset

Both classification and multinomial training follows the same pattern. For example, in `gz_evo/classification/train.py`:

```python
cfg = baseline_training.get_config(architecture_name, dataset_name, save_dir)

datamodule = set_up_task_data(cfg)

lightning_model = get_lightning_model(cfg)

baseline_training.run_training(cfg, lightning_model, datamodule)
```

We use `baseline_training.py` for shared training functions: `get_config` creates the omegaconf config used throughout, and `run_training` to execute the training given a config, a datamodule, and a model.

`classification_baseline.py` includes classification-specific functions: `set_up_task_data` creates a Lightning DataModule with dataloaders yielding (image_batch, classification_label_batch), and `get_lightning_model` creates a Lightning module with a classification-specific head.

`gz_evo/classification/test.py` is a script for making test predictions, and `gz_evo/classification/metrics.ipynb` visualizes performance (similarly for multinomial).

## Training on Downstream datasets

`finetune.py` loads configuration options from `gz_evo/downstream/conf` (via hydra) and then executes the requested finetuning. 

`get_encoder` is the function to adapt for your own code. By default, it will load any `timm` encoder (e.g. one trained with the Core code, above). This encoder is placed into a `LightningModule` and finetuned (according to e.g. `n_blocks`, etc, from config) using AdamW.

## Shared Code

The remaining code is shared across tasks:

- Generic Lightning `DataModule`s (consuming HuggingFace datasets) are imported from the `galaxy-datasets` repo
- A generic supervised `LightningModule` (`forward`, `training_step`, `configure_optimizers`, etc) is imported from the `zoobot` repo. This includes utility code for e.g. loss functions
- `baseline_configs.py` defines per-model training choices e.g. the learning rate, the weight decay, etc. Each model needs a dictlike. Add new models by making a new dictlike here. `baseline_training.py` will look here for instructions on training your chosen model.

## Example Installation on Cluster

Load your standard modules (CUDA, Python, etc.). Depends on cluster, but e.g.

```bash
module load StdEnv/2023 gcc cuda/12.2 arrow python/3.11 opencv
```

Make a new venv (anywhere, for example my venv dir)

```bash
virtualenv /project/def-bovy/walml/envs/gz-evo
source /project/def-bovy/walml/envs/gz-evo/bin/activate
```

Clone this repo (to anywhere, for example my repo directory), install Python pip dependencies (and deal with any failed installs)

```bash
cd /project/def-bovy/walml/repos

git clone -b derived-tasks git@github.com:mwalmsley/gz-evo.git
pip install -r gz-evo/slurm_examples/requirements.txt
```

Clone and editable-install these repos (which we use for generic Lightning code)

```bash
git clone -b dev git@github.com:mwalmsley/galaxy-datasets.git
git clone -b dev git@github.com:mwalmsley/zoobot.git
pip install --no-deps -e galaxy-datasets
pip install --no-deps -e zoobot
```
