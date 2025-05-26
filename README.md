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

## Quick Installation

```bash
pip install -r requirements.txt
pip install "zoobot[pytorch-cu121]" --extra-index-url https://download.pytorch.org/whl/cu121
```

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

Clone and editable-install these repos

```bash
git clone -b dev git@github.com:mwalmsley/galaxy-datasets.git
git clone -b dev git@github.com:mwalmsley/zoobot.git
pip install --no-deps -e galaxy-datasets
pip install --no-deps -e zoobot
```

Download the dataset and models (useful for clusters with offline worker nodes, like ours). Adjust the paths in each script as you like.

```bash
python download/download_wds_from_hub.py
python download/download_timm_model.py
```

<!-- ## Training

baseline_models.py includes PyTorch Lightning models. GenericBaseline is an abstract LightningModel that sets up the general structure: we create a model with self.encoder, self.head,  -->