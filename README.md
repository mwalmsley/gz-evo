# gz-evo
Baselines and loading code for Galaxy Zoo Evo

    pip install -r requirements.txt
    pip install "zoobot[pytorch-cu121]" --extra-index-url https://download.pytorch.org/whl/cu121


## Example setup on cluster

Load your standard modules (CUDA, Python, etc.). Depends on cluster, but e.g.

    module load StdEnv/2023 gcc cuda/12.2 arrow python/3.11 opencv

Make a new venv (anywhere, for example my venv dir)

    virtualenv /project/def-bovy/walml/envs/gz-evo
    source /project/def-bovy/walml/envs/gz-evo/bin/activate

Clone this repo (to anywhere, for example my repo directory), install Python pip dependencies (and deal with any failed installs)

    cd /project/def-bovy/walml/repos

    git clone -b derived-tasks git@github.com:mwalmsley/gz-evo.git
    pip install -r gz-evo/slurm_examples/requirements.txt

Clone  and editable-install these repos


    git clone -b dev git@github.com:mwalmsley/galaxy-datasets.git
    git clone -b dev git@github.com:mwalmsley/zoobot.git
    pip install --no-deps -e galaxy-datasets
    pip install --no-deps -e zoobot

Download the dataset and models (useful for clusters with offline worker nodes, like ours). Adjust the paths in each script as you like.

    python download/download_wds_from_hub.py
    python download/download_timm_model.py

## Training

baseline_models.py includes PyTorch Lightning models. GenericBaseline is an abstract LightningModel that sets up the general structure: we create a model with self.encoder, self.head, 