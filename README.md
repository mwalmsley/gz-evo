# gz-evo
Baselines and loading code for Galaxy Zoo Evo

    pip install -r requirements.txt
    pip install "zoobot[pytorch-cu121]" --extra-index-url https://download.pytorch.org/whl/cu121


## Example setup on cluster

Load your standard modules (CUDA, Python, etc.). Depends on cluster, but e.g.

    module load StdEnv/2023 gcc cuda/12.2 arrow python/3.11 opencv

Make a new venv

    virtualenv gz-evo
    source venv/bin/activate

Install Python pip dependencies (and deal with any failed installs)

    pip install -r slurm_examples/requirements.txt

Clone (to anywhere) and editable-install these repos

    cd repos

    git clone -b dev git@github.com:mwalmsley/galaxy-datasets.git
    git clone -b dev git@github.com:mwalmsley/zoobot.git
    pip install --no-deps -e galaxy-datasets
    pip install --no-deps -e zoobot
