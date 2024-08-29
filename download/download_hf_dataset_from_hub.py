
import os


if __name__ == '__main__':

    # adjust these as you like

    if os.path.isdir("/project/def-bovy/walml"):
        os.environ["HF_HOME"] = "/project/def-bovy/walml/cache/huggingface"
        os.environ["HF_DATASETS_CACHE"] = "/project/def-bovy/walml/cache/huggingface/datasets"
    else:
        assert os.path.isdir('/share/nas2/walml', 'Please add your own path')
        os.environ['HF_HOME']="/share/nas2/walml/cache/huggingface"
        os.environ['HF_DATASETS_CACHE']="/share/nas2/walml/cache/huggingface/datasets"

    from datasets import load_dataset
    from torch.utils.data import DataLoader

    # ds = load_dataset("mwalmsley/gz_evo", split="train", name='tiny')
    # ds = load_dataset("mwalmsley/gz_hubble", name='tiny')
    # ds = load_dataset("mwalmsley/gz_hubble", name='default')
    # ds = load_dataset("mwalmsley/gz2", name='default')
    ds = load_dataset("mwalmsley/gz_evo", name='default')
    ds.set_format("torch")  # also supports numpy, jax, etc
    dataloader = DataLoader(ds['train'], batch_size=4, num_workers=1)
