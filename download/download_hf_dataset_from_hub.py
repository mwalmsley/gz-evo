from datasets import load_dataset
from torch.utils.data import DataLoader


if __name__ == '__main__':

    # ds = load_dataset("mwalmsley/gz_evo", split="train", name='tiny')
    # ds = load_dataset("mwalmsley/gz_hubble", name='tiny')
    ds = load_dataset("mwalmsley/gz_hubble", name='default')
    ds.set_format("torch")  # also supports numpy, jax, etc
    dataloader = DataLoader(ds['train'], batch_size=4, num_workers=1)