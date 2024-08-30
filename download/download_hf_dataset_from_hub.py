
import os


if __name__ == '__main__':

    # adjust these as you like

    if os.path.isdir("/project/def-bovy/walml"):
        # os.environ["HF_HOME"] = "/project/def-bovy/walml/cache/huggingface"
        # os.environ["HF_DATASETS_CACHE"] = "/project/def-bovy/walml/cache/huggingface/datasets"
        gz_evo_only_cache = '/project/def-bovy/walml/tmp'
    else:
        assert os.path.isdir('/share/nas2/walml'), 'Please add your own path'
        # os.environ['HF_HOME']="/share/nas2/walml/cache/huggingface"
        # os.environ['HF_DATASETS_CACHE']="/share/nas2/walml/cache/huggingface/datasets"
        gz_evo_only_cache = '/share/nas2/walml/tmp'

    from datasets import load_dataset, DownloadConfig
    from huggingface_hub import snapshot_download
    # from torch.utils.data import DataLoader

    # ds = load_dataset("mwalmsley/gz_evo", split="train", name='tiny')
    # ds = load_dataset("mwalmsley/gz_hubble", name='tiny')
    # ds = load_dataset("mwalmsley/gz_hubble", name='default')
    # ds = load_dataset("mwalmsley/gz2", name='default')
    # load_dataset("mwalmsley/gz_evo", name='default', split='test')
    # load_dataset("mwalmsley/gz_evo", name='default', split='train')
    # ds.set_format("torch")  # also supports numpy, jax, etc
    # dataloader = DataLoader(ds['train'], batch_size=4, num_workers=1)
    
    # s all downloaded files are also cached on your local disk

    snapshot_download(
        repo_id="mwalmsley/gz_evo", 
        repo_type="dataset", 
        cache_dir=gz_evo_only_cache
    )
    load_dataset(
        'mwalmsley/gz-evo',
        name='default', 
        # cache_dir=gz_evo_only_cache,
        download_config=DownloadConfig(
            cache_dir=gz_evo_only_cache,
            local_files_only=True
        )
    )  # may run out of mem on head node, that's okay