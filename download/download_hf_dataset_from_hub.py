
import os
import glob

"""useful to download HF datasets for offline use, e.g. on a cluster"""

if __name__ == '__main__':

    from torch.utils.data import DataLoader
    import datasets

    for hub_dataset, hub_name in [
        ('mwalmsley/gz_evo', 'default'),
        ('mwalmsley/gz_evo', 'tiny'),
    ]:
        print(f"Downloading {hub_dataset} {hub_name} dataset")
        ds: datasets.DatasetDict = datasets.load_dataset(hub_dataset, name=hub_name)  # type: ignore
        print(f"Downloaded {hub_dataset} {hub_name} dataset with {len(ds['train'])} train examples, {len(ds['test'])} test examples")

        ds.set_format("torch")  # also supports numpy, jax, etc

        dataloader = DataLoader(ds['train'], batch_size=4, num_workers=1) # type: ignore
    


    # adjust these as you like

    # if os.path.isdir("/project/def-bovy/walml"):
    #     os.environ["HF_HOME"] = "/project/def-bovy/walml/cache/huggingface"
    #     os.environ["HF_DATASETS_CACHE"] = "/project/def-bovy/walml/cache/huggingface/datasets"
    #     os.environ["HF_LOCAL_DATASETS_CACHE"] = os.environ.get('SLURM_TMPDIR', '') + '/cache/huggingface/datasets'
    #     os.environ['GZ_EVO_MANUAL_DOWNLOAD_LOC'] = '/project/def-bovy/walml/tmp/gz-evo'
    # else:
    #     assert os.path.isdir('/share/nas2/walml'), 'Please add your own path'
    #     os.environ['HF_HOME']="/share/nas2/walml/cache/huggingface"
    #     os.environ['HF_DATASETS_CACHE']="/share/nas2/walml/cache/huggingface/datasets"
    #     os.environ["HF_LOCAL_DATASETS_CACHE"] = '/state/partition1/cache/huggingface/datasets'
    #     os.environ['GZ_EVO_MANUAL_DOWNLOAD_LOC'] = '/share/nas2/walml/tmp/gz-evo'







    # manual download - I don't remember why I did this?

    # from datasets import load_dataset, DownloadConfig
    # from huggingface_hub import snapshot_download

    # gz_evo_manual_download_loc = os.environ['GZ_EVO_MANUAL_DOWNLOAD_LOC']
    # # saves here, simple folder with /data/train*.parquet files
    # # should be on shared filesystem to ensure internet access
    # snapshot_download(
    #     repo_id="mwalmsley/gz_evo", 
    #     repo_type="dataset", 
    #     local_dir=gz_evo_manual_download_loc 
    # )

    # # point to those parquet files
    # train_locs = glob.glob(gz_evo_manual_download_loc + '/data/train*.parquet')
    # test_locs = glob.glob(gz_evo_manual_download_loc + '/data/test*.parquet')
    # load_dataset(
    #     path=gz_evo_manual_download_loc,
    #     # data_files must be explicit paths seemingly, not just glob strings. Weird.
    #     data_files={'train': train_locs, 'test': test_locs},
    #     # and place in LOCAL cache
    #     cache_dir=os.environ['HF_LOCAL_DATASETS_CACHE']
    # )
    # load_dataset(
    #     'mwalmsley/gz-evo',
    #     name='default', 
    #     # cache_dir=gz_evo_only_cache,
    #     download_config=DownloadConfig(
    #         cache_dir=gz_evo_only_cache,
    #         local_files_only=True
    #     )
    # )  # may run out of mem on head node, that's okay