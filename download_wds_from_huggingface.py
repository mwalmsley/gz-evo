import glob

import os
import webdataset as wds

from huggingface_hub import HfFolder, HfApi, snapshot_download

# def stream():
    # TODO the docs are outdated, it moved here
    # hf_token = HfFolder.get_token()

    # url = "https://huggingface.co/datasets/timm/imagenet-12k-wds/resolve/main/imagenet12k-train-{{0000..1023}}.tar"
    # url = "https://huggingface.co/datasets/timm/imagenet-12k-wds/resolve/main/imagenet12k-train-0000.tar"

    # -L to send to stdin
    # -s I'm not sure?
    # url = f"pipe:curl -s -L {url} -H 'Authorization:Bearer {hf_token}'"
    # url = f"curl {url} -H 'Authorization:Bearer {hf_token}' -O temp.tar"

    # header = f'Authorization:Bearer {hf_token}'

    # print(f'wget {url} --header="{header}" -O temp.tar')

    # api =  HfApi()


if __name__ == "__main__":



    if os.path.isdir("/share/nas2"):
        cache_dir = "/share/nas2/walml/cache/huggingface"
        max_workers = 8
    else:
        cache_dir = "/media/walml/ssd/huggingface"
        max_workers = 8

    default_pattern = "data/*.tar"
    evo_pattern = "evo/*.tar"
    all_wds_urls = []
    dataset_pattern_pairs = [
        ('gz_ukidss_wds', default_pattern),
        ('gz_desi_wds', default_pattern),
        ('gz_candels_wds', default_pattern),
        ('gz_hubble_wds', default_pattern),
        ('gz_h2o_wds', default_pattern),
        ('gz2_wds', evo_pattern)
    ]
    # dataset_pattern_pairs = [("gz_ukidss_wds", default_pattern)]
    for dataset_name, pattern in dataset_pattern_pairs:

        local_dir = f'/share/nas2/walml/webdatasets/huggingface/{dataset_name}'

        # download latest commit to cache
        # e.g. $HF_HOME/hub/datasets--mwalmsley--gz_ukidss_wds/snapshots/{long_commit_hash}/data/*.tar
        # https://huggingface.co/docs/huggingface_hub/v0.23.2/en/package_reference/file_download#huggingface_hub.snapshot_download
        snapshot_download(
            f"mwalmsley/{dataset_name}",
            allow_patterns=pattern,
            repo_type="dataset",
            max_workers=max_workers,
            cache_dir=cache_dir,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )

        print(local_dir)
        print(glob.glob(f'{local_dir}/{pattern}'))

    #     all_wds_urls += glob.glob(f'{local_dir}/{pattern}')

    # print(all_wds_urls)

    # print(wds.WebDataset(all_wds_urls).decode())
