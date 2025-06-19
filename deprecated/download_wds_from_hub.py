import os
import glob

import webdataset as wds
from huggingface_hub import snapshot_download


def download_wds(local_dir, cache_dir, max_workers=8):
    allow_patterns = "data/*.tar"  # include these files
    # allow_patterns = "tiny/*.tar"  # for debugging

    dataset_names = [
        # "gz_ukidss_wds",
        # "gz_desi_wds",
        "gz_candels_wds",
        # "gz_hubble_wds",
        # "gz_h2o_wds",
        # "gz2_wds",  # TODO needs remaking, config is evo not default
    ]
    for dataset_name in dataset_names:
        local_dataset_dir = f"{local_dir}/{dataset_name}"

        # download latest commit to cache
        # e.g. $HF_HOME/hub/datasets--mwalmsley--gz_ukidss_wds/snapshots/{long_commit_hash}/data/*.tar
        # https://huggingface.co/docs/huggingface_hub/v0.23.2/en/package_reference/file_download#huggingface_hub.snapshot_download
        snapshot_download(
            f"mwalmsley/{dataset_name}",
            allow_patterns=allow_patterns,
            repo_type="dataset",
            max_workers=max_workers,
            cache_dir=cache_dir,
            local_dir=local_dataset_dir,
            # local_dir_use_symlinks=False,  # now deprecated, HF no longer uses symlinks
        )


def test_download_wds():

    local_dir = "data/webdatasets"  # TODO replace with your own path
    cache_dir = os.environ.get("HF_HOME", None)  # default is ~/.cache/huggingface

    download_wds(local_dir, cache_dir)

    all_wds_urls = glob.glob(f"{local_dir}/**/*train*.tar", recursive=True)
    assert all_wds_urls, f"No tar files found in {local_dir}"
    print(all_wds_urls)

    print(wds.WebDataset(all_wds_urls).decode())


if __name__ == "__main__":

    # MW's actual download script, for further illustration
    if os.path.isdir("/share/nas2"):  # galahad UK cluster
        local_dir = "/share/nas2/walml/webdatasets/huggingface"
        cache_dir = "/share/nas2/walml/cache/huggingface"
    elif os.path.isdir('/media/walml/ssd'):  # mike@dunlap
        local_dir = "/media/walml/ssd/webdatasets/huggingface"  # TODO
        cache_dir = "/media/walml/ssd/huggingface"
    else:  # default
        local_dir = "data/webdatasets"
        cache_dir = os.environ.get("HF_HOME", None)

    # download_wds(local_dir, cache_dir)

    test_download_wds()
