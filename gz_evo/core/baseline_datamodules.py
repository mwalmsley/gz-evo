import pytorch_lightning as pl
from typing import Optional
import logging
import datasets as hf_datasets

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

LABEL_ORDER = ['smooth_round', 'smooth_cigar', 'unbarred_spiral', 'edge_on_disk', 'barred_spiral', 'featured_without_bar_or_spiral']
# convert to integers
LABEL_ORDER_DICT = {label_: i for i, label_ in enumerate(LABEL_ORDER)}



# https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
class GenericDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dict: hf_datasets.DatasetDict,  # must have train and test keys
        train_transform,
        test_transform,
        target_transform=None,
        # hardware params
        batch_size=256,
        num_workers=4,
        prefetch_factor=4,
        seed=42,
        iterable=False,  # whether to use IterableDataset (faster, no indexed access)
        dataset_kwargs={}
    ):
        # super().__init__()
        pl.LightningDataModule.__init__(self)

        logging.info("Initializing GenericDataModule")

        self.batch_size = batch_size

        self.num_workers = num_workers
        self.seed = seed

        # dataset_dict.set_format(None)  # clear any previous format 
        self.dataset_dict = dataset_dict
        # torchvision transforms, expect an image
        self.train_transform = train_transform
        self.test_transform = test_transform

        self.target_transform = target_transform
        self.dataset_kwargs = dataset_kwargs

        self.iterable = iterable

        if self.num_workers == 0:
            logging.warning(
                "num_workers=0, setting prefetch=None and timeout=0 as no multiprocessing"
            )
            self.prefetch_factor = None
            self.dataloader_timeout = 0
        else:
            self.prefetch_factor = prefetch_factor
            self.dataloader_timeout = 600  # seconds aka 10 mins

        logging.info("Num workers: {}".format(self.num_workers))
        logging.info("Prefetch factor: {}".format(self.prefetch_factor))

    # torchvision acts on image key but HF dataset returns dicts
    def train_transform_wrapped(self, example: dict):
        # REPLACES set_transform('torch') so we also need to make torch tensors
        # https://huggingface.co/docs/datasets/v3.6.0/en/package_reference/main_classes#datasets.Dataset.with_transform
        # best with pil_to_tensor=True
        # print(example['image'], 'before')
        # print(example['image'].shape)
        example['image'] = self.train_transform(example['image'])
        # example = self.target_transform(example) if self.target_transform is not None else example
        # print(example['image'].shape, 'after')
        return example
    def test_transform_wrapped(self, example: dict):
        example['image'] = self.test_transform(example['image'])
        # example = self.target_transform(example) if self.target_transform is not None else example
        return example
    # .map sends example as dict
    # .set_transform sends example as dict of lists, i.e. a batched dict
    # torch collate func will handle the final dict-of-lists-to-tensor, but image transforms only get applied to the first img
    def train_transform_wrapped_batch(self, examples: dict):
        assert len(examples['image']) > 1
        # assert len(examples['image']) == 1, "Expected a batch of size 1 in train, got {}".format(len(examples['image']))
        # return train_transform_wrapped(examples[0])
        examples['image'] = [self.train_transform(im) for im in examples['image']]
        # print(examples)
        return examples
    def test_transform_wrapped_batch(self, examples: dict):
        # assert len(examples['image']) == 1, "Expected a batch of size 1 in test, got {}".format(len(examples['image']))
        # return test_transform_wrapped(examples[0])
        examples['image'] = [self.test_transform(im) for im in examples['image']]
        return examples



    # called on every gpu
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:

            # first we have to split while it's still a normal dataset
            logging.warning('Creating validation split from 20%% of train dataset')
            train_and_val_dict = self.dataset_dict["train"].train_test_split(test_size=0.2, shuffle=True, seed=self.seed, keep_in_memory=self.seed != 42)
            # now shuffled, so flatten indices
            # breaks (silently hangs) if you have already done set_format
            # https://huggingface.co/docs/datasets/en/about_mapstyle_vs_iterable#speed-differences
            logging.info('Flattening indices for train and val datasets, may take a while...')
            train_and_val_dict = train_and_val_dict.flatten_indices(num_proc=self.num_workers, keep_in_memory=self.seed != 42)
            # don't cache for every random seed, or it will fill up disk space

            if self.iterable:
                # convert to iterable datasets
                logging.info('Converting to iterable datasets')
                # these have been split above, is really train and val
                train_dataset_hf = train_and_val_dict["train"].to_iterable_dataset(num_shards=64)
                val_dataset_hf = train_and_val_dict["test"].to_iterable_dataset(num_shards=64)
                del train_and_val_dict

                # apply transforms with map
                # map passes each example through the transform function as a dict
                # (while with_transform sends a list...)
                train_dataset_hf = train_dataset_hf.map(self.train_transform_wrapped)
                val_dataset_hf = val_dataset_hf.map(self.test_transform_wrapped)
                # https://huggingface.co/docs/datasets/en/image_process
                # for dataset, map is cached and intended for "do once" transforms
                # with_format('torch') is (probably) a map
                # set_transform is intended for "on-the-fly" transforms and so is not cached (less disk space, faster)
                # for iterabledataset, map is applied on-the-fly (on every yield) and not cached
                # so there's no need for set_transform: everything is a non-cached map
    
            else:  # leave as not iterable, fast reads 
                train_dataset_hf = train_and_val_dict["train"]
                val_dataset_hf = train_and_val_dict['test']  # actually used as val
                del train_and_val_dict

                # set transforms to use on-the-fly
                # with transform only works with dataset, not iterabledataset
                train_dataset_hf = train_dataset_hf.with_transform(self.train_transform_wrapped_batch)
                val_dataset_hf = val_dataset_hf.with_transform(self.train_transform_wrapped_batch)
                # these act individually, dataloader will handle batching afterwards

            self.train_dataset = train_dataset_hf
            self.val_dataset = val_dataset_hf

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            test_dataset_hf = self.dataset_dict['test']
            # not shuffled, so no need to flatten indices
            # never iterable, for now
            test_dataset_hf = test_dataset_hf.with_transform(self.test_transform_wrapped)
            self.test_dataset = test_dataset_hf


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # assume preshuffled
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            timeout=self.dataloader_timeout,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            timeout=self.dataloader_timeout,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            timeout=self.dataloader_timeout,
        )


if __name__ == "__main__":


    # for testing purposes

    import datasets as hf_datasets
    import time

    ds_dict = hf_datasets.load_dataset("mwalmsley/gz-evo", 'tiny')
    ds_dict['train'] = ds_dict['train']#.repeat(5)
    # print(ds_dict)

    # transform = v2.Compose([
    #     v2.ToImage(),  # Convert to tensor
    #     v2.ToDtype(torch.uint8, scale=True),  # probably already uint8
    #     v2.Resize((224, 224), antialias=True),
    #     v2.ToDtype(torch.float32, scale=True)  # float for models
    # ])
    from galaxy_datasets.transforms import GalaxyViewTransform, default_view_config
    transform = GalaxyViewTransform(default_view_config()).transform

    # most minimal
    # pure dataset (no dataloader) gives examples which are simple dicts, as I guessed

    # def transform_wrapped(example):
    #     example['image'] = transform(example['image'])
    #     return example

    # ds_dict.set_transform(transform_wrapped)

    # dataloader = DataLoader(
    #     ds_dict['train'],
    #     batch_size=2,  
    #     num_workers=0,
    # )

    # for batch in dataloader:
    #     print(batch['image'].shape)
    #     print(batch['spiral-winding-ukidss_medium_fraction'])
    #     break
    # exit()

    def target_transform(example):
        # print(example)
        # exit()
        example['label'] = LABEL_ORDER_DICT[example['summary']]
        # optionally could delete the other keys besides image and id_str
        return example
    
    ds_dict = ds_dict.filter(
        lambda x: x != '',
        input_columns='summary',  # important to specify, for speed
        # load_from_cache_file=False
        # num_proc=cfg.num_workers
    )
    
    ds_dict = ds_dict.map(
        target_transform
    )

    datamodule = GenericDataModule(
        dataset_dict=ds_dict,
        train_transform=transform,
        test_transform=transform,
        # target_transform=target_transform,
        batch_size=8, # applies AFTER transform in iter mode, transform still gets row-by-row examples
        num_workers=0,
        prefetch_factor=None,
        iterable=False
    )
    datamodule.setup()

    dataloader = datamodule.train_dataloader()
    start_time = time.time()
    for batch in dataloader:
        print(batch['label'])
    end_time = time.time()
    print(f"Time taken to iterate over train_dataloader: {end_time - start_time:.2f} seconds. Iterable: {datamodule.iterable}, num_workers: {datamodule.num_workers}, prefetch_factor: {datamodule.prefetch_factor}")
    print('Complete')
    exit()
