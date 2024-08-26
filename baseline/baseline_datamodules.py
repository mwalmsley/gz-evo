import pytorch_lightning as pl
from typing import Optional
import logging
import datasets as hf_datasets

from PIL import Image
import torch
from torch.utils.data import DataLoader

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
        batch_size=256,  # careful - will affect final performance
        num_workers=4,
        prefetch_factor=4,
        seed=42,
        dataset_kwargs={}
    ):
        # super().__init__()
        pl.LightningDataModule.__init__(self)

        self.batch_size = batch_size

        self.num_workers = num_workers
        self.seed = seed
        self.dataset_dict = dataset_dict
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.target_transform = target_transform
        self.dataset_kwargs = dataset_kwargs

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

        # self.prepare_data_per_node = False  # run prepare_data below only on master node (and one process)

    # only called on main process
    # def prepare_data(self):
        # pass  # 

    # called on every gpu
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            logging.warning('Creating validation split from 20%% of train dataset')
            train_and_val_dict = self.dataset_dict["train"].train_test_split(test_size=0.2, shuffle=True)
            train_dataset_hf = train_and_val_dict["train"]
            val_dataset_hf = train_and_val_dict['test']  # actually used as val
            del train_and_val_dict

            self.train_dataset = GenericDataset(
                    dataset=train_dataset_hf,
                    transform=self.train_transform,
                    target_transform=self.target_transform,
                    **self.dataset_kwargs
            )
            self.val_dataset = GenericDataset(
                dataset=val_dataset_hf,
                transform=self.test_transform,
                target_transform=self.target_transform,
                **self.dataset_kwargs
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = GenericDataset(
                dataset=self.dataset_dict['test'],
                transform=self.test_transform,
                target_transform=self.target_transform,
                **self.dataset_kwargs
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
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

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            timeout=self.dataloader_timeout,
        )

class GenericDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: hf_datasets.Dataset,  # HF Dataset
        transform=None,
        target_transform=None,
        **kwargs
    ):
        """
        Create custom PyTorch Dataset using HuggingFace dataset


        Args:
            name (str): Name of HF dataset
            transform (callable, optional): See Pytorch Datasets. Defaults to None.
            target_transform (callable, optional): See Pytorch Datasets. Defaults to None.
        """
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        example:dict = self.dataset[idx]
        
        if self.transform:
             # torchvision transforms operate on PIL images
             example['image'] = self.transform(example['image'])  # example['image'] is already a torch.Tensor via hf.set_format('torch')

        if self.target_transform:
            example = self.target_transform(example)  # slightly generalised: target_transform to expect and yield example, changing the targets (labels)

        return example  # dict like {'image': image, 'foobar', ..., 'label': ...}

