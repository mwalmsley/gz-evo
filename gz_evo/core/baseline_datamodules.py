
import pytorch_lightning as pl
from typing import Optional
import logging
import datasets as hf_datasets

# now all refactored into
from galaxy_datasets.pytorch import galaxy_datamodule, dataset_utils

LABEL_ORDER = ['smooth_round', 'smooth_cigar', 'unbarred_spiral', 'edge_on_disk', 'barred_spiral', 'featured_without_bar_or_spiral']
# convert to integers
LABEL_ORDER_DICT = {label_: i for i, label_ in enumerate(LABEL_ORDER)}




if __name__ == "__main__":


    # for testing purposes

    import datasets as hf_datasets
    import time

    ds_dict = hf_datasets.load_dataset("mwalmsley/gz-evo", 'tiny')
    ds_dict['train'] = ds_dict['train']#.repeat(5)
    # print(ds_dict)

    # logging.info("Transforming images to tensors")
    # ds_dict = pil_to_tensors(ds_dict, num_workers=1) 

    # 10 seconds with this fairly minimal transform
    # transform = v2.Compose([
    #     v2.ToImage(),  # Convert to tensor
    #     v2.ToDtype(torch.uint8, scale=True),  # probably already uint8
    #     v2.Resize((224, 224), antialias=True),
    #     v2.ToDtype(torch.float32, scale=True)  # float for models
    # ])


    from galaxy_datasets.transforms import default_view_config, minimal_view_config, fast_view_config
    cfg = default_view_config()
    # cfg.pil_to_tensor = True
    # cfg.erase_iterations = 0
    # cfg = minimal_view_config()
    cfg.interpolation_method='nearest'

    # from torchvision.transforms import InterpolationMode
    # interpolation_mode = InterpolationMode.NEAREST

    # with image at the start...

    # without dtype casts, still 19 seconds (no speedup from uint8 apparently)
    # wit toimage at the end, 10 seconds! (9 seconds with only resize)
    # So despite the official advice, it's 2x faster to use PIL backend for me here
    # https://docs.pytorch.org/vision/stable/transforms.html#performance-considerations
    # this may be because affine (like resize) is quicker with channels-last PIL format


    # with only resize, 8 seconds with toimage first, 8 seconds with toimage at the end
    # with affine+resize, 18/19 seconds with toimage first, 10 seconds with toimage at the end
    # so affine is drmatically slower if toimage is first (channels-first tensor) than if it is last (channels-last PIL image)
    # transform = v2.Compose([
    #     # v2.ToImage(),  # Convert to tensor
    #     # v2.ToDtype(torch.uint8, scale=True),  # probably already uint8
    #     v2.RandomAffine(**cfg.random_affine),  # no resize, random affine
    #     v2.Resize(cfg.output_size, antialias=True),  # resize to output size
    #     # v2.ToDtype(torch.float32, scale=True)  # float for models
    #     v2.ToImage(),  # Convert to tensor
    # ])


    # cfg = fast_view_config()

    # 30 seconds with default -> 10.6 seconds with PIL backend
    # 21 seconds with minimal, 17.6 with nearest interpolation -> 7.6 seconds with PIL backend
    # 8 seconds with fast, mostly decoding and thread lock
    # cfg = fast_view_config()
    transform = GalaxyViewTransform(cfg).transform

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

    datamodule = galaxy_datamodule.HuggingFaceDataModule(
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
        # print(batch['label'])
        pass
    end_time = time.time()
    print(f"Time taken to iterate over train_dataloader: {end_time - start_time:.2f} seconds. Iterable: {datamodule.iterable}, num_workers: {datamodule.num_workers}, prefetch_factor: {datamodule.prefetch_factor}")
    print('Complete')
    exit()
