import time
import logging

import torch
import pytorch_lightning as pl


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    logging.info("Running CUDA debug script...")

    logging.info(torch.cuda.is_available())

    logging.info(torch.cuda.device_count())

    logging.info(torch.cuda.current_device())

    from lightning.pytorch.accelerators import find_usable_cuda_devices

    # create dummy model and training data
    class DummyModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(10, 1)
            logging.info("Dummy model initialized on device:", self.layer.weight.device)

        def forward(self, x):
            time.sleep(1)
            return self.layer(x)
        
        def training_step(self, batch, batch_idx):
            x = batch
            output = self.forward(x)
            loss = torch.nn.functional.mse_loss(output, torch.zeros_like(output))
            return loss
        
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)
        
    class DummyDataModule(pl.LightningDataModule):
        def __init__(self):
            super().__init__()

        def train_dataloader(self):
            return torch.utils.data.DataLoader(torch.randn(10000, 10), batch_size=32)



    # this is not very useful - only checks if a tensor can be placed
    # not if you can put the whole model on the device
    # devices = find_usable_cuda_devices(1)
    # logging.info("Usable CUDA devices:", devices)

    device_mem = {}
    for device in range(torch.cuda.device_count()):
        logging.info(f"Device {device}: {torch.cuda.get_device_name(device)}")
        logging.info(f"  Memory Allocated: {torch.cuda.memory_allocated(device)} bytes")
        logging.info(f"  Memory Cached: {torch.cuda.memory_reserved(device)} bytes")
        device_mem[device] = {
            'allocated': torch.cuda.memory_allocated(device),
            'cached': torch.cuda.memory_reserved(device)
        }

    lowest_mem_device = min(device_mem, key=lambda k: device_mem[k]['allocated'])
    logging.info(f"Lowest memory device: {lowest_mem_device} with {device_mem[lowest_mem_device]['allocated']} bytes allocated")
    devices = [lowest_mem_device]  # use the device with the lowest memory allocation
    logging.info("Using devices:", devices)

    x = torch.randn(100000, 10).to(f'cuda:{lowest_mem_device}')
    logging.info("Tensor created on device:", x.device)
    logging.info("new memory: ", torch.cuda.memory_allocated(lowest_mem_device))

    # time.sleep(1000)

    logging.info('Releasing memory...')
    

    # trainer = pl.Trainer(
    #     devices=devices,
    #     accelerator='gpu',
    #     max_epochs=1
    # )

    # trainer.fit(
    #     model=DummyModel(),
    #     datamodule=DummyDataModule()
    # )
    # logging.info("Training completed.")