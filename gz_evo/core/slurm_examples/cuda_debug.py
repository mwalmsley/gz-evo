import time

import torch
import pytorch_lightning as pl


if __name__ == "__main__":
    print("Running CUDA debug script...")

    print(torch.cuda.is_available())

    print(torch.cuda.device_count())

    print(torch.cuda.current_device())

    from lightning.pytorch.accelerators import find_usable_cuda_devices

    # create dummy model and training data
    class DummyModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(10, 1)
            print("Dummy model initialized on device:", self.layer.weight.device)

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
    # print("Usable CUDA devices:", devices)

    device_mem = {}
    for device in range(torch.cuda.device_count()):
        print(f"Device {device}: {torch.cuda.get_device_name(device)}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(device)} bytes")
        print(f"  Memory Cached: {torch.cuda.memory_reserved(device)} bytes")
        device_mem[device] = {
            'allocated': torch.cuda.memory_allocated(device),
            'cached': torch.cuda.memory_reserved(device)
        }

    lowest_mem_device = min(device_mem, key=lambda k: device_mem[k]['allocated'])
    print(f"Lowest memory device: {lowest_mem_device} with {device_mem[lowest_mem_device]['allocated']} bytes allocated")
    devices = [lowest_mem_device]  # use the device with the lowest memory allocation
    print("Using devices:", devices)

    trainer = pl.Trainer(
        devices=devices,
        accelerator='gpu',
        max_epochs=1
    )

    trainer.fit(
        model=DummyModel(),
        datamodule=DummyDataModule()
    )
    print("Training completed.")