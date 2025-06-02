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
            print("Dummy model initialized on device:", self.device)

        def forward(self, x):
            time.sleep(1)
            return self.layer(x)
        
        def training_step(self, batch, batch_idx):
            x = batch
            output = self.forward(x)
            loss = torch.nn.functional.mse_loss(output, torch.zeros_like(output))
            return loss
        
    class DummyDataModule(pl.LightningDataModule):
        def __init__(self):
            super().__init__()

        def train_dataloader(self):
            return torch.utils.data.DataLoader(torch.randn(10000, 10), batch_size=32)

    devices = find_usable_cuda_devices(1)
    print("Usable CUDA devices:", devices)

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