import time

import torch
import pytorch_lightning as pl


if __name__ == "__main__":
    print("Running CUDA debug script...")

    print(torch.cuda.is_available())

    print(torch.cuda.device_count())

    print(torch.cuda.current_device())

    from lightning.pytorch.accelerators import find_usable_cuda_devices

    trainer = pl.Trainer(
        devices=find_usable_cuda_devices(1),
        accelerator='cuda',
        max_epochs=1
    )
    print("Trainer created with usable CUDA devices.")
    print(trainer.devices)

    time.sleep(60)  # Sleep to allow time to check the output