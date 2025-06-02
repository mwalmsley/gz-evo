import torch

if __name__ == "__main__":
    print("Running CUDA debug script...")

    print(torch.cuda.is_available())

    print(torch.cuda.device_count())

    print(torch.cuda.current_device())
