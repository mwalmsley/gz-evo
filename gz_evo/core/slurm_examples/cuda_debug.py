import time
import logging
import subprocess

import numpy as np
import torch


def get_available_memory_by_device():
    # uses nvidia-smi as torch.cuda.memory_allocated() seems broken, always 0
    # requires export CUDA_DEVICE_ORDER=PCI_BUS_ID to ensure nvidia-smi order = torch.cuda device order
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    memory_by_device = [int(x) for x in result.stdout.decode('utf-8').strip().split('\n')]

    logging.info("Available memory by device: {}".format(zip(
        range(len(memory_by_device)), 
        memory_by_device)
    ))

    return memory_by_device


def get_highest_free_memory_device() -> int:

    memory_by_device = get_available_memory_by_device()

    highest_free_mem_device = np.argmax(memory_by_device)

    logging.info(f"Lowest memory device: {highest_free_mem_device} with {memory_by_device[highest_free_mem_device]} bytes allocated")
    return highest_free_mem_device  # type: ignore



if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    logging.info("Running CUDA debug script...")

    logging.info(torch.cuda.is_available())

    logging.info(torch.cuda.device_count())

    logging.info(torch.cuda.current_device())

    # this is not very useful - only checks if a tensor can be placed
    # not if you can put the whole model on the device
    # devices = find_usable_cuda_devices(1)
    # logging.info("Usable CUDA devices:", devices)

    # this is not useful as it only seems to work within the same process, always returns 0
    # device_mem = {}
    # for device in range(torch.cuda.device_count()):
    #     logging.info(f"Device {device}: {torch.cuda.get_device_name(device)}")
    #     logging.info(f"  Memory Allocated: {torch.cuda.memory_allocated(device)} bytes")
    #     logging.info(f"  Memory Cached: {torch.cuda.memory_reserved(device)} bytes")
    
    # custom version with nvidia-smi
    highest_free_memory_device = get_highest_free_memory_device()


    x = torch.randn(500000, 10).to(f'cuda:{highest_free_memory_device}')
    logging.info(f"Tensor created on device: {x.device}")

    time.sleep(1000)

    logging.info('Releasing memory...')
    
