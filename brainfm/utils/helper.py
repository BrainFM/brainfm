import os
import torch


def set_seed(seed: int) -> None:
    pass
   
def get_device(device_str: str):
    return torch.device(device_str if torch.cuda.is_available() else "cpu")
