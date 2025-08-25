import os
import random
import numpy as np
import torch

def set_seed(seed: int, reproduce: bool = False) -> None:
    # Python built-ins
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # Torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN (for reproducibility at the cost of speed)
    if reproduce:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True) # might raise error if some operation is not deterministic
   
def get_device(device_str: str):
    return torch.device(device_str if torch.cuda.is_available() else "cpu")
