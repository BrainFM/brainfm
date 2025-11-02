import os
import random
import json
import yaml
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

def to_3tuple(x):
    if isinstance(x, (tuple, list)):
        assert len(x) == 3, "Must be a single value or a tuple of 3 values"
        return x
    return (x, x, x)

def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file {path} does not exist.")
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data: dict, path: str):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_yaml(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML file {path} does not exist.")
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)