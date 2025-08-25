import copy
import os
import yaml
import json
from types import SimpleNamespace

class Config(SimpleNamespace):
    """Dot-access wrapper around a dict, with .raw for the original dict."""
    def __init__(self, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                v = Config(v)
            setattr(self, k, v)
        self.raw = copy.deepcopy(d)   # keep original dict for saving

    def to_dict(self):
        return copy.deepcopy(self.raw)

def validate_config_path(path: str) -> None:
    if not (path.endswith('.yaml') or path.endswith('.yml') or path.endswith('.json')):
        raise ValueError("Configuration file must be .yaml, .yml, or .json format.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file {path} does not exist.")
    
def load_config(path: str) -> dict:
    d = None
    
    if path.endswith('.yaml') or path.endswith('.yml'):
        with open(path, 'r') as f:
            d = yaml.safe_load(f)
    elif path.endswith('.json'):
        with open(path, 'r') as f:
            d = json.load(f)

    if not isinstance(d, dict):
        raise ValueError("Cannot load config file into a dictionary. Check the file format.")
    return Config(d)

def get_config(path: str) -> Config:
    validate_config_path(path)
    return load_config(path)