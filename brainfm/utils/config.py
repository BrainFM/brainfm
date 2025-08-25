import copy
import os
from types import SimpleNamespace
from .helper import load_json, load_yaml

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
  
def get_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file {path} does not exist.")
    
    config_dict = None
    
    if path.endswith('.yaml') or path.endswith('.yml'):
        config_dict = load_yaml(path)
    elif path.endswith('.json'):
        config_dict = load_json(path)
    else:
        raise ValueError("Configuration file must be .yaml, .yml, or .json format.")

    if not isinstance(config_dict, dict):
        raise ValueError("Cannot load config file into a dictionary. Check the file format.")
    return Config(config_dict)