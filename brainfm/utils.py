import os
import yaml
import json


def set_seed(seed: int) -> None:
    pass

def load_config(path: str) -> dict:
    if path.endswith('.yaml') or path.endswith('.yml'):
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    elif path.endswith('.json'):
        import json
        with open(path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported config file format. Use .yaml, .yml, or .json")

def get_logger(log_dir: str, experiment_name: str):
    pass

def validate_config_path(path: str) -> None:
    if not (path.endswith('.yaml') or path.endswith('.yml') or path.endswith('.json')):
        raise ValueError("Configuration file must be .yaml, .yml, or .json format.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file {path} does not exist.")