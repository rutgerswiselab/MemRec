"""
Utility functions for MemRec
"""
import os
import random
import numpy as np
import torch
import yaml
from pathlib import Path


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration file with environment variable substitution"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle inheritance
    if 'inherit' in config:
        base_path = config['inherit']
        base_config = load_config(base_path)
        # Merge configs (current config overrides base)
        base_config.update(config)
        config = base_config
        del config['inherit']
    
    # Substitute environment variables
    config = _substitute_env_vars(config)
    
    return config


def _substitute_env_vars(obj):
    """
    Recursively substitute environment variables in config
    Supports ${ENV:VAR_NAME} syntax
    """
    if isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        # Handle ${ENV:VAR_NAME} pattern
        if obj.startswith('${ENV:') and obj.endswith('}'):
            var_name = obj[6:-1]  # Extract VAR_NAME
            return os.getenv(var_name)
        return obj
    else:
        return obj


def ensure_dir(directory: str):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_device(device_str: str = 'cuda:0') -> torch.device:
    """Get PyTorch device"""
    if device_str.startswith('cuda') and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device('cpu')
