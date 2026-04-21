import os
import json

from typing import Any, List, Union
from datetime import datetime
from omegaconf import OmegaConf

from utils.omega_parser import IUSConfig


def get_next_experiment_id(log_root: str, base_name: str, pad: int = 4) -> str:
    """
    Finds the next zero-padded experiment ID based on existing folders.

    Args:
        log_root:   Root directory where experiment folders are stored.
        base_name:  Name of experiment e.g., "experiment_ius"
        pad:        Number of digits to pad the ID (default: 4).

    Returns:
        Zero-padded string for next experiment ID, e.g., "0000", "0001".
    """
    existing = [
        d for d in os.listdir(log_root)
        if os.path.isdir(os.path.join(log_root, d)) and d.startswith(base_name)
    ]

    # Extract numeric ID
    existing_ids = []
    for d in existing:
        try:
            num_str = d.split("_")[-3]
            existing_ids.append(int(num_str))
        except (ValueError, IndexError):
            continue

    next_id = max(existing_ids, default=-1) + 1
    return str(next_id).zfill(pad)


def create_experiment_folder(log_root: str, model: str, experiment: str, timestamp: str = None) -> str:
    """
    Creates experiment folder with professional naming:
    experiment_0000_YYYYMMDD_HHMMSS

    Args:
        log_root: Root directory for logs.
        task: Downstream task name.
        model: Model architecture name.
        dataset: Dataset name.

    Returns:
        experiment_name
        @param timestamp:
    """
    base_name = f"{experiment}_{model}"
    exp_id = get_next_experiment_id(log_root, base_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp is None else timestamp
    experiment_name = f"{base_name}_{exp_id}_{timestamp}"

    return experiment_name


# def create_output_folders(log_root: str, ckpt_root: str, experiment_name: str) -> None:
#     os.makedirs(os.path.join(log_root, experiment_name), exist_ok=True)
#     os.makedirs(os.path.join(ckpt_root, experiment_name), exist_ok=True)
def create_output_folders(folder_list: Union[List[str], str]) -> None:
    if isinstance(folder_list, str):
        folder_list = [folder_list]
    for folder_name in folder_list:
        os.makedirs(folder_name, exist_ok=True)


def save_config_to_output_folder(out_folder: str, cfg: IUSConfig, cfg_filename="user_config.yaml") -> None:
    config_path = os.path.join(out_folder, cfg_filename)
    OmegaConf.save(config=OmegaConf.structured(cfg), f=config_path)


def update_experiment_metadata(cfg: Any,
                               experiment_name: str = None,
                               timestamp: str = None) -> None:
    """
    Update experiment metadata fields. Uses given timestamp if provided,
    otherwise does not override.
    """
    if timestamp:
        cfg.timestamp = timestamp

    if experiment_name and not getattr(cfg, "experiment_name", None):
        cfg.experiment_name = experiment_name

    if hasattr(cfg, "train_params"):
        train_params = cfg.train_params
        train_params.experiment_name = experiment_name
    if hasattr(cfg, "model"):
        model_cfg = cfg.model
        model_cfg.experiment_name = experiment_name

    cfg.experiment_saved_folder_name = experiment_name


def save_to_json(diction_to_save: dict, saving_path: str) -> None:
    try:
        with open(saving_path, "w") as f:
            json.dump(diction_to_save, f, indent=4)
        print(f"Saved json file at {saving_path}")
    except Exception as e:
        print(f"Failed to save json file at {saving_path}: {e}")
