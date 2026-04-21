from omegaconf import OmegaConf
from omegaconf import MISSING

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Tuple, Optional


@dataclass
class DataLoading:
    batch_size:         int = 64
    shuffle:            bool = True
    num_workers:        int = 0
    pin_memory:         bool = False
    persistent_workers: bool = False


@dataclass
class DataPreprocessing:
    data_parser:    str = MISSING
    label_mapping:  Dict[str, int] = field(default_factory=dict)
    resize_dims:    Tuple[int, int] = (128, 128)
    data_mode:      str = "rgb"                # auto-completed from input_channels in SubnetworkParams
    medmnist_csv_file:  Optional[str] = None


@dataclass
class DataParams:
    dataset_path:       str = MISSING
    images_extension:   str = MISSING
    data_preprocessing: DataPreprocessing = field(default_factory=DataPreprocessing)
    data_loading:       DataLoading = field(default_factory=DataLoading)


@dataclass
class TrainingParams:
    mode:           str = "binary"
    loss:           str = "binary_cross_entropy"
    epochs:         int = 10
    optimizer:      str = "adam"
    learning_rate:  float = 0.001
    momentum:       float = 0.0
    weight_decay:   float = 0.001
    early_stopping_patience: int = 4
    early_stopping_monitor: str = "val_loss"
    early_stopping_mode:    str = "min"


@dataclass
class SubnetworkParams:
    architecture:       str = "base_one"
    input_channels:     int = 3
    # fc_pred_units:    int = 1            # num_classes
    base_channels:      int = 32
    fc_hidden_units:    int = 64
    pred_activation:    str = "tanh"


@dataclass
class EPUCNNParams:
    num_subnetworks:   int = 4
    num_classes:       int = 1
    epu_activation:    str = "sigmoid"
    subnetwork_config: SubnetworkParams = field(default_factory=SubnetworkParams)


@dataclass
class IUSConfig:
    model:          EPUCNNParams = field(default_factory=EPUCNNParams)
    train_params:   TrainingParams = field(default_factory=TrainingParams)
    data_params:    DataParams = field(default_factory=DataParams)
    log_dir:            Optional[str] = "./logs"
    checkpoint_dir:     Optional[str] = "./checkpoints"
    experiment_name:    Optional[str] = "ius_experiment"
    timestamp:          Optional[str] = None
    experiment_saved_folder_name: Optional[str] = None

    @staticmethod
    def from_yaml(filepath: str) -> "IUSConfig":
        yaml_cfg = OmegaConf.load(filepath)
        merged = OmegaConf.merge(OmegaConf.structured(IUSConfig), yaml_cfg)
        merged = OmegaConf.to_object(merged)
        if isinstance(merged, IUSConfig):
            cfg = merged
        else:               # manually
            cfg = IUSConfig(**merged)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": vars(self.model),
            "train_params": vars(self.train_params),
            "data_params": vars(self.data_params),
        }

    def __repr__(self):
        return OmegaConf.to_yaml(OmegaConf.structured(self), resolve=True)


if __name__ == "__main__":
     cfg_file = "configs/train_config.yaml"
     cfg = IUSConfig.from_yaml(cfg_file)
     print(cfg.model)
     print(cfg.train_params)
     print(cfg.data_params)

     # yaml_cfg = IUSConfig.from_yaml("checkpoints/test_experiment_unet_0032_20260203_192949/user_config.yaml")
     # print(yaml_cfg)
     # print(type(yaml_cfg))
     # print(yaml_cfg.data_params.test_split)
