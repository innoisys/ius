import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from pathlib import Path
from datetime import datetime

# mine
from utils.omega_parser import IUSConfig
from utils.train_utils import (create_output_folders, create_experiment_folder,
                               update_experiment_metadata, save_config_to_output_folder)
from utils.sanity_utils import SanityChecker
from utils.callbacks import setup_callbacks
from utils.config_utils import model_cfg_to_epucnn
from utils.trainer import EPUTrainer
from utils.metrics import EPUMetrics
from model.epu import EPUCNN
from model.module_mapping import layer_mapping
from data.loading import EPUDatasetFromConfig
from data.dataloader import to_dataloader


BASE_PATH = Path(__file__).resolve().parent


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filepath", type=str, required=True, help="Path containing configuration")
    parser.add_argument("--tensorboard", action="store_true", help="Enable tensorboard")
    args = parser.parse_args()
    return args


def main():
    args = parse_options()

    print('Loading configuration...')
    # Load configuration .yaml
    cfg = IUSConfig.from_yaml(args.config_filepath)

    # Sanity Check User's Config
    SanityChecker(cfg).sanity_check()

    # Set User's params
    model_cfg, train_params, data_params = cfg.model, cfg.train_params, cfg.data_params

    print('Setup directories...')
    # Setup Experiment Name & Saving Directories
    cfg.log_dir = str((BASE_PATH / cfg.log_dir).resolve())
    cfg.checkpoint_dir = str((BASE_PATH / cfg.checkpoint_dir).resolve())
    create_output_folders([cfg.log_dir, cfg.checkpoint_dir])  # skipped if exists
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = create_experiment_folder(log_root=cfg.log_dir, model=model_cfg.subnetwork_config.architecture,
                                               experiment=cfg.experiment_name, timestamp=timestamp)
    logs_folder = os.path.join(cfg.log_dir, experiment_name)
    ckpt_folder = os.path.join(cfg.checkpoint_dir, experiment_name)
    create_output_folders([logs_folder, ckpt_folder])
    update_experiment_metadata(cfg, experiment_name=experiment_name, timestamp=timestamp,)
    save_config_to_output_folder(out_folder=ckpt_folder, cfg=cfg, cfg_filename="epu_config.yaml")

    # Set Device
    print('Set device ...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    print('Build detection model...')
    epu_model = EPUCNN(**model_cfg_to_epucnn(model_cfg))

    # # Setup Dataset & Dataloader
    print('Load data...')
    data_params.dataset_path = str((BASE_PATH / data_params.dataset_path).resolve())
    dset = EPUDatasetFromConfig(dataconfig=data_params)
    dataset_train = dset.get_dataset(dataset_mode="train")
    dataset_val = dset.get_dataset(dataset_mode="validation")
    dataloader_train = to_dataloader(dataset=dataset_train, loading_cfg=data_params.data_loading)
    dataloader_val = to_dataloader(dataset=dataset_val, loading_cfg=data_params.data_loading)

    print('Setup optimizer and callbacks ...')
    # Setup callbacks loss & optimizer & metrics
    calls = setup_callbacks(ckpt_path=os.path.join(ckpt_folder, f"ckpt_{experiment_name}.pt"),
                            log_dir=logs_folder,
                            early_patience=train_params.early_stopping_patience,
                            early_mode=train_params.early_stopping_mode,
                            early_monitor=train_params.early_stopping_monitor,
                            use_tensorboard=args.tensorboard,
                            )   # other kwargs to pass, override defaults:
                                # delta=0, verbose=True, restore_best_weights=False,save_final_model=True)
                                # log_histograms=False, tb_port=6006, tb_browser=False

    loss_fun = nn.BCEWithLogitsLoss() if train_params.mode == "binary" else nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=epu_model.parameters(),
                          lr=train_params.learning_rate,
                          momentum=train_params.momentum,
                          weight_decay=train_params.weight_decay,)
    metrics = EPUMetrics(mode=train_params.mode,
                         n_classes=model_cfg.num_classes,
                         activation=layer_mapping(model_cfg.epu_activation)()
                         )

    # # launch training
    print('Start training...')
    trainer = EPUTrainer(model=epu_model,
                         device=device,
                         optimizer=optimizer,
                         criterion=loss_fun,
                         epochs=train_params.epochs,
                         train_loader=dataloader_train,
                         val_loader=dataloader_val,
                         callbacks=calls,
                         metrics=metrics,
                         checkpoint_dir=ckpt_folder,
                         )
    trainer.train()


if __name__ == "__main__":
    # python -m scripts.train_epu --config_filepath configs/train_config.yaml
    main()
