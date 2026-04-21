import os
import torch
import argparse
import numpy as np

from pathlib import Path

# mine
from data.loading import EPUDatasetFromConfig
from data.dataloader import to_dataloader
from utils.omega_parser import IUSConfig
from utils.train_utils import create_output_folders
from utils.eval_utils import EPUCNNEval



BASE_PATH = Path(__file__).resolve().parent


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_folder_name", type=str, required=True,
                        help="Folder name containing epu configuration & saved ckpt "
                             "eg ius_dataset_name_base_one_0000_timestamp")
    parser.add_argument("--cb_data", type=str, default=None,
                        help="cb vector data. If not specified estimates all cb_vectors")
    args = parser.parse_args()
    return args


def main():
    args = parse_options()

    # Load saved config
    print('Loading configuration...')

    saved_epu_folder = (BASE_PATH / "../results/checkpoints").resolve()
    saved_epu_folder = os.path.join(saved_epu_folder, args.experiment_folder_name)
    cfg_path = os.path.join(saved_epu_folder, "epu_config.yaml")
    cfg = IUSConfig.from_yaml(cfg_path)

    model_cfg, data_params = cfg.model, cfg.data_params

    # Create model
    print('Loading trained model...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_ckpt = os.path.join(saved_epu_folder, f"ckpt_{cfg.experiment_saved_folder_name}.pt")
    epu = EPUCNNEval(epu_cfg=cfg.model)
    trained_epu = epu.load_ckpt(device=device, ckpt_path=trained_ckpt)

    # Setup Saving Dir
    output_folder = str((BASE_PATH / "../results/cb_vectors").resolve())
    output_folder = os.path.join(output_folder, args.experiment_folder_name)
    create_output_folders(output_folder)        # skipped if exists

    # Setup Dataset & Dataloader
    print('Load data...')

    if args.cb_data is not None:
        group_by = [args.cb_data]
    else:
        group_by = data_params.data_preprocessing.label_mapping.keys()

    for group_by_item in group_by:
        data_params.dataset_path = str((BASE_PATH / data_params.dataset_path).resolve())
        dset = EPUDatasetFromConfig(dataconfig=data_params, group_by=group_by_item)
        # eval_dataset = dset.get_dataset(dataset_mode="test")
        eval_dataset = dset.get_dataset(dataset_mode="validation")
        dataloader_eval = to_dataloader(dataset=eval_dataset, loading_cfg=data_params.data_loading)

        cb_vector = trained_epu.create_baseline_feature_contribution_profile(
            data_loader=dataloader_eval,
            device=device
        )
        cb_vector = cb_vector.detach().cpu().numpy()
        cb_path = os.path.join(output_folder, f"cb_vector_{group_by_item}.npy")
        np.save(cb_path, cb_vector)
        print(f'cb vector saved at {cb_path}')

    print('cb vector computation finished.')


if __name__ == "__main__":
    # python -m scripts.infer_cb_vector --experiment_folder_name ius_dataset_name_base_one_0000_timestamp
    main()
