import os
import torch
import argparse

from pathlib import Path
from datetime import datetime

# mine
from data.loading import EPUDatasetFromConfig
from data.dataloader import to_dataloader
from model.module_mapping import layer_mapping
from utils.metrics import EPUMetrics

from utils.eval_utils import EPUCNNEval
from utils.eval_utils import InferenceRunnerEPUCNN
from utils.train_utils import create_output_folders, save_to_json
from utils.omega_parser import IUSConfig


BASE_PATH = Path(__file__).resolve().parent


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_folder_name", type=str, required=True,
                        help="Folder name containing epu configuration & saved ckpt")
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "test", "validation"],)
    args = parser.parse_args()
    return args


def main():
    args = parse_options()
    data_split = args.data_split
    group_by = None

    # Load saved config
    print('Loading configuration...')
    saved_epu_folder = (BASE_PATH / "../results/checkpoints").resolve()
    saved_epu_folder = os.path.join(saved_epu_folder, args.experiment_folder_name)
    cfg_path = os.path.join(saved_epu_folder, "epu_config.yaml")
    cfg = IUSConfig.from_yaml(cfg_path)

    # Setup Dataset & Dataloader
    print('Load data...')
    dset = EPUDatasetFromConfig(dataconfig=cfg.data_params, group_by=group_by)
    eval_dataset = dset.get_dataset(dataset_mode=data_split)

    # update cfg.data_params.data_loading
    cfg.data_params.data_loading.shuffle = False
    cfg.data_params.data_loading.batch_size = 1
    dataloader_eval = to_dataloader(dataset=eval_dataset, loading_cfg=cfg.data_params.data_loading)

    # Setup Saving Dir
    output_folder = str((BASE_PATH / "../results/classification_performance").resolve())
    output_folder = os.path.join(output_folder, args.experiment_folder_name)
    create_output_folders(output_folder)        # skipped if exists
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load model
    print('Loading trained model...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_ckpt = os.path.join(saved_epu_folder, f"ckpt_{cfg.experiment_saved_folder_name}.pt")
    epu = EPUCNNEval(epu_cfg=cfg.model)
    trained_epu = epu.load_ckpt(device=device, ckpt_path=trained_ckpt)

    trained_epu = InferenceRunnerEPUCNN(epu_model=trained_epu,
                                        device=device,
                                        mode=cfg.train_params.mode)

    # metrics = EPUMetrics(mode=cfg.train_params.mode,
    #                      n_classes=cfg.model.num_classes,
    #                      activation=layer_mapping(cfg.model.epu_activation)())
    # epu_results = trained_epu.predict(dataloader=dataloader_eval,
    #                                   raw_logits=True,                  # not epu activation
    #                                   return_predictions=True)

    epu_results = trained_epu.predict(dataloader=dataloader_eval,
                                      raw_logits=False,                # apply epu activation
                                      return_predictions=True)
    confidence_level = 0.5
    metrics = EPUMetrics(mode=cfg.train_params.mode,
                         n_classes=cfg.model.num_classes,
                         confidence_level=confidence_level,
                         activation=layer_mapping("none")())

    metric_scores = metrics.compute(y_true=epu_results["targets"], y_pred=epu_results["predictions"],)

    save_info_dict = {
        "epu_ckpt": trained_ckpt,
        "config": cfg_path,
        "data_split": data_split,
        "batch_size": cfg.data_params.data_loading.batch_size,
        "dataloader_samples": len(epu_results["targets"]),
        "classification_performance": metric_scores,
        "confidence": confidence_level,
        "timestamp": timestamp,
        "experiment_id": cfg.experiment_saved_folder_name,
    }
    json_file_path = os.path.join(output_folder, f"epu_classification_performance_{timestamp}.json")
    save_to_json(save_info_dict, json_file_path)


if __name__ == "__main__":
    # python -m scripts.infer_epu --experiment_folder_name ius_dataset_name_base_one_0000_timestamp
    main()
