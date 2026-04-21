import os
import torch
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime

# mine
from ius.ius import IUS
from ius.ius_eval_parser import IUSEvalParser
from data.loading import IUSEvalDataset
from data.dataloader import to_dataloader
from utils.eval_utils import EPUCNNEval
from utils.train_utils import create_output_folders, save_to_json
from utils.omega_parser import IUSConfig


BASE_PATH = Path(__file__).resolve().parent


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_folder_name", type=str, required=True,
                        help="Folder name containing epu configuration & saved ckpt")
    parser.add_argument("--cb_vector_tag", type=str, default=None, required=True,
                        help="cb vector data. If not specified estimates all cb_vectors")
    parser.add_argument("--synthetic_images", type=str, default='png',
                        help="It can be either a single synthetic image file or a folder path containing multiple "
                             "synthetic images")
    parser.add_argument("--synthetic_img_extension", type=str, default='png',
                        help="Extension of synthetic images")
    args = parser.parse_args()
    return args


def main():
    args = parse_options()
    cb_vec_tag = args.cb_vector_tag
    synthetic_img_path = args.synthetic_images
    synthetic_img_extension = args.synthetic_img_extension
    experiment_folder_name = args.experiment_folder_name

    # Load saved config
    print('Loading configuration...')
    saved_epu_folder = (BASE_PATH / "../results/checkpoints").resolve()
    saved_epu_folder = os.path.join(saved_epu_folder, experiment_folder_name)
    cfg_path = os.path.join(saved_epu_folder, "epu_config.yaml")
    cfg = IUSConfig.from_yaml(cfg_path)

    # Load EPU for obtaining Contribution Feature Profiles of synthetic images
    print('Loading trained model...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_ckpt = os.path.join(saved_epu_folder, f"ckpt_{experiment_folder_name}.pt")
    epu = EPUCNNEval(epu_cfg=cfg.model)
    trained_epu = epu.load_ckpt(device=device, ckpt_path=trained_ckpt)

    # Load Baseline Contribution Feature Profile (cb_vector)
    print('IUS class...')
    saved_cb_vec_folder = (BASE_PATH / "../results/cb_vectors").resolve()
    saved_cb_vec_folder = os.path.join(saved_cb_vec_folder, experiment_folder_name)
    cb_vec_path = os.path.join(saved_cb_vec_folder, f"cb_vector_{cb_vec_tag}.npy")
    ius = IUS(cb_path=cb_vec_path, cb_tag=cb_vec_tag, device=device)

    # Load Synthetic Images & Estimate their Contribution Feature Profiles (c^_vectors)
    synthetic_img_path = str((BASE_PATH / ".." / synthetic_img_path).resolve())
    parser = IUSEvalParser(path=synthetic_img_path, image_ext=synthetic_img_extension,)
    synthetic_dset = IUSEvalDataset(dataconfig=cfg.data_params).get_dataset(parser=parser)

    cfg.data_params.data_loading.shuffle = False         # update cfg.data_params.data_loading
    cfg.data_params.data_loading.batch_size = 1
    synthetic_dataloader = to_dataloader(dataset=synthetic_dset, loading_cfg=cfg.data_params.data_loading)

    synthetic_c_vecs = trained_epu.calculate_feature_contribution_profiles(
        data_loader=synthetic_dataloader,
        device=device,
    )

    ius_score_list = ius.ius_measure(c_vectors=synthetic_c_vecs)
    synthetic_filenames = parser.image_filenames

    # Setup Saving Dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = str((BASE_PATH / "../results/ius_eval").resolve())
    output_folder = os.path.join(output_folder, experiment_folder_name)
    create_output_folders(output_folder)        # skipped if exists

    ius_score_pd = pd.DataFrame(
        {
            "filename": synthetic_filenames,
            "ius_measure_score":  ius_score_list
        }
    )
    file_saved_ius_scores = os.path.join(output_folder, f"ius_scores_{timestamp}.csv")
    ius_score_pd.to_csv(file_saved_ius_scores, index=False)

    save_info_dict = {
        "epu_ckpt": trained_ckpt,
        "config": cfg_path,
        "experiment_id": cfg.experiment_saved_folder_name,
        "cb_vector_path": cb_vec_path,
        "synthetic_images_path": synthetic_img_path,
        "synthetic_img_extension": synthetic_img_extension,
        "synthetic_samples_num": len(synthetic_filenames),
        "timestamp": timestamp,
        "ius_scores": file_saved_ius_scores,
    }
    json_file_path = os.path.join(output_folder, f"ius_scores_info_{timestamp}.json")
    save_to_json(save_info_dict, json_file_path)


if __name__ == "__main__":
    # python - m scripts.eval_ius - -experiment_folder_name ius_dataset_name_base_one_0000_timestamp - -cb_vector_tag normal - -synthetic_images datasets_synthetic/dataset_name/normal - -synthetic_img_extension png
    # python - m scripts.eval_ius - -experiment_folder_name ius_dataset_name_base_one_0000_timestamp - -cb_vector_tag normal - -synthetic_images datasets_synthetic/dataset_name/normal/image_0001.png - -synthetic_img_extension png

    main()
