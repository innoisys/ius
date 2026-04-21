from utils.omega_parser import EPUCNNParams
from utils.omega_parser import DataParams, DataPreprocessing


def model_cfg_to_epucnn(model_cfg: EPUCNNParams):
    return {
        "num_classes": model_cfg.num_classes,
        "subnetwork_name": model_cfg.subnetwork_config.architecture,
        "num_subnetworks": model_cfg.num_subnetworks,
        "epu_activation":  model_cfg.epu_activation,
        "subnet_activation": model_cfg.subnetwork_config.pred_activation,
        "subnet_cfg": {
            "input_channels": model_cfg.subnetwork_config.input_channels,
            "base_channels": model_cfg.subnetwork_config.base_channels,
            "fc_units": model_cfg.subnetwork_config.fc_hidden_units,
        }
    }


def data_cfg_to_dataparser(dataset_path: str,
                           images_extension: str,
                           data_mode: str,
                           preprocessing_cfg: DataPreprocessing,
                           group_by: str = None,):

    group_by_key = None
    group_by_value = None
    if group_by is not None:
        if preprocessing_cfg.data_parser in ["filename", "folder"]:
            group_by_key = group_by                                             # eg "normal"
        elif preprocessing_cfg.data_parser in ["medmnist"]:
            group_by_value = preprocessing_cfg.label_mapping.get(group_by)      # eg "0" from item ("normal": "0")

    return {
        "dataset_folder": dataset_path,
        "mode": data_mode,
        "image_ext": images_extension,
        "label_mapping": preprocessing_cfg.label_mapping,
        "csv_file": preprocessing_cfg.medmnist_csv_file,
        "group_by_key": group_by_key,
        "group_by_value": group_by_value
    }


# def data_cfg_to_dataparser(data_cfg: DataParams, data_mode: str = 'train'):
#     return {
#         "dataset_folder": data_cfg.dataset_path,
#         "mode": data_mode,
#         "image_ext": data_cfg.images_extension,
#         "kwargs": {
#             "label_mapping": data_cfg.data_preprocessing.label_mapping,
#             "csv_file": data_cfg.data_preprocessing.medmnist_csv_file
#         }
#     }
