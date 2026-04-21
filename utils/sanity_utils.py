from .omega_parser import IUSConfig


class SanityChecker:
    VALID_MODELS = ["base_one"]
    VALID_DATA_MODES = ["grayscale", "rgb"]
    VALID_DATA_PARSERS = ["filename", "folder", "medmnist"]

    def __init__(self, cfg: IUSConfig):
        self.cfg = cfg

    def model_cfg(self) -> None:
        architecture = self.cfg.model.subnetwork_config.architecture

        if architecture not in self.VALID_MODELS:
            raise ValueError(
                f"EPU-CNN backbone implemented {self.VALID_MODELS}. Original IUS paper was using 'base_one'"
            )

        num_classes = self.cfg.model.num_classes
        epu_activation = self.cfg.model.epu_activation

        if epu_activation == "sigmoid" and num_classes > 1:
            raise ValueError(
                f"epu_activation = {epu_activation} and num_classes = {num_classes}. "
                f"For sigmoid epu_activation you have to set num_classes=1"
            )

        num_subnets = self.cfg.model.num_subnetworks
        if num_subnets != 4:
            raise ValueError(
                "IUS measure was implemented using 4 PFMS (either for grayscale or rgb modalities)"
                "class PerceptualFeatureMapTransform in data.perceptual_transforms yields 4 PFM representations. "
            )

    def train_cfg(self) -> None:
        mode = self.cfg.train_params.mode
        loss = self.cfg.train_params.loss

        if mode == "binary" and loss != "binary_cross_entropy":
            raise ValueError(
                f"In train_params: mode={mode}, loss={loss}. For mode = 'binary' set loss = 'binary_cross_entropy'"
            )

    def data_preprocessing(self) -> None:
        data_mode = self.cfg.data_params.data_preprocessing.data_mode
        channels = self.cfg.model.subnetwork_config.input_channels

        if data_mode not in self.VALID_DATA_MODES:
            raise ValueError(
                f"data_mode should be one of {self.VALID_DATA_MODES}. "
            )

        if channels != 1:
            raise ValueError(
                f"In train_params:input_channels={channels} But Perceptual Feature decomposition yields PFMs "
                f"with 1 output channel. Please set input_channels=1"
            )

        num_classes = self.cfg.model.num_classes
        labels = set(self.cfg.data_params.data_preprocessing.label_mapping.values())

        if num_classes > 1 and len(labels) != num_classes:
            raise ValueError(
                f"In train_params: num_classes={num_classes}, in label mapping found {labels}. "
            )
        if num_classes == 1 and len(labels) != 2:
            raise ValueError(
                f"In train_params: num_classes={num_classes}, in label mapping found {labels}. "
            )

        data_parser = self.cfg.data_params.data_preprocessing.data_parser
        if data_parser not in self.VALID_DATA_PARSERS:
            raise ValueError(
                f"data_parser should be one of {self.VALID_DATA_PARSERS}. "
                f"Otherwise implement your own data_parser in data.parsers and update utils.sanity_utils"
            )

    def sanity_check(self) -> None:
        self.model_cfg()
        self.train_cfg()
        self.data_preprocessing()
        print("All sanity checks passed!")


if __name__ == "__main__":
    from .omega_parser import IUSConfig
    cfg_file = "configs/train_config.yaml"
    cfg = IUSConfig.from_yaml(cfg_file)

    # print(cfg.model)
    # print(cfg.train_params)
    # print(cfg.data_params)

    checker = SanityChecker(cfg)
    checker.sanity_check()