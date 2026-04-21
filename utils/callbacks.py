from typing import List

from utils.early_stopping import EarlyStoppingCallback
from utils.tensorboard import TensorboardLoggerCallback


def setup_callbacks(ckpt_path: str,
                    log_dir: str,
                    early_patience: int,
                    early_monitor: str = "val_loss",
                    early_mode: str = "min",
                    use_tensorboard: bool = False,
                    **kwargs) -> List[object]:

    delta = kwargs.get("delta", 0)
    verbose = kwargs.get("verbose", True)
    restore_best_weights = kwargs.get("restore_best_weights", False)
    save_final_model = kwargs.get("save_final_model", True)

    es_call = EarlyStoppingCallback(patience=early_patience,
                                    delta=delta,
                                    checkpoint_path=ckpt_path,
                                    verbose=verbose,
                                    restore_best_weights=restore_best_weights,
                                    monitor=early_monitor,
                                    mode=early_mode,
                                    save_final_model=save_final_model
                                    )

    log_histograms = kwargs.get("log_histograms", False)
    log_images_every = kwargs.get("log_images_every", 10)
    tb_port = kwargs.get("tb_port", 6006)
    tb_browser = kwargs.get("tb_browser", False)

    tb_logger = TensorboardLoggerCallback(log_dir=log_dir,
                                          log_histograms=log_histograms,
                                          launch_tb=use_tensorboard,
                                          tb_port=tb_port,
                                          open_tb_in_browser=tb_browser
                                          )
    return [es_call, tb_logger]