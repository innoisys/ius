import torch.nn as nn

REGISTERED_MODELS = {}


def register_model(name: str):
    def decorator(model_class: nn.Module) -> nn.Module:
        key = name.lower()
        if key in REGISTERED_MODELS:
            raise ValueError(
                f'Model {name} already registered'
            )
        REGISTERED_MODELS[key] = model_class
        return model_class
    return decorator


def get_registered_model(name: str) -> nn.Module:
    key = name.lower()
    if key not in REGISTERED_MODELS:
        raise ValueError(
            f'Unknown model name: {name}. '
            f'Available models: {list(REGISTERED_MODELS.keys())}'
        )
    return REGISTERED_MODELS[key]
