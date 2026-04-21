import torch.nn as nn


def layer_mapping(layer: str) -> nn.Module:
    layer = layer.lower()

    mappings = {
        # Normalization
        "batch": nn.BatchNorm2d,
        "instance": nn.InstanceNorm2d,
        "layer": lambda c: nn.GroupNorm(1, c),  # nn.LayerNorm,

        # Identity
        "none": nn.Identity,
        "linear": nn.Identity,

        # Activations
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "softmax": lambda: nn.Softmax(dim=1),
    }
    try:
        return mappings[layer]
    except KeyError as e:
        available = list(mappings.keys())
        raise ValueError(
            f"{layer} not found in the existing mapping."
            f"Existing available: {available}"
            f"Update model.model_utils.layer_mapping()"
        ) from e
