import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import List, Tuple, Union

from .subnetwork import Subnet
from .module_mapping import layer_mapping
from .register_modules import get_registered_model


class BaseAdditiveNetwork(nn.Module):
    def __init__(self,
                 subnetworks: nn.ModuleList,
                 num_classes: int,
                 activation: nn.Module
                 ):
        super(BaseAdditiveNetwork, self).__init__()

        self.subnetworks = subnetworks
        self.n_classes = num_classes
        self.activation = activation

        self.bias = nn.Parameter(torch.randn(self.n_classes), requires_grad=True)

        self.interpretations = None
        self.output = None

    def forward(self, pfms: Union[Tuple[torch.Tensor], torch.Tensor], ret_raw_logits=True) -> torch.Tensor:
        if isinstance(pfms, torch.Tensor) and pfms.ndim == 5:
            pfms = torch.unbind(pfms, dim=1)
        self.interpretations = [subnetwork(_x) for _x, subnetwork in zip(pfms, self.subnetworks)]
        output = torch.sum(torch.stack(self.interpretations), dim=0) + self.bias
        if not ret_raw_logits:                          # in case that ret_raw_logits = False
            output = self.activation(output)            # [bs, num_classes]
        self.output = output
        return output

    def get_interpretations(self) -> List[torch.Tensor]:
        return self.interpretations

    def get_outputs(self) -> torch.Tensor:
        return self.output

    def get_bias(self) -> torch.Tensor:
        return self.bias

    
class EPUCNN(BaseAdditiveNetwork):
    def __init__(self,
                 num_classes: int,
                 subnetwork_name: str,
                 num_subnetworks: int = 4,
                 epu_activation: str = "sigmoid",
                 subnet_activation: str = "tanh",
                 **kwargs,
                 ):

        subnet_cfg = kwargs.pop("subnet_cfg", None)
        default_cfg = {
            "input_channels": 3,
            "base_channels": 64,
            "fc_units": 32
        }
        if subnet_cfg is None:
            subnet_cfg = default_cfg.copy()
        else:
            _unknown_key = set(subnet_cfg.keys()) - set(default_cfg.keys())
            if _unknown_key:
                raise ValueError(
                    f"Unused subnet_cfg key: {list(_unknown_key)}"
                    f"Expected keys: {list(default_cfg.keys())}"
                )
            subnet_cfg = {**default_cfg, **subnet_cfg}

        subnet_class = get_registered_model(subnetwork_name)
        subnets = nn.ModuleList(
            [subnet_class(
                input_channels=subnet_cfg["input_channels"],
                base_channels=subnet_cfg["base_channels"],
                fc_hidden_units=subnet_cfg["fc_units"],
                fc_pred_units=num_classes,
                pred_activation=subnet_activation,
            ) for _ in range(num_subnetworks)]
        )
        epu_activation = layer_mapping(epu_activation)()

        super(EPUCNN, self).__init__(
            subnetworks=subnets,
            num_classes=num_classes,
            activation=epu_activation,
        )

        self.subnetworks = subnets
        self.num_classes = num_classes
        self.subnet_name = subnetwork_name
        self.epu_activation = epu_activation
        self.num_subnetworks = num_subnetworks
        self.subnet_activation = subnet_activation

    def create_baseline_feature_contribution_profile(self, data_loader: DataLoader, device: str) -> torch.Tensor:
        # [dataloader_items_unbatched, num_subnets, num_classes]
        profiles = self.calculate_feature_contribution_profiles(data_loader=data_loader, device=device)
        return torch.mean(profiles, dim=0)                           # [num_subnets, num_classes]

    def calculate_feature_contribution_profiles(self, data_loader: DataLoader, device: str) -> torch.Tensor:
        self.eval()
        profiles = []
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(device)
                x_profile = self.feature_contribution_profile(x)    # [batch, num_subnets, num_classes]
                profiles.append(x_profile)
        profiles = torch.cat(profiles, dim=0)                   # [dataloader_items_unbatched, num_subnets, num_classes]
        return profiles

    def feature_contribution_profile(self, pfm_tuple: Tuple[torch.Tensor]) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            _ = self.forward(pfm_tuple)
            profile = self.get_interpretations()        # List of 4 (num_subnets) items each [batch, subnet_pred_units]
            # profile = torch.stack(profile, dim=0)              # [num_subnets, batch, num_classes]
            # profile = torch.permute(profile, (1, 0, 2))        # [batch, num_subnets, num_classes]
            profile = torch.stack(profile, dim=1)                # [batch, num_subnets, num_classes]
            return profile


if __name__ == '__main__':
    cfg = {
        "input_channels": 3,
        "base_channels": 64,
        "fc_units": 32
    }
    epu = EPUCNN(
        num_classes=1,
        subnetwork_name="base_one",
        num_subnetworks=4,
        subnet_activation="sigmoid",
        epu_activation="tanh",
        subnet_cfg=cfg
    )
    x = torch.randn(16, 4, 3, 32, 32)                   # [batch, num_subnets, ch, h, w]
    y = epu(x)
    interpretation_vec = epu.get_interpretations()
    bias = epu.get_bias()

    vector_c = epu.feature_contribution_profile(x)
