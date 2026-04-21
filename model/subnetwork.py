import torch
import torch.nn as nn

from .subnetwork_utils import BaseBlockConvBN, TopHead
from .register_modules import register_model


class BaseSubNetwork(nn.Module):
    def __init__(self,
                 input_channels: int,
                 base_channels: int,
                 fc_hidden_units: int,
                 fc_pred_units: int,
                 pred_activation: str,
                 ):
        super(BaseSubNetwork, self).__init__()

        self.input_channels = input_channels
        self.base_channels = base_channels
        self.fc_hidden_units = fc_hidden_units
        self.fc_pred_units = fc_pred_units
        self.pred_activation = pred_activation

        self.intermediate_features = None

    def get_intermediate_features(self) -> torch.Tensor:
        return self.intermediate_features


@register_model("base_one")
class Subnet(BaseSubNetwork):
    def __init__(self, input_channels=3, base_channels=32, fc_hidden_units=64, fc_pred_units=1, pred_activation="sigmoid"):

        super(Subnet, self).__init__(
            input_channels=input_channels,
            base_channels=base_channels,
            fc_hidden_units=fc_hidden_units,
            fc_pred_units=fc_pred_units,
            pred_activation=pred_activation,
        )

        self.block_one = BaseBlockConvBN(in_ch=input_channels,
                                         out_ch=base_channels,
                                         conv_layers=2,
                                         kernel_size=(3, 3),
                                         stride=(2, 2),
                                         padding=(1, 1),
                                         activation="relu",
                                         normalization=True,)

        self.block_two = BaseBlockConvBN(in_ch=base_channels,
                                         out_ch=base_channels*2,
                                         conv_layers=2,
                                         kernel_size=(3, 3),
                                         stride=(2, 2),
                                         padding=(1, 1),
                                         activation="relu",
                                         normalization=True,)
        
        self.block_three = BaseBlockConvBN(in_ch=base_channels*2,
                                           out_ch=base_channels*4,
                                           conv_layers=3,
                                           kernel_size=(3, 3),
                                           stride=(2, 2),
                                           padding=(1, 1),
                                           activation="relu",
                                           normalization=True,)
        self.flatten = nn.Flatten()

        self.head = TopHead(fc_units=fc_hidden_units,
                            num_classes=fc_pred_units,
                            hidden_layers=1,
                            dropout_rate=0.6,
                            fc_activation="relu",
                            pred_activation=pred_activation)

        self.intermediate_features = None

    def forward(self, x):
        x = self.block_one(x)
        x = self.block_two(x)
        self.intermediate_features = self.block_two.get_block_feats()
        x = self.block_three(x)
        x = self.flatten(x)
        x = self.head(x)
        return x
