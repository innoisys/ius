import torch
import torch.nn as nn

from typing import Tuple, Optional

from model.module_mapping import layer_mapping


class BaseBlockConvBN(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 stride: Tuple[int, int] = (2, 2),
                 padding: Tuple[int, int] = (1, 1),
                 conv_layers: int = 2,
                 pool_layer: bool = True,
                 normalization: bool = True,
                 activation: Optional[str] = "linear",
                 # return_feats: bool = False,
                 ):
        super(BaseBlockConvBN, self).__init__()

        self.conv_layers = torch.nn.ModuleList()
        # Same params for all convolutional layers in ModuleList
        #      activation, out_channels, kh, kw, stride, padding
        for _ in range(conv_layers):
            self.conv_layers.append(
                nn.Conv2d(in_channels=in_ch,
                          out_channels=out_ch,
                          kernel_size=kernel_size,
                          stride=(1, 1),
                          padding=padding,
                          bias=True)
            )
            in_ch = out_ch
        self._activation = layer_mapping(activation.lower())()

        self._batch_norm = nn.BatchNorm2d(out_ch) if normalization else None
        self._max_pooling = nn.MaxPool2d(kernel_size=(2, 2), stride=stride) if pool_layer else None

        self._intermediate_feats = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.conv_layers:
            x = conv(x)
            x = self._activation(x)
        self._intermediate_feats = x

        if self._max_pooling is not None:
            x = self._max_pooling(x)

        if self._batch_norm is not None:
            x = self._batch_norm(x)

        return x

    def get_block_feats(self) -> torch.Tensor:
        return self._intermediate_feats


class TopHead(nn.Module):
    def __init__(self,
                 # in_feats: int,
                 fc_units: int = 64,
                 num_classes: int = 1,
                 hidden_layers: int = 1,
                 fc_activation: Optional[str] = "relu",
                 pred_activation: Optional[str] = "sigmoid",
                 dropout_rate: Optional[float] = None,
                 ):
        super(TopHead, self).__init__()

        self._dense_layers = nn.ModuleList()

        for i in range(hidden_layers + 1):
            if i == 0:
                self._dense_layers.append(
                    nn.LazyLinear(out_features=fc_units, bias=True)
                )
            else:
                self._dense_layers.append(
                    nn.Linear(in_features=fc_units, out_features=fc_units, bias=True)
                )
            self._dense_layers.append(
                layer_mapping(fc_activation.lower())()
            )
            if dropout_rate is not None:
                self._dense_layers.append(
                    nn.Dropout(dropout_rate)
                )

        # prediction layer
        self._dense_layers.append(
            nn.Linear(in_features=fc_units, out_features=num_classes, bias=True),
        )
        self._dense_layers.append(
            layer_mapping(pred_activation.lower())()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._dense_layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    block = BaseBlockConvBN(in_ch=3, out_ch=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), normalization=True,
                            activation="relu", conv_layers=2)
    dummy_input = torch.randn((1, 3, 32, 32))
    out = block(dummy_input)
