"""
Different output decoders for different datasets/tasks.
"""

import torch.nn as nn
from torch import Tensor
from .basic import Linear


def _weights_init(m: nn.Module):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()


class GraphClassification(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(GraphClassification, self).__init__()
        self.classifier = nn.Sequential(
            Linear(in_channels, in_channels // 2), nn.GELU(),
            Linear(in_channels // 2, out_channels)
        )

    def reset_parameters(self):
        self.classifier.apply(_weights_init)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)


class GraphRegression(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(GraphRegression, self).__init__()
        self.regressor = nn.Sequential(
            Linear(in_channels, in_channels // 2), nn.GELU(),
            Linear(in_channels // 2, in_channels // 4), nn.GELU(),
            Linear(in_channels // 4, out_channels)
        )

    def reset_parameters(self):
        self.regressor.apply(_weights_init)

    def forward(self, x: Tensor) -> Tensor:
        return self.regressor(x)


class NodeClassification(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(NodeClassification, self).__init__()
        self.classifier = nn.Sequential(
            Linear(in_channels, in_channels // 2), nn.GELU(),
            Linear(in_channels // 2, out_channels)
        )

    def reset_parameters(self):
        self.classifier.apply(_weights_init)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)
