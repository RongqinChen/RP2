"""
Multi layer perceptron.
"""

from typing import Optional

from torch import nn, Tensor
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.norm import BatchNorm, LayerNorm, InstanceNorm, GraphSizeNorm, PairNorm


def get_norm(norm_type: str, channels: int):
    norm_type = norm_type.lower()
    if norm_type == "batch":
        return BatchNorm(channels)
    if norm_type == "layer":
        return LayerNorm(channels)
    if norm_type == "instance":
        return InstanceNorm(channels)
    if norm_type == "graphsize":
        return GraphSizeNorm()
    if norm_type == "pair":
        return PairNorm()

    raise NotImplementedError()


def get_activation(act_type: str):
    act_type = act_type.lower()
    if act_type == "relu":
        return nn.ReLU()
    if act_type == "leakyrelu":
        return nn.LeakyReLU()
    if act_type == "prelu":
        return nn.PReLU()

    raise NotImplementedError()


class JK_MLP(nn.Module):
    r"""Multi-layer perceptron.
    Args:
        in_channels (int): Input feature size.
        out_channels (int): Output feature size.
        norm_type (str, optional): Method of normalization, choose from (Batch, Layer, Instance, GraphSize, Pair).
    """

    def __init__(
        self, in_channels: int, out_channels: int, drop_prob: Optional[float] = 0.,
        norm_type: Optional[str] = "Batch", act_type: Optional[str] = "ReLU",
    ):

        super(JK_MLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_type = norm_type
        self.linear1 = Linear(self.in_channels, self.out_channels)
        self.linear2 = Linear(self.out_channels, self.out_channels)
        self.norm1 = get_norm(norm_type, out_channels)
        self.norm2 = get_norm(norm_type, out_channels)
        self.activation = get_activation(act_type)
        self.dropout = nn.Dropout(drop_prob)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
