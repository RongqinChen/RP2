from typing import List
import torch
from torch import nn, Tensor
from .block_layer import BlockMatmulConv, BlockMLP


def _init_weights(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
    elif hasattr(m, "reset_parameters"):
        m.reset_parameters()


class SeperatedBlockUpdateLayer(nn.Module):
    def __init__(self, channels, mlp_depth, drop_prob) -> None:
        super().__init__()
        self.matmul_conv = BlockMatmulConv(channels, mlp_depth, drop_prob)
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.ReLU()
        self.update = BlockMLP(2 * channels, channels, 2, drop_prob)

    def reset_parameters(self):
        self.matmul_conv.apply(_init_weights)
        self.norm.apply(_init_weights)
        self.update.apply(_init_weights)

    def forward(self, x: Tensor, len1d, size3d):
        xs = torch.split(x, len1d, 0)
        bs = [x.reshape(s + (-1,)) for x, s in zip(xs, size3d)]
        bs = [b.permute((0, 3, 1, 2)) for b in bs]

        hs: List[Tensor] = [self.matmul_conv(b) for b in bs]
        hs = [h.permute((0, 2, 3, 1)).flatten(0, 2) for h in hs]
        h = torch.cat(hs).contiguous()

        h = self.activation(self.norm(h))
        h = torch.cat((x, h), 1)
        h = self.update(h) + x
        return h
