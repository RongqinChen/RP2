from typing import List
import torch
from torch import nn, Tensor
from .padded_block_conv import PaddedBlockMatmulConv


class SeperatedBlockConv(nn.Module):
    def __init__(self, channels, mlp_depth, drop_prob) -> None:
        super().__init__()
        self.matmul_conv = PaddedBlockMatmulConv(channels, mlp_depth, drop_prob)
        self.skip = nn.Linear(channels, channels)
        self.update = nn.Sequential(
            nn.Linear(channels, channels), nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(channels, channels), nn.BatchNorm1d(channels), nn.ReLU(),
        )

    def forward(self, x: Tensor, len1d, size3d):
        xs = torch.split(x, len1d, 0)
        bs = [x.reshape(s + (-1,)) for x, s in zip(xs, size3d)]
        bs = [b.permute((0, 3, 1, 2)) for b in bs]

        hs: List[Tensor] = [self.matmul_conv(b) for b in bs]
        hs = [h.permute((0, 2, 3, 1)).flatten(0, 2) for h in hs]

        hs2 = torch.cat(hs).contiguous() + self.skip(x)
        h = self.update(hs2) + hs2
        return h
