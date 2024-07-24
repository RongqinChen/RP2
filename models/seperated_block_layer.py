from typing import List
import torch
from torch import nn, Tensor
# from .block_layer import BlockMatmulConv
from .basic import Linear


def _init_weights(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
    elif hasattr(m, "reset_parameters"):
        m.reset_parameters()


class BlockMLP(nn.Module):

    def __init__(self, in_channels, out_channels, mlp_depth, drop_prob=0.0):
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(mlp_depth):
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True))
            # self.norms.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

    def reset_parameters(self):
        self.convs.apply(_init_weights)
        # self.norms.apply(_init_weights)

    def forward(self, inputs: Tensor):
        out = inputs
        for idx in range(len(self.convs)):
            out = self.convs[idx](out)
            # if out.shape[0] == out.shape[2] == out.shape[3] == 1:
            #     pass
            # else:
            #     out = self.norms[idx](out)
            out = self.activation(out)

        return out


class BlockMatmulConv(nn.Module):
    def __init__(self, channels, mlp_depth, drop_prob=0.0) -> None:
        super().__init__()
        self.mlp1 = BlockMLP(channels, channels, mlp_depth, drop_prob)
        self.mlp2 = BlockMLP(channels, channels, mlp_depth, drop_prob)

    def reset_parameters(self):
        self.mlp1.apply(_init_weights)
        self.mlp2.apply(_init_weights)

    def forward(self, x):  # x: B, H, N, N
        mlp1 = self.mlp1(x)
        mlp2 = self.mlp2(x)
        x = torch.matmul(mlp1, mlp2)
        x = torch.sqrt(torch.relu(x))
        return x


class SeperatedBlockUpdateLayer(nn.Module):
    def __init__(self, channels, mlp_depth, drop_prob) -> None:
        super().__init__()
        self.matmul_conv = BlockMatmulConv(channels, mlp_depth, drop_prob)
        self.update = nn.Sequential(
            Linear(2 * channels, channels, True, bias_initializer='zeros'),
            nn.BatchNorm1d(channels), nn.ReLU(),
            Linear(channels, channels, True, bias_initializer='zeros'),
            nn.BatchNorm1d(channels), nn.ReLU(),
        )

    def reset_parameters(self):
        self.matmul_conv.apply(_init_weights)
        self.update.apply(_init_weights)

    def forward(self, x: Tensor, len1d, size3d):
        xs = torch.split(x, len1d, 0)
        bs = [x.reshape(s + (-1,)) for x, s in zip(xs, size3d)]
        bs = [b.permute((0, 3, 1, 2)).contiguous() for b in bs]

        hs: List[Tensor] = [self.matmul_conv(b) for b in bs]
        hs = [h.permute((0, 2, 3, 1)).contiguous().view((-1, x.shape[-1])) for h in hs]
        h = torch.cat(hs).contiguous()

        h = torch.cat((x, h), 1)
        h = self.update(h) + x
        return h
