import torch
from torch import nn, Tensor


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
            self.norms.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

    def reset_parameters(self):
        self.convs.apply(_init_weights)
        self.norms.apply(_init_weights)

    def forward(self, inputs: Tensor):
        out = inputs
        for idx in range(len(self.convs)):
            out = self.convs[idx](out)
            out = self.norms[idx](out)
            out = self.activation(out)
            out = self.dropout(out)

        return out


class BlockMatmulConv(nn.Module):
    def __init__(self, channels, mlp_depth, drop_prob=0.0) -> None:
        super().__init__()
        self.mlp1 = BlockMLP(channels, channels, mlp_depth, drop_prob)
        self.mlp2 = BlockMLP(channels, channels, mlp_depth, drop_prob)
        # self.scale_weight = nn.Parameter(torch.zeros(1, channels, 1, 1, 2))

    def reset_parameters(self):
        self.mlp1.apply(_init_weights)
        self.mlp2.apply(_init_weights)
        # nn.init.kaiming_normal_(self.scale_weight)

    def forward(self, x, log_degrees):  # x: B, H, N, N
        mlp1 = self.mlp1(x)
        mlp2 = self.mlp2(x)
        # x = torch.matmul(mlp1, mlp2) / x.shape[-1]
        x = torch.matmul(mlp1, mlp2)
        x = torch.sqrt(torch.relu(x))

        # x = torch.stack([x, x * log_degrees], dim=-1)
        # x = (x * self.scale_weight).sum(dim=-1)
        return x


class BlockUpdateLayer(nn.Module):
    def __init__(self, channels, mlp_depth, drop_prob) -> None:
        super().__init__()
        self.matmul_conv = BlockMatmulConv(channels, mlp_depth, drop_prob)
        # self.skip = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.norm = nn.BatchNorm2d(channels)
        self.update = BlockMLP(channels, 2, drop_prob)

    def reset_parameters(self):
        self.matmul_conv.apply(_init_weights)
        # self.skip.apply(_init_weights)
        self.norm.apply(_init_weights)
        self.update.apply(_init_weights)

    def forward(self, inputs, log_degrees):
        h = self.matmul_conv(inputs, log_degrees)
        h = self.norm(h)
        h = torch.cat((inputs, h), 1)
        h = self.update(h) + inputs
        return h
