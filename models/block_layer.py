import torch
from torch import nn, Tensor


def _init_weights(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
    elif isinstance(m, nn.Parameter):
        nn.init.kaiming_normal_(m.data)
    elif hasattr(m, "reset_parameters"):
        m.reset_parameters()


class BlockMLP(nn.Module):

    def __init__(self, in_channels, out_channels, mlp_depth, drop_prob=0.0):
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
        self.norms = nn.ModuleList()
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

        return out


class BlockMatmulConv(nn.Module):
    def __init__(self, channels, mlp_depth, drop_prob=0.0) -> None:
        super().__init__()
        self.mlp1 = BlockMLP(channels, channels, mlp_depth, drop_prob)
        self.mlp2 = BlockMLP(channels, channels, mlp_depth, drop_prob)
        # self.scale_weight = nn.Parameter(torch.zeros(1, channels, 1, 1, 2))

    def reset_parameters(self):
        # def _init_conv2d(m: nn.Module):
        #     nn.init.xavier_normal_(m.weight)
        #     if m.bias is not None:
        #         nn.init.zeros_(m.bias)
        self.mlp1.apply(_init_weights)
        self.mlp2.apply(_init_weights)

    def forward(self, x, log_deg):  # x: B, H, N, N
        mlp1 = self.mlp1(x)
        mlp2 = self.mlp2(x)
        x = torch.matmul(mlp1, mlp2) / x.shape[-1]
        x = torch.sqrt(torch.relu(x))
        # x = torch.stack([x, x * log_deg], dim=-1)
        # x = (x * self.scale_weight).sum(dim=-1)
        return x


class BlockUpdateLayer(nn.Module):
    def __init__(self, channels, mlp_depth, drop_prob) -> None:
        super().__init__()
        self.matmul_conv = BlockMatmulConv(channels, mlp_depth, drop_prob)
        self.skip = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.update = BlockMLP(channels, channels, 2, drop_prob)

    def reset_parameters(self):
        self.matmul_conv.apply(_init_weights)
        self.skip.apply(_init_weights)
        self.update.apply(_init_weights)

    def forward(self, x, log_deg):
        h = self.matmul_conv(x, log_deg) + self.skip(x)
        h = self.update(h) + h
        return h
