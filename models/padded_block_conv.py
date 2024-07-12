import torch
from torch import nn, Tensor


class MlpBlock(nn.Module):
    """
    Block of MLP layers with activation function after each (1x1 conv2d layers).
    """

    def __init__(self, channels, mlp_depth, activation_fn=nn.functional.relu):
        super().__init__()
        self.activation = activation_fn
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(mlp_depth):
            self.convs.append(nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True))
            self.norms.append(nn.BatchNorm2d(channels))
            _init_weights(self.convs[-1])
            self.norms[-1].reset_parameters()

    def forward(self, inputs):
        out = inputs
        for idx in range(len(self.convs)):
            out = self.convs[idx](out)
            out = self.norms[idx](out)
            out = self.activation(out)

        return out


class PaddedBlockMatmulConv(nn.Module):
    def __init__(self, channels, mlp_depth) -> None:
        super().__init__()
        self.mlp1 = MlpBlock(channels, mlp_depth)
        self.mlp2 = MlpBlock(channels, mlp_depth)

    def forward(self, inputs) -> Tensor:  # inputs: B, H, N, N
        mlp1 = self.mlp1(inputs)
        mlp2 = self.mlp2(inputs)
        mult = torch.matmul(mlp1, mlp2) / inputs.shape[-1]
        # mult = torch.sqrt(torch.relu(mult)) - torch.sqrt(torch.relu(-mult))
        mult = torch.sqrt(torch.relu(mult))
        return mult


class PaddedBlockConv(nn.Module):
    def __init__(self, channels, mlp_depth) -> None:
        super().__init__()
        self.matmul_conv = PaddedBlockMatmulConv(channels, mlp_depth)
        self.skip = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True)
        self.update = MlpBlock(channels, 2)

    def forward(self, inputs):
        h = self.matmul_conv(inputs) + self.skip(inputs)
        h = self.update(h) + h
        return h


def _init_weights(layer):
    """
    Init weights of the layer
    :param layer:
    :return:
    """
    nn.init.xavier_uniform_(layer.weight)
    # nn.init.xavier_normal_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
