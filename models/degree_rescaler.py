import torch
from torch import nn, Tensor


class DegreeRescaler(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.scale_weight = nn.Parameter(torch.zeros(1, channels, 2))

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.scale_weight)

    def forward(self, x: Tensor, log_degrees: Tensor):
        x = torch.stack([x, x * log_degrees], dim=-1)
        x = (x * self.scale_weight).sum(dim=-1)
        return x
