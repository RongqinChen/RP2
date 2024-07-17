import torch
from torch import Tensor


def block_avg_pooling(inputs: Tensor) -> Tensor:
    # inputs.shape: B, H, N, N
    N = inputs.shape[-1]
    diag_sum = torch.sum(torch.diagonal(inputs, dim1=-2, dim2=-1), dim=2)  # B, H
    diag_avg = diag_sum / N
    if N == 1:
        offdiag_avg = torch.zeros_like(diag_avg)
    else:
        offdiag_avg = (torch.sum(inputs, dim=[-1, -2]) - diag_sum) / (N * N - N)
    outputs = torch.cat((diag_avg, offdiag_avg), dim=1)  # B, 2H
    return outputs


def block_sum_pooling(inputs: Tensor) -> Tensor:
    # inputs.shape: B, H, N, N
    diag_sum = torch.sum(torch.diagonal(inputs, dim1=-2, dim2=-1), dim=2)  # B, H
    offdiag_sum = (torch.sum(inputs, dim=[-1, -2]) - diag_sum)
    outputs = torch.cat((diag_sum, offdiag_sum), dim=1)  # B, 2H
    return outputs
