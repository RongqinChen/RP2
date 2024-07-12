from typing import List
import torch
from torch import Tensor


def padded_block_avg_pool(inputs: Tensor) -> Tensor:
    # inputs.shape: B, H, N, N
    N = inputs.shape[-1]
    diag_sum = torch.sum(torch.diagonal(inputs, dim1=-2, dim2=-1), dim=2)  # B, H
    diag_avg = diag_sum / N
    offdiag_avg = (torch.sum(inputs, dim=[-1, -2]) - diag_sum) / (N * N - N)
    outputs = torch.cat((diag_avg, offdiag_avg), dim=1)  # B, 2H
    return outputs


def padded_block_sum_pool(inputs: Tensor) -> Tensor:
    # inputs.shape: B, H, N, N
    diag_sum = torch.sum(torch.diagonal(inputs, dim1=-2, dim2=-1), dim=2)  # B, H
    offdiag_sum = (torch.sum(inputs, dim=[-1, -2]) - diag_sum)
    outputs = torch.cat((diag_sum, offdiag_sum), dim=1)  # B, 2H
    return outputs


def seperated_block_avg_pool(inputs: List[Tensor]) -> List[Tensor]:
    outputs = [padded_block_avg_pool(ele) for ele in inputs]
    return outputs


def seperated_block_sum_pool(inputs: List[Tensor]) -> List[Tensor]:
    outputs = [padded_block_sum_pool(ele) for ele in inputs]
    return outputs
