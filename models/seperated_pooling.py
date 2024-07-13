from typing import List
import torch
from torch import Tensor


def block_avg_pool(inputs: Tensor) -> Tensor:
    # inputs.shape: B, N, N, H
    N = inputs.shape[-1]
    diag_sum = torch.sum(torch.diagonal(inputs, dim1=-3, dim2=-2), dim=2)  # B, H
    diag_avg = diag_sum / N
    offdiag_avg = (torch.sum(inputs, dim=[-2, -3]) - diag_sum) / (N * N - N)
    outputs = torch.cat((diag_avg, offdiag_avg), dim=1)  # B, 2H
    return outputs


def block_sum_pool(inputs: Tensor) -> Tensor:
    # inputs.shape: B, N, N, H
    diag_sum = torch.sum(torch.diagonal(inputs, dim1=-3, dim2=-2), dim=2)  # B, H
    offdiag_sum = (torch.sum(inputs, dim=[-2, -3]) - diag_sum)
    outputs = torch.cat((diag_sum, offdiag_sum), dim=1)  # B, 2H
    return outputs


def seperated_graph_avg_pool(inputs: List[Tensor]) -> List[Tensor]:
    outputs = [block_avg_pool(ele) for ele in inputs]
    outputs = torch.cat(outputs, 0)
    return outputs


def seperated_graph_sum_pool(inputs: List[Tensor]) -> List[Tensor]:
    outputs = [block_sum_pool(ele) for ele in inputs]
    outputs = torch.cat(outputs, 0)
    return outputs
