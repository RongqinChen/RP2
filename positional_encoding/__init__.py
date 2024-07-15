import torch
from torch_geometric.data import Data

from .adj_powers import compute_adjacency_power_series
from .rrwp import compute_rrwp
from .bernstein import compute_bernstein_polynomials
from .bern_mixed_smooth import compute_bern_mixed_smooth
from .bern_mixed_sym3 import compute_bern_mixed_sym3
from .bern_mixed_smooth import compute_bern_mixed_smooth


pe_computer_dict = {
    "adj_powers": compute_adjacency_power_series,       # output_len: K+1
    "rrwp": compute_rrwp,                               # output_len: K+1
    "bernstein": compute_bernstein_polynomials,         # output_len: K+2
    "bern_mixed_smooth": compute_bern_mixed_smooth,         # output_len: K*2+1
    "bern_mixed_sym3": compute_bern_mixed_sym3,         # output_len: K//2*5+1
    "bern_mixed_smooth": compute_bern_mixed_smooth,     # output_len: K+1
}

pe_len_dict = {
    "adj_powers": lambda K: K + 1,
    "rrwp": lambda K: K + 1,
    "bernstein": lambda K: K + 2,
    "bern_mixed_smooth": lambda K: K * 2 + 1,
    "bern_mixed_sym3": lambda K: K // 2 * 5 + 1,
    "bern_mixed_smooth": lambda K: K + 1,
}


class PositionalEncodingComputation(object):
    r"""Positional encoding computation.
    Args:
        pe_method (str): the method computes positional encoding.
        pe_power (int): highest power.
        flat (bool): whether to flatten the PE and remove the null entries.
    """

    def __init__(self, pe_method: str, pe_power: int, flat=False):
        self.compute_pe = pe_computer_dict[pe_method]
        self.pe_power = pe_power
        self.pe_len = pe_len_dict[pe_method](pe_power)
        self.flat = flat

    def __call__(self, data: Data) -> Data:
        N = data.num_nodes
        adj = torch.zeros((N, N))
        edge_index = data["edge_index"]
        adj[edge_index[0], edge_index[1]] = torch.ones((edge_index.size(1),))
        pe = self.compute_pe(adj, self.pe_power)
        if self.flat:
            pe = pe.permute((1, 2, 0))
            pe_index = pe.abs().sum([i for i in range(2, pe.dim())]).nonzero().t_()
            pe_val = pe[pe_index[0], pe_index[1], :].contiguous()
            data["pe_index"] = pe_index
            data["pe_val"] = pe_val
        else:
            data["pe"] = pe
        return data
