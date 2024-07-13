"""
Seperated_SpecDistGNN framework.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch_sparse
from torch import Tensor

from .basic import JK_MLP
from .seperated_block_layer import SeperatedBlockUpdateLayer
from .seperated_pooling import seperated_avg_pooling, seperated_sum_pooling
from .degree_rescaler import DegreeRescaler
from .output_decoder import GraphClassification, GraphRegression, NodeClassification


class Seperated_SpecDistGNN(nn.Module):
    r"""An implementation of Seperated_SpecDistGNN.
    Args:
        pe_len (int): the length of positional embedding.
        node_encoder (nn.Module): initial node feature encoding.
        edge_encoder (nn.Module): initial edge feature encoding.
        hidden_channels (int): hidden_channels
        num_layers (int): the number of layers
        mlp_depth (int): the number of layers in each MLP in RPGN
        norm_type (str, optional): Method of normalization, choose from (Batch, Layer, Instance, GraphSize, Pair).
        drop_prob (float, optional): dropout rate.
        graph_pool (str): Method of graph pooling, last,concat,max or sum.
        jumping_knowledge (str, optional): Method of jumping knowledge, last,concat,max or sum.
        task_type (str): Task type, graph_classification, graph_regression, node_classification.
    """

    def __init__(
        self,
        node_encoder: nn.Module,
        edge_encoder: nn.Module,
        pe_encoder: nn.Module,
        hidden_channels: int,
        num_layers: int,
        mlp_depth: int,
        norm_type: Optional[str] = "Batch",
        graph_pool: Optional[str] = 'avg',
        drop_prob: Optional[float] = 0.0,
        jumping_knowledge: bool = False,
        degree_rescalling: bool = True,
        task_type: str = "graph_classfication",
        num_tasks: int = 1,
    ):

        super(Seperated_SpecDistGNN, self).__init__()
        self.hidden_channels = hidden_channels
        self.norm_type = norm_type
        self.drop_prob = drop_prob
        self.dropout = nn.Dropout(drop_prob)

        # 1st part - mapping input features
        assert len({node_encoder.out_channels,
                    edge_encoder.out_channels,
                    pe_encoder.out_channels,
                    hidden_channels}) == 1

        self.node_encoder = node_encoder
        self.edge_encoder = edge_encoder
        self.pe_encoder = pe_encoder

        # 2st part - conv
        self.blocks = nn.ModuleList([
            SeperatedBlockUpdateLayer(hidden_channels, mlp_depth, drop_prob)
            for _ in range(num_layers)
        ])

        # 3st part - readout
        if graph_pool in {"mean", "avg"}:
            self.seperated_pooling = seperated_avg_pooling
        else:
            self.seperated_pooling = seperated_sum_pooling

        jk_channels = hidden_channels * 2 * (num_layers + 1)
        self.jk_mlp = JK_MLP(jk_channels, hidden_channels) if jumping_knowledge == "concat" else None

        # 4th part - degree rescalling
        if degree_rescalling:
            assert graph_pool in {"avg", "mean"}
            self.degree_rescaler = DegreeRescaler(hidden_channels * 2)
        else:
            self.degree_rescaler = None

        # 5th part - output decoding
        if task_type == "graph_classification":
            self.out_decoder = GraphClassification(hidden_channels * 2, num_tasks)
        elif task_type == "graph_regression":
            self.out_decoder = GraphRegression(hidden_channels * 2, num_tasks)
        elif task_type == "node_classification":
            self.out_decoder = NodeClassification(hidden_channels * 2, num_tasks)
        else:
            raise NotImplementedError()

        self.reset_parameters()

    def _init_weights(self, m: nn.Module):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def reset_parameters(self):
        self.node_encoder.apply(self._init_weights)
        self.edge_encoder.apply(self._init_weights)
        self.pe_encoder.apply(self._init_weights)
        self.blocks.apply(self._init_weights)
        if self.jk_mlp is not None:
            self.jk_mlp.apply(self._init_weights)
        self.out_decoder.apply(self._init_weights)

    def forward(self, data: Dict[str, Tensor]) -> Tensor:
        idx_list = [data["batch_full_index"]]
        val_list = [idx_list[0].new_zeros((idx_list[0].size(1), self.hidden_channels))]
        idx_list += [data["batch_pe_index"]]
        val_list += [self.pe_encoder(data["batch_pe_val"])]
        idx_list += [data["batch_edge_index"]]
        val_list += [self.edge_encoder(data["batch_edge_val"])]
        idx_list += [data["batch_eye_index"]]
        val_list += [self.node_encoder(data["batch_node_val"])]

        N = data["total_num_nodes"]
        _, h = torch_sparse.coalesce(
            torch.cat(idx_list, dim=1), torch.cat(val_list, dim=0), N, N, op="add",
        )

        degrees = data["batch_num_nodes"]
        log_degrees = torch.log(degrees + 1.)
        log_degrees = log_degrees.view((-1, 1, 1, 1))
        len1d = data["len1d"]
        size3d = data["size3d"]
        sep_size = [s[0] for s in size3d]
        sep_log_degs = torch.split(log_degrees, sep_size)
        h_list = [h]
        for block in self.blocks:
            h = block(h, sep_log_degs, len1d, size3d)  # shape: total_num_edges, self.hidden_channels
            h_list.append(h)

        if self.jk_mlp is not None:
            h = self.jk_mlp(torch.cat(h_list, 1))
        else:
            h = h_list[-1]

        hs = torch.split(h, len1d, 0)
        bs = [h.reshape(s + (-1,)) for h, s in zip(hs, size3d)]
        z = self.seperated_pooling(bs)

        if self.degree_rescaler is not None:
            log_degrees = log_degrees.view((-1, 1))
            z = self.degree_rescaler(z, log_degrees)

        out = self.out_decoder(z)
        return out
