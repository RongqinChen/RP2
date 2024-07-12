"""
SpecDistGNN framework.
"""

from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from .mlp import MLP
from .padded_block_conv import PaddedBlockConv
from .block_pooling import padded_block_avg_pool
from .output_decoder import GraphClassification, GraphRegression, NodeClassification


class SpecDistGNN(nn.Module):
    r"""An implementation of SpecDistGNN.
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
        pe_len: int,
        node_encoder: nn.Module,
        edge_encoder: nn.Module,
        hidden_channels: int,
        num_layers: int,
        mlp_depth: int,
        norm_type: Optional[str] = "Batch",
        graph_pool: nn.Module = None,
        drop_prob: Optional[float] = 0.0,
        jumping_knowledge: bool = False,
        task_type: str = "graph_classfication",
        num_tasks: int = 1,
    ):

        super(SpecDistGNN, self).__init__()
        self.norm_type = norm_type
        self.drop_prob = drop_prob
        self.dropout = nn.Dropout(drop_prob)
        self.node_encoder = node_encoder
        self.edge_encoder = edge_encoder

        # 1st part - mapping input features
        in_channels = pe_len + node_encoder.out_channels + edge_encoder.out_channels
        self.conv0 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)

        # 2st part - conv
        self.blocks = nn.ModuleList([
            PaddedBlockConv(hidden_channels, mlp_depth) for _ in range(num_layers)
        ])

        self.graph_pool = padded_block_avg_pool
        jk_channels = hidden_channels * 2 * (num_layers + 1)
        self.jk_mlp = MLP(jk_channels, hidden_channels) if jumping_knowledge == "concat" else None

        self.act = nn.ReLU()
        if task_type == "graph_classification":
            self.out_decoder = GraphClassification(hidden_channels * 2, num_tasks)
        elif task_type == "graph_regression":
            self.out_decoder = GraphRegression(hidden_channels * 2, num_tasks)
        elif task_type == "node_classification":
            self.out_decoder = NodeClassification(hidden_channels * 2, num_tasks)
        else:
            raise NotImplementedError()

        self.reset_parameters()

    def weights_init(self, m: nn.Module):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def reset_parameters(self):
        self.blocks.apply(self.weights_init)
        if self.jk_mlp is not None:
            self.jk_mlp.apply(self.weights_init)
        self.out_decoder.apply(self.weights_init)

    def forward(self, data: Data) -> Tensor:
        hs = [data["batch_full_pe"]]
        hs += [self.node_encoder(data)] if self.node_encoder is not None else []
        hs += [self.edge_encoder(data)] if self.edge_encoder is not None else []
        h = torch.cat(hs, 1)
        h = self.conv0(h)

        h_list = []
        for block in self.blocks:
            h = block(h)
            h_list.append(h)

        if self.jk_mlp is not None:
            z_list = [self.graph_pool(h) for h in h_list]
            z = self.jk_mlp(torch.cat(z_list, 1))
        else:
            z = self.graph_pool(h_list[-1])

        out = self.out_decoder(z)
        return out
