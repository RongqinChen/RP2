"""
Model construction.
"""

from argparse import ArgumentParser
from torch import nn
from models.padded_network import Padded_SpecDistGNN
from models.seperated_network import Seperated_SpecDistGNN


def make_padded_model(
    args: ArgumentParser,
    node_encoder: nn.Module,
    edge_encoder: nn.Module,
) -> nn.Module:
    r"""Make GNN model given input parameters.
    Args:
        args (ArgumentParser): Arguments dict from argparser.
        node_encoder (nn.Module): Node feature input encoder.
        edge_encoder (nn.Module): Edge feature input encoder.
    """

    gnn = Padded_SpecDistGNN(
        args.pe_len,
        node_encoder, edge_encoder,
        args.hidden_channels,
        args.num_layers,
        args.mlp_depth,
        args.norm_type,
        args.graph_pool,
        args.drop_prob,
        args.jumping_knowledge,
        args.task_type,
        args.num_task,
    )
    return gnn


def make_seperated_model(
    args: ArgumentParser,
    node_encoder: nn.Module,
    edge_encoder: nn.Module,
    pe_encoder: nn.Module,
) -> nn.Module:
    r"""Make GNN model given input parameters.
    Args:
        args (ArgumentParser): Arguments dict from argparser.
        node_encoder (nn.Module): Node feature input encoder.
        edge_encoder (nn.Module): Edge feature input encoder.
        pe_encoder (nn.Module): Positional encoding input encoder.
    """

    gnn = Seperated_SpecDistGNN(
        node_encoder, edge_encoder, pe_encoder,
        args.hidden_channels,
        args.num_layers,
        args.mlp_depth,
        args.norm_type,
        args.graph_pool,
        args.drop_prob,
        args.jumping_knowledge,
        args.task_type,
        args.num_task,
    )
    return gnn
