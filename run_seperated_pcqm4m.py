"""
script to run on PCQM4Mv2 tasks.
"""

import os

# import time
import numpy as np
import swanlab
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Timer
from lightning.pytorch.callbacks.progress import TQDMProgressBar
# from tqdm import tqdm
from ogb.lsc import PygPCQM4Mv2Dataset
from swanlab.integration.pytorch_lightning import SwanLabLogger
# import wandb
# from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch_geometric.transforms import Compose
from torchmetrics import MeanAbsoluteError

import utils
from models.model_construction import make_seperated_model
from pl_modules.loader import BatchSamplerWithGrouping, PlPyGDataTestonValModule
from pl_modules.model import PlGNNTestonValModule
from positional_encoding import PositionalEncodingComputation
from torchmetrics.functional.regression.mae import _mean_absolute_error_compute

torch.set_num_threads(8)
torch.set_float32_matmul_precision('high')


def main():
    parser = utils.args_setup()
    parser.add_argument("--config_file", type=str, default="configs/padded_pcqm4m.yaml",
                        help="Additional configuration file for different dataset and models.")
    args = parser.parse_args()
    args = utils.update_args(args)

    dataset = PygPCQM4Mv2Dataset("data")
    slices = dataset.slices
    num_nodes = torch.diff(slices['x'])
    split_idx = dataset.get_idx_split()
    rng = np.random.default_rng(seed=42)
    train_idx = rng.permutation(split_idx["train"].numpy())
    # Leave out 150k graphs for a new validation set.
    val_idx, train_idx = train_idx[:150000], train_idx[150000:]
    test_idx = split_idx["valid"]
    train_num_nodes = num_nodes[train_idx].tolist()
    val_num_nodes = num_nodes[val_idx].tolist()
    test_num_nodes = num_nodes[test_idx].tolist()

    train_dataset = dataset[train_idx]
    train_sampler = BatchSamplerWithGrouping(train_num_nodes, args.batch_size, True, False)

    val_idx = [val_idx[idx] for idx in sorted(range(len(val_idx)), key=lambda idx: val_num_nodes[idx])]
    test_idx = [test_idx[idx] for idx in sorted(range(len(test_idx)), key=lambda idx: test_num_nodes[idx])]
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]

    pe_computation = PositionalEncodingComputation(args.pe_method, args.pe_power, True)
    args.pe_len = pe_computation.pe_len
    y_train = dataset._data.y[train_idx]
    mean, std = y_train.mean(), y_train.std()
    evaluator = MeanAbsoluteErrorPCQM4M(std)
    set_y_fn = SetY(mean, std)
    print("mean, std", mean.item(), std.item())
    transform = Compose([set_y_fn, pe_computation])
    train_dataset.transform = transform
    val_dataset.transform = transform
    test_dataset.transform = transform

    MACHINE = os.environ.get("MACHINE", "") + "-"
    for i in range(args.runs):
        # logger = WandbLogger(f"Run-{i}", args.save_dir, offline=args.offline, project=MACHINE + args.project_name)
        logger = SwanLabLogger(experiment_name=f"Run-{i}",
                               project=MACHINE + args.project_name,
                               logdir="results/PCQM4Mv2/swanlab",
                               save_dir=args.save_dir,
                               mode="local" if args.offline else None)
        logger.log_hyperparams(args)
        timer = Timer(duration=dict(weeks=4))

        # Set random seed
        seed = utils.get_seed(args.seed)
        seed_everything(seed)

        datamodule = PlPyGDataTestonValModule(
            train_dataset, val_dataset, test_dataset,
            args.batch_size, args.num_workers, args.drop_last,
            pad2same=False, train_sampler=train_sampler
        )
        # sorting training set because the large number of samples

        loss_criterion = nn.L1Loss()
        evaluator = MeanAbsoluteError()
        node_encoder = AtomEncoder(args.hidden_channels)
        edge_encoder = BondEncoder(args.hidden_channels)
        pe_encoder = PELinear(args.pe_len, args.hidden_channels)
        model = make_seperated_model(args, node_encoder, edge_encoder, pe_encoder)
        modelmodule = PlGNNTestonValModule(args, model, loss_criterion, evaluator)

        trainer = Trainer(
            accelerator="auto",
            devices="auto",
            max_epochs=args.num_epochs,
            enable_checkpointing=True,
            enable_progress_bar=True,
            logger=logger,
            callbacks=[
                TQDMProgressBar(refresh_rate=20),
                ModelCheckpoint(monitor="val/metric", mode="min"),
                LearningRateMonitor(logging_interval="epoch"),
                timer
            ]
        )
        print(model)

        trainer.fit(modelmodule, datamodule=datamodule)
        val_result, test_result = trainer.test(modelmodule, datamodule=datamodule, ckpt_path="best")
        results = {
            "final/best_val_metric": val_result["val/metric"],
            "final/best_test_metric": test_result["test/metric"],
            "final/avg_train_time_epoch": timer.time_elapsed("train") / args.num_epochs,
        }
        print("Positional encoding:", f"({args.pe_method}, {args.pe_power})")
        # print("PE computation time:", pe_elapsed)
        print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024))
        logger.log_metrics(results)
        swanlab.finish()

    return


class AtomEncoder(torch.nn.Module):
    r"""The atom encoder used in OGB molecule dataset.

    Args:
        emb_dim (int): The output embedding dimension.

    Example:
        >>> encoder = AtomEncoder(emb_dim=16)
        >>> batch = torch.randint(0, 10, (10, 3))
        >>> encoder(batch).size()
        torch.Size([10, 16])
    """

    def __init__(self, emb_dim, *args, **kwargs):
        super().__init__()
        self.out_channels = emb_dim
        from ogb.utils.features import get_atom_feature_dims
        self.atom_embedding_list = torch.nn.ModuleList()
        for dim in get_atom_feature_dims():
            emb = torch.nn.Embedding(dim + 1, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, node_val):
        node_h = 0
        for idx, embedding in enumerate(self.atom_embedding_list):
            node_h += embedding(node_val[:, idx])
        return node_h


class BondEncoder(torch.nn.Module):
    r"""The bond encoder used in OGB molecule dataset.

    Args:
        emb_dim (int): The output embedding dimension.

    Example:
        >>> encoder = BondEncoder(emb_dim=16)
        >>> batch = torch.randint(0, 10, (10, 3))
        >>> encoder(batch).size()
        torch.Size([10, 16])
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        self.out_channels = emb_dim
        from ogb.utils.features import get_bond_feature_dims
        self.bond_embedding_list = torch.nn.ModuleList()
        for dim in get_bond_feature_dims():
            emb = torch.nn.Embedding(dim + 1, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_val):
        node_h = 0
        for idx, embedding in enumerate(self.bond_embedding_list):
            node_h += embedding(edge_val[:, idx])
        return node_h


class PELinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, pe_val):
        pe_h = self.linear(pe_val)
        return pe_h


class MeanAbsoluteErrorPCQM4M(MeanAbsoluteError):
    def __init__(self, std, **kwargs):
        super().__init__(**kwargs)
        self.std = std

    def compute(self):
        return (_mean_absolute_error_compute(self.sum_abs_error, self.total) * self.std)


class SetY(object):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data.y = (data.y - self.mean) / self.std
        return data


if __name__ == "__main__":
    main()
