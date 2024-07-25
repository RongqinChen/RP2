"""
script to run on Peptides-Functional task.
"""
import os
import time
import numpy as np
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Timer
from tqdm import tqdm
# from lightning.pytorch.callbacks.progress import TQDMProgressBar
# import wandb
# from lightning.pytorch.loggers import WandbLogger
import swanlab
from swanlab.integration.pytorch_lightning import SwanLabLogger
from sklearn.metrics import average_precision_score
from torch import Tensor, nn
from torchmetrics.metric import Metric

import utils
from models.model_construction import make_seperated_model
from datasets.peptides_functional import PeptidesFunctionalDataset
from pl_modules.loader import PlPyGDataTestonValModule
from pl_modules.model import PlGNNTestonValModule
from positional_encoding import PositionalEncodingComputation

torch.set_num_threads(8)
torch.set_float32_matmul_precision('high')


def main():
    parser = utils.args_setup()
    parser.add_argument("--config_file", type=str, default="configs/pep_functional.yaml",
                        help="Additional configuration file for different dataset and models.")
    args = parser.parse_args()
    args = utils.update_args(args)

    pe_computation = PositionalEncodingComputation(args.pe_method, args.pe_power, flat=True)
    args.pe_len = pe_computation.pe_len
    dataset = PeptidesFunctionalDataset("data", transform=pe_computation)

    # pre-compute positional encoding
    time_start = time.perf_counter()
    dataset._data_list = [pe_computation(data) for data in tqdm(dataset, "Computing PE ...")]
    pe_elapsed = time.perf_counter() - time_start
    pe_elapsed = time.strftime("%H:%M:%S", time.gmtime(pe_elapsed)) + f"{pe_elapsed:.2f}"[-3:]
    print(f"Took {pe_elapsed} to compute positional encoding ({args.pe_method}, {args.pe_power}).")

    split_dict = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_dict['train'], split_dict['val'], split_dict['test']
    train_dataset, val_dataset, test_dataset = dataset[train_idx], dataset[val_idx], dataset[test_idx]

    num_nodes = torch.diff(dataset.slices['x'])
    val_num_nodes = num_nodes[val_idx]
    test_num_nodes = num_nodes[test_idx]
    val_idx = [idx for idx in sorted(range(len(val_num_nodes)), key=lambda idx: val_num_nodes[idx])]
    test_idx = [idx for idx in sorted(range(len(test_num_nodes)), key=lambda idx: test_num_nodes[idx])]
    val_dataset = [val_dataset[idx] for idx in val_idx]
    test_dataset = [test_dataset[idx] for idx in test_idx]

    project = os.environ.get("MACHINE", "") + "-Sep-" + args.project_name
    for i in range(args.runs):
        # logger = WandbLogger(f"Run-{i}", args.save_dir, offline=args.offline, project=project)
        logger = SwanLabLogger(experiment_name=f"func-Run{i}",
                               project=project,
                               logdir="results/func/swanlab",
                               save_dir=args.save_dir,
                               mode="local" if args.offline else None)

        logger.log_hyperparams(args)
        timer = Timer(duration=dict(weeks=4))

        # Set random seed
        seed = utils.get_seed(args.seed)
        seed_everything(seed)

        datamodule = PlPyGDataTestonValModule(
            train_dataset, val_dataset, test_dataset,
            args.batch_size, args.num_workers, args.drop_last, pad2same=False,
        )
        loss_criterion = nn.BCEWithLogitsLoss()
        evaluator = AveragePrecision()
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
            enable_progress_bar=False,
            logger=logger,
            callbacks=[
                # TQDMProgressBar(refresh_rate=20),
                ModelCheckpoint(monitor="val/metric", mode="min"),
                LearningRateMonitor(logging_interval="epoch"),
                timer
            ]
        )

        trainer.fit(modelmodule, datamodule=datamodule)
        val_result, test_result = trainer.test(modelmodule, datamodule=datamodule, ckpt_path="best")
        results = {
            "final/best_val_metric": val_result["val/metric"],
            "final/best_test_metric": test_result["test/metric"],
            "final/avg_train_time_epoch": timer.time_elapsed("train") / args.num_epochs,
        }
        print("Positional encoding:", f"({args.pe_method}, {args.pe_power})")
        print("PE computation time:", pe_elapsed)
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
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        node_h = 0
        for idx, embedding in enumerate(self.atom_embedding_list):
            node_h += embedding(x[:, idx])

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
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, x):
        edge_h = 0
        for idx, embedding in enumerate(self.bond_embedding_list):
            edge_h += embedding(x[:, idx])

        return edge_h


class PELinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, pe_val: Tensor):
        pe_h = self.linear(pe_val)
        return pe_h


class AveragePrecision(Metric):
    def __init__(self):
        super().__init__()
        self.preds_list = []
        self.targets_list = []

    def update(self, preds: Tensor, targets: Tensor) -> None:
        self.preds_list.append(preds.detach().cpu().numpy())
        self.targets_list.append(targets.detach().cpu().numpy())

    def compute(self) -> Tensor:
        y_pred = np.concatenate(self.preds_list, 0)
        y_true = np.concatenate(self.targets_list, 0)

        ap_list = []
        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                # ignore nan values
                is_labeled = y_true[:, i] == y_true[:, i]
                ap = average_precision_score(y_true[is_labeled, i], y_pred[is_labeled, i])
                ap_list.append(ap)

        if len(ap_list) == 0:
            raise RuntimeError(
                'No positively labeled data available. Cannot compute Average Precision.')

        ap = sum(ap_list) / len(ap_list)
        ap = torch.tensor(ap)
        return ap

    def reset(self) -> None:
        self.preds_list.clear()
        self.targets_list.clear()


if __name__ == "__main__":
    main()
