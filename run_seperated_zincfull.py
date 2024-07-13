"""
script to run on ZINC task.
"""
import os
import time
import torch
import torchmetrics
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Timer
# from lightning.pytorch.callbacks.progress import TQDMProgressBar
import wandb
from lightning.pytorch.loggers import WandbLogger
# import swanlab
# from swanlab.integration.pytorch_lightning import SwanLabLogger
from torch import Tensor, nn
from torch_geometric.datasets import ZINC

import utils
from models.model_construction import make_seperated_model
from pl_modules.loader import PlPyGDataTestonValModule
from pl_modules.model import PlGNNTestonValModule
from positional_encoding import PositionalEncodingComputation

torch.set_num_threads(8)
torch.set_float32_matmul_precision('high')


def main():
    parser = utils.args_setup()
    parser.add_argument("--config_file", type=str, default="configs/seperated_zincfull.yaml",
                        help="Additional configuration file for different dataset and models.")
    parser.add_argument("--runs", type=int, default=5, help="Number of repeat run.")
    args = parser.parse_args()

    args = utils.update_args(args)
    args.full = True

    path = "data/ZINC"
    train_dataset = ZINC(path, not args.full, "train")
    val_dataset = ZINC(path, not args.full, "val")
    test_dataset = ZINC(path, not args.full, "test")

    # pre-compute Positional encoding
    time_start = time.perf_counter()
    pe_computation = PositionalEncodingComputation(args.pe_method, args.pe_power, True)
    args.pe_len = pe_computation.pe_len
    train_dataset._data_list = [pe_computation(data) for data in train_dataset]
    val_dataset._data_list = [pe_computation(data) for data in val_dataset]
    test_dataset._data_list = [pe_computation(data) for data in test_dataset]
    pe_elapsed = time.perf_counter() - time_start
    pe_elapsed = time.strftime("%H:%M:%S", time.gmtime(pe_elapsed)) + f"{pe_elapsed:.2f}"[-3:]
    print(f"Took {pe_elapsed} to compute positional encoding ({args.pe_method}, {args.pe_power}).")

    project = os.environ.get("MACHINE", "") + "-Sep-" + args.project_name
    for i in range(args.runs):
        logger = WandbLogger(f"Run-{i}", args.save_dir, offline=args.offline, project=project)
        # logger = SwanLabLogger(experiment_name=f"Run-{i}", project=MACHINE + args.project_name,
        #                        logdir=args.save_dir + "/swanlab",
        #                        save_dir=args.save_dir, offline=args.offline)
        logger.log_hyperparams(args)
        timer = Timer(duration=dict(weeks=4))

        # Set random seed
        seed = utils.get_seed(args.seed)
        seed_everything(seed)

        datamodule = PlPyGDataTestonValModule(
            train_dataset, val_dataset, test_dataset,
            args.batch_size, args.num_workers, args.drop_last, pad2same=False,
        )
        loss_criterion = nn.L1Loss()
        evaluator = torchmetrics.MeanAbsoluteError()
        node_encoder = NodeEncoder(28, args.hidden_channels)
        edge_encoder = EdgeEncoder(4, args.hidden_channels)
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
        wandb.finish()

    return


class NodeEncoder(nn.Module):
    def __init__(self, num_types, out_channels):
        super().__init__()
        self.emb = nn.Embedding(num_types, out_channels)
        self.out_channels = out_channels
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()

    def forward(self, node_val: Tensor):
        # Encode just the first dimension if more exist
        node_h = self.emb(node_val[:, 0])
        return node_h


class EdgeEncoder(nn.Module):
    def __init__(self, num_types, out_channels):
        super().__init__()
        self.emb = nn.Embedding(num_types, out_channels)
        self.out_channels = out_channels
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()

    def forward(self, edge_val: Tensor):
        # Encode just the first dimension if more exist
        edge_h = self.emb(edge_val[:, 0])
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


if __name__ == "__main__":
    main()
