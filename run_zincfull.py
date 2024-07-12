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
from models.model_construction import make_model
from pl_modules.loader import PlPyGDataTestonValModule
from pl_modules.model import PlGNNTestonValModule
from positional_encoding import PositionalEncodingComputation

torch.set_num_threads(8)
torch.set_float32_matmul_precision('high')


def main():
    parser = utils.args_setup()
    parser.add_argument("--config_file", type=str, default="configs/zincfull.yaml",
                        help="Additional configuration file for different dataset and models.")
    parser.add_argument("--runs", type=int, default=10, help="Number of repeat run.")
    args = parser.parse_args()

    args = utils.update_args(args)
    args.full = True
    if args.full:
        args.project_name = "full_" + args.project_name

    path = "data/ZINC"
    train_dataset = ZINC(path, not args.full, "train")
    val_dataset = ZINC(path, not args.full, "val")
    test_dataset = ZINC(path, not args.full, "test")

    # pre-compute Positional encoding
    time_start = time.perf_counter()
    pe_computation = PositionalEncodingComputation(args.pe_method, args.pe_power)
    args.pe_len = pe_computation.pe_len
    train_dataset._data_list = [pe_computation(data) for data in train_dataset]
    val_dataset._data_list = [pe_computation(data) for data in val_dataset]
    test_dataset._data_list = [pe_computation(data) for data in test_dataset]
    pe_elapsed = time.perf_counter() - time_start
    pe_elapsed = time.strftime("%H:%M:%S", time.gmtime(pe_elapsed)) + f"{pe_elapsed:.2f}"[-3:]
    print(f"Took {pe_elapsed} to compute positional encoding ({args.pe_method}, {args.pe_power}).")

    MACHINE = os.environ.get("MACHINE", "") + "-"
    for i in range(args.runs):
        logger = WandbLogger(f"Run-{i}", args.save_dir, offline=args.offline, project=MACHINE + args.project_name)
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
            args.batch_size, args.num_workers, args.drop_last, pad2same=True,
        )
        loss_criterion = nn.L1Loss()
        evaluator = torchmetrics.MeanAbsoluteError()
        node_encoder = NodeEncoder(28, args.emb_channels)
        edge_encoder = EdgeEncoder(4, args.emb_channels)
        model = make_model(args, node_encoder, edge_encoder)
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


class NodeEncoder(torch.nn.Module):
    def __init__(self, num_types, out_channels, padding_idx=0):
        super().__init__()
        self.emb = nn.Embedding(num_types + 1, out_channels, padding_idx)
        self.out_channels = out_channels
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()

    def forward(self, batch: dict):
        # Encode just the first dimension if more exist
        batch_node_attr = batch["batch_node_attr"]
        B, N, _ = batch_node_attr.size()
        node_h: Tensor = self.emb(batch_node_attr[:, :, 0].flatten())
        batch_node_h = node_h.reshape((B, N, -1))  # B, N, H
        batch_node_h = batch_node_h.permute((0, 2, 1))  # B, H, N
        batch_full_node_h = torch.diag_embed(batch_node_h)  # B, H, N, N
        return batch_full_node_h


class EdgeEncoder(torch.nn.Module):
    def __init__(self, num_types, out_channels, padding_idx=0):
        super().__init__()
        self.emb = nn.Embedding(num_types + 1, out_channels, padding_idx)
        self.out_channels = out_channels
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()

    def forward(self, batch: dict):
        # Encode just the first dimension if more exist
        batch_full_edge_attr = batch["batch_full_edge_attr"]
        B, N, N, _ = batch_full_edge_attr.size()
        edge_h: Tensor = self.emb(batch_full_edge_attr[:, :, :, 0].flatten())
        batch_full_edge_h = edge_h.reshape((B, N, N, -1))
        batch_full_edge_h = batch_full_edge_h.permute((0, 3, 1, 2)).contiguous()
        return batch_full_edge_h


if __name__ == "__main__":
    main()
