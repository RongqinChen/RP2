"""
Utils file for training.
"""

import argparse
import os
import time
import yaml


def args_setup():
    r"""Setup argparser.
    """
    parser = argparse.ArgumentParser("arguments for training and testing")
    # common args
    parser.add_argument("--save_dir", type=str, default="./results", help="Base directory for saving information.")
    parser.add_argument("--seed", type=int, default=234, help="Random seed for reproducibility.")
    parser.add_argument("--runs", type=int, help="Number of repeat run.")

    # training args
    parser.add_argument("--batch_size", type=int, help="Batch size per GPU.")
    parser.add_argument("--num_workers", type=int, help="Number of worker.")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--min_lr", type=float, help="Minimum learning rate.")
    parser.add_argument("--l2_wd", type=float, help="L2 weight decay.")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs.")
    parser.add_argument("--num_warmup_epochs", type=int, help="Number of warmup epochs.")
    parser.add_argument("--test_eval_interval", type=int,
                        help="Interval between validation on test dataset.")
    parser.add_argument("--factor", type=float,
                        help="Factor in the ReduceLROnPlateau learning rate scheduler.")
    parser.add_argument("--patience", type=int,
                        help="Patience in the ReduceLROnPlateau learning rate scheduler.")
    parser.add_argument("--offline", action="store_true", help="If true, save the wandb log offline. "
                                                               "Mainly use for debug.")
    parser.add_argument("--drop_last", action="store_true", help="If true, drop the last batch that is smaller than `batchsize`.")

    # data args
    parser.add_argument("--pe_method", type=str, help="Positional encoding computation method.")
    parser.add_argument("--pe_power", type=int, help="Positional encoding power.")

    # model args
    parser.add_argument("--emb_channels", type=int, help="Embedding size.")
    parser.add_argument("--hidden_channels", type=int, help="Hidden size of the model.")
    parser.add_argument("--num_layers", type=int, help="Number of layer for GNN.")
    parser.add_argument("--mlp_depth", type=int, help="Number of MLP layer of RPGN.")
    parser.add_argument("--norm_type", type=str, default="Batch",
                        choices=("Batch", "Layer", "Instance", "GraphSize", "Pair", "None"),
                        help="Normalization method in model.")
    parser.add_argument("--drop_prob", type=float,
                        help="Probability of zeroing an activation in dropout models.")
    parser.add_argument("--graph_pool", type=str, choices=("mean", "sum", "attention"),
                        help="Pooling method in graph level tasks.")
    parser.add_argument("--jumping_knowledge", type=str, choices=("last", "concat"),
                        help="Jumping knowledge method.")
    return parser


def update_args(args: argparse.ArgumentParser) -> argparse.ArgumentParser:
    r"""Update argparser given config file.
    Args:
        args (ArgumentParser): Arguments dict from argparser.
    """

    if args.config_file is not None:
        with open(args.config_file) as f:
            cfg = yaml.safe_load(f)
        for key, value in cfg.items():
            if getattr(args, key, None) is not None:
                continue
            if isinstance(value, list):
                for v in value:
                    getattr(args, key, []).append(v)
            else:
                setattr(args, key, value)

    arg_list = [
        str(args.num_layers),
        str(args.hidden_channels),
        str(args.mlp_depth),
        str(args.pe_method),
        str(args.pe_power),
    ]
    model_cfg = "-".join(arg_list)
    args.project_name = f"{args.dataset_name}-{model_cfg}"
    args.save_dir = args.save_dir + "/" + args.dataset_name
    os.makedirs(args.save_dir, exist_ok=True)
    return args


def get_seed(seed=234) -> int:
    r"""Return random seed based on current time.
    Args:
        seed (int): base seed.
    """
    t = int(time.time() * 1000.0)
    seed = seed + ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >> 8) + ((t & 0x0000ff00) << 8) + ((t & 0x000000ff) << 24)
    return seed
