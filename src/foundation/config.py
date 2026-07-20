"""Hyperparameters for the foundation-model runs.

Values are the paper's Var-GNN settings for each dataset (Appendix E.2, Table 12),
changed only where the task set demands it -- per the study's brief, everything except
the forward operators stays as published.

METR-LA  batch 128, cglsIter 8,  channels 32, layers 8, lr 2.50e-3, wd 9.89e-6
CPOX     batch  64, cglsIter 32, channels 32, layers 8, lr 2.80e-4, wd 7.77e-5

Epoch budgets and patience are Appendix E.2: METR-LA 100/35, CPOX 250/50.
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# Paper Var-GNN hyperparameters, per dataset.
DATASET_DEFAULTS = {
    "METRLA": dict(
        train_batch_size=128,
        cglsIter=8,
        channels=32,
        layers=8,
        lr=2.50e-3,
        wd=9.89e-6,
        solveIter=8,
        epochs=100,
        max_patience=35,
    ),
    "CPOX": dict(
        train_batch_size=64,
        cglsIter=32,
        channels=32,
        layers=8,
        lr=2.80e-4,
        wd=7.77e-5,
        solveIter=8,
        epochs=250,
        max_patience=50,
    ),
}

# Largest per-step batch that leaves room for two concurrent runs on a 6 GB card.
# CPOX graphs are 20 nodes, so its whole batch fits and needs no accumulation.
MAX_MICRO_BATCH = {"METRLA": 32, "CPOX": 64}


def build_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--dataset", type=str, default="METRLA", choices=["METRLA", "CPOX"])
    p.add_argument("--datapath", type=str, default=os.path.join(REPO, "data"))
    p.add_argument("--runs_root", type=str, default=os.path.join(REPO, "runs"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)

    # Overridable; None means "take the dataset default above".
    p.add_argument(
        "--train_batch_size",
        type=int,
        default=None,
        help="effective batch size (the paper's value); realised via gradient accumulation",
    )
    p.add_argument(
        "--micro_batch",
        type=int,
        default=None,
        help="actual per-step batch; effective/micro gives the accumulation count",
    )
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--wd", type=float, default=None)
    p.add_argument("--cglsIter", type=int, default=None)
    p.add_argument("--solveIter", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max_patience", type=int, default=None)

    # Task-specific operator settings.
    p.add_argument(
        "--blur_count",
        type=int,
        default=16,
        help="diffusion steps k for source localization (paper's headline setting)",
    )
    # The data is z-normalised, so a trivial "return the observation" estimator scores
    # nMSE = sigma^2. At sigma=0.1 that baseline is already 0.01 and the task is
    # uninformative; sigma=0.5 puts it at 0.25, comparable to the other tasks' scale.
    p.add_argument(
        "--noise_sigma", type=float, default=0.5, help="observation noise for denoising"
    )
    p.add_argument(
        "--alpha", type=float, default=1.0, help="data-fit weight in the loss (App. E.1)"
    )

    # Fractions, useful for smoke tests.
    p.add_argument("--train_frac", type=float, default=1.0)
    p.add_argument("--test_frac", type=float, default=1.0)
    return p


def finalize(args):
    """Fill unset options from the dataset defaults and add the fixed upstream fields."""
    defaults = DATASET_DEFAULTS[args.dataset]
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    # Gradient accumulation: the paper's METR-LA batch of 128 means 26,496 nodes per
    # step, which peaks at 5.3 GB -- fine on the paper's 48 GB A6000, but this study
    # runs two models concurrently on a 6 GB card. Measured peaks are 0.67/1.33/2.72 GB
    # at micro-batches of 16/32/64, so 32 leaves comfortable headroom for two processes
    # while accumulation keeps the effective batch, and therefore the optimisation,
    # identical to the published setting.
    args.effective_batch_size = args.train_batch_size
    if args.micro_batch is None:
        args.micro_batch = min(MAX_MICRO_BATCH.get(args.dataset, 32), args.train_batch_size)
    if args.effective_batch_size % args.micro_batch != 0:
        raise ValueError(
            f"effective batch {args.effective_batch_size} must be a multiple of "
            f"micro batch {args.micro_batch}"
        )
    args.accum_steps = args.effective_batch_size // args.micro_batch

    # The loaders and the rest of the pipeline see the micro-batch.
    args.train_batch_size = args.micro_batch
    args.test_batch_size = args.micro_batch

    # Fixed upstream fields that get_network / process_data / get_data_and_loaders read.
    args.method = "drip"          # Var-GNN, the paper's best model
    args.regnet = "hyper"
    args.task = "foundation"      # only used for naming; operators are swapped explicitly
    args.classify = 0             # every task here is regression, scored by nMSE
    args.dropout = 0.0
    args.rnfPE = 1
    args.mu = 0.01
    args.use_meta_data = 1
    args.CPOX_lags = 1
    args.cluster = 0
    args.pathLength = 32
    args.mask_per_class_budget = 4
    args.mask_per_snapshot_budget = 16
    return args


def parse(description):
    return finalize(build_parser(description).parse_args())
