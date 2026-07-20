"""Time one full-size training epoch, to estimate a sweep's wall-clock cost.

Measures the real thing rather than extrapolating from a data fraction: a complete
epoch over every training snapshot, plus one validation pass, plus the end-of-run
five-task scoring. Run two copies concurrently to capture GPU contention, which is how
main_parallel.py actually executes.

    python measure_epoch.py --dataset METRLA              # single process
    python measure_epoch.py --dataset METRLA --tag A &    # two at once, for contention
    python measure_epoch.py --dataset METRLA --tag B &
"""

import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "foundation"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="METRLA", choices=["METRLA", "CPOX"])
    p.add_argument("--tag", default="")
    p.add_argument("--epochs_budget", type=int, default=None,
                   help="epochs per model to project (defaults to the dataset's setting)")
    args_cli = p.parse_args()

    sys.argv = ["measure", "--dataset", args_cli.dataset]
    from config import parse
    from common import evaluate, evaluate_all, train_epoch
    from tasks import TASKS, build_forward_ops, experiment_split, mask_budget_for, nodes_per_graph_for
    from utils import get_data_and_loaders, get_network

    args = parse("measure")
    budget_epochs = args_cli.epochs_budget or args.epochs
    tag = f"[{args_cli.tag}] " if args_cli.tag else ""

    train_tasks, val_task, _ = experiment_split(0)
    npg = nodes_per_graph_for(args.dataset)
    mask_budget = mask_budget_for(args.dataset)

    t0 = time.time()
    train_ds, test_ds, train_loader, test_loader, lc, fc = get_data_and_loaders(args)
    load_t = time.time() - t0

    ops, _ = build_forward_ops(
        TASKS, nodes_per_graph=npg, hid_channels=args.channels, label_channels=lc,
        learn_emb=True, device=args.device, blur_count=args.blur_count,
        noise_sigma=args.noise_sigma,
    )
    net = get_network(args, ops[train_tasks[0]], args.channels, lc, fc, args.device).to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd,
                           amsgrad=True, eps=1e-3)

    n_train, n_test = len(list(train_ds)), len(list(test_ds))
    print(f"{tag}dataset {args.dataset}  train {n_train}  test {n_test}  "
          f"batches/epoch {len(train_loader)}  micro-batch {args.micro_batch}", flush=True)

    # First epoch pays CUDA warm-up and operator construction, so it is timed
    # separately and excluded from the projection.
    t = time.time()
    train_epoch(net, ops, train_tasks, train_loader, opt, args, mask_budget, npg)
    torch.cuda.synchronize()
    warm_t = time.time() - t
    print(f"{tag}epoch 1 (with warm-up)  {warm_t:7.1f}s", flush=True)

    t = time.time()
    train_epoch(net, ops, train_tasks, train_loader, opt, args, mask_budget, npg)
    torch.cuda.synchronize()
    train_t = time.time() - t

    t = time.time()
    evaluate(net, ops, val_task, test_loader, args, mask_budget, npg)
    torch.cuda.synchronize()
    val_t = time.time() - t

    t = time.time()
    evaluate_all(net, ops, test_loader, args, mask_budget, npg)
    torch.cuda.synchronize()
    scoring_t = time.time() - t

    peak = torch.cuda.max_memory_allocated() / 1e9
    per_epoch = train_t + val_t
    per_model = warm_t + per_epoch * (budget_epochs - 1) + scoring_t

    print(f"{tag}steady train epoch     {train_t:7.1f}s", flush=True)
    print(f"{tag}validation pass        {val_t:7.1f}s", flush=True)
    print(f"{tag}per epoch              {per_epoch:7.1f}s", flush=True)
    print(f"{tag}end-of-run 5-task score{scoring_t:7.1f}s  (once, not per epoch)", flush=True)
    print(f"{tag}data load              {load_t:7.1f}s", flush=True)
    print(f"{tag}peak VRAM              {peak:7.2f} GB", flush=True)
    print(f"{tag}", flush=True)
    print(f"{tag}PROJECTION at {budget_epochs} epochs/model:", flush=True)
    print(f"{tag}  1 model            {per_model / 3600:6.2f} h", flush=True)
    print(f"{tag}  6 models, 2 at a time (3 waves)  {per_model * 3 / 3600:6.2f} h", flush=True)
    print(f"{tag}  (upper bound: assumes no run early-stops)", flush=True)


if __name__ == "__main__":
    main()
