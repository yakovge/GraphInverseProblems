"""Reproduction of the paper's own experiment, as the study's baseline.

Var-GNN on METR-LA inverse graph transport (``path``, pl=32) -- the paper's Table 5,
which reports nMSE 0.004. Hyperparameters are Table 12's exact values.

Two things this gives us:

1. A **fidelity check**. If our port of the training loop is sound, this run should land
   near 0.004. A large miss means something is wrong upstream of every other result.
2. A **single-task baseline**. ``path`` is none of our five tasks, so afterwards the same
   model is evaluated zero-shot on all five. That is the comparison that makes the
   foundation models interpretable: a model trained on one operator versus models trained
   on three.

This script deliberately reproduces the article's protocol *including* its selection on
the test set (``main_3_linear_inv_problems.py`` early-stops on test, there being no
validation split). That is what keeps the number comparable to the published table; it is
recorded in the metrics as ``selection='test'`` so the comparison never presents it as
though it were a clean held-out figure.

Run:  python exp_original.py --dataset METRLA
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import evaluate_all, nmse, observe, set_operator  # noqa: E402
from config import DATASET_DEFAULTS, parse  # noqa: E402
from graphForwardOps import graphPath  # noqa: E402
from progress import ProgressReporter  # noqa: E402
from tasks import (  # noqa: E402
    TASK_SHORT,
    TASKS,
    build_forward_ops,
    build_shared_embedding,
    mask_budget_for,
    nodes_per_graph_for,
)
from utils import (  # noqa: E402
    count_trainable_parameters,
    get_data_and_loaders,
    get_network,
    process_data,
)

RUN_NAME = "ORIGINAL_paper_path_pl32"

# The paper reports inverse graph transport (path) only on METR-LA, at nMSE 0.004
# (Table 5). CPOX appears in the paper under inverse *source* estimation (Table 4), so
# 'path' on CPOX reproduces nothing and gets no reference value.
#
# We deliberately keep 'path' as the baseline task on both datasets rather than
# switching CPOX to the published 'deblur' setting: deblur is our source_localization
# task, and training the baseline on one of the five would destroy the property that
# makes it a useful control -- that it trained on none of them.
PAPER_REFERENCE = {"METRLA": 0.004}


def run_path_batch(net, op, graph, args):
    """One batch of the paper's inverse-graph-transport task."""
    graph = process_data(args, graph)
    graph = graph.to(args.device)
    # graphPath redraws its random walks every batch, exactly as upstream does.
    op.gen_paths(nnodes=graph.x.shape[0], edge_index=graph.edge_index)
    set_operator(net, op)

    d_obs = observe(op, graph.y, graph)
    X, _, _ = net(d_obs, graph.edge_index, graph.edge_weight, graph.x)
    d_rec = op(X, graph.edge_index, graph.edge_weight, emb=False)

    loss_x = nmse(X, graph.y)
    loss_data = nmse(d_rec, d_obs)
    return 0.5 * (loss_x + args.alpha * loss_data), loss_x.item(), loss_data.item(), len(
        graph.batch.unique()
    )


def main():
    args = parse(__doc__)
    # The reproduction keeps the article's patience. config.finalize applies the tighter
    # v2 default for --selection train, but this run selects on its own test metric (the
    # article's protocol) where the published value is the right one.
    if "--max_patience" not in sys.argv:
        args.max_patience = DATASET_DEFAULTS[args.dataset]["max_patience"]
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    run_name = f"{RUN_NAME}_s{args.seed}"
    run_dir = os.path.join(args.runs_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    npg = nodes_per_graph_for(args.dataset)
    budget = mask_budget_for(args.dataset)
    args.task = "path"

    _, _, train_loader, test_loader, label_channels, feat_channels = get_data_and_loaders(
        args
    )

    # One embedding, shared by the path operator and by the five evaluation tasks, so the
    # zero-shot comparison uses weights this model actually trained.
    shared_emb = build_shared_embedding(
        args.channels, label_channels, learn_emb=True, device=args.device
    )
    path_op = graphPath(
        embdsize=args.channels,
        nin=label_channels,
        learnEmb=True,
        device=args.device,
        pathLength=args.pathLength,
    ).to(args.device)
    path_op.Emb = shared_emb

    eval_ops, _ = build_forward_ops(
        TASKS,
        nodes_per_graph=npg,
        hid_channels=args.channels,
        label_channels=label_channels,
        shared_emb=shared_emb,
        learn_emb=True,
        device=args.device,
        blur_count=args.blur_count,
        noise_sigma=args.noise_sigma,
    )

    net = get_network(
        args, path_op, args.channels, label_channels, feat_channels, args.device
    ).to(args.device)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True, eps=1e-3
    )

    n_params = count_trainable_parameters(net)
    reference = PAPER_REFERENCE.get(args.dataset)
    print(f"run        : {run_name}")
    print(f"task       : path (inverse graph transport), pl={args.pathLength}")
    print(
        f"paper ref  : Table 5, Var-GNN METR-LA nMSE {reference}"
        if reference is not None
        else f"paper ref  : none -- the paper does not report 'path' on {args.dataset}"
    )
    print(f"batch size : {args.train_batch_size}")
    print(f"parameters : {n_params}\n", flush=True)

    reporter = ProgressReporter(
        run_dir,
        run_name,
        args.epochs,
        extra={
            "kind": "original",
            "task": "path",
            "dataset": args.dataset,
            "parameters": n_params,
        },
    )

    best_test = float("inf")
    best_epoch = -1
    patience = 0
    started = time.time()
    last_epoch = 0

    try:
        for epoch in range(args.epochs):
            last_epoch = epoch + 1
            net.train()
            total, count = 0.0, 0
            for graph in train_loader:
                optimizer.zero_grad()
                loss, lx, _, bs = run_path_batch(net, path_op, graph, args)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                total += lx * bs
                count += bs
            train_nmse = total / max(count, 1)

            net.eval()
            t_total, t_count = 0.0, 0
            with torch.no_grad():
                for graph in test_loader:
                    _, lx, _, bs = run_path_batch(net, path_op, graph, args)
                    t_total += lx * bs
                    t_count += bs
            test_nmse = t_total / max(t_count, 1)

            # The article's criterion, selecting on test. Preserved for comparability;
            # flagged as such in the metrics. The first epoch is forced to count, since
            # with best_test = inf the relative test degenerates to inf > inf.
            if best_epoch < 0 or (best_test - test_nmse) > abs(best_test * 0.01):
                best_test = test_nmse
                best_epoch = epoch
                patience = 0
                # Checkpoint only; the five-task zero-shot scoring happens once at the
                # end from this checkpoint. See common.train_foundation_model.
                torch.save(
                    {"model": net.state_dict(), "epoch": epoch, "args": vars(args)},
                    os.path.join(run_dir, "model.pth"),
                )
            else:
                patience += 1

            reporter.update(
                epoch=epoch + 1,
                status="training",
                train_loss=train_nmse,
                val_nmse=test_nmse,
                val_task="path",
                best_val_nmse=best_test,
                patience=patience,
            )
            print(
                f"epoch {epoch:3d}  train {train_nmse:.4f}  test[path] {test_nmse:.4f}  "
                f"best {best_test:.4f}  patience {patience}",
                flush=True,
            )

            if patience > args.max_patience:
                print(f"early stop at epoch {epoch}")
                break

    except Exception as exc:  # noqa: BLE001
        reporter.finish(status="failed", error=str(exc))
        raise

    # Zero-shot scoring on the five study tasks, from the selected checkpoint.
    checkpoint = os.path.join(run_dir, "model.pth")
    if os.path.isfile(checkpoint):
        state = torch.load(checkpoint, map_location=args.device, weights_only=False)
        net.load_state_dict(state["model"])
        print(f"\nscoring five tasks from best checkpoint (epoch {state['epoch']})...", flush=True)
    else:
        best_epoch = 0
    best_all = evaluate_all(net, eval_ops, test_loader, args, budget, npg)

    metrics = {
        "run_name": run_name,
        "kind": "original",
        "protocol": 2,
        "dataset": args.dataset,
        "train_tasks": ["path"],
        "val_task": None,
        "test_task": None,
        "selection": "test",  # the article's protocol, not a clean holdout
        "paper_reference_nmse": PAPER_REFERENCE.get(args.dataset),
        "own_task_nmse": best_test,
        "best_epoch": best_epoch,
        "epochs_run": last_epoch,
        "epochs_budget": args.epochs,
        "stopped_early": last_epoch < args.epochs,
        "parameters": n_params,
        "seed": args.seed,
        "batch_size": args.train_batch_size,
        "wall_seconds": round(time.time() - started, 1),
        "per_task": best_all,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    reporter.finish(status="completed", best_val_nmse=best_test)

    if reference is not None:
        print(f"\npath nMSE {best_test:.4f}   (paper Table 5: {reference})")
    else:
        print(f"\npath nMSE {best_test:.4f}   (no published value for {args.dataset})")
    print("zero-shot nMSE on the five study tasks, none of them trained:")
    for task in TASKS:
        print(f"  {TASK_SHORT[task]:<8} {best_all[task]['nmse']:.4f}")


if __name__ == "__main__":
    main()
