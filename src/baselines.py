"""Trivial-predictor baselines for each task.

An nMSE of 0.59 means nothing on its own. These are the reference points that make the
model numbers interpretable -- specifically, they answer "did the learned prior do
anything at all?"

Four baselines per task:

``zero``
    Predict 0 everywhere. By construction nMSE = 1.0; a model at or above this has
    learned nothing useful.
``mean``
    Predict the training-set mean. The bar any regressor must clear.
``adjoint``
    ``F^T(d_obs)``, rescaled. Uses the operator but no prior at all.
``pinv``
    Least-squares fit via CGLS with no learned regularizer -- exactly what the
    foundation models' data-fit step computes before the GNN contributes anything.
    **This is the important one**: a model that fails to beat it has added nothing over
    the parameter-free solve it is built on.

    python baselines.py --dataset CPOX
"""

import argparse
import csv
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "foundation"))


@torch.no_grad()
def run_baselines(args, ops, loader, tasks, budget, npg, train_mean):
    from common import nmse, observe
    from networks import graph_CGLS
    from tasks import resample_task
    from utils import process_data

    totals = {t: {k: 0.0 for k in ("zero", "mean", "adjoint", "pinv")} for t in tasks}
    counts = {t: 0 for t in tasks}

    for graph in loader:
        # process_data mutates in place (it overwrites graph.x and graph.y), so it must
        # run exactly once per batch -- calling it per task corrupts the second call.
        g = process_data(args, graph).to(args.device)
        bs = len(g.batch.unique())

        for task in tasks:
            op = ops[task]
            resample_task(task, g, op, nodes_per_graph=npg, budget=budget)

            d_obs = observe(op, g.y, g)
            y = g.y

            totals[task]["zero"] += nmse(torch.zeros_like(y), y).item() * bs
            totals[task]["mean"] += nmse(torch.full_like(y, train_mean), y).item() * bs

            # Adjoint back-projection, scaled to best match the target. Without the
            # scaling the comparison would be dominated by an arbitrary magnitude.
            adj = op.adjoint(d_obs, g.edge_index, g.edge_weight, emb=False)
            denom = (adj * adj).sum()
            scale = ((adj * y).sum() / denom) if denom > 0 else 0.0
            totals[task]["adjoint"] += nmse(scale * adj, y).item() * bs

            # Unregularised least squares -- the data-fit step with no learned prior.
            cgls = graph_CGLS(op, CGLSit=args.cglsIter, eps=1e-5)
            x0 = torch.zeros_like(y)
            x_ls, _ = cgls(d_obs, x0, g.edge_index, g.edge_weight, emb=False)
            totals[task]["pinv"] += nmse(x_ls, y).item() * bs

            counts[task] += bs

    return {
        t: {k: v / max(counts[t], 1) for k, v in totals[t].items()} for t in tasks
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="CPOX", choices=["METRLA", "CPOX"])
    p.add_argument("--out", default=None)
    cli = p.parse_args()

    sys.argv = ["baselines", "--dataset", cli.dataset]
    from config import parse
    from tasks import TASK_SHORT, TASKS, build_forward_ops, mask_budget_for, nodes_per_graph_for
    from utils import get_data_and_loaders

    args = parse("baselines")
    npg = nodes_per_graph_for(args.dataset)
    budget = mask_budget_for(args.dataset)

    train_ds, _, train_loader, test_loader, lc, fc = get_data_and_loaders(args)

    # Training-set mean, for the mean predictor. Computed on train only -- using the
    # test set would give the baseline information the models never had.
    total, n = 0.0, 0
    from utils import process_data

    for g in train_loader:
        g = process_data(args, g)
        total += g.y.sum().item()
        n += g.y.numel()
    train_mean = total / max(n, 1)

    ops, _ = build_forward_ops(
        TASKS, nodes_per_graph=npg, hid_channels=args.channels, label_channels=lc,
        learn_emb=False, device=args.device, blur_count=args.blur_count,
        noise_sigma=args.noise_sigma,
    )

    print(f"dataset {args.dataset}   train mean {train_mean:.4f}")
    print(f"computing baselines over the test snapshots...\n", flush=True)
    results = run_baselines(args, ops, test_loader, TASKS, budget, npg, train_mean)

    print(f"  {'task':<9} {'zero':>9} {'mean':>9} {'adjoint':>9} {'pinv':>9}")
    print("  " + "-" * 50)
    for t in TASKS:
        r = results[t]
        print(f"  {TASK_SHORT[t]:<9} {r['zero']:>9.4f} {r['mean']:>9.4f} "
              f"{r['adjoint']:>9.4f} {r['pinv']:>9.4f}")

    print("\n  pinv = unregularised least squares (the data-fit step with no prior).")
    print("  A trained model must beat pinv to have contributed anything.")

    out_dir = cli.out or os.path.join(REPO, "results", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "baselines.json"), "w") as fh:
        json.dump({"dataset": args.dataset, "train_mean": train_mean,
                   "per_task": results}, fh, indent=2)

    csv_path = os.path.join(out_dir, "baselines.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "zero", "mean", "adjoint", "pinv"])
        for t in TASKS:
            r = results[t]
            w.writerow([TASK_SHORT[t]] + [round(r[k], 6) for k in
                                          ("zero", "mean", "adjoint", "pinv")])
        w.writerow([])
        for line in [
            "Trivial predictors, for interpreting the model nMSE values.",
            "zero = predict 0 (nMSE 1.0 by construction); mean = predict the training mean.",
            "adjoint = F^T(d_obs), optimally rescaled; uses the operator but no prior.",
            "pinv = unregularised least squares via CGLS -- the data-fit step with no",
            "       learned regularizer. A model that does not beat pinv added nothing.",
        ]:
            w.writerow([f"# {line}"])
    print(f"\n  wrote {csv_path}")


if __name__ == "__main__":
    main()
