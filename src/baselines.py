"""Trivial-predictor and classical-regularization baselines for each task.

An nMSE of 0.59 means nothing on its own. These are the reference points that make the
model numbers interpretable -- specifically, they answer "did the learned prior do
anything at all?"

Six baselines per task:

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
``tikhonov`` / ``laplacian``
    Classical regularized least squares, min ||Fx-d||^2 + lam*||x||^2 and
    min ||Fx-d||^2 + lam*x^T L x (the paper's own baseline family). The lambda is
    **oracle-tuned**: swept over LAMBDA_GRID and chosen per task on the aggregate test
    nMSE. Deliberately optimistic -- a ceiling for what classical, non-learned
    regularization can do. **These are the important ones**: a learned prior that fails
    to beat the best of these has added nothing a textbook method would not supply.

Note the grid is strictly positive: the lam -> 0 limit is not included because CG on the
normal equations semi-converges on the ill-conditioned tasks (it starts fitting noise),
and the "no penalty" point is already represented -- more honestly -- by ``pinv``, whose
truncated CGLS iterations regularize implicitly. Consequently tikhonov/laplacian are NOT
guaranteed to beat pinv on every task; compare.py takes the min over all baselines.

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

# Strictly positive (see module docstring for why there is no lam=0 point).
LAMBDA_GRID = [10.0 ** e for e in range(-6, 4)]

REG_NAMES = ("tikhonov", "laplacian")


def block_laplacian(edge_index, edge_weight, n):
    """Dense [n, n] unnormalized weighted Laplacian of batch block 0.

    Both datasets are static graphs, so block 0 is every block; built once per run.
    """
    from networks import compute_weighted_laplacian

    mask = (edge_index[0] < n) & (edge_index[1] < n)
    return compute_weighted_laplacian(edge_index[:, mask], edge_weight[mask], n)


def apply_blockwise(mat, x, n):
    """Apply a single-graph [n, n] matrix to a batched [B*n, C] signal block-wise."""
    b = x.shape[0] // n
    return torch.einsum("ij,bjc->bic", mat, x.reshape(b, n, -1)).reshape(x.shape)


def solve_regularized(op, d_obs, edge_index, edge_weight, lam, L=None, npg=None,
                      iters=300, tol=1e-6):
    """CG for the regularized normal equations (A^T A + lam*R) x = A^T d.

    Matrix-free via the operator's own forward/adjoint (emb=False), which is what
    handles per-graph masks exactly. R = I when L is None, else block-wise L. Returns
    ``(x, rel_residual, converged)`` -- convergence metadata is reported so a bad
    laplacian score can be told apart from a stagnating solve (L is singular on
    constants, so at large lam the masking tasks have a near-nullspace CG can wander).
    """

    def matvec(p):
        Ap = op(p, edge_index, edge_weight, emb=False)
        AtAp = op.adjoint(Ap, edge_index, edge_weight, emb=False)
        reg = p if L is None else apply_blockwise(L, p, npg)
        return AtAp + lam * reg

    b = op.adjoint(d_obs, edge_index, edge_weight, emb=False)
    b_norm = b.norm()
    x = torch.zeros_like(b)
    if b_norm == 0:
        return x, 0.0, True

    r = b.clone()
    p = r.clone()
    rs = (r * r).sum()
    rel = 1.0
    for _ in range(iters):
        Mp = matvec(p)
        denom = (p * Mp).sum()
        if denom <= 0:  # numerically semi-definite direction: stop rather than diverge
            return x, rel, False
        alpha = rs / denom
        x = x + alpha * p
        r = r - alpha * Mp
        rs_new = (r * r).sum()
        rel = (rs_new.sqrt() / b_norm).item()
        if rel < tol:
            return x, rel, True
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, rel, False


@torch.no_grad()
def run_baselines(args, ops, loader, tasks, budget, npg, train_mean):
    from common import nmse, observe
    from networks import graph_CGLS
    from tasks import resample_task
    from utils import process_data

    totals = {t: {k: 0.0 for k in ("zero", "mean", "adjoint", "pinv")} for t in tasks}
    counts = {t: 0 for t in tasks}
    reg_sums = {
        t: {r: {lam: 0.0 for lam in LAMBDA_GRID} for r in REG_NAMES} for t in tasks
    }
    reg_resid = {
        t: {r: {lam: 0.0 for lam in LAMBDA_GRID} for r in REG_NAMES} for t in tasks
    }
    reg_noconv = {
        t: {r: {lam: 0 for lam in LAMBDA_GRID} for r in REG_NAMES} for t in tasks
    }
    L = None

    for graph in loader:
        # process_data mutates in place (it overwrites graph.x and graph.y), so it must
        # run exactly once per batch -- calling it per task corrupts the second call.
        g = process_data(args, graph).to(args.device)
        bs = len(g.batch.unique())
        if L is None:  # static graphs: block 0 of the first batch is every block
            L = block_laplacian(g.edge_index, g.edge_weight, npg)

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

            # Classical regularized least squares over the lambda grid, reusing the SAME
            # d_obs (noise and masks identical across lambdas and vs pinv/adjoint).
            for reg in REG_NAMES:
                Lreg = None if reg == "tikhonov" else L
                for lam in LAMBDA_GRID:
                    x_reg, rel, ok = solve_regularized(
                        op, d_obs, g.edge_index, g.edge_weight, lam, L=Lreg, npg=npg
                    )
                    reg_sums[task][reg][lam] += nmse(x_reg, y).item() * bs
                    reg_resid[task][reg][lam] = max(reg_resid[task][reg][lam], rel)
                    if not ok:
                        reg_noconv[task][reg][lam] += 1

            counts[task] += bs

    results = {
        t: {k: v / max(counts[t], 1) for k, v in totals[t].items()} for t in tasks
    }

    # Oracle lambda: ONE value per (task, regularizer), chosen on the aggregate test
    # nMSE -- the strongest classical setting, deliberately.
    reg_info = {}
    for t in tasks:
        reg_info[t] = {}
        for reg in REG_NAMES:
            curve = {lam: reg_sums[t][reg][lam] / max(counts[t], 1) for lam in LAMBDA_GRID}
            best_lam = min(curve, key=curve.get)
            results[t][reg] = curve[best_lam]
            # A grid-edge argmin only matters if the curve is still descending there,
            # and the two edges mean different things. At the LOW edge the curve is
            # descending into the lam -> 0 unregularized limit, which pinv already
            # represents -- no wider grid can beat it. Only the HIGH edge suggests a
            # better lambda outside the grid.
            neighbour = (
                LAMBDA_GRID[1] if best_lam == LAMBDA_GRID[0] else LAMBDA_GRID[-2]
            )
            still_descending = (
                best_lam in (LAMBDA_GRID[0], LAMBDA_GRID[-1])
                and (curve[neighbour] - curve[best_lam]) > 0.005 * curve[best_lam]
            )
            edge = None
            if still_descending:
                edge = "low" if best_lam == LAMBDA_GRID[0] else "high"
            reg_info[t][reg] = {
                "lambda": best_lam,
                "nmse": curve[best_lam],
                "on_grid_edge": edge,
                "curve": {f"{lam:g}": v for lam, v in curve.items()},
                "residual_max": {f"{lam:g}": reg_resid[t][reg][lam] for lam in LAMBDA_GRID},
                "unconverged_batches": {
                    f"{lam:g}": reg_noconv[t][reg][lam] for lam in LAMBDA_GRID
                },
            }
    return results, reg_info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="CPOX", choices=["METRLA", "CPOX"])
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    cli = p.parse_args()

    # The masking tasks redraw their observed sets per batch and denoising redraws its
    # noise; fix the seed so the published baseline numbers are reproducible.
    torch.manual_seed(cli.seed)

    sys.argv = ["baselines", "--dataset", cli.dataset, "--device", cli.device]
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
    results, reg_info = run_baselines(args, ops, test_loader, TASKS, budget, npg, train_mean)

    cols = ("zero", "mean", "adjoint", "pinv") + REG_NAMES
    print("  " + f"{'task':<9}" + "".join(f"{c:>10}" for c in cols))
    print("  " + "-" * (9 + 10 * len(cols)))
    for t in TASKS:
        r = results[t]
        print(f"  {TASK_SHORT[t]:<9}" + "".join(f"{r[c]:>10.4f}" for c in cols))

    print("\n  pinv = unregularised least squares (the data-fit step with no prior).")
    print("  tikhonov/laplacian = oracle-tuned classical regularization (lambda chosen")
    print("  per task on the test nMSE). A learned prior must beat the best of these")
    print("  to have contributed anything a textbook method would not.")

    for t in TASKS:
        for reg in REG_NAMES:
            info = reg_info[t][reg]
            lam = info["lambda"]
            if info["on_grid_edge"] == "high":
                print(f"  WARNING {TASK_SHORT[t]}/{reg}: chosen lambda {lam:g} sits on "
                      f"the upper grid edge -- widen LAMBDA_GRID upward.")
            elif info["on_grid_edge"] == "low":
                print(f"  note {TASK_SHORT[t]}/{reg}: optimum is the lam->0 "
                      f"(unregularized) limit; pinv represents it in best-classical.")
            hits = info["unconverged_batches"][f"{lam:g}"]
            if hits:
                print(f"  WARNING {TASK_SHORT[t]}/{reg}: CG hit the iteration cap on "
                      f"{hits} batch(es) at the chosen lambda {lam:g} "
                      f"(worst residual {info['residual_max'][f'{lam:g}']:.2e}) -- "
                      f"treat this score as a numerical floor, not the method's best.")

    out_dir = cli.out or os.path.join(REPO, "results", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "baselines.json"), "w") as fh:
        json.dump(
            {
                "dataset": args.dataset,
                "train_mean": train_mean,
                "lambda_grid": LAMBDA_GRID,
                "per_task": results,
                "regularized": reg_info,
            },
            fh,
            indent=2,
        )

    csv_path = os.path.join(out_dir, "baselines.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task"] + list(cols))
        for t in TASKS:
            r = results[t]
            w.writerow([TASK_SHORT[t]] + [round(r[k], 6) for k in cols])
        w.writerow([])
        for line in [
            "Trivial predictors and classical ceilings, for interpreting model nMSE.",
            "zero = predict 0 (nMSE 1.0 by construction); mean = predict the training mean.",
            "adjoint = F^T(d_obs), optimally rescaled; uses the operator but no prior.",
            "pinv = unregularised least squares via CGLS -- the data-fit step with no",
            "       learned regularizer.",
            "tikhonov / laplacian = classical regularized least squares with the lambda",
            "       ORACLE-TUNED per task on the test nMSE -- an optimistic ceiling for",
            "       non-learned regularization, deliberately. A model that does not beat",
            "       the best classical baseline added nothing a textbook method lacks.",
        ]:
            w.writerow([f"# {line}"])
    print(f"\n  wrote {csv_path}")


if __name__ == "__main__":
    main()
