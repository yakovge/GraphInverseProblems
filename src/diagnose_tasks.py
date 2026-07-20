"""Measure how ill-posed each task's forward operator actually is.

An inverse problem is only interesting if the forward operator destroys information. If
F is close to the identity, recovering x from F(x) is trivial and the task tells us
nothing about the learned prior -- so the five tasks have to be checked and calibrated
against each other before any of the comparison numbers mean anything.

Reported per operator, on the real graph:

* ``||Fx - x|| / ||x||``   how far F is from the identity (0 = trivial task)
* ``sigma_min / sigma_max`` conditioning; small = ill-posed
* ``rank(tol)``            effective rank, i.e. how many directions survive
* ``energy kept``          ||Fx|| / ||x||

Run:  python diagnose_tasks.py --dataset METRLA
"""

import argparse
import os

import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import remove_self_loops

from tasks import TASK_SHORT, build_forward_ops, mask_budget_for, nodes_per_graph_for, resample_task

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


class FakeGraph:
    def __init__(self, x, edge_index, edge_weight):
        self.x = x
        self.edge_index = edge_index
        self.edge_weight = edge_weight


def real_graph(dataset, datapath, device):
    """Block-0 graph of the real dataset, normalised exactly as process_data does."""
    if dataset == "METRLA":
        from customMETRLA import METRLADatasetLoader

        path = os.path.join(datapath, "temporal_data", "METRLA")
        ds = METRLADatasetLoader(raw_data_dir=path).get_dataset(1, 0)
    else:
        from customCPOX import ChickenpoxDatasetLoader

        cache = os.path.join(datapath, "temporal_data", "CPOX", "chickenpox.json")
        ds = ChickenpoxDatasetLoader(cache_path=cache).get_dataset(lags=1)

    g = next(iter(ds))
    ei, _ = remove_self_loops(g.edge_index)
    ei, ew = gcn_norm(ei, add_self_loops=True)
    return ei.to(device), ew.to(device)


def dense_matrix(op, edge_index, edge_weight, n, device):
    """Materialise the operator by applying it to the identity, column by column."""
    cols = []
    for i in range(n):
        e = torch.zeros(n, 1, device=device)
        e[i] = 1.0
        cols.append(op(e, edge_index, edge_weight, emb=False).squeeze(-1))
    return torch.stack(cols, dim=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="METRLA", choices=["METRLA", "CPOX"])
    p.add_argument("--datapath", default=os.path.join(REPO, "data"))
    p.add_argument("--blur_count", type=int, default=16)
    p.add_argument("--tau", type=float, default=None, help="override CDR integration time")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = nodes_per_graph_for(args.dataset)
    budget = mask_budget_for(args.dataset)
    ei, ew = real_graph(args.dataset, args.datapath, device)

    cdr = {"tau": args.tau} if args.tau is not None else {}
    tasks = ["denoising", "inpainting", "source_localization", "sensor_recovery", "pde_state"]
    ops, _ = build_forward_ops(
        tasks,
        nodes_per_graph=n,
        hid_channels=32,
        label_channels=1,
        learn_emb=False,
        device=device,
        blur_count=args.blur_count,
        cdr=cdr,
    )

    graph = FakeGraph(torch.zeros(n, 1, device=device), ei, ew)
    x = torch.randn(n, 1, device=device)

    print(f"dataset={args.dataset}  nodes={n}  mask budget={budget}  k={args.blur_count}")
    if args.tau is not None:
        print(f"CDR tau override = {args.tau}")
    print()
    print(f"  {'task':<9} {'||Fx-x||/||x||':>15} {'cond':>12} {'rank':>8} {'energy kept':>12}")
    print("  " + "-" * 60)

    for task in tasks:
        resample_task(task, graph, ops[task], nodes_per_graph=n, budget=budget)
        op = ops[task]

        Fx = op(x, ei, ew, emb=False)
        energy = (Fx.norm() / x.norm()).item()
        # Masking changes the output dimension, so identity-distance only applies
        # where the shapes match.
        deviation = ((Fx - x).norm() / x.norm()).item() if Fx.shape == x.shape else float("nan")

        A = dense_matrix(op, ei, ew, n, device)
        sv = torch.linalg.svdvals(A.float())
        cond = (sv.min() / sv.max()).item() if sv.max() > 0 else 0.0
        rank = int((sv > sv.max() * 1e-6).sum())

        dev_str = "  n/a (masked)" if deviation != deviation else f"{deviation:>15.4f}"
        print(
            f"  {TASK_SHORT[task]:<9} {dev_str} {cond:>12.2e} {rank:>4}/{n:<3} {energy:>12.4f}"
        )

    print()
    print("  A task with deviation ~0 and rank ~n is near-invertible, i.e. trivial.")
    print("  Ill-posed tasks show small conditioning and/or reduced rank.")


if __name__ == "__main__":
    main()
