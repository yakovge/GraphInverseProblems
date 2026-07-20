"""Correctness checks for the five task operators.

Run directly::

    python test_taskOps.py

The adjoint test is the important one. ``graph_CGLS`` is a Krylov method: it assumes
``adjoint`` is the exact transpose of ``forward``. If it is not, CGLS does not error --
it quietly converges to the wrong thing, and the only symptom is a model that trains
badly for no visible reason. So every operator is checked before anything trains.
"""

import re
import sys

import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import remove_self_loops

from baselines import block_laplacian, solve_regularized
from graphForwardOps import graph_smooth
from tasks import TASKS, build_forward_ops, experiment_split, resample_task, run_name
from taskOps import build_adjacency_list, sample_random_observed, sample_structured_observed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = 20          # nodes per graph
B = 3           # batch size
C = 4           # channels
TOL = 1e-5


class FakeGraph:
    """Minimal stand-in for a PyG batch: the masking samplers only need x and edge_index."""

    def __init__(self, x, edge_index):
        self.x = x
        self.edge_index = edge_index


def build_batch(n=N, b=B, seed=0, avg_degree=7.0):
    """A block-diagonal batch of b identical random graphs, normalised as process_data does.

    ``avg_degree`` defaults to METR-LA's real sparsity (1515 edges / 207 nodes ~ 7.3).
    Density matters: on a dense graph every node is everyone's neighbour and any
    clustering metric saturates at 1.0.
    """
    g = torch.Generator().manual_seed(seed)
    p = min(1.0, avg_degree / max(n - 1, 1))
    dense = (torch.rand(n, n, generator=g) < p).float()
    dense = ((dense + dense.t()) > 0).float()  # undirected
    dense.fill_diagonal_(0)
    single = dense.nonzero().t()

    blocks = [single + i * n for i in range(b)]
    edge_index = torch.cat(blocks, dim=1).to(DEVICE)

    edge_index, _ = remove_self_loops(edge_index)
    edge_index, edge_weight = gcn_norm(edge_index, add_self_loops=True)
    return edge_index, edge_weight


def make_ops(edge_index, learn_emb=False):
    """All five operators sharing one embedding, as the experiments build them."""
    tasks = ["denoising", "inpainting", "source_localization", "sensor_recovery", "pde_state"]
    ops, shared = build_forward_ops(
        tasks,
        nodes_per_graph=N,
        hid_channels=C,
        label_channels=C if not learn_emb else 1,
        learn_emb=learn_emb,
        device=DEVICE,
        blur_count=4,
    )
    return ops, shared


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    return ok


# --------------------------------------------------------------------------------------


def test_adjoints():
    """<Ax, y> == <x, A^T y> for every operator."""
    print("\nAdjoint identity  <Ax,y> == <x,A^T y>")
    edge_index, edge_weight = build_batch()
    graph = FakeGraph(torch.zeros(B * N, C, device=DEVICE), edge_index)
    ops, _ = make_ops(edge_index)
    passed = True

    for task, op in ops.items():
        resample_task(task, graph, op, nodes_per_graph=N, budget=6)

        x = torch.randn(B * N, C, device=DEVICE)
        Ax = op(x, edge_index, edge_weight, emb=False)
        y = torch.randn_like(Ax)
        Aty = op.adjoint(y, edge_index, edge_weight, emb=False)

        lhs = (Ax * y).sum().item()
        rhs = (x * Aty).sum().item()
        rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12)
        passed &= check(task, rel < TOL, f"<Ax,y>={lhs:.6f} <x,Aty>={rhs:.6f} rel={rel:.2e}")

    return passed


def test_block_equivalence():
    """The block-diagonal fast path must match upstream's dense whole-batch operator."""
    print("\nBlock-diagonal fast path == upstream dense graph_smooth")
    edge_index, edge_weight = build_batch()
    ops, _ = make_ops(edge_index)

    dense = graph_smooth(nin=C, embdsize=C, learnEmb=False, device=DEVICE, k=4).to(DEVICE)
    x = torch.randn(B * N, C, device=DEVICE)

    fast = ops["source_localization"](x, edge_index, edge_weight, emb=False)
    slow = dense(x, edge_index, edge_weight, emb=False)

    rel = ((fast - slow).norm() / slow.norm()).item()
    return check("source_localization", rel < TOL, f"relative difference {rel:.2e}")


def test_cdr_is_distinct():
    """CDR must not collapse into the pure-diffusion source-localization operator."""
    print("\nPDE task is genuinely distinct from source localization")
    edge_index, edge_weight = build_batch()
    ops, _ = make_ops(edge_index)
    x = torch.randn(B * N, C, device=DEVICE)

    cdr = ops["pde_state"](x, edge_index, edge_weight, emb=False)
    smooth = ops["source_localization"](x, edge_index, edge_weight, emb=False)
    rel = ((cdr - smooth).norm() / smooth.norm()).item()
    ok = check("CDR differs from P^k", rel > 0.1, f"relative difference {rel:.2f}")

    # The convection term is what keeps the two apart, and it is the piece with no
    # Fourier analogue on a graph -- so verify the generator really is non-symmetric.
    op = ops["pde_state"]._operator(edge_index, edge_weight)
    asym = ((op - op.t()).norm() / op.norm()).item()
    ok &= check("operator is non-symmetric", asym > 1e-3, f"asymmetry {asym:.3f}")
    return ok


def test_masks_differ():
    """Observed sensors must be clustered for sensor_recovery, scattered for inpainting."""
    print("\nStructured vs random masking, at equal budget")
    # Use METR-LA's real scale and sparsity: at 207 nodes with a budget of 16 the
    # observed set is only 7.7% of the graph, which is where the two samplers must
    # actually differ. A small dense test graph saturates any clustering metric.
    n, budget, trials = 207, 16, 20
    edge_index, _ = build_batch(n=n, b=1, seed=1)
    graph = FakeGraph(torch.zeros(n, C, device=DEVICE), edge_index)
    adj = build_adjacency_list(edge_index, n)

    rand_ind = sample_random_observed(graph, budget, n)
    struct_ind = sample_structured_observed(graph, budget, n, adj)

    ok = check(
        "equal observation budget",
        rand_ind.numel() == struct_ind.numel() == budget,
        f"random={rand_ind.numel()} structured={struct_ind.numel()} expected={budget}",
    )

    # Fraction of observed nodes having at least one observed neighbour. Patches grown
    # by BFS should be far more contiguous than a uniform scatter.
    def contiguity(observed):
        idx = observed.cpu().tolist()
        oset = set(idx)
        return sum(any(nb in oset for nb in adj[i]) for i in idx) / len(idx)

    c_rand = sum(contiguity(sample_random_observed(graph, budget, n)) for _ in range(trials)) / trials
    c_struct = (
        sum(contiguity(sample_structured_observed(graph, budget, n, adj)) for _ in range(trials))
        / trials
    )
    # Contiguity is capped at 1.0, so a ratio test is meaningless once the random
    # baseline clears 0.5. Require patches to be near-fully contiguous *and* clearly
    # above the scatter baseline.
    ok &= check(
        "observed sensors are more clustered",
        c_struct > 0.9 and c_struct - c_rand > 0.2,
        f"structured={c_struct:.2f} vs random={c_rand:.2f} (mean of {trials})",
    )
    return ok


def test_no_task_specific_params():
    """The zero-shot claim: no parameter may belong to a single task."""
    print("\nZero task-specific parameters")
    edge_index, _ = build_batch()
    ops, shared = make_ops(edge_index, learn_emb=True)

    ok = check(
        "all operators share one embedding",
        all(op.Emb is shared for op in ops.values()),
        f"{sum(op.Emb is shared for op in ops.values())}/{len(ops)} share it",
    )

    shared_ids = {id(p) for p in shared.parameters()}
    stray = {
        task: [n for n, p in op.named_parameters() if id(p) not in shared_ids]
        for task, op in ops.items()
    }
    offenders = {t: names for t, names in stray.items() if names}
    ok &= check(
        "no operator owns private parameters",
        not offenders,
        f"offenders: {offenders}" if offenders else "",
    )
    return ok


def test_regularized_solver():
    """baselines.solve_regularized must match the direct dense solve."""
    print("\nRegularized CG solver == direct solve of (A^T A + lam R) x = A^T d")
    edge_index, edge_weight = build_batch()
    graph = FakeGraph(torch.zeros(B * N, 1, device=DEVICE), edge_index)
    ops, _ = make_ops(edge_index)
    torch.manual_seed(0)
    passed = True

    m = B * N
    L = block_laplacian(edge_index, edge_weight, N)
    regs = {"tikhonov": (None, torch.eye(m, device=DEVICE)),
            "laplacian": (L, torch.block_diag(*[L] * B))}

    def dense_operator(op):
        """Materialise the batched operator column-by-column, as diagnose_tasks does."""
        cols = []
        for i in range(m):
            e = torch.zeros(m, 1, device=DEVICE)
            e[i] = 1.0
            cols.append(op(e, edge_index, edge_weight, emb=False).squeeze(-1))
        return torch.stack(cols, dim=1)

    for task in ("denoising", "inpainting", "source_localization"):
        op = ops[task]
        resample_task(task, graph, op, nodes_per_graph=N, budget=6)
        A = dense_operator(op)
        d = torch.randn(A.shape[0], 1, device=DEVICE)
        for reg_name, (Lreg, R) in regs.items():
            for lam in (1e-2, 1.0):
                x_cg, rel, _ = solve_regularized(
                    op, d, edge_index, edge_weight, lam, L=Lreg, npg=N
                )
                x_direct = torch.linalg.solve(A.t() @ A + lam * R, A.t() @ d)
                err = ((x_cg - x_direct).norm() / x_direct.norm()).item()
                passed &= check(
                    f"{task}/{reg_name} lam={lam:g}", err < 1e-4,
                    f"rel diff {err:.2e} (cg residual {rel:.2e})",
                )

    # Free analytic cross-check: for the identity operator the Tikhonov optimum is
    # plain shrinkage, x = d / (1 + lam).
    d = torch.randn(m, 1, device=DEVICE)
    lam = 0.5
    x_cg, _, _ = solve_regularized(ops["denoising"], d, edge_index, edge_weight, lam)
    err = ((x_cg - d / (1 + lam)).norm() / d.norm()).item()
    passed &= check("denoising tikhonov == d/(1+lam)", err < 1e-5, f"rel diff {err:.2e}")
    return passed


def test_run_name_seed():
    """Seed suffix round-trips through the compare.py config grouping regex."""
    print("\nSeed-suffixed run names")
    train, val, _ = experiment_split(0)
    base = run_name(train, val)
    ok = check("no suffix without seed", not re.search(r"_s\d+$", base), base)
    named = run_name(train, val, seed=3)
    ok &= check("suffix with seed", named == f"{base}_s3", named)
    # Mirrors compare.config_of; kept literal so this file stays import-light.
    ok &= check("config grouping round-trip", re.sub(r"_s\d+$", "", named) == base)
    return ok


def test_experiment_split_invariants():
    """Every task is val exactly once and test exactly once; train sets are disjoint."""
    print("\nExperiment rotation invariants")
    ok = True
    vals, tests = [], []
    for i in range(5):
        train, val, test = experiment_split(i)
        vals.append(val)
        tests.append(test)
        ok &= check(
            f"split {i} well-formed",
            len(train) == 3 and val != test and val not in train and test not in train,
            f"train={train} val={val} test={test}",
        )
    ok &= check("each task is val exactly once", sorted(vals) == sorted(TASKS))
    ok &= check("each task is test exactly once", sorted(tests) == sorted(TASKS))
    return ok


def main():
    print(f"device = {DEVICE}")
    results = [
        test_adjoints(),
        test_block_equivalence(),
        test_cdr_is_distinct(),
        test_masks_differ(),
        test_no_task_specific_params(),
        test_regularized_solver(),
        test_run_name_seed(),
        test_experiment_split_invariants(),
    ]
    print(f"\n{sum(results)}/{len(results)} groups passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
