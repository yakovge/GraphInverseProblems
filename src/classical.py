"""Classical regularised least-squares baselines with a validation-tuned weight (shared by plot_fixed.py / plot_2.py).

    x_lam = argmin_x ||A x - b||^2 + lam * x^T R x,   R = I (Tikhonov)  or  R = L = I - A_hat (graph Laplacian)

solved by conjugate gradients on the normal equations (A^T A + lam R) x = A^T b from x = 0. For every task and
regulariser, (lam, number of CG steps) is chosen from LAMBDAS x ITERS by the lowest mean nMSE on a validation split
that the learned models also used for checkpoint selection (never on the test split). The CG budget is tuned
because on the ill-conditioned operators an early-stopped CG run is itself a regulariser (the plain least-squares
row, 25 CGLS steps, is one), so lam -> 0 with few steps has to be a candidate. These are the "meaningful classical
baselines" of the course brief: a tuned prior with no learning, against which the learned prior has to be measured.

Also here: `measure`, the one place where a measurement b = A x (+ noise) is produced for evaluation. The Gaussian
noise is drawn on the CPU generator (then moved), so the evaluation tables are identical on CPU and GPU; the GPU
run of 2026-10-21 drew it on the device and its tables differ from the CPU ones in the third decimal.
"""
import torch
import torch.nn.functional as F

LAMBDAS = [0.0] + [10 ** (e / 2) for e in range(-12, 5)]  # 0, 1e-6, 3e-6, ..., 30, 100
ITERS = [5, 25, 100]
REGULARISERS = ['tikhonov', 'laplacian']
ROW_OF = {'tikhonov': 'BASELINE: Tikhonov LS (lam, CG steps tuned on val)',
          'laplacian': 'BASELINE: Laplacian LS (lam, CG steps tuned on val)'}


def nmse(X, y):
    return (F.mse_loss(X, y) / F.mse_loss(torch.zeros_like(y), y)).item()


def measure(op, g):
    """b = A y (+ noise). Noise comes from the (seeded) CPU generator so tables do not depend on the device."""
    b = op(g.y, g.edge_index, g.edge_weight, emb=False)
    if hasattr(op, 'corrupt'):
        b = b + op.noise_std * torch.randn(b.shape).to(b.device)
    return b


def laplacian_apply(x, edge_index, edge_weight):
    """L x = x - A_hat x with A_hat the gcn-normalised adjacency (self loops included, symmetric: graphs are undirected)."""
    n = x.shape[0]
    A = torch.sparse_coo_tensor(edge_index, edge_weight, (n, n))
    return x - torch.sparse.mm(A, x)


def regularised_ls(op, b, edge_index, edge_weight, lam, reg, iters, snapshots=None):
    """CG on (A^T A + lam R) x = A^T b, x0 = 0, `iters` steps. With snapshots = [k1, k2, ...] (sorted) returns the
    iterates after k1, k2, ... steps instead (one run serves every iteration budget)."""
    A = lambda v: op(v, edge_index, edge_weight, emb=False)
    At = lambda v: op.adjoint(v, edge_index, edge_weight, emb=False)
    R = (lambda v: v) if reg == 'tikhonov' else (lambda v: laplacian_apply(v, edge_index, edge_weight))
    M = lambda v: At(A(v)) + lam * R(v)
    x = torch.zeros_like(At(b))
    r = At(b) - M(x)
    p, rr = r, (r * r).sum()
    out = []
    for k in range(1, iters + 1):
        if rr.sqrt() > 1e-9:
            q = M(p)
            alpha = rr / ((p * q).sum() + 1e-12)
            x, r = x + alpha * p, r - alpha * q
            rr_new = (r * r).sum()
            p, rr = r + rr_new / (rr + 1e-12) * p, rr_new
        if snapshots is not None and k in snapshots:
            out.append(x.clone())
    return out if snapshots is not None else x


@torch.no_grad()
def tune_lambdas(batches, log=print):
    """batches: iterable of (task, g, b, op) validation measurements (already on the device).
    -> {reg: {task: (lam, iters)}} minimising the mean validation nMSE, and the curves {reg: {task: {(lam, iters): nMSE}}}."""
    sums = {reg: {} for reg in REGULARISERS}
    count = {}
    for task, g, b, op in batches:
        count[task] = count.get(task, 0) + 1
        for reg in REGULARISERS:
            d = sums[reg].setdefault(task, {(lam, it): 0.0 for lam in LAMBDAS for it in ITERS})
            for lam in LAMBDAS:
                for it, x in zip(ITERS, regularised_ls(op, b, g.edge_index, g.edge_weight, lam, reg, max(ITERS), snapshots=ITERS)):
                    d[(lam, it)] += nmse(x, g.y)
    curves = {reg: {t: {k: v / count[t] for k, v in d.items()} for t, d in sums[reg].items()} for reg in REGULARISERS}
    best = {reg: {t: min(c, key=c.get) for t, c in curves[reg].items()} for reg in REGULARISERS}
    for reg in REGULARISERS:
        log(f"  tuned {reg:9s}: " + "  ".join(f"{t[:8]} lam={best[reg][t][0]:g} it={best[reg][t][1]} (val {curves[reg][t][best[reg][t]]:.4f})" for t in best[reg]))
    return best, curves
