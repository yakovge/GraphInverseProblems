"""Forward operators for the foundation-model (cross-operator transfer) study.

These extend the operators in graphForwardOps.py with:
  - graphIdentity   : denoising (A = I; corrupt() adds Gaussian noise)
  - graphMaskSmooth : PDE-state reconstruction holdout — observe a node subset of
                      the k-step diffused state (composition mask ∘ smooth)
  - graphMaxPool    : nonlinear far-transfer holdout — y_i = max_{j in N(i)∪{i}} x_j;
                      the "adjoint" is the vector-Jacobian product at the last
                      forward linearization point (same stash pattern as
                      graph_edgeRecovery / contactMap use for their autodiff adjoints)

All operators follow the existing interface:
    forward(I, edge_index, edge_weights, emb=False)
    adjoint(r, edge_index, edge_weights, emb=False)
plus corrupt(I, ...) used only for observation generation (adds noise where the
task calls for it; defaults to forward()).

The factory make_foundation_op() also wraps the upstream operators
(graph_smooth, graphMask, graphPath) so one code path builds any operator in the
pretraining mixture or the holdout set. All ops are built with learnEmb=False:
the training loop always calls them with emb=False, and parameter-free operators
can be swapped per batch without touching the optimizer.
"""
import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from graphForwardOps import graph_smooth, graphMask, graphPath

# Operator-type ids for the optional conditioning embedding. UNKNOWN is reserved
# for held-out operators so the model is never handed an untrained category.
OP_TYPE_IDS = {"denoise": 0, "mask": 1, "smooth": 2, "path": 3, "UNKNOWN": 4}
NUM_OP_TYPES = len(OP_TYPE_IDS)


class graphIdentity(nn.Module):
    """Denoising: A = I. The observation is y + sigma * noise (corrupt only)."""

    op_name = "denoise"

    def __init__(self, sigma=0.3):
        super().__init__()
        self.sigma = sigma

    def forward(self, I, edge_index=None, edge_weights=None, emb=False):
        return I

    def adjoint(self, r, edge_index=None, edge_weights=None, emb=False):
        return r

    def corrupt(self, I, edge_index=None, edge_weights=None):
        return I + self.sigma * torch.randn_like(I)


class graphMaskSmooth(nn.Module):
    """PDE-state reconstruction: observe a subset of nodes of the diffused state.

    forward = graphMask ∘ graph_smooth (A^k diffusion then node subsampling);
    adjoint = graph_smooth.adjoint ∘ graphMask.adjoint. Composed from the two
    upstream pieces so the professor's operator code is reused verbatim.
    `ind` is resampled per batch by the training loop (same protocol as 'mask').
    """

    op_name = "maskSmooth"

    def __init__(self, nin, embdsize, device="cuda", k=4, ind=None):
        super().__init__()
        self.smooth = graph_smooth(nin=nin, embdsize=embdsize, learnEmb=False,
                                   device=device, k=k)
        self.ind = ind if ind is not None else torch.arange(25, device=device)
        self.k = k

    def forward(self, I, edge_index=None, edge_weights=None, emb=False):
        S = self.smooth.forward(I, edge_index, edge_weights, emb=False)
        return S[self.ind, :]

    def adjoint(self, r, edge_index=None, edge_weights=None, emb=False):
        nnodes = int(edge_index.max()) + 1
        full = torch.zeros(nnodes, r.shape[1], device=r.device, dtype=r.dtype)
        full[self.ind, :] = r
        return self.smooth.adjoint(full, edge_index, edge_weights, emb=False)

    def corrupt(self, I, edge_index=None, edge_weights=None):
        return self.forward(I, edge_index, edge_weights, emb=False)


class graphMaxPool(nn.Module):
    """Nonlinear far-transfer operator: y_i = max over N(i) ∪ {i} of x_j (per channel).

    There is no linear adjoint. adjoint(r) returns the vector-Jacobian product
    J(x0)^T r where x0 is the linearization point stashed by the most recent
    forward() call (falls back to r itself before any forward call, e.g. for the
    initial back-projection in the solvers). Wrapped in enable_grad so it also
    works inside eval/no_grad contexts.
    """

    op_name = "maxpool"

    def __init__(self):
        super().__init__()
        self._point = None  # linearization point for the VJP adjoint

    def _pool(self, I, edge_index):
        n = I.shape[0]
        loops = torch.arange(n, device=edge_index.device).unsqueeze(0).repeat(2, 1)
        ei = torch.cat([edge_index, loops], dim=1)  # ensure self in neighborhood
        return scatter(I[ei[0]], ei[1], dim=0, dim_size=n, reduce="max")

    def forward(self, I, edge_index=None, edge_weights=None, emb=False):
        self._point = I.detach()
        return self._pool(I, edge_index)

    def adjoint(self, r, edge_index=None, edge_weights=None, emb=False):
        point = self._point if (self._point is not None
                                and self._point.shape == r.shape) else r.detach()
        keep_graph = torch.is_grad_enabled() and r.requires_grad
        with torch.enable_grad():
            x = point.detach().requires_grad_(True)
            y = self._pool(x, edge_index)
            (g,) = torch.autograd.grad(y, x, grad_outputs=r,
                                       create_graph=keep_graph)
        return g if keep_graph else g.detach()

    def corrupt(self, I, edge_index=None, edge_weights=None):
        # Do NOT leave the ground truth as the linearization point: observation
        # generation must not leak y into the solver's adjoint (the solver may
        # only linearize at its own iterates or at the observation).
        out = self.forward(I, edge_index, edge_weights, emb=False)
        self._point = None
        return out


def make_foundation_op(name, args, label_channels, hid_channels, device):
    """Build any operator (upstream or new) by name, parameter-free (learnEmb=False)."""
    if name == "denoise":
        return graphIdentity(sigma=getattr(args, "denoise_sigma", 0.3))
    if name == "mask":
        return graphMask(ind=torch.arange(25), embdsize=hid_channels,
                         nin=label_channels, device=device, learnEmb=False)
    if name == "smooth":
        return graph_smooth(nin=label_channels, embdsize=hid_channels,
                            learnEmb=False, device=device, k=int(args.blur_count))
    if name == "path":
        return graphPath(embdsize=hid_channels, nin=label_channels,
                         learnEmb=False, device=device, pathLength=args.pathLength)
    if name == "maskSmooth":
        return graphMaskSmooth(nin=label_channels, embdsize=hid_channels,
                               device=device, k=int(args.blur_count))
    if name == "maxpool":
        return graphMaxPool()
    raise ValueError(f"unknown operator name: {name}")


def op_type_id(name):
    """Conditioning type id; held-out operators map to UNKNOWN."""
    return OP_TYPE_IDS.get(name, OP_TYPE_IDS["UNKNOWN"])


def sample_mask_indices(graph, args, batch_size):
    """Observed-node sampling for mask-type operators.

    Mirrors utils.task_specific_modifiers: class-stratified for classification
    datasets (per_class_budget nodes per class), per-snapshot for regression
    datasets (e.g. METR-LA sensor recovery), but uses the actual number of
    graphs in the batch instead of args.train_batch_size.
    """
    if args.classify:
        n_classes = graph.y.shape[-1]
        budget = int(batch_size * args.mask_per_class_budget)
        mask_indices = []
        for c in range(n_classes):
            class_ind = torch.where(graph.y.argmax(dim=-1) == c)[0]
            perm = torch.randperm(len(class_ind))[:budget]
            mask_indices += list(class_ind[perm])
    else:
        mask_indices = []
        for b in range(batch_size):
            snap_ind = torch.where(graph.batch == b)[0]
            perm = torch.randperm(len(snap_ind))[:args.mask_per_snapshot_budget]
            mask_indices += list(snap_ind[perm])
    return torch.tensor(mask_indices, device=graph.y.device).long()


def apply_batch_modifiers(graph, op, args):
    """Per-batch operator state: resample mask indices / regenerate paths."""
    batch_size = len(graph.batch.unique()) if hasattr(graph, "batch") else 1
    name = getattr(op, "op_name", None)
    if isinstance(op, graphMask) or name == "maskSmooth":
        op.ind = sample_mask_indices(graph, args, batch_size)
    elif isinstance(op, graphPath):
        op.gen_paths(nnodes=graph.x.shape[0], edge_index=graph.edge_index)
    return op
