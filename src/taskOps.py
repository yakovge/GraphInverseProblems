"""Forward operators for the five graph inverse tasks of the foundation-model study.

Every operator implements the contract already used by ``graph_CGLS`` and
``graph_inverseSolveNet``::

    forward(I, edge_index, edge_weight, emb=True)  -> observed data
    adjoint(Ic, edge_index, edge_weight, emb=True) -> back-projection

so the existing networks consume them unchanged.

Two properties matter for the foundation-model experiment:

1. **No task-specific parameters.** The only parameters an operator touches live in
   ``self.Emb``, and every task is handed the *same* shared ``graphEmbed`` instance
   (see ``tasks.build_forward_ops``). A held-out task therefore runs with weights that
   were fully trained by the other tasks -- which is what makes zero-shot meaningful.

2. **Static-graph block structure.** METR-LA and CPOX are static graphs: a batch of B
   snapshots is a block-diagonal graph of B identical N x N blocks. The batched operator
   is exactly ``I_B (x) op_single``, so we build the single N x N block once and apply it
   block-wise. Upstream's ``graph_smooth`` instead densifies the *whole batch*
   (B*N x B*N); at the paper's METR-LA batch size of 128 that is a 26496^2 matrix, ~2.8 GB,
   which does not fit on a 6 GB card. Block-wise it is 207^2, ~171 KB.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import get_laplacian

from graphForwardOps import graphEmbed, graphMask, graph_smooth  # noqa: F401 (re-exported)


# --------------------------------------------------------------------------------------
# Static-graph block machinery
# --------------------------------------------------------------------------------------


class BlockOperatorMixin(nn.Module):
    """Applies a single-graph N x N operator across a block-diagonal batch.

    PyG batches disjoint graphs by offsetting node indices, so for a static-graph
    dataset the batched adjacency is block diagonal with identical blocks. We extract
    block 0 (the edges whose endpoints both fall in ``[0, nodes_per_graph)``), build the
    dense operator there once, and reuse it for every batch.
    """

    def __init__(self, nodes_per_graph):
        super().__init__()
        self.nodes_per_graph = nodes_per_graph
        self._cached_op = None
        self._cache_key = None

    def _block_adjacency(self, edge_index, edge_weight, n):
        """Dense [n, n] adjacency of the first block. A[src, dst] = w, as upstream."""
        mask = (edge_index[0] < n) & (edge_index[1] < n)
        ei, ew = edge_index[:, mask], edge_weight[mask]
        A = torch.zeros(n, n, device=edge_weight.device, dtype=edge_weight.dtype)
        A[ei[0], ei[1]] = ew
        return A

    def _build_operator(self, edge_index, edge_weight, n):
        """Return the dense [n, n] single-graph operator. Subclasses implement this."""
        raise NotImplementedError

    def _operator(self, edge_index, edge_weight):
        n = self.nodes_per_graph
        # The graph is static, so the operator only needs rebuilding if the device or
        # dtype changes. Keying on those keeps the cache correct without rebuilding
        # every batch.
        key = (n, edge_weight.device, edge_weight.dtype)
        if self._cached_op is None or self._cache_key != key:
            self._cached_op = self._build_operator(edge_index, edge_weight, n)
            self._cache_key = key
        return self._cached_op

    def _apply_blockwise(self, op, x):
        """op: [N, N]; x: [B*N, C] -> [B*N, C]."""
        n = self.nodes_per_graph
        total, c = x.shape
        if total % n != 0:
            raise ValueError(
                f"{total} nodes is not a multiple of nodes_per_graph={n}; "
                "the block-diagonal fast path assumes a static graph."
            )
        b = total // n
        return torch.matmul(op, x.view(b, n, c)).reshape(total, c)

    # -- shared operator API ------------------------------------------------------------

    def forward(self, I, edge_index=None, edge_weight=None, emb=True):
        if emb and self.learnEmb:
            I = self.Emb(I)
        return self._apply_blockwise(self._operator(edge_index, edge_weight), I)

    def adjoint(self, Ic, edge_index=None, edge_weight=None, emb=True):
        op = self._operator(edge_index, edge_weight)
        I = self._apply_blockwise(op.t(), Ic)
        if emb and self.learnEmb:
            I = self.Emb.backward(I)
        return I

    def perturb_observation(self, d):
        """Hook for Eq. (1)'s noise term. Only the denoising task overrides it."""
        return d


# --------------------------------------------------------------------------------------
# Task 1: denoising
# --------------------------------------------------------------------------------------


class graphDenoise(nn.Module):
    """F(x) = x, with observation noise.

    The operator itself stays the identity -- adding noise inside ``forward`` would be
    wrong, because CGLS evaluates the operator many times per solve and fresh noise on
    each call would destroy the Krylov iteration. Following Eq. (1), ``d = F(x) + eps``,
    the noise is drawn once and applied to the observation via ``perturb_observation``.
    """

    def __init__(self, nin, embdsize, learnEmb=True, device="cuda", sigma=0.1):
        super().__init__()
        self.nin = nin
        self.sigma = sigma
        self.Emb = graphEmbed(embdsize, nin, learned=learnEmb, device=device)
        self.learnEmb = learnEmb

    def forward(self, I, edge_index=None, edge_weight=None, emb=True):
        if emb and self.learnEmb:
            I = self.Emb(I)
        return I

    def adjoint(self, Ic, edge_index=None, edge_weight=None, emb=True):
        if emb and self.learnEmb:
            Ic = self.Emb.backward(Ic)
        return Ic

    def perturb_observation(self, d):
        return d + self.sigma * torch.randn_like(d)


# --------------------------------------------------------------------------------------
# Task 2/4: masking -- random (inpainting) vs structured (sensor recovery)
# --------------------------------------------------------------------------------------


def sample_random_observed(graph, budget_per_graph, nodes_per_graph):
    """Uniformly sample the observed nodes -- the paper's property-completion setting."""
    device = graph.x.device
    total = graph.x.shape[0]
    n_graphs = total // nodes_per_graph
    picks = []
    for b in range(n_graphs):
        offset = b * nodes_per_graph
        perm = torch.randperm(nodes_per_graph, device=device)[:budget_per_graph]
        picks.append(perm + offset)
    return torch.cat(picks)


def sample_structured_observed(
    graph, budget_per_graph, nodes_per_graph, adj_list, n_clusters=3
):
    """Observe a few contiguous patches of sensors instead of a uniform scatter.

    The paper's budget is deliberately tight (16 observed nodes out of METR-LA's 207),
    so the interesting structural axis is *where the working sensors are*, not where the
    failures are -- at 92% failure, "clustered failures" would be indistinguishable from
    random ones.

    So this grows the observed set by BFS from a few seeds, modelling a network that is
    densely instrumented in a handful of neighbourhoods and blind everywhere else. That
    is a genuinely harder inverse problem than the uniform scatter of ``inpainting``:
    large regions sit far from any measurement, whereas random sampling gives roughly
    even coverage. The budget is identical, so the two tasks differ purely in the
    *structure* of what is seen, not in how much.
    """
    device = graph.x.device
    total = graph.x.shape[0]
    n_graphs = total // nodes_per_graph

    picks = []
    for b in range(n_graphs):
        offset = b * nodes_per_graph
        observed = torch.zeros(nodes_per_graph, dtype=torch.bool, device=device)
        n_observed = 0
        per_cluster = max(1, budget_per_graph // n_clusters)

        while n_observed < budget_per_graph:
            seed = int(torch.randint(nodes_per_graph, (1,), device=device))
            grown = 0
            frontier = [seed]
            # Grow one patch outward, then reseed elsewhere for the next patch.
            while frontier and grown < per_cluster and n_observed < budget_per_graph:
                node = frontier.pop(0)
                if observed[node]:
                    continue
                observed[node] = True
                n_observed += 1
                grown += 1
                frontier.extend(adj_list[node])

        picks.append(observed.nonzero(as_tuple=True)[0] + offset)
    return torch.cat(picks)


def build_adjacency_list(edge_index, n):
    """Neighbour lists for block 0, used to grow structured failures."""
    mask = (edge_index[0] < n) & (edge_index[1] < n)
    ei = edge_index[:, mask]
    adj = [[] for _ in range(n)]
    for src, dst in zip(ei[0].tolist(), ei[1].tolist()):
        if src != dst:
            adj[src].append(dst)
    return adj


# --------------------------------------------------------------------------------------
# Task 3: source localization -- P^k, block-diagonal fast path
# --------------------------------------------------------------------------------------


class graphSmoothBlock(BlockOperatorMixin):
    """``graph_smooth`` (F(x) = P^k x) using the static-graph block fast path.

    Numerically identical to upstream ``graph_smooth``; see the module docstring for why
    the dense whole-batch version is not viable at the paper's batch size.
    """

    def __init__(self, nin, embdsize, nodes_per_graph, learnEmb=True, device="cuda", k=4):
        super().__init__(nodes_per_graph)
        self.nin = nin
        self.k = k
        self.Emb = graphEmbed(embdsize, nin, learned=learnEmb, device=device)
        self.learnEmb = learnEmb

    def _build_operator(self, edge_index, edge_weight, n):
        A = self._block_adjacency(edge_index, edge_weight, n)
        return torch.linalg.matrix_power(A, self.k)


# --------------------------------------------------------------------------------------
# Task 5: PDE-state reconstruction -- convection-diffusion-reaction
# --------------------------------------------------------------------------------------


class graphCDR(BlockOperatorMixin):
    """F(x) = exp(tau * M) x for a convection-diffusion-reaction generator.

    Ports the PDE-SSM operator (arXiv 2603.13663) to a graph. In that paper the PDE is
    solved on a regular grid, where the Fourier symbol is

        G(k) = exp(tau * (-k^T K k + r + i(b.k)))

    The three terms port to a graph unevenly:

    * **Diffusion** ``-k^T K k`` -> ``-kappa * L`` on the graph Laplacian. Exact analogue.
    * **Reaction** ``r`` -> ``r * I``. Exact analogue (it is a scalar).
    * **Convection** ``i(b.k)`` is a *phase shift*, relying on the Fourier shift theorem.
      A graph Laplacian is symmetric, so its eigenbasis is real and there is no phase to
      shift -- this term has **no spectral analogue**. Dropping it would reduce the
      operator to pure diffusion, i.e. exactly the source-localization task, making this
      fifth task degenerate. So convection is built directly in the node basis as a
      genuinely non-symmetric upwind advection operator.

    Advection uses an antisymmetric edge velocity ``v_ij = phi_j - phi_i`` derived from a
    fixed seeded potential ``phi``, discretised upwind so that mass is conserved (columns
    of ``Adv`` sum to zero). The physics parameters are fixed constants, not learned: a
    forward operator must be known.
    """

    def __init__(
        self,
        nin,
        embdsize,
        nodes_per_graph,
        learnEmb=True,
        device="cuda",
        kappa=1.0,
        gamma=1.0,
        r=0.0,
        tau=4.0,
        seed=0,
    ):
        # tau is calibrated, not guessed: it sets how much information the PDE destroys.
        # Measured on METR-LA with diagnose_tasks.py, tau=0.25 leaves the operator
        # full-rank at condition 1.3e-1 -- effectively invertible, so the task is
        # trivial and scores ~0 nMSE regardless of what the model learned. tau=4.0
        # gives condition 6.2e-11 while staying full-rank, making this severely
        # ill-posed in a way that is genuinely different from the rank-deficient
        # masking tasks (rank 16/207) and from source localization (rank 25/207).
        super().__init__(nodes_per_graph)
        self.nin = nin
        self.kappa = kappa
        self.gamma = gamma
        self.r = r
        self.tau = tau
        self.seed = seed
        self.Emb = graphEmbed(embdsize, nin, learned=learnEmb, device=device)
        self.learnEmb = learnEmb

    def _potential(self, n, device, dtype):
        """Fixed velocity potential -- seeded so the operator is reproducible."""
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        return torch.randn(n, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)

    def _advection(self, edge_index, edge_weight, n):
        """Non-symmetric, mass-conserving upwind advection."""
        A = self._block_adjacency(edge_index, edge_weight, n)
        phi = self._potential(n, edge_weight.device, edge_weight.dtype)

        # v_ij = phi_j - phi_i, antisymmetric by construction.
        v = phi.unsqueeze(0) - phi.unsqueeze(1)  # v[i, j]
        inflow = A * torch.relu(v)  # mass arriving at i from j
        outflow = A * torch.relu(-v)  # mass leaving i towards j

        Adv = inflow.clone()
        Adv.fill_diagonal_(0.0)
        # Diagonal carries the total outflow, so column sums vanish (conservation).
        Adv -= torch.diag(outflow.sum(dim=1))
        return Adv

    def _laplacian(self, edge_index, edge_weight, n):
        mask = (edge_index[0] < n) & (edge_index[1] < n)
        ei, ew = edge_index[:, mask], edge_weight[mask]
        li, lw = get_laplacian(ei, ew, normalization="sym", num_nodes=n)
        L = torch.zeros(n, n, device=edge_weight.device, dtype=edge_weight.dtype)
        L[li[0], li[1]] = lw
        return L

    def _build_operator(self, edge_index, edge_weight, n):
        L = self._laplacian(edge_index, edge_weight, n)
        Adv = self._advection(edge_index, edge_weight, n)
        eye = torch.eye(n, device=edge_weight.device, dtype=edge_weight.dtype)

        M = -self.kappa * L + self.gamma * Adv + self.r * eye
        return torch.matrix_exp(self.tau * M)
