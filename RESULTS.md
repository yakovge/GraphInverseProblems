# Foundation-model transfer study — preliminary results

**Question** (per Prof. Eliasof's guidance): keep the dataset / signal family
fixed and vary only the forward operator A. Pretrain ONE unrolled solver on a
mixture of operators (the operator is resampled per batch), then test on
operators it never saw. Does it learn something general about solving graph
inverse problems?

## Setup

- Base: this repo (official GRIP code); backbone `graph_NeuralProximalGradient`
  (PGD / Prox-GNN) with the iterate lifted to hidden width so the regularizer
  carries ~155k parameters (`--channels 64 --layers 12`).
- Dataset: CLUSTER (SBM-style benchmark; node classes are the signal). METR-LA
  (real road-sensor graph, regression) is the planned second graph type.
- Pretraining mixture: `denoise` (identity + Gaussian noise), `mask` (property
  completion), `smooth` (A^k diffusion — GRIP's 'deblur' / inverse source
  estimation), `path` (inverse graph transport).
- Held out: `maskSmooth` = observe a node subset of the diffused state
  (PDE-state reconstruction; compositional/near holdout) and `maxpool` =
  neighborhood max (nonlinear; genuinely new operator, far holdout).
- Methods compared on each holdout, all with the same support budget N:
  pretrained zero-shot, pretrained few-shot, pretrained few-shot with the
  denoiser-only recipe (data-fidelity step disabled — the v1-project mechanism
  finding re-tested at this scale), from-scratch few-shot, classical Tikhonov.
- Protocol: N ∈ {5, 20, 100} support graphs, fixed fine-tuning schedule
  (150 steps, no validation peeking), 3 seeds × 2 support draws, node accuracy.

Reproduce: `bash scripts/run_prelim.sh` (GPU; `DEVICE=cpu` works but is slow).

## Results

*(pending — filled in from `results/foundation_CLUSTER.csv` after the GPU run)*

| holdout | method | N=5 | N=20 | N=100 |
|---|---|---|---|---|
| ... | | | | |

### CPU validation run (2026-07-17; wiring check, NOT results)

Reduced config (1 seed, 25% data, 1 epoch, N∈{5,20}, 1 support draw;
`results/foundation_CLUSTER_cpuval.csv`). Numbers are underscaled, but two
qualitative observations already replicate the v1-project mechanism finding at
~100x the model size, now inside GRIP's own solver:

1. **Naive transfer to the nonlinear holdout diverges.** On `maxpool`, the
   pretrained solver's data-fidelity step explodes (zero-shot data loss ~1e20;
   few-shot fine-tuning goes to NaN and collapses to a constant predictor,
   acc 0.159 < chance-ish). Notably, GRIP's PGD computes its step size by exact
   line search per iteration — there are no stale *learned* step sizes here —
   yet it still diverges: the line-search step is exact only for linear A, and
   overshoots under the VJP linearization of a nonlinear operator. The fragile
   component under operator shift is the data step itself.
2. **The denoiser-only recipe restores stability.** With the data step disabled,
   zero-shot and few-shot on `maxpool` are sane (acc ~0.23, loss ~1.7).
3. On the compositional holdout (`maskSmooth`), pretrained few-shot ≥
   from-scratch at both N (0.28/0.30 vs 0.27/0.27) and above classical (0.26) —
   directionally right, far too small a run to claim anything.

In-distribution sanity after 1 short epoch: denoise 0.96, path 0.82, mask 0.38,
smooth 0.28 (smooth/deblur is the hardest pretraining task, consistent with GRIP).

## Notes / what failed so far

- **Target-leakage pitfall (fixed):** generating the observation stashes a
  linearization point inside the nonlinear operator; if the solver's VJP
  "adjoint" then linearizes at the ground truth, zero-shot maxpool accuracy
  jumps to ~0.97 (oracle). The operator now clears the stash after observation
  generation; the solver only linearizes at the observation or its own iterates.
- Upstream PGD keeps the iterate in label space (~8k params total); the
  100–200k target required open/close linear maps around the regularizer.
