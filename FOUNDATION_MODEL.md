# A foundation model for graph inverse problems

Extension of *Learning Regularization for Graph Inverse Problems* (AAAI-2025,
[arXiv:2408.10436](https://arxiv.org/abs/2408.10436)).

The paper trains **one model per inverse problem**. This asks whether a **single** model
can learn a prior shared across inverse problems: train on three tasks and report
zero-shot on the two it has never seen — transfer to unseen forward operators.

**Protocol v2** (current). The first sweep early-stopped each model on a held-out val
task; on METR-LA that signal never improved for 4/5 models, so patience fired and an
*untrained* epoch-0 checkpoint was scored (visible as `best_epoch 0–1` in the old
`comparison.csv`). Selection now tracks the mean training-task nMSE (`--selection train`,
the default), with a `--min_epochs` floor as a backstop. Since the val-slot task no
longer does model selection, it becomes a **second clean zero-shot task** — 10 zero-shot
numbers from 5 runs. Runs are tagged `protocol: 2` in `metrics.json`; `compare.py`
excludes the old runs by default (`--include_legacy` restores them). The v1 behaviour is
kept under `--selection val_task` for comparison.

## Quick start

```bash
python src/data_prep.py                      # download + prepare METR-LA and CPOX
python src/test_taskOps.py                   # verify the operators (do this first)
python src/diagnose_tasks.py --dataset METRLA   # check each task is actually ill-posed

python src/main_parallel.py --dataset METRLA # train all 6 models, 2 at a time
python src/watch_progress.py                 # live progress, in another terminal
```

`main_parallel.py` runs `compare.py` and `plots.py` automatically when training finishes.
Outputs land in `results/<dataset>/`. It is resumable — any run with a `metrics.json` is
skipped, so an interrupted sweep can just be restarted.

## The five tasks

| Task | Paper equivalent | Forward operator |
|---|---|---|
| `denoising` | — | `F(x) = x`, with observation noise |
| `inpainting` | property completion | random node mask |
| `source_localization` | inverse source estimation | `F(x) = Pᵏx` |
| `sensor_recovery` | — | structured (clustered) node mask |
| `pde_state` | — | `F(x) = exp(τM)x`, convection-diffusion-reaction |

Only two existed upstream. Three are new, in `src/taskOps.py`, registered in
`src/tasks.py` and re-exported from `utils`.

### The PDE task

Ports the convection-diffusion-reaction operator of PDE-SSM
([arXiv:2603.13663](https://arxiv.org/abs/2603.13663), same authors) onto a graph. That
paper solves on a regular grid, where the Fourier symbol is
`G(k) = exp(τ(−kᵀKk + r + i(b·k)))`. The three terms port unevenly:

- **Diffusion** `−kᵀKk` → `−κL` on the graph Laplacian. Exact analogue.
- **Reaction** `r` → `rI`. Exact analogue.
- **Convection** `i(b·k)` is a *phase shift* relying on the Fourier shift theorem. A graph
  Laplacian is symmetric, so its eigenbasis is real and **there is no phase to shift** —
  this term has no spectral analogue. Dropping it would reduce the operator to pure
  diffusion, i.e. exactly `source_localization`, making the fifth task degenerate.

So convection is built directly in the node basis as a non-symmetric, mass-conserving
upwind advection operator, and the generator `M = −κL + γ·Adv(b) + rI` is exponentiated
with `torch.matrix_exp`. The physics parameters are fixed constants, not learned: a
forward operator must be known.

### Sensor recovery vs inpainting

Both mask nodes with an identical budget; they differ in *structure*. `inpainting` samples
uniformly (the paper's setting). `sensor_recovery` grows observed patches by BFS from a
few seeds — a network densely instrumented in a handful of neighbourhoods and blind
elsewhere, which leaves large regions far from any measurement.

At METR-LA's budget (16 observed of 207) the structural axis has to be *where the working
sensors are*, not where the failures are: at 92% failure, "clustered failures" are
indistinguishable from random ones.

## Experiment design

`val = (i+1) mod 5`, `test = (i+2) mod 5`, so each task takes each held-out slot exactly
once, and no pair recurs reversed. Asserted in `test_taskOps.py`. Under protocol v2 both
held-out columns are zero-shot; the "val"/"test" names only identify the rotation slot.

| Script | Train | Zero-shot (val slot) | Zero-shot (test slot) |
|---|---|---|---|
| `exp1.py` | denois, sensor, pde | inpaint | source |
| `exp2.py` | denois, inpaint, pde | source | sensor |
| `exp3.py` | denois, inpaint, source | sensor | pde |
| `exp4.py` | inpaint, source, sensor | pde | denois |
| `exp5.py` | source, sensor, pde | denois | inpaint |
| `exp_original.py` | path (the paper's task) | — | — |

Multi-seed: `main_parallel.py --seeds 0,1,2` suffixes run dirs with `_s{seed}` and skips
finished (config, seed) pairs, so seeds can be added incrementally. `compare.py` groups
seeds and reports mean ± std.

## Three things worth knowing

### 1. Validation is at the task level, following the article

The article has **no snapshot-level validation split**, and `main_3_linear_inv_problems.py`
early-stops directly on the test set. We keep its splits unchanged. The train/holdout
structure here lives at the **task** level: a zero-shot task contributed no gradients and
no model selection (under v2, selection uses only the training tasks' epoch-averaged
nMSE). All held-out metrics are computed on the same test snapshots, as in the article —
noted in every CSV footer.

`exp_original.py` reproduces the paper's protocol *including* its test-set selection, so
its number stays comparable to Table 5. It is recorded as `selection: "test"` so it is
never presented as a clean holdout.

### 2. Zero-shot required one refactor

`graphEmbed` holds the only parameters a forward operator owns — the lift/project between
the 1-channel physical space and the network's hidden space. If each task built its own, a
held-out task would be evaluated with a **randomly initialised** embedding and its
zero-shot number would be meaningless.

So every task shares one `graphEmbed` instance. It is trained by the training tasks and
equally valid for the held-out ones, leaving **zero task-specific parameters**. The
operator reaches the network only through the parameter-free CGLS data-fit step.
`train_foundation_model` asserts that swapping operators does not change the parameter
set, and `test_taskOps.py` checks no operator owns private parameters.

### 3. Tasks were calibrated, not guessed

An inverse problem is only informative if the forward operator destroys information. Run
`diagnose_tasks.py` to measure it. Two initial settings were badly wrong:

- **`pde` at τ=0.25** was full-rank at condition 1.3e-1 — effectively invertible, scoring
  0.0000 nMSE regardless of what the model had learned. Recalibrated to **τ=4.0**
  (condition 6.2e-11).
- **`denoising` at σ=0.1** — since the data is z-normalised, the trivial "return the
  observation" estimator scores nMSE = σ² = 0.01. Raised to **σ=0.5** (baseline 0.25).

Final profile on METR-LA, deliberately spanning different failure modes:

| task | conditioning | rank | character |
|---|---|---|---|
| denois | 1.0 | 207/207 | well-posed, noise-limited |
| inpaint | — | 16/207 | rank-deficient |
| sensor | — | 16/207 | rank-deficient, structured |
| source | 2.0e-13 | 25/207 | rank-deficient **and** ill-conditioned |
| pde | 6.2e-11 | 203/207 | full-rank, severely ill-conditioned |

## Hardware notes

The paper used a 48 GB A6000; this targets a **6 GB RTX 4050** running two models
concurrently.

- **Operators are block-diagonal.** `graph_smooth` upstream densifies the whole batch
  (`B·N × B·N`); at METR-LA batch 128 that is a 26,496² matrix, ~2.8 GB. Both datasets are
  static graphs, so the batched operator is exactly `I_B ⊗ op_single` — build the 207×207
  block once and apply block-wise. Verified numerically identical to the dense path;
  memory drops to ~171 KB.
- **Gradient accumulation** preserves the paper's effective batch of 128 via 32×4.
  Measured peaks: 0.67 / 1.33 / 2.72 GB at micro-batch 16 / 32 / 64. At 32, two concurrent
  runs use ~2.7 GB of 6 GB.

## Data

METR-LA's host (graphmining.ai) no longer serves a valid TLS certificate. `data_prep.py`
rebuilds the identical `adj_mat.npy` / `node_values.npy` from the canonical DCRNN release
mirrored by torch-spatiotemporal, using the standard thresholded Gaussian kernel
(`normalized_k = 0.1`). Sanity check: 34,272 snapshots × 207 nodes, 1722 edges including
self-loops → **1515** after `remove_self_loops`, matching the paper.

## Outputs

| File | Contents |
|---|---|
| `comparison.csv` | every configuration × every task (mean ± std across seeds), with each cell's role (train/zeroshot/unseen) |
| `transfer_gap.csv` | one row per (model, zero-shot task) vs mean of its training tasks |
| `original_zeroshot.csv` | the paper's single-task model on the five tasks it never trained on |
| `per_task_ranking.csv` | who wins each task, and whether they trained on it |
| `prior_value.csv` | best zero-shot model vs the best classical baseline, per task |
| `baselines.csv` | trivial predictors + oracle Tikhonov/Laplacian (see below) |
| `*.png` | the same, as report figures (light and `--dark`) |

Each run directory additionally keeps `history.json` (per-epoch training curves — the
diagnosability the v1 sweep lacked), the selected checkpoint `model.pth`, and
`model_last.pth` (the final state, for re-scoring a run post hoc). The checkpoints are
~120 KB each and are exempted from the `*.pth` ignore rule, so they get committed.

## Read the baselines first

**Do not interpret any nMSE without `baselines.py`.** Every model here sits on top of a
CGLS data-fit step that already solves the inverse problem with *no learned component*;
the GNN only adds regularization. And beating `pinv` (unregularised least squares) is
still not enough: `baselines.py` also computes **oracle-tuned Tikhonov and
graph-Laplacian regularized least squares** — the paper's own classical baseline family,
with the λ swept and chosen *per task on the test nMSE*. That is a deliberate ceiling
for what non-learned regularization can do. A learned prior that fails to beat the best
classical baseline has contributed nothing a textbook method would not supply.

`compare.py` writes `prior_value.csv` with the verdict driven by the best **zero-shot**
model against the best classical baseline, and prints a warning when no model beats it
by more than 5% on a task.

### CPOX result

| task | type | pinv | best model | prior helps? |
|---|---|---|---|---|
| inpaint | rank-deficient | 0.634 | 0.483 | **yes, 24%** |
| sensor | rank-deficient, structured | 0.571 | 0.517 | **yes, 9%** |
| denois | well-posed, noise-limited | 0.227 | 0.242 | no — *worse* |
| source | ill-conditioned | 0.591 | 0.594 | no |
| pde | full-rank, ill-conditioned | 0.258 | 0.256 | no |

The learned prior earns its keep only where information is genuinely **missing** and must
be supplied from a prior. Where the operator merely distorts recoverable information,
least squares already extracts it.

**This makes the apparent zero-shot success on CPOX an artifact.** All six models score
alike on denoising, source and pde because they collapse onto the same prior-free
solution — which is also why `ORIGINAL`, trained on none of the five, matches models that
trained on the task. Similarity between models is *not* evidence of transfer.

On denoising the models are beaten by the *optimal linear estimator*: the rescaled adjoint
scores 0.179, matching Wiener shrinkage σ²/(1+σ²) = 0.20.

This is not a port bug — our `source_localization` numbers (0.59) match the paper's
published CPOX Var-GNN result of 0.59 (Table 4). The paper's own CPOX number also sits at
the least-squares floor; its baselines were LaplacianReg/TikhonovReg (0.84), not plain
least squares, so this comparison does not appear there.

CPOX is 20 nodes and 468 training snapshots, so this may be a small-data effect. METR-LA
(207 nodes, 24k snapshots) is the real test.

## Caveat

A model trained on masking and diffusion has no particular reason to invert an advection
operator well, so **zero-shot transfer may simply be weak**. That is a real finding if it
is what the numbers say. The single-task `ORIGINAL` baseline is what makes the gap
interpretable — it shows what one operator's worth of training buys on the same five
tasks.
