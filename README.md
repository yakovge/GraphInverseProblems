# One Prior for Many Graph Inverse Problems

**Course project (Learning on Graphs, Project 13), built on the official code of
*Learning Regularization for Graph Inverse Problems* (GRIP, Eliasof et al., AAAI-2025).**
Upstream: https://github.com/nafi007/GraphInverseProblems. Everything that is not upstream is listed
under [Repository layout](#repository-layout); the full engineering log is in `CHANGES_SUMMARY.txt`.

> **Status.** Code and evaluation harness are final. The result tables committed under `old_models/`
> were produced *before* the operator fixes of 2026-10-07 (see [Results](#results)) and are
> superseded by the tables that `run_gpu.sh` writes to `results/`. See `GPU_RUN.md`.

## Question

GRIP trains one Var-GNN solver per inverse problem. We ask whether **a single learned prior can
serve several graph inverse problems at once**, and how far it transfers:

* **A. to an unseen forward operator** on a fixed signal family (CPOX, Hungarian chickenpox counts
  on a 20-county graph): train on four tasks, hold out the fifth;
* **B. to an unseen dataset / graph** with all five operators known: train on three
  spatio-temporal datasets, hold out a fourth.

The model never sees the held-out task (A) or the held-out dataset (B) during backbone training.

## The five tasks

All tasks share one target: the node signal `x` of a graph snapshot. Each task is a forward
operator `A` with measurements `b = A x (+ noise)`; the model recovers `x` from `b`.

| Task | Operator `A` | Setting | What makes it hard |
|---|---|---|---|
| denoising | identity, Gaussian noise σ = 0.25 | `AddNoise` | noise on every node |
| inpainting | random node mask, 6 of 20 nodes observed per snapshot | `graphMask`, `--mask_per_snapshot_budget 6` | null space: 70 % of nodes unobserved |
| sensor recovery | one BFS-grown patch of 5 sensors per snapshot | `SensorRecovery`, `--n_sensors 5` | null space, spatially clustered |
| source localization | k = 4 steps of graph diffusion `Â⁴ x` (GRIP's inverse source estimation) | `graph_smooth`, `--blur_count 4` | ill-conditioned (cond ≈ 7·10⁵) |
| PDE-state reconstruction | heat kernel `exp(−τ L) x`, τ = 20 | `PDESSM`, `--pde_tau 20` | very ill-conditioned (cond ≈ 10⁹) |

Masks and sensor patches are resampled every batch. Regression graphs are made undirected before
normalisation and the PDE Laplacian is symmetrised (both were sources of blow-ups, see the log).

## Model

`GraphInverseFoundationModel` (`src/networks.py`) = GRIP's unrolled Var-GNN solver with

* a **shared backbone** (`graphHyperResNet`, the learned prior), fed the measurement once;
* a small **per-task encoder/head** pair, plus a shared encoder/head used before any task head exists;
* a **data-consistency step** per unrolled iteration: `--dc prox` solves
  `argmin_x ‖A x − b‖² + λ_task ‖x − x_ref‖²` with 5 conjugate-gradient steps from the network's
  guess `x_ref`, with `λ_task` learned (`--dc cgls` is GRIP's original fixed CGLS step).

Sizes: experiment A uses 128 channels × 32 layers (1.58 M parameters), experiment B uses
128 × 16 (0.79 M). GRIP's default 32 × 16 is 51 k. Head/encoder parameters are < 0.1 % of the
total, so the prior lives in the shared backbone.

**Training protocol** (`src/main_3_linear_inv_problems.py`, method `foundation`):

1. `--epochs 40`: backbone + shared head, cycling through the training tasks one per epoch, loss
   `½ (nMSE(x̂, x) + nMSE(A x̂, A x))`. The best checkpoint is saved as `no_head_training_<run>.pth`.
2. `--head_epochs 16`: backbone frozen, task-specific heads trained on the training tasks.
3. `--test_head_epochs 4`: backbone still frozen, the head of the **held-out task** is trained on
   training snapshots (experiment A only; in B the held-out dataset is never trained on).
   Saved as `<run>.pth`.

So in experiment A, `no_head_training_*` is the **zero-shot** number and `*_test.pth` is
**few-shot head adaptation** with a frozen prior. In experiment B both are zero-shot on the
held-out dataset.

## Experiments and how to run them

Everything is driven by two scripts; `run_gpu.sh` chains them with the evaluation (details and
environment setup in **`GPU_RUN.md`**).

| | Train on | Test on | Script | Table |
|---|---|---|---|---|
| **A** task hold-out | CPOX, 4 tasks | CPOX test split (last 10 % of weeks), held-out task | `src/script.py` (5 runs) | `src/plot_fixed.py <dir>` |
| **B** dataset hold-out | 3 of {CPOX, PedalMe, WikiMaths, MontevideoBus}, all 5 tasks | the held-out dataset, all 5 tasks (Windmill: test-only, 5 % of 17 k snapshots) | `src/script2.py` (5 runs) | `src/plot_2.py <dir>` |

Datasets come from PyTorch-Geometric-Temporal and download on first use. In B every dataset is
z-normalised with its own mean/std and node features are constant (the native features are lagged
copies of the target, which would leak it); batches never mix datasets.

**Metric:** nMSE = ‖x̂ − x‖² / ‖x‖² on the test split, averaged over batches (lower is better;
1.0 = predicting zero). **Baselines** in every table: `X = 0` and either `X = b` (A) or
**least squares** (B: CGLS from zero with no prior, which equals zero-fill for masks and `b` for
denoising). A model that does not beat least squares has learned nothing beyond the data term.
Within a table every row sees identical masks, sensors and noise.

```bash
source .venv/bin/activate
bash run_gpu.sh            # A + B, ~1 h + several hours on a 6 GB GPU
# -> results/cpox_task_holdout/loss_matrix_table.png
# -> results/dataset_holdout/comparison_table.png
```

## Results

**Final tables:** `results/cpox_task_holdout/loss_matrix_table.png` (A) and
`results/dataset_holdout/comparison_table.png` (B), produced by `run_gpu.sh` on the current code.
*(To be filled in after the GPU run: one sentence per experiment on where the model beats the
least-squares / `X = b` baseline and where it does not.)*

**Preliminary tables** (`old_models/`, trained 2026-09-27 to 2026-10-05, *before* the final
operator fixes, kept for the record):

* `old_models/cpox_task_holdout_final/loss_matrix_table.png` (A, 64 × 32 channels/layers,
  σ = 0.25): the foundation model beats the `X = b` baseline on every task for every held-out
  rotation, e.g. zero-shot inpainting 0.19–0.28 vs 0.20 for `X = b` and sensor recovery 0.73 vs
  0.77. Gains are modest because CPOX has little spatial structure at 6/20 observed nodes
  (neighbour averaging is *worse* than zero-fill there, see the log).
* `old_models/dataset_holdout/comparison_table.png` (B): mixed. The model beats least squares on
  denoising and on the mask tasks for MontevideoBus, PedalMe and Windmill, but not on CPOX, and
  the WikiMaths source-localization column blew up (nMSE 255–10 866) because WikiMaths is a
  directed graph with a 533-out-degree hub that the undirected normalisation amplified. The
  PDE column reads 0.0000 for every row because the operator was still trivially invertible
  (τ = 1). Both bugs are fixed in the code (2026-10-07) and are the reason for the rerun.
* `old_models/cpox_task_holdout_noise_0.10`, `..._0.25`: earlier A runs at two noise levels.

## Repository layout

```
run_gpu.sh, GPU_RUN.md          full pipeline and GPU instructions (ours)
CHANGES_SUMMARY.txt             engineering log: every bug found, why, and the fix (ours)
src/main_3_linear_inv_problems.py  training/eval loop; method 'foundation' and its phases (modified)
src/networks.py                 GraphInverseFoundationModel + prox data-consistency (modified)
src/graphForwardOps.py          AddNoise, graphMask, SensorRecovery, graph_smooth, PDESSM (modified)
src/utils.py                    data loading (CPOX, MULTI), per-batch mask/sensor sampling (modified)
src/script.py / src/script2.py  experiment A / B drivers (ours)
src/plot_fixed.py / src/plot_2.py  evaluation harnesses + tables for A / B (ours)
old_models/                     preliminary checkpoints + tables (pre-fix, see Results)
results/                        final checkpoints + tables written by run_gpu.sh
configurations/, environment.yml, src/forwardOps.py, src/custom*.py, src/pygt_dataloader.py   upstream
```

## Limitations (please read before quoting numbers)

* **Model selection uses the test split.** As in the upstream code, the best epoch is chosen on
  the test loss (there is no validation split for CPOX). A validation-split variant was tried and
  reverted (commit `ac8ceec`); numbers are therefore optimistic by the usual selection bias.
* **Single seed.** `--num_seeds 1`; no error bars.
* **Model size** is above the 100–200 k parameter budget discussed with the supervisor
  (1.58 M for A, 0.79 M for B); `--channels 32`/`64` reproduces smaller variants.
* **Denoising** on CPOX is largely solved by the data-consistency step (the operator is the
  identity), so its column mostly measures the data term, not the prior.
* **PedalMe and Windmill are complete graphs**, so their Laplacian has two eigenvalues and PDE
  reconstruction and source localization both reduce to "recover the mean"; identical baselines
  there are inherent.
* GRIP's own `--method drip` path is untouched except for a constructor-name typo fix.

## Citation of the upstream work

```bibtex
@inproceedings{eliasof2025learning,
  title={Learning Regularization for Graph Inverse Problems},
  author={Eliasof, Moshe and Siddiqui, Md Shahriar Rahim and Sch{\"o}nlieb, Carola-Bibiane and Haber, Eldad},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39}, number={16}, pages={16471--16479}, year={2025}
}
```
