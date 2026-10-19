# One Prior for Many Graph Inverse Problems

**Course project (Learning on Graphs, Project 13), built on the official code of
*Learning Regularization for Graph Inverse Problems* (GRIP, Eliasof et al., AAAI-2025).**
Upstream: https://github.com/nafi007/GraphInverseProblems. Everything that is not upstream is listed
under [Repository layout](#repository-layout); the full engineering log is in `CHANGES_SUMMARY.txt`.

> **Status (2026-10-19).** Code and evaluation harness are final after five correctness fixes
> (validation-based checkpoint selection, a checkpoint-loading bug that swapped task heads, matching
> train/eval inputs, fixed PDE physics, working seeds; details in `CHANGES_SUMMARY.txt`). The tables
> under `old_models/` predate those fixes and the 2026-10-07 operator fixes and are kept only for the
> record; the submission tables are the ones `run_gpu.sh` writes to `results/`. See `GPU_RUN.md`.

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

Size: 32 channels × 32 layers, about 101 k parameters, in both experiments (the supervisor's
100–200 k brief; GRIP's default 32 × 16 is 51 k). Head/encoder parameters are < 1 % of the total,
so the prior lives in the shared backbone. The PDE coefficients are fixed buffers, not parameters.

**Training protocol** (`src/main_3_linear_inv_problems.py`, method `foundation`):

1. `--epochs 40`: backbone + shared head, cycling through the training tasks one per epoch, loss
   `½ (nMSE(x̂, x) + nMSE(A x̂, A x))`. Saved as `no_head_training_<run>_seed_k.pth`.
2. `--head_epochs 16`: backbone frozen, task-specific heads trained on the training tasks.
3. `--test_head_epochs 4`: backbone still frozen, the head of the **held-out task** is trained on
   the full CPOX training split (experiment A only; in B the held-out dataset is never trained on).
   Saved as `<run>_seed_k.pth`.

**Checkpoint selection** in every phase uses a **validation split**: the last 10 % of the training
weeks (CPOX) or the tail of each training dataset (B), scored on the tasks being trained in that
phase. The held-out test split is evaluated once per phase with the selected checkpoint and never
drives selection.

So in experiment A, `no_head_training_*` is the **zero-shot** number and `*_test_seed_k.pth` is
**supervised head adaptation** with a frozen prior (not few-shot: it sees the whole training
split). In experiment B both are zero-shot on the held-out dataset. Each run is repeated with
3 seeds; tables show mean ± std.

## Experiments and how to run them

Everything is driven by two scripts; `run_gpu.sh` chains them with the evaluation (details and
environment setup in **`GPU_RUN.md`**).

| | Train on | Test on | Script | Table |
|---|---|---|---|---|
| **A** task hold-out | CPOX, 4 tasks | CPOX test split (last 10 % of weeks), held-out task | `src/script.py` (5 runs) | `src/plot_fixed.py <dir>` |
| **B** dataset hold-out | 3 of {CPOX, PedalMe, WikiMaths, MontevideoBus}, all 5 tasks | the held-out dataset, all 5 tasks (Windmill: test-only, 5 % of 17 k snapshots) | `src/script2.py` (5 runs) | `src/plot_2.py <dir>` |

Datasets come from PyTorch-Geometric-Temporal and download on first use. In B every dataset is
z-normalised with its own target mean/std, including the held-out one, so B is an
**oracle-normalised** benchmark (two scalars of the held-out targets are used; nothing else).
Node features are constant (the native features are lagged copies of the target, which would leak
it); batches never mix datasets.

**Metric:** nMSE = ‖x̂ − x‖² / ‖x‖² on the test split, averaged over batches (lower is better;
1.0 = predicting zero). **Baselines** in every table: `X = 0`, `X = b` (A, where shapes allow) and
**least squares** (CGLS from zero with no learned prior, which equals zero-fill for masks and `b`
for denoising). A model that does not beat least squares has learned nothing beyond the data term.
Within a table every row inverts identical measurements (same masks, sensors, noise), produced by
the same preprocessing path as training. In the A table the held-out task's column is marked blue:
only that cell is a transfer number, the other columns are tasks the row trained on.

```bash
source .venv/bin/activate
bash run_gpu.sh            # A + B, 3 seeds each; see GPU_RUN.md for timing
# -> results/cpox_task_holdout/loss_matrix_table.png
# -> results/dataset_holdout/comparison_table.png
```

## Results

**Experiment A (done, 2026-10-20, CPU run of the current code, 3 seeds):**
`results/cpox_task_holdout/loss_matrix_table.png`, log in `results/run_cpu.log`.

| CPOX test nMSE | denoising | inpainting | sensor rec. | source loc. | PDE rec. |
|---|---|---|---|---|---|
| `X = b` baseline | 0.383 | 0.687 | 0.758 | 0.742 | 0.817 |
| least squares (25 CGLS it.) | 0.383 | 0.687 | 0.758 | **0.421** | **0.613** |
| model, task seen in training (range over rows) | **0.28–0.30** | **0.65–0.66** | **0.72–0.74** | 0.61–0.63 | 0.73 |
| model, task held out, zero-shot | 0.317 ± 0.018 | 0.683 ± 0.020 | 0.732 ± 0.006 | 0.652 ± 0.002 | 0.805 ± 0.001 |
| model, task held out, adapted head (4 epochs) | 0.301 ± 0.005 | 0.668 ± 0.005 | 0.725 ± 0.005 | 0.639 ± 0.003 | 0.804 ± 0.001 |

Reading: on the tasks it was trained on, the shared model beats `X = b` everywhere, but it beats
the least-squares solver only on the three tasks where least squares cannot do better than `X = b`
(denoising, and the two mask tasks with a null space). On the two well-posed-but-ill-conditioned
operators (source localization A⁴, PDE exp(−20 L)) the unrolled solver, which runs only 5 CGLS
steps per iteration, is clearly *worse* than 25 plain CGLS steps. Transfer to a held-out task
costs 0.01–0.07 nMSE relative to training on it, and 4 epochs of head adaptation recover a third
to a half of that gap, except for the PDE task, where the held-out model stays at the `X = b`
level (0.80 vs 0.82). Seed spread is small (std ≤ 0.03), so these differences are stable.

**Experiment B (pending, must run on the GPU machine):**
`results/dataset_holdout/comparison_table.png`, produced by `bash run_gpu.sh B`. The CPU attempt
was killed by the out-of-memory killer at 21 GB on the first WikiMaths batch; the backbone now
checkpoints its layers for batches with more than 20k edges (see `CHANGES_SUMMARY.txt`), and one
WikiMaths training step costs 8–15 s on an 8-core CPU, which makes B (≈ 900 epochs × 340
batches) a GPU-only experiment.

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
src/checkpoints.py              checkpoint naming + loading shared by both evaluators (ours)
src/check_save_load.py          save/load round-trip test: run it after touching networks.py (ours)
old_models/                     preliminary checkpoints + tables (pre-fix, see Results)
results/                        final checkpoints + tables written by run_gpu.sh
configurations/, environment.yml, src/forwardOps.py, src/custom*.py, src/pygt_dataloader.py   upstream
```

## Limitations (please read before quoting numbers)

* **Baselines are sanity floors, not competitors.** `X = 0`, `X = b` and unregularised least
  squares show whether the prior adds anything beyond the data term. A validation-tuned classical
  regulariser (Tikhonov / graph-Laplacian) and the same network trained from scratch on the target
  task, plus a single- vs multi-task pretraining ablation at equal budget, are still to be added.
* **Head adaptation is supervised**, on the full training split; no few-shot claim is made.
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
