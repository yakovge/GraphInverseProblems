# One Prior for Many Graph Inverse Problems

**Course project (Learning on Graphs, Project 13), built on the official code of
*Learning Regularization for Graph Inverse Problems* (GRIP, Eliasof et al., AAAI-2025).**
Upstream: https://github.com/nafi007/GraphInverseProblems. Everything that is not upstream is listed
under [Repository layout](#repository-layout); the full engineering log is in `CHANGES_SUMMARY.txt`.

> **Status (2026-10-21): complete.** All experiments are trained (A on CPU, C and B on a GTX 1080 Ti,
> 3 seeds each) and evaluated; the paper is `paper/main.pdf`. The submission tables are the ones in
> `results/` (`loss_matrix_table.png/.csv`, `comparison_table.png/.csv`, `solver_ablation.png/.csv`);
> the tables under `old_models/` predate the correctness fixes of 2026-10-19 (`CHANGES_SUMMARY.txt`)
> and are kept only for the record.

## Question

GRIP trains one Var-GNN solver per inverse problem. We ask whether **a single learned prior can
serve several graph inverse problems at once**, and how far it transfers:

* **A. to an unseen forward operator** on a fixed signal family (CPOX, Hungarian chickenpox counts
  on a 20-county graph): train on four tasks, hold out the fifth;
* **B. to an unseen dataset / graph** with all five operators known: train on the other
  spatio-temporal training datasets (two or three of CPOX, PedalMe, MontevideoBus), hold out one of
  five (WikiMaths and Windmill are test-only).

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
split). In experiment B both are zero-shot on the held-out dataset: `no_head_training_*` uses the
shared head, `*_dataset_test_*` the task heads after 16 + 4 more epochs on the *training* datasets
(nothing is ever trained on the held-out dataset). Each run is repeated with 3 seeds; tables show
mean ± std.

## Experiments and how to run them

Everything is driven by two scripts; `run_gpu.sh` chains them with the evaluation (details and
environment setup in **`GPU_RUN.md`**).

| | Train on | Test on | Script | Table |
|---|---|---|---|---|
| **A** task hold-out | CPOX, 4 tasks | CPOX test split (last 10 % of weeks), held-out task | `src/script.py` (5 runs) | `src/plot_fixed.py <dir>` |
| **B** dataset hold-out | the other training datasets among {CPOX, PedalMe, MontevideoBus}, all 5 tasks | the held-out dataset, all 5 tasks (WikiMaths: test-only, 20 % of its snapshots; Windmill: test-only, 5 % of 17 k snapshots) | `src/script2.py` (5 runs) | `src/plot_2.py <dir>` |
| **C** from-scratch baselines | CPOX, one task per model, 60 epochs | CPOX test split, that task | `src/script_scratch.py` (5 runs) | row of the A table |
| **ablation** test-time solver budget | (no training) | CPOX test split, held-out-task cells | `src/ablate_solver.py <dir>` | `solver_ablation.png/.csv` |

Datasets come from PyTorch-Geometric-Temporal and download on first use. In B every dataset is
z-normalised with its own target mean/std, including the held-out one, so B is an
**oracle-normalised** benchmark (two scalars of the held-out targets are used; nothing else).
Node features are constant (the native features are lagged copies of the target, which would leak
it); batches never mix datasets.

**Metric:** nMSE = ‖x̂ − x‖² / ‖x‖² on the test split, averaged over batches (lower is better;
1.0 = predicting zero). **Baselines** in every table: `X = 0`, `X = b` (A, where shapes allow),
**least squares** (25 CGLS steps from zero with no prior, which equals zero-fill for masks and `b`
for denoising, and is an early-stopped, implicitly regularised solution on the ill-conditioned
operators), **Tikhonov** and **graph-Laplacian** regularised least squares with `(λ, CG steps)`
tuned per task on the validation split the models were selected on (`src/classical.py`), and in A
the **same network trained from scratch** on the target task for 60 epochs (= the 40+16+4 budget of
a foundation run). Within a table every row inverts identical measurements (same masks, sensors,
noise; masks, sensors and noise are seeded on the CPU generator, so the same measurements are drawn
on any device and CPU/GPU tables agree up to floating-point differences in the model forward pass),
produced by the same preprocessing path as training. In the A table the held-out task's column is marked blue:
only that cell is a transfer number, the other columns are tasks the row trained on.

```bash
source .venv/bin/activate
bash run_gpu.sh            # A, C, B, 3 seeds each, then both tables; see GPU_RUN.md for timing
# -> results/cpox_task_holdout/loss_matrix_table.png (+ .csv)
# -> results/dataset_holdout/comparison_table.png (+ .csv)
python src/ablate_solver.py results/cpox_task_holdout   # -> solver_ablation.png (+ .csv), no training
python paper/make_tables.py                              # LaTeX tables of the paper from the CSVs
```

**Evaluation only** (no training; reproduces every table and number of the paper from the released
checkpoints in `results/`; CPU is enough, about two hours in total; logged in `results/run_eval_cpu.log`):

```bash
export CUDA_VISIBLE_DEVICES=""                                  # optional: force CPU
.venv/bin/python src/plot_fixed.py results/cpox_task_holdout    # A table (+ tuned classical rows)   -> loss_matrix_table.png/.csv
.venv/bin/python src/ablate_solver.py results/cpox_task_holdout # test-time solver ablation          -> solver_ablation.png/.csv
.venv/bin/python src/plot_2.py results/dataset_holdout --model_rows_from results/run_gpu_CB.log --val_batches 5
#   B table: tuned classical rows computed here, learned rows copied from the GPU run's log (see Results)
#   -> comparison_table.png/.csv.  Add `--only WINDMILL` etc. to run the datasets in parallel, then `--merge`.
#   Without --model_rows_from the 30 checkpoints are re-evaluated too (minutes on a GPU, a day on a CPU).
.venv/bin/python paper/make_tables.py                           # LaTeX tables of the paper from the CSVs
.venv/bin/python paper/make_figures.py                          # appendix figure from solver_ablation.csv
```

## Results

All numbers below are the CPU re-evaluation of the released checkpoints (`results/run_eval_cpu.log`,
2026-10-21), except the learned rows of experiment B (see its provenance note); the `.csv` next to each
table holds the per-seed values (or `mean+-std` for the copied B rows). The paper (`paper/main.pdf`)
discusses them; this is the summary.

**Experiment A (CPOX task hold-out, 3 seeds):** `results/cpox_task_holdout/loss_matrix_table.png`.
The row "held out" is the model that never trained on that column's task.

| CPOX test nMSE | denoising | inpainting | sensor rec. | source loc. | PDE rec. |
|---|---|---|---|---|---|
| `X = b` | 0.383 | 0.687 | 0.758 | 0.742 | 0.817 |
| least squares (25 CGLS steps) | 0.383 | 0.687 | 0.758 | **0.421** | **0.613** |
| Tikhonov LS, (λ, CG steps) tuned on val | **0.281** | 0.687 | 0.758 | 0.428 | 0.714 |
| graph-Laplacian LS, tuned on val | 0.302 | 0.687 | 0.758 | 0.428 | 0.714 |
| same net from scratch, 60 epochs on that task | 0.313 ± 0.012 | **0.658 ± 0.011** | 0.733 ± 0.007 | 0.609 ± 0.007 | 0.730 ± 0.006 |
| pretrained, task held out, zero-shot | 0.317 ± 0.018 | 0.683 ± 0.020 | 0.732 ± 0.006 | 0.652 ± 0.002 | 0.805 ± 0.001 |
| pretrained, task held out, + adapted head (4 ep.) | 0.301 ± 0.005 | 0.668 ± 0.005 | **0.725 ± 0.005** | 0.639 ± 0.003 | 0.804 ± 0.001 |

Reading. On the two null-space tasks no classical solver beats zero-fill (the tuned λ is 0) and the
learned prior is the only thing that does; the pretrained model transfers there (sensor recovery:
zero-shot ties from-scratch, adapted head is ahead; inpainting: zero-shot is at zero-fill, the adapted
head recovers most of the gap). On denoising a tuned scalar shrinkage (`x = b / 1.32`) beats every
learned model. On the two ill-conditioned operators every learned model, from scratch included, is far
behind plain CGLS, and the pretrained models are a further 0.03–0.07 behind from-scratch. Seed spread
is ≤ 0.02 everywhere.

**Test-time solver ablation (A, no retraining):** `results/cpox_task_holdout/solver_ablation.png`.
The same checkpoints run with a different data-consistency budget, held-out-task cells:

| nMSE | source loc. | PDE rec. |
|---|---|---|
| least squares, 5 / 25 / 100 CGLS steps | 0.632 / 0.421 / **0.173** | 0.716 / 0.613 / 0.610 |
| from scratch, 1 / 5 / 25 CG steps per unrolled iteration | 0.778 / 0.609 / 0.514 | 0.829 / 0.730 / 0.724 |
| pretrained + task head, 1 / 5 / 25 CG steps per iteration | 0.784 / 0.639 / 0.627 | 0.830 / 0.804 / 0.804 |
| from scratch, 10 instead of 5 unrolled iterations | 0.612 | 0.924 ± 0.287 (diverges on one seed) |
| from scratch, λ_task := 1e-8, 5 / 25 CG steps per iteration | 0.609 / 0.471 | 0.729 / 0.722 |
| pretrained + task head, λ_task := 1e-8, 5 / 25 CG steps per iteration | 0.627 / 0.471 | 0.728 / 0.717 |

Source localization is noise-free, so plain CG keeps improving with budget (0.17 at 100 steps); the
learned models do not follow. Neither the budget nor the data-consistency weight explains it: with
λ_task overridden to ~0 (a pure least-squares step from the prior's output) and 25 CG steps per
iteration, every learned model lands at 0.47 on source localization and 0.72 on PDE, the error of a
25-step CG run from zero, because each unrolled iteration restarts the CG solve at the backbone's
output and so never accumulates more than one iteration's worth of progress. Removing the weight does
lift the pretrained PDE model from 0.80 to 0.73 (its held-out λ_task is the untrained initial 0.01).
With one CG step all models fall to the `X = b` level: the data-consistency step, not the prior, does
the work on these operators. Mask tasks are unaffected by every setting.

**Experiment B (leave-one-dataset-out, 3 seeds):** `results/dataset_holdout/comparison_table.png`.
Every row is zero-shot on the held-out dataset (the model trained on the other two or three datasets);
"task heads" = 20 more epochs of head training on the *training* datasets. Bold = best in that column.
Provenance: the classical rows were computed on CPU; the learned rows are the ones the same evaluator
printed at the end of the GPU run (`results/run_gpu_CB.log`, mean ± std over 3 seeds), copied by
`plot_2.py --model_rows_from`, because re-running the 30 checkpoints on the Windmill test set (a complete
graph) takes about a day on a CPU. The only difference between the two is the noise draw of the denoising
task (third decimal); all other measurements are identical. The B regularisers are tuned on the first 5
validation batches (20 snapshots) of each training dataset.

| held-out dataset | method | denoising | inpainting | sensor rec. | source loc. | PDE rec. |
|---|---|---|---|---|---|---|
| CPOX (20 nodes) | least squares (25 CGLS) | 0.474 | 0.703 | **0.757** | **0.384** | **0.563** |
|  | Tikhonov LS, tuned | 0.400 | 0.703 | **0.757** | 0.398 | 0.687 |
|  | graph-Laplacian LS, tuned | 0.413 | **0.694** | 0.888 | 0.398 | 0.687 |
|  | pretrained, shared head | 0.466 ± 0.060 | 0.831 ± 0.188 | 0.793 ± 0.052 | 0.642 ± 0.095 | 0.770 ± 0.044 |
|  | pretrained, task heads | **0.400 ± 0.014** | 0.812 ± 0.159 | 0.789 ± 0.055 | 0.805 ± 0.228 | 0.726 ± 0.017 |
| Montevideo (675) | least squares (25 CGLS) | 0.127 | **0.991** | 0.992 | 0.385 | 0.565 |
|  | Tikhonov LS, tuned | **0.120** | **0.991** | 0.992 | **0.269** | **0.528** |
|  | graph-Laplacian LS, tuned | 0.122 | 0.991 | **0.992** | **0.269** | 0.528 |
|  | pretrained, shared head | 0.148 ± 0.013 | 1.059 ± 0.055 | 1.073 ± 0.060 | 0.457 ± 0.002 | 0.612 ± 0.001 |
|  | pretrained, task heads | 0.146 ± 0.018 | 1.023 ± 0.032 | 1.050 ± 0.069 | 0.446 ± 0.009 | 0.611 ± 0.002 |
| PedalMe (15, complete) | least squares (25 CGLS) | 0.102 | 0.599 | 0.710 | **0.706** | **0.706** |
|  | Tikhonov LS, tuned | 0.092 | 0.599 | 0.710 | **0.706** | **0.706** |
|  | graph-Laplacian LS, tuned | **0.090** | 0.545 | **0.575** | **0.706** | **0.706** |
|  | pretrained, shared head | 0.130 ± 0.040 | 0.499 ± 0.003 | 0.606 ± 0.007 | 0.706 ± 0.000 | 0.706 ± 0.000 |
|  | pretrained, task heads | 0.116 ± 0.038 | **0.493 ± 0.004** | 0.603 ± 0.011 | 0.706 ± 0.000 | 0.706 ± 0.000 |
| WikiMaths (1068) | least squares (25 CGLS) | 0.068 | 0.995 | 0.995 | 0.516 | 0.554 |
|  | Tikhonov LS, tuned | 0.064 | 0.995 | 0.995 | **0.471** | **0.552** |
|  | graph-Laplacian LS, tuned | **0.061** | 0.995 | 0.991 | **0.471** | 0.578 |
|  | pretrained, shared head | 0.080 ± 0.010 | 0.989 ± 0.007 | 0.988 ± 0.008 | 0.572 ± 0.036 | 0.590 ± 0.019 |
|  | pretrained, task heads | 0.075 ± 0.002 | **0.986 ± 0.005** | **0.986 ± 0.004** | 0.575 ± 0.017 | 0.586 ± 0.011 |
| Windmill (319, complete) | least squares (25 CGLS) | 0.100 | 0.981 | 0.985 | **0.354** | 0.354 |
|  | Tikhonov LS, tuned | 0.095 | 0.981 | 0.985 | **0.354** | **0.354** |
|  | graph-Laplacian LS, tuned | **0.085** | 0.981 | **0.830** | **0.354** | 0.354 |
|  | pretrained, shared head | 0.096 ± 0.007 | **0.873 ± 0.024** | 0.889 ± 0.021 | 0.354 ± 0.000 | 0.354 ± 0.000 |
|  | pretrained, task heads | 0.094 ± 0.008 | 0.890 ± 0.025 | 0.908 ± 0.027 | 0.354 ± 0.000 | 0.354 ± 0.000 |

Reading. Mostly negative. On the two complete graphs (PedalMe, Windmill) the Laplacian has two
eigenvalues, so source localization and PDE reconstruction reduce to recovering the mean and every method
coincides; there the prior wins inpainting (0.49 vs 0.55, 0.89 vs 0.98) but loses sensor recovery to the
tuned graph-Laplacian solver (0.60 vs 0.58, 0.91 vs 0.83). On the sparse Montevideo graph the model is
behind the tuned solvers on every task and above zero-fill on the mask tasks; on WikiMaths it is ahead only
on inpainting, by 0.01. CPOX held out (trained on PedalMe + Montevideo only) is the worst case: worse than
least squares on four tasks with a seed spread up to 0.23.

**Preliminary tables** (`old_models/`, trained 2026-09-27 to 2026-10-05, *before* the final
operator and evaluation fixes, kept for the record): see `CHANGES_SUMMARY.txt` for what was wrong with
them (test-set checkpoint selection, head-aliasing bug on load, trainable PDE physics, a WikiMaths
blow-up from directed normalisation, PDE at τ = 1 trivially invertible). They are not comparable to
the tables above.

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
src/script_scratch.py           experiment C driver: from-scratch single-task baselines (ours)
src/classical.py                tuned Tikhonov / graph-Laplacian baselines, device-independent measurements (ours)
src/ablate_solver.py            test-time solver-budget ablation on the A checkpoints (ours)
src/checkpoints.py              checkpoint naming + loading shared by both evaluators (ours)
paper/                          the 4-page paper (main.tex/main.pdf); make_tables.py builds its tables from results/*.csv
src/check_save_load.py          save/load round-trip test: run it after touching networks.py (ours)
old_models/                     preliminary checkpoints + tables (pre-fix, see Results)
results/                        final checkpoints + tables written by run_gpu.sh
configurations/, environment.yml, src/forwardOps.py, src/custom*.py, src/pygt_dataloader.py   upstream
```

## Limitations (please read before quoting numbers)

* **Baselines.** `X = 0`, `X = b` and 25-step least squares are floors; the tuned Tikhonov /
  graph-Laplacian solvers and the from-scratch network are the competitors. The classical weights
  come from a grid (`λ ∈ {0, 1e-6 … 100}`, CG steps `∈ {5, 25, 100}`) and are selected on the
  validation split, which for PDE reconstruction picks a CG budget (100) that is not the test-optimal
  one (25); both rows are reported as they are.
* **Head adaptation is supervised**, on the full training split; no few-shot claim is made.
* **Denoising** on CPOX is largely solved by the data-consistency step (the operator is the
  identity), so its column mostly measures the data term, not the prior.
* **PedalMe and Windmill are complete graphs**, so their Laplacian has two eigenvalues and PDE
  reconstruction and source localization both reduce to "recover the mean"; identical baselines
  there are inherent.
* GRIP's own `--method drip` path is untouched except for a constructor-name typo fix.

## Compute note

| what | hardware | wall time | log |
|---|---|---|---|
| experiment A (15 runs) | 8-core laptop CPU (`GRIP_DEVICE=cpu`) | 2 h 05 min | `results/run_cpu.log` |
| experiments C + B (30 runs) | 1× NVIDIA GTX 1080 Ti (11 GB), torch 2.7.1+cu126 | 23 h 15 min | `results/run_gpu_CB.log` |
| all tables + ablation (no training) | 8-core laptop CPU | < 2 h | `results/run_eval_cpu.log` |

Model: 32 channels × 32 layers, 101 227 (A, C) / 101 259 (B) trainable parameters. No run hit NaN,
divergence or out-of-memory. The pinned `requirements.txt` is the CPU environment; the GPU machine
used torch 2.7.1+cu126 with the matching PyG wheels (see `GPU_RUN.md`).

## Citation of the upstream work

```bibtex
@inproceedings{eliasof2025learning,
  title={Learning Regularization for Graph Inverse Problems},
  author={Eliasof, Moshe and Siddiqui, Md Shahriar Rahim and Sch{\"o}nlieb, Carola-Bibiane and Haber, Eldad},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39}, number={16}, pages={16471--16479}, year={2025}
}
```
