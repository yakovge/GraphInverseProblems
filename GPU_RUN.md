# Running the experiments on the GPU machine

Everything below runs from the repository root on branch `final` of
**https://github.com/yakovge/GraphInverseProblems** (not the upstream nafi007 repo, not branch
`abra` or `main`). Each experiment is 5 runs × 3 seeds × 60 epochs at 32 channels × 32 layers.
On a laptop CPU one CPOX epoch takes ~20 s, so expect A in the order of 1–2 hours on a GPU and B
longer (WikiMaths batches dominate); run both inside `tmux`/`screen`.

## 1. Get the code

```bash
git clone -b final https://github.com/yakovge/GraphInverseProblems.git
cd GraphInverseProblems
git log --oneline -1        # must show the "final" branch's latest commit (see the hand-off message)
```

If the repo is already cloned with the fork as a remote: `git fetch <remote> && git checkout -B final <remote>/final`.

## 2. Environment (Python 3.10–3.12)

Install torch for your CUDA version first, then the PyG wheels built for that torch, then the rest.
Check your CUDA with `nvidia-smi` (top-right "CUDA Version").

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# 1) torch -- pick the index for your CUDA (cu126 / cu128 / cu130 ...), see https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2) PyG wheels compiled for THAT torch build (torch-cluster is imported by the code, it is not optional)
TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA=$(python -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))")
pip install torch-geometric torch-geometric-temporal
pip install torch-cluster torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

# 3) the rest
pip install numpy scipy matplotlib pandas networkx tqdm six wandb
```

Sanity checks (must print `cuda True`, `cluster ok`, and `OK: save/load round trip is exact`):

```bash
python -c "import torch, torch_geometric, torch_geometric_temporal, torch_cluster; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('cluster ok')"
python src/check_save_load.py
```

`requirements.txt` lists the exact versions the code was last run with, if you need to match them.

## 3. Weights & Biases

Training logs to wandb. `run_gpu.sh` sets `WANDB_MODE=offline` so no account is needed.
To sync to your wandb account instead: `wandb login` once, then run with `WANDB_MODE=online bash run_gpu.sh`.
Never commit an API key (the `keys` file is git-ignored).

## 4. Smoke test first (2–3 minutes)

Proves the GPU, the datasets (auto-downloaded on first use) and the full train → save → evaluate path work:

```bash
WANDB_MODE=offline python src/main_3_linear_inv_problems.py --epochs 1 --head_epochs 1 --test_head_epochs 1 \
  --num_seeds 2 --dc prox --channels 32 --layers 32 --noise True --painting True --blurring True --pdessm True \
  --method foundation --cglsIter 5 --solveIter 5 --classify 0 --max_patience 10000 \
  --mask_per_snapshot_budget 6 --project_name smoke_sensoring_test
ls models            # 4 files: [no_head_training_]smoke_sensoring_test_seed_{0,1}.pth
python src/plot_fixed.py models --max_batches 2      # writes models/loss_matrix_table.png (2 rows, mean ± std)
rm -rf models                                         # discard the smoke-test checkpoints
```

Expected in the training log: three lines `[phase 'pretrain' done] ...`, `[phase 'heads' done] ...`,
`[phase 'adapt' done] ... TEST loss` per seed. Those are the only test-set evaluations.

## 5. Run everything

```bash
bash run_gpu.sh          # A then B; or `bash run_gpu.sh A` / `bash run_gpu.sh B`
```

What it does:

| Step | Command | Output |
|---|---|---|
| A: 5 CPOX runs × 3 seeds, one held-out task each | `src/script.py` | `results/cpox_task_holdout/*_seed_k.pth` (30 checkpoints) |
| A: table | `src/plot_fixed.py results/cpox_task_holdout` | `results/cpox_task_holdout/loss_matrix_table.png` |
| B: 5 runs × 3 seeds, one held-out dataset each | `src/script2.py` | `results/dataset_holdout/*_seed_k.pth` (30 checkpoints) |
| B: table | `src/plot_2.py results/dataset_holdout` | `results/dataset_holdout/comparison_table.png` |

Each run prints its progress live. If a step fails, the finished runs stay in `models/`; fix the
problem and rerun only the missing run (edit the loop in `src/script.py` / `src/script2.py`),
then move the checkpoints into `results/...` and rerun the table command.

## 6. Commit the results

```bash
git add results/
git commit -m "Add final results: CPOX task hold-out and leave-one-dataset-out (retrained on fixed operators)"
git push fork final      # or: git push origin final, depending on your remote name
```

Also keep the training logs: `run_gpu.sh 2>&1 | tee results/run_gpu.log` captures the per-phase
`TEST loss` lines, which are the numbers the tables are built from. Then paste the two tables'
numbers into the README's Results section.

## Troubleshooting

- **CUDA out of memory in experiment B.** At 32 channels WikiMaths batches should fit in a few GB
  (128 channels needed ~6 GB). If not, add `'train_batch_size': 2` to the dict in `src/script2.py`.
- **A run crashed mid-way.** `script.py` stops at the first failing run; finished checkpoints stay in
  `models/`. Rerun only the missing run (edit the command list at the bottom of the script) with the
  same `--num_seeds 3`, then move everything into `results/...` and run the table command.
- **`ModuleNotFoundError: torch_cluster`.** The PyG wheel index did not match your torch/CUDA;
  rerun step 2 with the exact `TORCH`/`CUDA` strings printed by the sanity check.
- **Dataset download fails.** PyTorch-Geometric-Temporal fetches CPOX, PedalMe, WikiMaths,
  MontevideoBus and Windmill from GitHub on first use; it needs internet once. METR-LA is not
  used (its host is dead).
- **`--noise False` etc.** Task flags accept `True/False`; omitting a flag also means False.
- **wandb asks to log in.** `export WANDB_MODE=offline` (run_gpu.sh does this for you).
