# Running the experiments on the GPU machine

Everything below runs from the repository root on branch `final`. Total time is roughly
1 hour for experiment A and several hours for experiment B (WikiMaths batches dominate), so
start B in a `tmux`/`screen` session or overnight.

## 1. Get the code

```bash
git clone -b final https://github.com/yakovge/GraphInverseProblems.git
cd GraphInverseProblems
```

If the repo is already cloned: `git fetch fork && git checkout final && git pull`.

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

Sanity check (must print `cuda True` and `cluster ok`):

```bash
python -c "import torch, torch_geometric, torch_geometric_temporal, torch_cluster; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('cluster ok')"
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
  --dc prox --channels 128 --layers 32 --noise True --painting True --blurring True --pdessm True \
  --method foundation --cglsIter 5 --solveIter 5 --classify 0 --max_patience 10000 \
  --mask_per_snapshot_budget 6 --project_name smoke_test
python src/plot_fixed.py models --max_batches 2      # writes models/loss_matrix_table.png
rm -rf models                                         # discard the smoke-test checkpoints
```

## 5. Run everything

```bash
bash run_gpu.sh          # A then B; or `bash run_gpu.sh A` / `bash run_gpu.sh B`
```

What it does:

| Step | Command | Output |
|---|---|---|
| A: 5 CPOX runs, one held-out task each | `src/script.py` | `results/cpox_task_holdout/*.pth` (10 checkpoints) |
| A: table | `src/plot_fixed.py results/cpox_task_holdout` | `results/cpox_task_holdout/loss_matrix_table.png` |
| B: 5 runs, one held-out dataset each | `src/script2.py` | `results/dataset_holdout/*.pth` (10 checkpoints) |
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

Then paste the two tables' numbers into the README's Results section.

## Troubleshooting

- **CUDA out of memory in experiment B.** script2.py at 128 channels needs ~6 GB on WikiMaths
  batches. Set `'channels': 64` in `src/script2.py` (and say so in the README), or lower
  `--train_batch_size` by adding `'train_batch_size': 2` to the dict.
- **`ModuleNotFoundError: torch_cluster`.** The PyG wheel index did not match your torch/CUDA;
  rerun step 2 with the exact `TORCH`/`CUDA` strings printed by the sanity check.
- **Dataset download fails.** PyTorch-Geometric-Temporal fetches CPOX, PedalMe, WikiMaths,
  MontevideoBus and Windmill from GitHub on first use; it needs internet once. METR-LA is not
  used (its host is dead).
- **`--noise False` etc.** Task flags accept `True/False`; omitting a flag also means False.
- **wandb asks to log in.** `export WANDB_MODE=offline` (run_gpu.sh does this for you).
