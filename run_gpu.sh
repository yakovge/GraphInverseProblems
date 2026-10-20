#!/usr/bin/env bash
# Full experiment pipeline (GPU). See GPU_RUN.md for setup and what each step produces.
#   bash run_gpu.sh            # everything: experiment A, its table, experiment B, its table
#   bash run_gpu.sh A          # only the CPOX task-holdout experiment + table
#   bash run_gpu.sh B          # only the leave-one-dataset-out experiment + table
#   bash run_gpu.sh C          # only the from-scratch single-task baselines (added to the A table)
#   bash run_gpu.sh AC         # A, then its baselines, then the A table
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}                       # interpreter of the environment with torch + PyG (activate the venv first)
export WANDB_MODE=${WANDB_MODE:-offline} # offline = no login needed; set WANDB_MODE=online after `wandb login` to sync
WHAT=${1:-ACB}

$PY src/check_save_load.py                # checkpoint round trip must be exact before anything is trained

if [[ "$WHAT" == *A* ]]; then
  echo "=== A) CPOX task hold-out: 5 runs (one held-out task each) ==="
  $PY src/script.py
  mkdir -p results/cpox_task_holdout
  mv models/*.pth results/cpox_task_holdout/
  if [[ "$WHAT" != *C* ]]; then
    echo "=== A) evaluation table ==="
    $PY src/plot_fixed.py results/cpox_task_holdout      # -> results/cpox_task_holdout/loss_matrix_table.png
  fi
fi

if [[ "$WHAT" == *C* ]]; then
  echo "=== C) from-scratch single-task baselines on CPOX: 5 runs ==="
  $PY src/script_scratch.py
  mkdir -p results/cpox_task_holdout
  mv models/*.pth results/cpox_task_holdout/
  echo "=== A+C) evaluation table ==="
  $PY src/plot_fixed.py results/cpox_task_holdout        # foundation rows + the single-task baseline row
fi

if [[ "$WHAT" == *B* ]]; then
  echo "=== B) leave-one-dataset-out: 5 runs (one held-out dataset each) ==="
  $PY src/script2.py
  mkdir -p results/dataset_holdout
  mv models/*.pth results/dataset_holdout/
  echo "=== B) evaluation table ==="
  $PY src/plot_2.py results/dataset_holdout              # -> results/dataset_holdout/comparison_table.png
fi

echo "Done. Tables: results/*/..._table.png  Checkpoints: results/*/*.pth"
