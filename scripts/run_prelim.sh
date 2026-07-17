#!/usr/bin/env bash
# Preliminary experiment grid for the foundation-model transfer study.
# Run from the repo root. Designed for a single GPU (falls back to CPU with
# DEVICE=cpu). Pretraining checkpoints are cached under checkpoints/, so
# re-runs skip completed pretraining. Results land in results/*.csv.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
DEVICE=${DEVICE:-cuda:0}
DATA=${DATA:-data}

# 1) Sanity: operator tests (seconds)
(cd src && "../$PY" test_foundation_ops.py)

# 2) Main preliminary run: CLUSTER, pretrain on {denoise,mask,smooth,path},
#    hold out {maskSmooth (PDE-state), maxpool (nonlinear)}; 3 seeds.
(cd src && "../$PY" main_foundation_transfer.py \
    --device "$DEVICE" --datapath "../$DATA" --dataset CLUSTER \
    --pretrain_epochs 10 --seeds 0,1,2 \
    --out ../results/foundation_CLUSTER.csv)

# 3) Conditioning ablation: same run without operator conditioning (1 seed).
(cd src && "../$PY" main_foundation_transfer.py \
    --device "$DEVICE" --datapath "../$DATA" --dataset CLUSTER \
    --pretrain_epochs 10 --seeds 0 --op_conditioning 0 --skip_classical 1 \
    --out ../results/foundation_CLUSTER_nocond.csv)

echo "done. See results/*.csv"
