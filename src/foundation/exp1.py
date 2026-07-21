"""Experiment 1 (protocol v3).

    train     : source_localization, sensor_recovery, pde_state
    zero-shot : inpainting, denoising   (no gradients, no model selection)

Denoising is eval-only in v3: its identity operator makes the CGLS data-projection
return the observation exactly, so its training gradient is identically zero.
See FOUNDATION_MODEL.md, "Protocol v3".

Run:  python exp1.py --dataset METRLA
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import train_foundation_model  # noqa: E402
from config import parse  # noqa: E402
from tasks import experiment_split_v3  # noqa: E402

EXPERIMENT_INDEX = 0

if __name__ == "__main__":
    args = parse(__doc__)
    train_tasks, holdout, eval_only = experiment_split_v3(EXPERIMENT_INDEX)
    train_foundation_model(args, train_tasks, val_task=holdout, test_task=eval_only)
