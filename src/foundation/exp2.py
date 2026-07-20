"""Experiment 2.

    train : denoising, inpainting, pde_state
    val   : source_localization   (early stopping only)
    test  : sensor_recovery       (never trained on, never selected on)

Run:  python exp2.py --dataset METRLA
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import train_foundation_model  # noqa: E402
from config import parse  # noqa: E402
from tasks import experiment_split  # noqa: E402

EXPERIMENT_INDEX = 1

if __name__ == "__main__":
    args = parse(__doc__)
    train_tasks, val_task, test_task = experiment_split(EXPERIMENT_INDEX)
    train_foundation_model(args, train_tasks, val_task, test_task)
