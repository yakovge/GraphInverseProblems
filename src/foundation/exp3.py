"""Experiment 3.

    train : denoising, inpainting, source_localization
    val   : sensor_recovery   (early stopping only)
    test  : pde_state         (never trained on, never selected on)

Run:  python exp3.py --dataset METRLA
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import train_foundation_model  # noqa: E402
from config import parse  # noqa: E402
from tasks import experiment_split  # noqa: E402

EXPERIMENT_INDEX = 2

if __name__ == "__main__":
    args = parse(__doc__)
    train_tasks, val_task, test_task = experiment_split(EXPERIMENT_INDEX)
    train_foundation_model(args, train_tasks, val_task, test_task)
