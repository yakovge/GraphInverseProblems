"""Experiment 4.

    train : inpainting, source_localization, sensor_recovery
    val   : pde_state   (early stopping only)
    test  : denoising   (never trained on, never selected on)

Run:  python exp4.py --dataset METRLA
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import train_foundation_model  # noqa: E402
from config import parse  # noqa: E402
from tasks import experiment_split  # noqa: E402

EXPERIMENT_INDEX = 3

if __name__ == "__main__":
    args = parse(__doc__)
    train_tasks, val_task, test_task = experiment_split(EXPERIMENT_INDEX)
    train_foundation_model(args, train_tasks, val_task, test_task)
