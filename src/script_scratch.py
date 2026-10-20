"""From-scratch single-task baselines for the CPOX task hold-out (experiment A, row "single-task model from scratch").

One model per task, same architecture as the foundation model (32 x 32, shared head, no task heads), trained
for 60 epochs on that task only = the 40 + 16 + 4 epochs a foundation-model run gets, so the comparison is at
equal training budget. Checkpoints: models/no_head_training_scratch60_<flag>_test[_seed_k].pth; plot_fixed.py
puts them in one yellow baseline row (each column from the model trained on that column's task).
"""
import os
from script import create_message, run_commands_sequentially

FLAGS = ['pdessm', 'blurring', 'painting', 'sensoring', 'noise']   # task flags, see checkpoints.TASK_OF_FLAG

dic = {
    'device': os.environ.get('GRIP_DEVICE', 'cuda:0'),
    'epochs': 60,
    'dc': 'prox',
    'heads': False,          # shared encoder/head only (there is a single task)
    'single_task': True,     # test operator = the trained operator
    'channels': 32,
    'num_seeds': int(os.environ.get('GRIP_SEEDS', 3)),
    'layers': 32,
    'method': 'foundation',
    'cglsIter': 5,
    'solveIter': 5,
    'classify': 0,
    'max_patience': 10000,
    'mask_per_snapshot_budget': 6,
}

if __name__ == "__main__":
    my_commands = []
    for flag in FLAGS:
        d = dict(dic, **{flag: True}, project_name=f'scratch60_{flag}_test')
        my_commands.append(create_message(d))
    run_commands_sequentially(my_commands)
