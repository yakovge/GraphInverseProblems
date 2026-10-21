"""Test-time solver ablation for the CPOX task hold-out (experiment A): no retraining.

    python src/ablate_solver.py results/cpox_task_holdout        # -> <dir>/solver_ablation.{csv,png}

Question: the learned models lose to 25 plain CGLS steps on the two ill-conditioned operators (source localization
A^4, PDE exp(-20 L)). Is that because the unrolled solver runs only 5 CG steps per data-consistency step, or
because of the data-consistency weight lam_task? The same checkpoints are evaluated with the CG budget, the
unrolling depth and lam_task changed at test time, and the least-squares baseline with 5 / 25 / 100 CGLS steps. Rows = held-out-task cells of the A table (the blue
diagonal) and the from-scratch baseline row; everything else (seeds, measurements, splits) is as in plot_fixed.py.
"""
import os
import argparse

import torch
import matplotlib.pyplot as plt

import plot_fixed as pf
from checkpoints import parse_checkpoint_name, fmt_cell, TASK_OF_FLAG

SETTINGS = [  # (label, model overrides, LS CGLS steps)
    ('trained setting: 5 unrolled it. x 5 CG steps; LS 25 steps', {'niter': 5, 'cgls_iter': 5}, 25),
    ('1 CG step per unrolled it.; LS 5 steps', {'niter': 5, 'cgls_iter': 1}, 5),
    ('25 CG steps per unrolled it.; LS 100 steps', {'niter': 5, 'cgls_iter': 25}, 100),
    ('10 unrolled it. x 5 CG steps; LS 25 steps', {'niter': 10, 'cgls_iter': 5}, 25),
    # data-consistency weight set to ~0 at test time (the held-out task's weight is 0.01 = its initial value in the
    # zero-shot checkpoints, 6e-3 after head adaptation, 1e-4..1e-5 when the task was trained in the backbone phase)
    ('lam = 1e-8 at test time, 5 x 5; LS 25 steps', {'niter': 5, 'cgls_iter': 5, 'lam': 1e-8}, 25),
    ('lam = 1e-8 at test time, 25 CG steps per unrolled it.; LS 100 steps', {'niter': 5, 'cgls_iter': 25, 'lam': 1e-8}, 100),
]
ROWS = ['least squares', 'single-task from scratch', 'held out, zero-shot', 'held out, task head']


def main(model_dir, max_batches=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    files, holdout_of, scratch, own_task = [], {}, [], {}
    for fn in sorted(os.listdir(model_dir)):
        info = parse_checkpoint_name(fn)
        if info is None or info['kind'] != 'test':
            continue
        task, path = TASK_OF_FLAG[info['holdout']], os.path.join(model_dir, fn)
        if info['run'].startswith('scratch'):
            scratch.append((pf.SCRATCH_ROW, path)); own_task[path] = task
        else:
            name = f"{task} held out | {info['variant']}"
            files.append((name, path)); holdout_of[name] = task
    table = {}  # (setting, row) -> task -> [per seed]
    for label, overrides, ls_iters in SETTINGS:
        print(f"\n== {label}", flush=True)
        rows, _, _ = pf.evaluate(scratch + files, device, max_batches, own_task, solver=overrides, ls_iters=ls_iters)
        out = {r: {t: [] for t in pf.TASKS} for r in ROWS}
        out['least squares'] = {t: rows['BASELINE: least squares'][t] for t in pf.TASKS}
        out['single-task from scratch'] = {t: rows[pf.SCRATCH_ROW][t] for t in pf.TASKS}
        for name, t in holdout_of.items():
            key = 'held out, zero-shot' if 'zero-shot' in name else 'held out, task head'
            out[key][t] = rows[name][t]
        for r in ROWS:
            table[(label, r)] = out[r]
            print(f"  {r:26s} " + "  ".join(f"{t[:8]} {fmt_cell(out[r][t])}" for t in pf.TASKS))
    with open(os.path.join(model_dir, 'solver_ablation.csv'), 'w') as f:
        f.write('setting,row,' + ','.join(pf.TASKS) + '\n')
        for (label, r), d in table.items():
            f.write(f'"{label}","{r}",' + ','.join(';'.join(f"{v:.6f}" for v in d[t]) for t in pf.TASKS) + '\n')
    labels = [f"{label} | {r}" for (label, r) in table]
    cells = [[fmt_cell(d[t]) for t in pf.TASKS] for d in table.values()]
    fig, ax = plt.subplots(figsize=(17, 0.36 * len(labels) + 1.2)); ax.axis('off')
    tb = ax.table(cellText=cells, rowLabels=labels, colLabels=pf.TASKS, loc='center', cellLoc='center',
                  colColours=['#40466e'] * len(pf.TASKS), rowColours=['#e9ecef'] * len(labels))
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1.0, 1.6)
    for (row, col), cell in tb.get_celld().items():
        if row == 0:
            cell.set_text_props(color='white', weight='bold')
    plt.title("Test-time solver ablation on the CPOX task hold-out (held-out-task cells), nMSE mean ± std over 3 seeds",
              weight='bold', size=11, pad=16)
    plt.savefig(os.path.join(model_dir, 'solver_ablation.png'), dpi=200, bbox_inches='tight')
    print(f"\nSaved {model_dir}/solver_ablation.csv and .png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('model_dir', nargs='?', default='results/cpox_task_holdout')
    p.add_argument('--max_batches', type=int, default=None)
    a = p.parse_args()
    main(a.model_dir, a.max_batches)
