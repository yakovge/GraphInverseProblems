"""
plot_fixed.py -- evaluation table for script.py (CPOX task hold-out) checkpoints.

    python src/plot_fixed.py results/cpox_task_holdout            # table -> <dir>/loss_matrix_table.png
    python src/plot_fixed.py models --max_batches 2               # smoke test

Every script.py checkpoint in the directory is evaluated on the CPOX TEST weeks on all five tasks. For a
checkpoint trained with task T held out, only the T column is a transfer number (marked in the table); the
other columns are tasks it trained on. Rows:
  zero-shot (shared head)  no_head_training_*: best-validation backbone + shared head, T never trained on
  adapted (task heads)     *_test.pth: + task heads, T's head trained on training snapshots (backbone frozen)
Seeds (…_seed_k.pth) of the same run are averaged: cells show mean ± std.

Measurements are produced EXACTLY as in training: utils.process_data + utils.task_specific_modifiers on
shared operator objects (same mask budget, clustered sensors, observation flag, noise std, tau). Within a
batch every row inverts the identical measurement. Baselines:
  X = 0           predict zero                       (nMSE = 1 by definition)
  X = b           return the measurement             (only where b has the shape of x)
  least squares   CGLS from zero, no learned prior   (= zero-fill for masks, = b for denoising)
  Tikhonov / Laplacian LS   regularised least squares, weight tuned per task on the CPOX validation weeks
                            (the split the learned models were selected on), see classical.py
Column headers carry the operator's condition number (inf = genuine null space).
Noise is drawn on the CPU generator (classical.measure), so the table is device independent.
"""
import os
import math
import argparse

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import get_data_and_loaders, process_data, task_specific_modifiers, get_forward_op
from networks import graph_CGLS
from checkpoints import load_foundation_checkpoint, parse_checkpoint_name, fmt_cell, TASK_OF_FLAG
from classical import measure, tune_lambdas, regularised_ls, ROW_OF, REGULARISERS

TASKS = ['denoising', 'inpainting', 'sensor_recovery', 'source_localization', 'pde_reconstruction']
BASELINES = ['BASELINE: X = 0', 'BASELINE: X = b', 'BASELINE: least squares'] + [ROW_OF[r] for r in REGULARISERS]
SEED = 0
VAL_SEED = 500000  # validation measurements (lambda tuning) use their own seeds, disjoint from the test ones
_TUNED = {}        # max_batches -> tuned classical (lam, CG steps); filled on the first evaluate() call


def make_args():
    # operator settings = script.py / main_3 defaults; layers / channels / dc are read from each checkpoint
    return argparse.Namespace(
        dataset='CPOX', use_meta_data=1, classify=0, CPOX_lags=1, val_frac=0.1,
        train_batch_size=4, test_batch_size=4, train_frac=1.0, test_frac=1.0,
        method='foundation', layers=32, channels=32, dc='prox', cglsIter=5, solveIter=5,
        rnfPE=1, dropout=0.0, task='mask', mask_per_snapshot_budget=6, n_sensors=5, blur_count='4', pde_tau=20.0,
        noise=True, painting=True, blurring=True, sensoring=True, pdessm=True)


def nmse(X, y):
    return (F.mse_loss(X, y) / F.mse_loss(torch.zeros_like(y), y)).item()


@torch.no_grad()
def condition_number(op, g, task):
    """Dense probe of the operator in data space on one batch (inf = null space)."""
    if task == 'denoising':
        return float('nan')  # stochastic; the linear part is the identity
    cols = [op(torch.eye(g.y.shape[0], device=g.y.device)[:, j:j + 1], g.edge_index, g.edge_weight, emb=False).reshape(-1)
            for j in range(g.y.shape[0])]
    s = torch.linalg.svdvals(torch.stack(cols, dim=1).float())
    return (s[0] / s[-1]).item() if s[-1] > 1e-10 else float('inf')


def fmt_cond(c):
    return 'stochastic' if c != c else ('inf (null space)' if c == float('inf') else f'{c:.1f}')


SCRATCH_ROW = 'BASELINE: single-task model from scratch (same net, 60 epochs on that task only)'


def measurements(loader, args, ops, device, seed, max_batches=None):
    """Yield (batch index, task, g, b, op) for every batch x task, built exactly as in training; the seed fixes
    masks / sensors / noise so that every row inverts the identical measurement."""
    for bi, raw in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        for k, (op, task) in enumerate(ops):
            torch.manual_seed(seed + 1000 * bi + k)
            g, op = task_specific_modifiers(process_data(args, raw.clone()), args, op, None)
            g = g.to(device)
            yield bi, task, g, measure(op, g), op


@torch.no_grad()
def evaluate(files, device, max_batches=None, own_task=None, solver=None, ls_iters=None):
    """-> rows[name][task] = [nMSE per seed], cond[task], lambdas. files = [(row name, path)];
    own_task[path] = the single task of a from-scratch baseline checkpoint (only that column is filled).
    solver = {'cgls_iter': k, 'niter': m} overrides the loaded models' data-consistency / unrolling depth at test
    time (ablate_solver.py); ls_iters overrides the CG budget of the least-squares baseline (default 5 x 5)."""
    own_task = own_task or {}
    args = make_args()
    _, _, _, test_loader, label_channels, feat_channels, val_loader = get_data_and_loaders(args)
    args.num_nodes = next(iter(test_loader)).num_nodes // 1
    ops = get_forward_op(args, args.channels, label_channels, device)  # one operator per task, shared by every model
    models = {path: load_foundation_checkpoint(path, args, ops, device, label_channels, feat_channels) for _, path in files}
    for net in models.values():
        for k, v in (solver or {}).items():
            if k == 'lam':  # test-time override of every data-consistency weight lam_task = exp(20 * dc_lam)
                for q in net.dc_lam.values():
                    q.data.fill_(math.log(v) / 20)
            else:
                setattr(net, k, v)
    if max_batches not in _TUNED:  # the classical weights depend only on the validation split: tune once per process
        print("tuning the classical regularisers on the validation weeks ...", flush=True)
        _TUNED[max_batches] = tune_lambdas((x[1:] for x in measurements(val_loader, args, ops, device, VAL_SEED, max_batches)))[0]
    lambdas = _TUNED[max_batches]
    ls_iters = ls_iters or args.cglsIter * args.solveIter
    sums = {r: {t: 0.0 for t in TASKS} for r in BASELINES + list(models)}
    count = {r: {t: 0 for t in TASKS} for r in sums}
    cond = {}
    for bi, task, g, b, op in measurements(test_loader, args, ops, device, SEED, max_batches):
        if task not in cond:
            cond[task] = condition_number(op, g, task)
        sums['BASELINE: X = 0'][task] += nmse(torch.zeros_like(g.y), g.y); count['BASELINE: X = 0'][task] += 1
        if b.shape == g.y.shape:
            sums['BASELINE: X = b'][task] += nmse(b, g.y); count['BASELINE: X = b'][task] += 1
        ls, _ = graph_CGLS(op, CGLSit=ls_iters, eps=1e-5)(b, torch.zeros_like(b), g.edge_index, g.edge_weight)
        sums['BASELINE: least squares'][task] += nmse(ls, g.y); count['BASELINE: least squares'][task] += 1
        for reg in REGULARISERS:
            xr = regularised_ls(op, b, g.edge_index, g.edge_weight, lambdas[reg][task][0], reg, lambdas[reg][task][1])
            sums[ROW_OF[reg]][task] += nmse(xr, g.y); count[ROW_OF[reg]][task] += 1
        for path, net in models.items():
            if task not in net.task_heads:  # single-task baseline: evaluated on its own task only
                continue
            net.set_task(task)
            X, _, _ = net(b, g.edge_index, g.edge_weight, g.x)
            sums[path][task] += nmse(torch.nan_to_num(X, nan=1e6), g.y); count[path][task] += 1
    per_file = {r: {t: (sums[r][t] / count[r][t] if count[r][t] else float('nan')) for t in TASKS} for r in sums}
    rows = {}
    for name, path in [(b, b) for b in BASELINES] + files:
        for t in TASKS:
            if path in own_task and t != own_task[path]:
                rows.setdefault(name, {}).setdefault(t, [])
                continue
            rows.setdefault(name, {}).setdefault(t, []).append(per_file[path][t])
    return rows, cond, lambdas


def plot_table(rows, cond, holdout_of, save_path):
    labels = list(rows)
    cells = [[fmt_cell(rows[r][t]) for t in TASKS] for r in labels]
    colors = []
    for r in labels:
        if r.startswith('BASELINE'):
            colors.append(['#fff3cd'] * len(TASKS))
        else:
            colors.append(['#cfe2ff' if t == holdout_of.get(r) else '#ffffff' for t in TASKS])
    n_seeds = max(len(v) for r in rows.values() for v in r.values())
    fig, ax = plt.subplots(figsize=(17, 0.38 * len(labels) + 1.5))
    ax.axis('off')
    table = ax.table(cellText=cells, rowLabels=labels, colLabels=[f"{t}\ncond {fmt_cond(cond[t])}" for t in TASKS],
                     cellColours=colors, loc='center', cellLoc='center',
                     colColours=['#40466e'] * len(TASKS), rowColours=['#e9ecef'] * len(labels))
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.0, 1.7)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(color='white', weight='bold')
    plt.title(f"CPOX test weeks, nMSE (lower is better){f', mean ± std over {n_seeds} seeds' if n_seeds > 1 else ''}\n"
              "blue = the held-out task of that row (transfer); white = tasks the row trained on; yellow = baseline",
              weight='bold', size=11, pad=20)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nTable saved to {save_path}")


def main(model_dir, max_batches=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    files, holdout_of, scratch, own_task = [], {}, [], {}
    for fn in sorted(os.listdir(model_dir)):
        info = parse_checkpoint_name(fn)
        if info is None or info['kind'] != 'test':
            if fn.endswith('.pth'):
                print(f"skipping {fn} ({'script2.py checkpoint: use plot_2.py' if info else 'not a script.py checkpoint'})")
            continue
        task = TASK_OF_FLAG[info['holdout']]
        if info['run'].startswith('scratch'):  # script_scratch.py: one model per task, trained on that task only
            scratch.append((SCRATCH_ROW, os.path.join(model_dir, fn)))
            own_task[os.path.join(model_dir, fn)] = task
            continue
        name = f"{task} held out | {info['variant']}"
        files.append((name, os.path.join(model_dir, fn)))
        holdout_of[name] = task
    if not files:
        print(f"No script.py checkpoints in '{model_dir}'.")
        return
    print(f"{len(files)} checkpoints on {device}: {sorted({n for n, _ in files})}")
    files = scratch + files  # baseline row right under the classical baselines
    rows, cond, lambdas = evaluate(files, device, max_batches, own_task)
    for r in rows:
        print(f"  {r:48s} " + "  ".join(f"{t[:8]} {fmt_cell(rows[r][t])}" for t in TASKS))
    print("operator condition numbers:", {t: fmt_cond(c) for t, c in cond.items()})
    print("tuned regularisation weights:", lambdas)
    plot_table(rows, cond, holdout_of, os.path.join(model_dir, 'loss_matrix_table.png'))
    with open(os.path.join(model_dir, 'loss_matrix_table.csv'), 'w') as f:  # the numbers behind the png
        f.write('row,' + ','.join(TASKS) + '\n')
        for r in rows:
            f.write('"' + r + '",' + ','.join(';'.join(f"{v:.6f}" for v in rows[r][t]) for t in TASKS) + '\n')


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('model_dir', nargs='?', default='models', help='directory with script.py checkpoints')
    p.add_argument('--max_batches', type=int, default=None, help='evaluate only this many test batches (smoke test)')
    a = p.parse_args()
    main(a.model_dir, a.max_batches)
