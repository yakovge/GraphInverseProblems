"""
plot_2.py -- comparison table for script2.py (leave-one-dataset-out) checkpoints.

    python src/plot_2.py old_models/dataset_holdout

Every *_<dataset>_dataset_test.pth in the directory is evaluated zero-shot on its held-out dataset,
on all 5 tasks, with the same operators and settings as training (script2.py). Within a dataset every
model and baseline sees identical masks and noise. Baselines:
  X = 0          -- predict the mean (data is z-normalised), nMSE ~ 1
  least squares  -- CGLS from zero, no learned prior (= zero-fill for masks, = b for denoising).
                    A model that does not beat this row has learned nothing beyond the data fit.
  Tikhonov / Laplacian LS -- regularised least squares; the weight is tuned per task on the validation tails of the
                    datasets the model trained on (never on the held-out dataset), see classical.py.
Noise is drawn on the CPU generator (classical.measure), so the table is device independent.
The table is saved as comparison_table.png (+ .csv) in the directory.
"""
import os
import random
import argparse

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from utils import load_pgt_snapshots, get_fractional_dataset, process_data, task_specific_modifiers, get_forward_op
from networks import graph_CGLS
from checkpoints import load_foundation_checkpoint, parse_checkpoint_name, fmt_cell
from classical import measure, tune_lambdas, regularised_ls, ROW_OF, REGULARISERS

TASKS = ['denoising', 'inpainting', 'sensor_recovery', 'source_localization', 'pde_reconstruction']
BASELINES = ['BASELINE: X = 0', 'BASELINE: least squares'] + [ROW_OF[r] for r in REGULARISERS]
VAL_SEED = 500000
_SNAPSHOTS = {}  # dataset name -> snapshots (the pgt loaders re-download on every call)
_LAMBDAS = {}    # training-dataset tuple -> tuned (lam, iters) per regulariser and task


def snapshots(name):
    if name not in _SNAPSHOTS:
        _SNAPSHOTS[name] = load_pgt_snapshots(name)
    return _SNAPSHOTS[name]


def measurements(loader, args, ops, device, seed, max_batches=None):
    """(task, g, b, op) per batch x task, built exactly as in training with seeded masks / sensors / noise."""
    for bi, raw in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        for k, (op, task) in enumerate(ops):
            torch.manual_seed(seed + 1000 * bi + k)
            g, op = task_specific_modifiers(process_data(args, raw.clone()), args, op, None)
            g = g.to(device)
            yield task, g, measure(op, g), op


def validation_batches(dataset, args, ops, device, max_batches=None):
    """Validation tails (last val_frac) of the datasets a model with `dataset` held out trained on (script2.py)."""
    from script2 import DATASETS, TEST_ONLY
    for d in [d for d in DATASETS if d != dataset and d not in TEST_ONLY]:
        full = snapshots(d)
        tail = full[len(full) - int(0.1 * len(full)):]
        yield from measurements(DataLoader(tail, batch_size=4, shuffle=False), args, ops, device, VAL_SEED, max_batches)


def make_args(layers=32, channels=32, dc='prox'):
    # operator settings must match script2.py / main_3 defaults; layers/channels/dc are read from each checkpoint
    return argparse.Namespace(dataset='MULTI', classify=0, use_meta_data=1, task='mask', method='foundation',
                              noise=True, painting=True, blurring=True, sensoring=True, pdessm=True,
                              mask_per_snapshot_budget=6, n_sensors=5, blur_count='4', pde_tau=20.0, cglsIter=5, solveIter=5,
                              rnfPE=1, dropout=0.0, train_batch_size=4, layers=layers, channels=channels, num_nodes=1, dc=dc)


def load_model(path, ops, device):
    return load_foundation_checkpoint(path, make_args(), ops, device, label_channels=1, feat_channels=2)  # MULTI: constant feature + observation flag


def nmse(X, y):
    return (F.mse_loss(X, y) / F.mse_loss(torch.zeros_like(y), y)).item()


class LogStat(list):
    """A row cell imported from a run log: [mean] (so sum / len / fmt_cell work) plus the std and the seed count."""
    def __init__(self, mean, std, n):
        super().__init__([mean])
        self.std, self.n = std, n


def rows_from_log(path):
    """{dataset: {row: {task: LogStat}}} parsed from the '== held-out X' blocks that plot_2.py printed in a run log
    (results/run_gpu_CB.log): lets the classical rows be (re)computed without re-running the 75 model evaluations."""
    import re
    out, dataset = {}, None
    rx = re.compile(r'(\S{8}) ([\d.]+)(?: \u00b1 ([\d.]+))?')
    for line in open(path):
        m = re.match(r'== held-out (\w+):', line)
        if m:
            dataset = m.group(1)
            out[dataset] = {}
            continue
        if dataset is None or not line.startswith('  ') or line.startswith('  BASELINE') or line.startswith('  tuning'):
            continue
        name, rest = line[2:28].strip(), line[28:]
        found = {t: (float(v), float(sd) if sd else None) for t, v, sd in rx.findall(rest)}
        if len(found) == len(TASKS) and name:
            out[dataset][name] = {t: LogStat(found[t[:8]][0], found[t[:8]][1], 3 if found[t[:8]][1] is not None else 1) for t in TASKS}
    return out


@torch.no_grad()
def evaluate_dataset(dataset, files, device, max_batches=None, val_batches=None):
    """nMSE[row][task] for every checkpoint trained with `dataset` held out, plus the baselines.
    `files` = [(row name, path)]; several seeds of one row are averaged afterwards."""
    args = make_args()
    ops = get_forward_op(args, args.channels, 1, device)  # shared by every model, so all invert the same measurement
    models = {path: load_model(path, ops, device) for _, path in files}
    if files:
        print(f"  evaluating {len(files)} checkpoints and {len(BASELINES)} classical rows on {len(snapshots(dataset))} snapshots", flush=True)
    random.seed(0)
    test = snapshots(dataset)
    from script2 import TEST_FRAC
    test = get_fractional_dataset(test, TEST_FRAC[dataset]) if dataset in TEST_FRAC else test  # as in script2.py
    from script2 import DATASETS, TEST_ONLY
    train_key = tuple(d for d in DATASETS if d != dataset and d not in TEST_ONLY)
    if train_key not in _LAMBDAS:  # the same training mix (-> same validation tails) serves several held-out datasets
        print(f"  tuning the classical regularisers on the validation tails of {train_key} ...", flush=True)
        _LAMBDAS[train_key] = tune_lambdas(validation_batches(dataset, args, ops, device, val_batches or max_batches))[0]
    lambdas = _LAMBDAS[train_key]
    sums = {r: {t: 0.0 for t in TASKS} for r in BASELINES + list(models)}
    count = {t: 0 for t in TASKS}
    for i, (task, g, b, op) in enumerate(measurements(DataLoader(test, batch_size=4, shuffle=False), args, ops, device, 0, max_batches)):
        if i % 250 == 0:
            print(f"    batch {i // len(TASKS)} / {(len(test) + 3) // 4} ...", flush=True)
        sums['BASELINE: X = 0'][task] += nmse(torch.zeros_like(g.y), g.y)
        ls, _ = graph_CGLS(op, CGLSit=args.cglsIter * args.solveIter, eps=1e-5)(b, torch.zeros_like(b), g.edge_index, g.edge_weight)
        sums['BASELINE: least squares'][task] += nmse(ls, g.y)
        for reg in REGULARISERS:
            sums[ROW_OF[reg]][task] += nmse(regularised_ls(op, b, g.edge_index, g.edge_weight, lambdas[reg][task][0], reg, lambdas[reg][task][1]), g.y)
        for path, net in models.items():
            net.set_task(task)
            X, _, _ = net(b, g.edge_index, g.edge_weight, g.x)
            sums[path][task] += nmse(torch.nan_to_num(X, nan=1e6), g.y)
        count[task] += 1
    per_file = {r: {t: v / max(count[t], 1) for t, v in row.items()} for r, row in sums.items()}
    rows = {}  # row name -> task -> [value per seed]
    for name, path in [(b, b) for b in BASELINES] + files:
        for t in TASKS:
            rows.setdefault(name, {}).setdefault(t, []).append(per_file[path][t])
    return rows


def plot_table(results, save_path):
    cols = TASKS + ['mean']
    labels, cells, colors = [], [], []
    for dataset, rows in results.items():
        avg = {name: {t: sum(v) / len(v) for t, v in r.items()} for name, r in rows.items()}  # mean over seeds
        best = {t: min(r[t] for name, r in avg.items() if not name.startswith('BASELINE')) for t in TASKS}
        for name, r in rows.items():
            means = [avg[name][t] for t in TASKS]
            labels.append(f"{dataset} | {name}")
            cells.append([fmt_cell(r[t]) for t in TASKS] + [f"{sum(means) / len(TASKS):.4f}"])
            base = name.startswith('BASELINE')
            colors.append(['#fff3cd' if base else ('#d4edda' if avg[name][t] == best[t] and avg[name][t] < avg['BASELINE: least squares'][t] else '#ffffff')
                           for t in TASKS] + ['#fff3cd' if base else '#f1f1f2'])
    fig, ax = plt.subplots(figsize=(16, 0.3 * len(labels) + 1))
    ax.axis('off')
    table = ax.table(cellText=cells, rowLabels=labels, colLabels=cols, cellColours=colors, loc='center', cellLoc='center',
                     colColours=['#40466e'] * len(cols), rowColours=['#e9ecef'] * len(labels))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(color='white', weight='bold')
    n_seeds = max(len(v) for rows in results.values() for r in rows.values() for v in r.values())
    plt.title(f"Zero-shot nMSE on the held-out dataset (lower is better){f', mean ± std over {n_seeds} seeds' if n_seeds > 1 else ''}\n"
              "green = best model in that dataset AND better than least squares, yellow = baseline",
              weight='bold', size=12, pad=24)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nTable saved to {save_path}")


def read_csv(path):
    """Inverse of the csv written by main(): {dataset: {row: {task: [per-seed values] | LogStat}}}."""
    import csv
    results = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            cells = {}
            for t in TASKS:
                if '+-' in r[t]:
                    m, sd = r[t].split('+-')
                    cells[t] = LogStat(float(m), float(sd), 3)
                else:
                    cells[t] = [float(v) for v in r[t].split(';')]
            results.setdefault(r['dataset'], {})[r['row']] = cells
    return results


def write_csv(results, path):
    with open(path, 'w') as f:  # the numbers behind the png
        f.write('dataset,row,' + ','.join(TASKS) + '\n')
        for dataset, rows in results.items():
            for r in rows:
                f.write(f'{dataset},"{r}",' + ','.join(  # per-seed values 'a;b;c', or 'mean+-std' for rows imported from a log
                    f"{v[0]:.6f}+-{v.std:.6f}" if isinstance(v, LogStat) and v.std is not None else ';'.join(f"{x:.6f}" for x in v)
                    for v in (rows[r][t] for t in TASKS)) + '\n')


def merge(model_dir):
    """Join the per-dataset tables of `--only` runs (comparison_table_<DATASET>.csv) into comparison_table.csv / .png."""
    results = {}
    for d in ['CPOX', 'MONTEVIDEO', 'PEDALME', 'WIKIMATHS', 'WINDMILL']:
        p = os.path.join(model_dir, f'comparison_table_{d}.csv')
        if os.path.exists(p):
            results.update(read_csv(p))
    plot_table(results, os.path.join(model_dir, 'comparison_table.png'))
    write_csv(results, os.path.join(model_dir, 'comparison_table.csv'))


def main(model_dir, max_batches=None, model_rows_from=None, only=None, val_batches=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    imported = rows_from_log(model_rows_from) if model_rows_from else None
    groups = {}
    for fn in sorted(os.listdir(model_dir)):
        info = parse_checkpoint_name(fn)
        if info is None or info['kind'] != 'dataset_test':
            if fn.endswith('.pth'):
                print(f"skipping {fn} (not a script2.py checkpoint)")
            continue
        # in B nothing is adapted to the held-out dataset: both rows are zero-shot, they differ in the head used
        variant = {'zero-shot (shared head)': 'shared head (after backbone training)',
                   'adapted (task heads)': 'task heads (20 more epochs on the training datasets)'}[info['variant']]
        groups.setdefault(info['holdout'].upper(), []).append((variant, os.path.join(model_dir, fn), info['run']))
    for d, files in groups.items():  # several runs (not seeds) for one dataset: tag the row with its run name
        runs = {r for _, _, r in files}
        groups[d] = [(f"{n} [{r}]" if len(runs) > 1 else n, p) for n, p, r in files]
    if not groups:
        print(f"No script2.py checkpoints found in '{model_dir}'.")
        return
    results = {}
    for dataset, files in groups.items():
        if only and dataset != only.upper():
            continue
        print(f"\n== held-out {dataset}: {[n for n, _ in files]}")
        if imported is None:
            results[dataset] = evaluate_dataset(dataset, files, device, max_batches, val_batches)
        else:  # classical rows computed here, model rows copied from the log of the run that trained the checkpoints
            results[dataset] = evaluate_dataset(dataset, [], device, max_batches, val_batches)
            for name, row in imported[dataset].items():
                results[dataset][{'zero-shot (shared head)': 'shared head (after backbone training)',
                                  'adapted (task heads)': 'task heads (20 more epochs on the training datasets)'}.get(name, name)] = row
        for row, r in results[dataset].items():
            print(f"  {row:26s} " + "  ".join(f"{t[:8]} {fmt_cell(r[t])}" for t in TASKS))
    suffix = f'_{only.upper()}' if only else ''
    plot_table(results, os.path.join(model_dir, f'comparison_table{suffix}.png'))
    write_csv(results, os.path.join(model_dir, f'comparison_table{suffix}.csv'))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('model_dir', type=str, help='directory with script2.py checkpoints')
    p.add_argument('--max_batches', type=int, default=None, help='evaluate only this many batches per dataset (quick check)')
    p.add_argument('--model_rows_from', type=str, default=None,
                   help='run log (results/run_gpu_CB.log) to copy the model rows from; only the classical rows are computed')
    p.add_argument('--only', type=str, default=None, help='evaluate one held-out dataset (writes comparison_table_<DATASET>.csv)')
    p.add_argument('--val_batches', type=int, default=None, help='tune the classical regularisers on at most this many validation batches per training dataset')
    p.add_argument('--merge', action='store_true', help='join the per-dataset tables of --only runs into comparison_table.csv/.png')
    a = p.parse_args()
    if a.merge:
        merge(a.model_dir)
    else:
        main(a.model_dir, a.max_batches, a.model_rows_from, a.only, a.val_batches)
