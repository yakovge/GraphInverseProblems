"""
plot_2.py -- comparison table for script2.py (leave-one-dataset-out) checkpoints.

    python src/plot_2.py "old models/script_2"

Every *_<dataset>_dataset_test.pth in the directory is evaluated zero-shot on its held-out dataset,
on all 5 tasks, with the same operators and settings as training (script2.py). Within a dataset every
model and baseline sees identical masks and noise. Baselines:
  X = 0          -- predict the mean (data is z-normalised), nMSE ~ 1
  least squares  -- CGLS from zero, no learned prior (= zero-fill for masks, = b for denoising).
                    A model that does not beat this row has learned nothing beyond the data fit.
The table is saved as comparison_table.png in the directory.
"""
import os
import re
import random
import argparse

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from utils import load_pgt_snapshots, get_fractional_dataset, process_data, task_specific_modifiers, get_forward_op, get_network
from networks import graph_CGLS

TASKS = ['denoising', 'inpainting', 'sensor_recovery', 'source_localization', 'pde_reconstruction']
OP_PREFIXES = ('current_forward_op.', 'solver.forOp.')  # save-time physics operator, not learned model state
BASELINES = ['BASELINE: X = 0', 'BASELINE: least squares']


def make_args(layers=16, channels=32):
    # must match script2.py / main_3 defaults
    return argparse.Namespace(dataset='MULTI', classify=0, use_meta_data=1, task='mask', method='foundation',
                              noise=True, painting=True, blurring=True, sensoring=True, pdessm=True,
                              mask_per_snapshot_budget=6, n_sensors=5, blur_count='4', cglsIter=5, solveIter=5,
                              rnfPE=1, dropout=0.0, train_batch_size=4, layers=layers, channels=channels, num_nodes=1)


def load_model(path, ops, device):
    sd = torch.load(path, map_location=device, weights_only=True)
    layers, channels = sd['backbone.K'].shape[:2]
    net = get_network(make_args(layers, channels), ops, channels, 1, sd['feat_embed.weight'].shape[1], device).to(device)
    res = net.load_state_dict({k: v for k, v in sd.items() if not k.startswith(OP_PREFIXES)}, strict=False)
    missing = [k for k in res.missing_keys if not k.startswith(OP_PREFIXES)]
    if missing or res.unexpected_keys:
        raise RuntimeError(f"{path}: missing {missing}, unexpected {res.unexpected_keys}")
    # not in the state_dict: main_3 saves no_head_training_* before freeze_backbone() switches to task heads
    net.use_specialized_heads = not os.path.basename(path).startswith('no_head_training_')
    return net.eval()


def nmse(X, y):
    return (F.mse_loss(X, y) / F.mse_loss(torch.zeros_like(y), y)).item()


@torch.no_grad()
def evaluate_dataset(dataset, files, device, max_batches=None):
    """nMSE[row][task] for every model trained with `dataset` held out, plus the baselines."""
    args = make_args()
    ops = get_forward_op(args, args.channels, 1, device)  # shared by every model, so all invert the same measurement
    models = {name: load_model(path, ops, device) for name, path in files}
    random.seed(0)
    test = load_pgt_snapshots(dataset)
    test = get_fractional_dataset(test, 0.05) if dataset == 'WINDMILL' else test  # as in script2.py
    sums = {r: {t: 0.0 for t in TASKS} for r in BASELINES + list(models)}
    count = {t: 0 for t in TASKS}
    for bi, raw in enumerate(DataLoader(test, batch_size=4, shuffle=False)):
        if max_batches is not None and bi >= max_batches:
            break
        for k, (op, task) in enumerate(ops):
            torch.manual_seed(1000 * bi + k)  # identical masks / sensors / noise for every row
            g, op = task_specific_modifiers(process_data(args, raw.clone()), args, op, None)  # exactly as in training
            g = g.to(device)
            b = op(g.y, g.edge_index, g.edge_weight, emb=False)
            if hasattr(op, 'corrupt'):
                b = op.corrupt(b)
            sums['BASELINE: X = 0'][task] += nmse(torch.zeros_like(g.y), g.y)
            ls, _ = graph_CGLS(op, CGLSit=args.cglsIter * args.solveIter, eps=1e-5)(b, torch.zeros_like(b), g.edge_index, g.edge_weight)
            sums['BASELINE: least squares'][task] += nmse(ls, g.y)
            for name, net in models.items():
                net.set_task(task)
                X, _, _ = net(b, g.edge_index, g.edge_weight, g.x)
                sums[name][task] += nmse(torch.nan_to_num(X, nan=1e6), g.y)
            count[task] += 1
    return {r: {t: v / max(count[t], 1) for t, v in row.items()} for r, row in sums.items()}


def plot_table(results, save_path):
    cols = TASKS + ['mean']
    labels, cells, colors = [], [], []
    for dataset, rows in results.items():
        best = {t: min(r[t] for name, r in rows.items() if not name.startswith('BASELINE')) for t in TASKS}
        for name, r in rows.items():
            mean = sum(r.values()) / len(TASKS)
            labels.append(f"{dataset} | {name}")
            cells.append([f"{r[t]:.4f}" for t in TASKS] + [f"{mean:.4f}"])
            base = name.startswith('BASELINE')
            colors.append(['#fff3cd' if base else ('#d4edda' if r[t] == best[t] and r[t] < rows['BASELINE: least squares'][t] else '#ffffff')
                           for t in TASKS] + ['#fff3cd' if base else '#f1f1f2'])
    fig, ax = plt.subplots(figsize=(16, 0.45 * len(labels) + 2))
    ax.axis('off')
    table = ax.table(cellText=cells, rowLabels=labels, colLabels=cols, cellColours=colors, loc='center', cellLoc='center',
                     colColours=['#40466e'] * len(cols), rowColours=['#e9ecef'] * len(labels))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(color='white', weight='bold')
    plt.title("Zero-shot nMSE on the held-out dataset (lower is better)\n"
              "green = best model in that dataset AND better than least squares, yellow = baseline",
              weight='bold', size=12)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nTable saved to {save_path}")


def main(model_dir, max_batches=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    groups = {}
    for fn in sorted(os.listdir(model_dir)):
        m = re.search(r'_([a-z]+)_dataset_test\.pth$', fn)
        if not m:
            if fn.endswith('.pth'):
                print(f"skipping {fn} (not a script2.py checkpoint)")
            continue
        name = 'pre-head (shared head)' if fn.startswith('no_head_training_') else 'final (task heads)'
        groups.setdefault(m.group(1).upper(), []).append((name, os.path.join(model_dir, fn)))
    if not groups:
        print(f"No script2.py checkpoints found in '{model_dir}'.")
        return
    results = {}
    for dataset, files in groups.items():
        print(f"\n== held-out {dataset}: {[n for n, _ in files]}")
        results[dataset] = evaluate_dataset(dataset, files, device, max_batches)
        for row, r in results[dataset].items():
            print(f"  {row:26s} " + "  ".join(f"{t[:8]} {r[t]:.4f}" for t in TASKS))
    plot_table(results, os.path.join(model_dir, 'comparison_table.png'))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('model_dir', type=str, help='directory with script2.py checkpoints')
    p.add_argument('--max_batches', type=int, default=None, help='evaluate only this many batches per dataset (quick check)')
    a = p.parse_args()
    main(a.model_dir, a.max_batches)
