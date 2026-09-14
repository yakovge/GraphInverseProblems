"""
plot_fixed.py -- corrected evaluation harness for the foundation model.

Fixes relative to the original plot.py
-------------------------------------
1. OPERATOR SYNC. The measurement `b` is produced by the *exact* operator object
   the model will invert (net.task_heads[task]). In the original these were two
   different instances with different indices, so the model was inverting the
   identity while the data had been masked.

2. PER-BATCH INDICES. Mask / sensor indices are drawn from graph.batch, not from
   a single 20-node snapshot. They are drawn once per (operator, batch) and then
   copied onto every model, so all models see identical measurements.

3. graphMask.adjoint PATCH. In the repo, graphMask.forward is shape-preserving
   ([N,C] -> [N,C]) but graphMask.adjoint expects a compressed measurement
   ([len(ind),C]). That mismatch is invisible when ind == arange(N) and crashes
   as soon as ind is a real subset. Masking is self-adjoint, so the adjoint is
   patched below to mask rather than scatter. SensorRecovery.adjoint is already
   correct and is left alone.

4. NON-TRIVIAL PDE OPERATOR. With tau=K=1 the PDESSM filter has condition number
   1.28 -- invertible, so CGLS recovers the ground truth exactly and the column
   reads 0.0000 for every model. PDE_TAU below controls how ill-posed it is.

5. DIAGNOSTICS. Per-operator condition number in the column header, plus
   `X = 0` and `X = b` baseline rows. If a model row sits on the `X = b` row,
   that column is measuring nothing.
"""

import os
import argparse

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import process_data, get_data_and_loaders, get_network, get_forward_op
from utils import (load_checkpoint, generate_measurement, accumulate_pergraph_ratios,
                   get_data_and_loaders_foundation)
from graphForwardOps import graphMask


# ---------------------------------------------------------------------------
# 0. (graphMask is now natively self-adjoint and accepts batch=; no patch needed)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

# column label -> foundation-model task-head key
TASK_MAP = {
    'noise': 'denoising',
    'painting': 'inpainting',
    'blurring': 'source_localization',
    'sensoring': 'sensor_recovery',
    'pdessm': 'pde_reconstruction',
}

CFG = {
    # inpainting: nodes KEPT per snapshot, out of 20 for CPOX. Random each batch.
    'mask_budget': 6,
    # sensor recovery: fixed sensor nodes per snapshot. Same positions every batch.
    'n_sensors': 5,
    # source localization: number of diffusion steps A^k
    'blur_k': 4,
    # denoising
    'noise_std': 0.1,
    # PDE reconstruction. cond(A) ~ exp(tau * K / 4) for this filter:
    #   tau=1   -> cond 1.3    (trivially invertible; this is the 0.0000 column)
    #   tau=20  -> cond ~150
    #   tau=50  -> cond ~3e5
    'pde_tau': 20.0,
    'pde_K': 1.0,
}

SEED = 0
MAX_BATCHES = None      # set to 1 or 2 for a smoke test


# ---------------------------------------------------------------------------
# operator setup
# ---------------------------------------------------------------------------

def sample_per_snapshot(graph, budget, generator):
    """Random `budget` node indices from every snapshot in the batch.

    Mirrors utils.task_specific_modifiers, which is what training uses.
    """
    kept = []
    for c in graph.batch.unique():
        nodes = torch.where(graph.batch == c)[0]
        perm = torch.randperm(len(nodes), generator=generator)
        kept.append(nodes[perm[:budget].to(nodes.device)])
    return torch.cat(kept)


def first_k_per_snapshot(graph, k):
    """Deterministic sensor positions: the first k nodes of every snapshot."""
    kept = []
    for c in graph.batch.unique():
        nodes = torch.where(graph.batch == c)[0]
        kept.append(nodes[:k])
    return torch.cat(kept)


def op_params_for_batch(op_name, graph, cfg, generator):
    """Everything that defines the operator for this (operator, batch).

    Drawn once and then shared by every model, so the comparison is fair.
    """
    if op_name == 'painting':
        return {'ind': sample_per_snapshot(graph, cfg['mask_budget'], generator)}
    if op_name == 'sensoring':
        return {'sensor_indices': first_k_per_snapshot(graph, cfg['n_sensors'])}
    if op_name == 'blurring':
        return {'k': cfg['blur_k']}
    if op_name == 'pdessm':
        return {'tau': cfg['pde_tau'], 'K': cfg['pde_K']}
    if op_name == 'noise':
        return {'noise_std': cfg['noise_std']}
    return {}


def configure_head(net, op_name, params):
    """Activate `op_name` on this model and stamp the shared params onto its head."""
    net.set_task(TASK_MAP[op_name])   # also re-points solver.forOp (alias-safe)
    head = net.current_forward_op
    for key, value in params.items():
        setattr(head, key, value)
    return head


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def operator_condition_number(head, graph, op_name):
    """Dense probe of the operator in data space, to see whether the inverse
    problem is ill-posed at all. inf means a genuine null space (a mask)."""
    if op_name == 'noise':
        return float('nan')          # stochastic; its linear part is the identity
    n = graph.y.shape[0]
    cols = []
    for j in range(n):
        e = torch.zeros_like(graph.y)
        e[j] = 1.0
        cols.append(head(e, graph.edge_index, graph.edge_weight, emb=False).reshape(-1))
    M = torch.stack(cols, dim=1).float()
    if not torch.isfinite(M).all():
        return float('nan')
    s = torch.linalg.svdvals(M)
    smin, smax = s[-1].item(), s[0].item()
    return smax / smin if smin > 1e-10 else float('inf')


def fmt_cond(c):
    if c is None or c != c:
        return "stochastic"
    if c == float('inf'):
        return "inf (null space)"
    return f"{c:.1f}"


def rel_mse(X, y):
    return (F.mse_loss(X, y) / F.mse_loss(torch.zeros_like(y), y)).item()


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------

def load_all_models(args, label_channels, feat_channels, device, model_dir="models"):
    print(f"Loading saved models from '{model_dir}' ...")
    loaded = []
    if not os.path.exists(model_dir):
        print(f"Directory '{model_dir}' not found. No models loaded.")
        return loaded

    for filename in sorted(os.listdir(model_dir)):
        if not filename.endswith(".pth"):
            continue
        filepath = os.path.join(model_dir, filename)
        net = get_network(args, None, args.channels, label_channels,
                          feat_channels, device)
        blob = torch.load(filepath, map_location=device, weights_only=False)
        if isinstance(blob, dict) and 'model_state_dict' in blob:
            load_checkpoint(filepath, net, device)
        else:
            net.load_state_dict(blob)
        net.eval()
        loaded.append({"name": filename, "model": net})
        print(f" -> {filename}")
    return loaded


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_operations_and_models(models, op_names, loader, args, device, cfg,
                              seed=SEED, max_batches=MAX_BATCHES):
    row_names = [m['name'] for m in models] + ['BASELINE: X = 0', 'BASELINE: X = b']
    loss_matrix = {r: {op: 0.0 for op in op_names} for r in row_names}
    counts = {r: {op: 0 for op in op_names} for r in row_names}
    cond = {op: None for op in op_names}
    nan_hits = set()

    generator = torch.Generator().manual_seed(seed)

    for batch_idx, graph in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        graph = process_data(args, graph)
        graph = graph.to(device)

        for op_name in op_names:
            params = op_params_for_batch(op_name, graph, cfg, generator)

            # configure every model's head identically
            heads = [configure_head(m['model'], op_name, params) for m in models]

            # one measurement, shared by every model (uses the shared generator so
            # noise is actually applied via corrupt() rather than a no-op forward)
            b = generate_measurement(heads[0], graph.y, graph.edge_index,
                                     graph.edge_weight, graph.batch, seed=seed + batch_idx)

            if cond[op_name] is None:
                cond[op_name] = operator_condition_number(heads[0], graph, op_name)

            for m in models:
                X, _, _ = m['model'](b, graph.edge_index, graph.edge_weight, graph.x,
                                     batch=graph.batch)
                if not torch.isfinite(X).all():
                    nan_hits.add((m['name'], op_name))
                    X = torch.nan_to_num(X)
                # graph-count-weighted per-graph relative MSE (production protocol)
                s, ng = accumulate_pergraph_ratios(X, graph.y, graph.batch)
                loss_matrix[m['name']][op_name] += s
                counts[m['name']][op_name] += ng

            s0, ng0 = accumulate_pergraph_ratios(torch.zeros_like(graph.y), graph.y, graph.batch)
            loss_matrix['BASELINE: X = 0'][op_name] += s0
            counts['BASELINE: X = 0'][op_name] += ng0
            if b.shape == graph.y.shape:
                sb, ngb = accumulate_pergraph_ratios(b, graph.y, graph.batch)
                loss_matrix['BASELINE: X = b'][op_name] += sb
                counts['BASELINE: X = b'][op_name] += ngb

    for r in loss_matrix:
        for op in loss_matrix[r]:
            loss_matrix[r][op] = (loss_matrix[r][op] / counts[r][op]
                                  if counts[r][op] > 0 else float('nan'))

    print("\nOperator condition numbers (data space):")
    for op in op_names:
        print(f"  {op:10s} {fmt_cond(cond[op])}")
    if nan_hits:
        print("\nWARNING: non-finite outputs from:")
        for name, op in sorted(nan_hits):
            print(f"  {op:10s} {name}")
    return loss_matrix, cond


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------

def plot_loss_matrix_table(loss_matrix, cond, save_path="loss_matrix_table.png"):
    row_names = list(loss_matrix.keys())
    if not row_names:
        print("Loss matrix is empty, nothing to plot.")
        return
    op_names = list(loss_matrix[row_names[0]].keys())

    col_labels = [f"{op}\ncond {fmt_cond(cond.get(op))}" for op in op_names]
    short_rows = [r if len(r) <= 46 else r[:21] + "..." + r[-21:] for r in row_names]
    cell_text = [[f"{loss_matrix[r][op]:.4f}" for op in op_names] for r in row_names]

    fig, ax = plt.subplots(figsize=(max(9, len(op_names) * 2.2 + 5),
                                    max(3, len(row_names) * 0.55 + 2)))
    ax.axis('off')
    ax.axis('tight')

    table = ax.table(cellText=cell_text,
                     rowLabels=short_rows,
                     colLabels=col_labels,
                     loc='center', cellLoc='center',
                     rowColours=['#e9ecef'] * len(row_names),
                     colColours=['#40466e'] * len(op_names))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.0)

    row_colors = ['#f1f1f2', '#ffffff']
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(color='white', weight='bold')
        elif col == -1:
            cell.set_text_props(weight='bold')
        else:
            is_baseline = row_names[row - 1].startswith('BASELINE')
            cell.set_facecolor('#fff3cd' if is_baseline else row_colors[(row - 1) % 2])

    plt.title("Reconstruction Loss (Relative MSE) by Operation",
              pad=20, weight='bold', size=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTable saved to {save_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    args = argparse.Namespace(
        dataset='CPOX', datapath='./data', use_meta_data=1, classify=0, CPOX_lags=1,
        train_batch_size=4, test_batch_size=4, train_frac=1.0, test_frac=1.0,
        method='foundation', layers=16, channels=32, cglsIter=5, solveIter=5,
        rnfPE=1, dropout=0.0, task='mask', blur_count='4',
        mask_per_snapshot_budget=CFG['mask_budget'], held_out_op=None,
        noise=True, painting=True, blurring=True, sensoring=True, pdessm=True,
    )

    # Use the same foundation split/loader the model was trained and reported on.
    (_, _, _, _, _, test_loader, label_channels, feat_channels,
     _split, _norm) = get_data_and_loaders_foundation(args)

    models = load_all_models(args, label_channels, feat_channels, device)
    print(f"Models ready: {len(models)}")
    if not models:
        return

    loss_matrix, cond = run_operations_and_models(
        models, list(TASK_MAP.keys()), test_loader, args, device, CFG
    )
    plot_loss_matrix_table(loss_matrix, cond)


if __name__ == "__main__":
    main()
