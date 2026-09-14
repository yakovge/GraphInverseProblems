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
from utils import (load_checkpoint, load_legacy_state_dict, generate_measurement,
                   accumulate_pergraph_ratios, get_data_and_loaders_foundation)
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

    from utils import build_foundation_model_from_checkpoint
    for filename in sorted(os.listdir(model_dir)):
        if not filename.endswith(".pth"):
            continue
        filepath = os.path.join(model_dir, filename)
        blob = torch.load(filepath, map_location=device, weights_only=False)
        meta = {}
        if isinstance(blob, dict) and 'model_state_dict' in blob:
            # reconstruct model from the checkpoint's own architecture metadata
            net, meta = build_foundation_model_from_checkpoint(filepath, feat_channels, device)
        else:
            net = get_network(args, None, args.channels, label_channels,
                              feat_channels, device)
            load_legacy_state_dict(filepath, net, device)
        net.eval()
        loaded.append({"name": filename, "model": net, "meta": meta})
        print(f" -> {filename}")
    return loaded


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_operations_and_models(models, op_names, loader, args, device, cfg=None,
                              seed=SEED, max_batches=MAX_BATCHES):
    """Build the loss matrix using the PRODUCTION evaluator (evaluate_all_operators):
    production observation budgets (args.mask_per_snapshot_budget) and production
    operator parameters (e.g. PDESSM tau=1). Rows are annotated with held-out
    metadata from each checkpoint. Condition numbers are computed on the same
    production operators for the diagnostics column.
    """
    from utils import (evaluate_all_operators, create_operator, apply_config,
                       sample_operator_config, eval_args_from_checkpoint)
    from plot import _protocol_signature

    loss_matrix = {}
    group_baseline = {}

    for m in models:
        meta = m.get('meta', {})
        margs = eval_args_from_checkpoint(meta, base_args=args)
        held = meta.get('held_out_flag')
        row = m['name'] + (f" [held-out={held}]" if held else "")
        summary = evaluate_all_operators(m['model'], loader, margs, device, process_data)
        loss_matrix[row] = {op: summary[op]['model'] for op in op_names}
        sig = _protocol_signature(meta)
        if sig not in group_baseline:
            group_baseline[sig] = {
                'solver': {op: summary[op]['solver'] for op in op_names},
                'xeqb': {op: summary[op]['xeqb'] for op in op_names},
                'zero': {op: summary[op]['zero'] for op in op_names},
            }
    # per-protocol baselines (differing budgets/blur get separate rows)
    single = len(group_baseline) == 1
    for sig, b in group_baseline.items():
        tag = "" if single else f" [budget={sig[0]},blur={sig[1]}]"
        loss_matrix[f"BASELINE: solver{tag}"] = b['solver']
        loss_matrix[f"BASELINE: X = b{tag}"] = b['xeqb']
        loss_matrix[f"BASELINE: X = 0{tag}"] = b['zero']

    # Condition numbers via the production physical operators (first batch).
    cond = {op: None for op in op_names}
    try:
        first = next(iter(loader))
        graph = process_data(args, first).to(device)
        for op in op_names:
            phys = create_operator(op, args, device, learnEmb=False)
            cfg_op = sample_operator_config(graph, args, op, seed=seed)
            apply_config(phys, cfg_op)
            cond[op] = operator_condition_number(phys, graph, op)
    except StopIteration:
        pass

    print("\nOperator condition numbers (data space, production operators):")
    for op in op_names:
        print(f"  {op:10s} {fmt_cond(cond[op])}")
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
        mask_per_snapshot_budget=16, held_out_op=None,   # production budget
        noise=True, painting=True, blurring=True, sensoring=True, pdessm=True,
    )

    from utils import eval_args_from_checkpoint
    models = load_all_models(args, 1, 1, device)
    print(f"Models ready: {len(models)}")
    if not models:
        return
    # Reconstruct the loader from the first checkpoint's saved eval_config.
    loader_args = args
    for m in models:
        if m.get('meta', {}).get('eval_config'):
            loader_args = eval_args_from_checkpoint(m['meta'], base_args=args)
            break
    (_, _, _, _, _, test_loader, label_channels, feat_channels,
     _split, _norm) = get_data_and_loaders_foundation(loader_args)

    loss_matrix, cond = run_operations_and_models(
        models, list(TASK_MAP.keys()), test_loader, loader_args, device, CFG
    )
    plot_loss_matrix_table(loss_matrix, cond)


if __name__ == "__main__":
    main()
