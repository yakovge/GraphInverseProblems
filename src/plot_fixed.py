"""
plot_fixed.py -- corrected evaluation harness for the foundation model.

Fixes relative to the original plot.py
-------------------------------------
1. OPERATOR SYNC. The measurement `b` is produced by the *exact* operator object
   the model will invert (net.current_forward_op after set_task). In the original
   these were two different instances with different indices, so the model was
   inverting the identity while the data had been masked.

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

6. ALL FIVE PHYSICS OPERATORS. The network is built with every operator flag on,
   so set_task() resolves each column to its own operator. With all flags off,
   get_forward_op returns a single graphMask and every column silently evaluated
   masking. configure_head now asserts the operator type.

7. CHECKPOINT LOADING. GraphInverseFoundationModel keeps its operators in a plain
   dict, and only the one active at save time is registered (as
   current_forward_op / solver.forOp). So each checkpoint carries a different
   operator's weights (a PDESSM checkpoint adds .K and .r), which broke strict
   loading. Those keys are operator state, not learned inverse-model state: they
   are dropped, every other key is still checked strictly, and the harness sets
   the shared operator itself. Saved PDESSM K/r are printed so drift is visible.

8. NOISE. AddNoise.forward is the identity; the noise lives in corrupt(), which
   training calls and the original harness did not. It is applied here.
   PDESSM.K / .r are nn.Parameters, so they are now filled in place rather than
   reassigned with setattr (which raises TypeError on a Parameter).

9. DEVICE + HEADS. The net is moved to `device` after loading (get_network
   leaves the encoders/heads on CPU). use_specialized_heads is restored from the
   checkpoint name, since it is not saved in the state_dict: without this every
   post-head-training checkpoint was evaluated with the shared encoder/head.
"""

import os
import argparse

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import process_data, get_data_and_loaders, get_network, get_forward_op
from graphForwardOps import graphMask, graph_smooth, SensorRecovery, AddNoise, PDESSM


# ---------------------------------------------------------------------------
# 0. Patch graphMask  (see note 3 in the module docstring)
# ---------------------------------------------------------------------------

def _graphmask_forward(self, I, edge_index=None, edge_weight=None, emb=True):
    """Same as the repo version, but device-safe on self.ind."""
    if emb and self.learnEmb:
        I = self.Emb(I)
    idx = self.ind.to(I.device)
    Ic = torch.zeros_like(I)
    Ic[idx] = I[idx]
    return Ic


def _graphmask_adjoint(self, Ic, edge_index=None, edge_weight=None, emb=True):
    """True adjoint of a shape-preserving mask: mask, then pull back through Emb."""
    idx = self.ind.to(Ic.device)
    I = torch.zeros_like(Ic)
    I[idx] = Ic[idx]
    if emb and self.learnEmb:
        I = self.Emb.backward(I)
    return I


graphMask.forward = _graphmask_forward
graphMask.adjoint = _graphmask_adjoint


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

# column label -> operator class that set_task() must resolve to
EXPECTED_OP = {
    'noise': AddNoise,
    'painting': graphMask,
    'blurring': graph_smooth,
    'sensoring': SensorRecovery,
    'pdessm': PDESSM,
}

# state_dict prefixes that hold the (single, save-time-active) physics operator
OP_PREFIXES = ('current_forward_op.', 'solver.forOp.')

CFG = {
    # inpainting: nodes KEPT per snapshot, out of 20 for CPOX. Random each batch.
    'mask_budget': 16,
    # sensor recovery: fixed sensor nodes per snapshot. Same positions every batch.
    'n_sensors': 5,
    # source localization: number of diffusion steps A^k
    'blur_k': 4,
    # denoising
    'noise_std': 0.25,
    # PDE reconstruction. cond(A) ~ exp(tau * K / 4) for this filter:
    #   tau=1   -> cond 1.3    (trivially invertible; this is the 0.0000 column)
    #   tau=20  -> cond ~150
    #   tau=50  -> cond ~3e5
    'pde_tau': 20.0,   # = training default (--pde_tau)
    'pde_K': 1.0,
    'pde_r': 0.0,
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
        return {'tau': cfg['pde_tau'], 'K': cfg['pde_K'], 'r': cfg['pde_r']}
    if op_name == 'noise':
        return {'noise_std': cfg['noise_std']}
    return {}


def configure_head(net, op_name, params):
    """Activate `op_name` on this model and stamp the shared params onto its head."""
    net.set_task(TASK_MAP[op_name])
    head = net.current_forward_op
    if not isinstance(head, EXPECTED_OP[op_name]):
        raise RuntimeError(
            f"set_task('{TASK_MAP[op_name]}') gave {type(head).__name__}, expected "
            f"{EXPECTED_OP[op_name].__name__}. Build the net with all operator flags on.")
    for key, value in params.items():
        current = getattr(head, key, None)
        if isinstance(current, torch.nn.Parameter):
            # PDESSM.K / .r: setattr(float) on a Parameter raises TypeError
            with torch.no_grad():
                current.fill_(value)
        else:
            setattr(head, key, value)
    net.solver.forOp = head          # set_task does this too; explicit for safety
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

def split_operator_keys(state_dict):
    """Separate the save-time physics operator's tensors from the model's own."""
    model_sd, op_sd = {}, {}
    for key, value in state_dict.items():
        (op_sd if key.startswith(OP_PREFIXES) else model_sd)[key] = value
    return model_sd, op_sd


def report_saved_operator(op_sd):
    """Print the saved PDESSM coefficients, if this checkpoint has them."""
    K = op_sd.get('current_forward_op.K')
    r = op_sd.get('current_forward_op.r')
    if K is None and r is None:
        return
    K = K.item() if K is not None else float('nan')
    r = r.item() if r is not None else float('nan')
    msg = f"    saved PDESSM operator: K={K:.4f}, r={r:.4f}"
    if abs(K - 1.0) > 1e-6 or abs(r) > 1e-6:
        msg += ("  <- changed from init (K=1, r=0): the optimizer trained the "
                "operator. Ignored here; the harness sets K and r.")
    print(msg)


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
        dummy_op = get_forward_op(args, args.channels, label_channels, device)
        net = get_network(args, dummy_op, args.channels, label_channels,
                          feat_channels, device)
        state_dict = torch.load(filepath, map_location=device, weights_only=True)
        model_sd, op_sd = split_operator_keys(state_dict)
        result = net.load_state_dict(model_sd, strict=False)
        missing = [k for k in result.missing_keys if not k.startswith(OP_PREFIXES) and k != 'denoise_mu']
        if missing or result.unexpected_keys:
            raise RuntimeError(
                f"{filename}: state_dict mismatch outside the physics operator.\n"
                f"  missing:    {missing}\n  unexpected: {result.unexpected_keys}")

        # get_network only moves the backbone and feat_embed to `device`; the
        # shared/task encoders and heads stay on CPU. main_3 calls net.to(device)
        # after get_network, so do the same here.
        net = net.to(device)

        # use_specialized_heads is a plain attribute, so it is not in the
        # state_dict and a fresh net always starts with False (shared encoder /
        # head). main_3 saves "no_head_training_*" right before freeze_backbone()
        # and the final checkpoint after head training, which ran with True.
        net.use_specialized_heads = not filename.startswith("no_head_training_")

        net.eval()
        loaded.append({"name": filename, "model": net})
        heads = "task-specific" if net.use_specialized_heads else "shared"
        print(f" -> {filename}  [{heads} encoder/head]")
        report_saved_operator(op_sd)
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

            # one measurement, shared by every model
            torch.manual_seed(seed + batch_idx)
            b = heads[0](graph.y, graph.edge_index, graph.edge_weight, emb=False)
            if hasattr(heads[0], 'corrupt'):      # AddNoise: noise is added here,
                b = heads[0].corrupt(b)           # exactly as in training

            if cond[op_name] is None:
                cond[op_name] = operator_condition_number(heads[0], graph, op_name)

            for m in models:
                X, _, _ = m['model'](b, graph.edge_index, graph.edge_weight, graph.x)
                if not torch.isfinite(X).all():
                    nan_hits.add((m['name'], op_name))
                    X = torch.nan_to_num(X)
                loss_matrix[m['name']][op_name] += rel_mse(X, graph.y)
                counts[m['name']][op_name] += 1

            loss_matrix['BASELINE: X = 0'][op_name] += rel_mse(torch.zeros_like(graph.y), graph.y)
            counts['BASELINE: X = 0'][op_name] += 1
            if b.shape == graph.y.shape:
                loss_matrix['BASELINE: X = b'][op_name] += rel_mse(b, graph.y)
                counts['BASELINE: X = b'][op_name] += 1

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
        dataset='CPOX', use_meta_data=1, classify=0, CPOX_lags=1,
        train_batch_size=4, test_batch_size=4, train_frac=1.0, test_frac=1.0,
        method='foundation', layers=32, channels=64, cglsIter=5, solveIter=5,
        rnfPE=1, dropout=0.0, task='mask',
        mask_per_snapshot_budget=CFG['mask_budget'],
        # All True: get_forward_op(test=False) then returns all five operators,
        # so the foundation model's set_task() can reach each one.
        noise=True, painting=True, blurring=True, sensoring=True, pdessm=True,
        blur_count=str(CFG['blur_k']),
    )

    (train_dataset, test_dataset, train_loader, test_loader,
     label_channels, feat_channels) = get_data_and_loaders(args)
    sample = next(iter(test_dataset))
    num_nodes = sample.x.shape[0]
    args.num_nodes = num_nodes
    

    models = load_all_models(args, label_channels, feat_channels, device, model_dir="models")
    print(f"Models ready: {len(models)}")
    if not models:
        return

    loss_matrix, cond = run_operations_and_models(
        models, list(TASK_MAP.keys()), test_loader, args, device, CFG
    )
    plot_loss_matrix_table(loss_matrix, cond)


if __name__ == "__main__":
    main()