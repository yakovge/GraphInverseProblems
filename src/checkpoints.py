"""Loading and naming of foundation-model checkpoints, shared by plot_fixed.py (CPOX task hold-out) and plot_2.py
(leave-one-dataset-out).

A checkpoint is the state_dict of GraphInverseFoundationModel. Older checkpoints also contain
  current_encoder.* / current_head.*      -- registered aliases of the active task's encoder/head at save time.
                                             Loading them into a fresh net overwrote the head the fresh alias
                                             pointed to (inpainting) with the saved task's head. Dropped here;
                                             task_encoders.* / task_heads.* are the canonical weights.
  current_forward_op.* / solver.forOp.*   -- the physics operator active at save time (its embedding, PDESSM K/r).
                                             Physics, not model state. Dropped; the evaluator sets the operators.
"""
import os
import re

import torch

from utils import get_network

DROP_PREFIXES = ('current_encoder.', 'current_head.', 'current_forward_op.', 'solver.forOp.')
TASK_OF_FLAG = {'noise': 'denoising', 'painting': 'inpainting', 'blurring': 'source_localization',
                'sensoring': 'sensor_recovery', 'pdessm': 'pde_reconstruction'}
_NAME = re.compile(r'^(?P<nohead>no_head_training_)?(?P<run>.+?)_(?P<holdout>[a-z]+)_(?P<kind>test|dataset_test)(?:_seed_(?P<seed>\d+))?\.pth$')


def parse_checkpoint_name(filename):
    """-> dict(run, holdout, kind, seed, variant) or None if this is not a script.py / script2.py checkpoint.

    script.py:  [no_head_training_]<run>_<noise|painting|blurring|sensoring|pdessm>_test[_seed_k].pth
    script2.py: [no_head_training_]<run>_<cpox|pedalme|...>_dataset_test[_seed_k].pth
    variant: 'zero-shot (shared head)' for no_head_training_* (saved before head training),
             'adapted (task heads)' otherwise.
    """
    m = _NAME.match(os.path.basename(filename))
    if not m:
        return None
    d = m.groupdict()
    d['seed'] = int(d['seed']) if d['seed'] is not None else 0
    d['variant'] = 'zero-shot (shared head)' if d['nohead'] else 'adapted (task heads)'
    del d['nohead']
    return d


def load_foundation_checkpoint(path, args, ops, device, label_channels=1, feat_channels=None):
    """Build the net with the architecture the checkpoint was trained with and load it strictly (minus the
    dropped physics / alias keys). `ops` is the shared list of (operator, task) every model inverts."""
    sd = torch.load(path, map_location=device, weights_only=True)
    args.layers, args.channels = sd['backbone.K'].shape[:2]
    args.dc = 'prox' if any(k.startswith('dc_lam.') for k in sd) else 'cgls'
    ckpt_feat = sd['feat_embed.weight'].shape[1]
    if feat_channels is not None and ckpt_feat != feat_channels:
        raise RuntimeError(f"{path}: trained with {ckpt_feat} node-feature channels, the dataset gives {feat_channels}")
    net = get_network(args, ops, args.channels, label_channels, ckpt_feat, device).to(device)
    res = net.load_state_dict({k: v for k, v in sd.items() if not k.startswith(DROP_PREFIXES)}, strict=False)
    missing = [k for k in res.missing_keys if not k.startswith(DROP_PREFIXES)]
    if missing or res.unexpected_keys:
        raise RuntimeError(f"{path}: missing {missing}, unexpected {res.unexpected_keys}")
    # not in the state_dict: main_3 saves no_head_training_* before freeze_backbone() switches to task heads
    net.use_specialized_heads = not os.path.basename(path).startswith('no_head_training_')
    return net.eval()


def fmt_cell(values):
    """mean ± std over seeds (std only with >1 seed)."""
    t = torch.tensor([float(v) for v in values])
    if len(t) > 1:
        return f"{t.mean():.4f} ± {t.std():.4f}"
    return f"{t[0]:.4f}"
