"""Registry for the five graph inverse tasks.

Single entry point for building and driving task operators, so experiment scripts (and
``utils``) never touch the operator classes directly::

    from utils import TASKS, build_forward_ops, resample_task

    ops = build_forward_ops(['denoising', 'inpainting'], nodes_per_graph=207, ...)
    resample_task('inpainting', graph, ops['inpainting'])   # per-batch mask resample
    d_obs = ops['inpainting'](graph.y, graph.edge_index, graph.edge_weight, emb=False)
    d_obs = ops['inpainting'].perturb_observation(d_obs)    # Eq. (1) noise, if any

Mapping to the paper's tasks:

===========================  ==========================  ====================
task                         paper equivalent            operator
===========================  ==========================  ====================
``denoising``                --                          ``graphDenoise``
``inpainting``               property completion         ``graphMask`` (random)
``source_localization``      inverse source estimation   ``graphSmoothBlock``
``sensor_recovery``          --                          ``graphMask`` (structured)
``pde_state``                --                          ``graphCDR``
===========================  ==========================  ====================
"""

import torch

from graphForwardOps import graphEmbed, graphMask
from taskOps import (
    build_adjacency_list,
    graphCDR,
    graphDenoise,
    graphSmoothBlock,
    sample_random_observed,
    sample_structured_observed,
)

TASKS = [
    "denoising",
    "inpainting",
    "source_localization",
    "sensor_recovery",
    "pde_state",
]

# Short labels for run names, table headers and filenames.
TASK_SHORT = {
    "denoising": "denois",
    "inpainting": "inpaint",
    "source_localization": "source",
    "sensor_recovery": "sensor",
    "pde_state": "pde",
}

# Per-dataset node counts. Both datasets are static graphs, which is what lets the
# operators use the block-diagonal fast path.
NODES_PER_GRAPH = {"METRLA": 207, "CPOX": 20}

# Observation budget for the two masking tasks. CPOX has only 20 nodes, so the paper's
# budget of 16 would leave just 4 failures -- too few for structured masking to differ
# meaningfully from random. Scaled down to keep both tasks distinguishable.
MASK_BUDGET = {"METRLA": 16, "CPOX": 8}


def nodes_per_graph_for(dataset):
    for key, n in NODES_PER_GRAPH.items():
        if key in dataset:
            return n
    raise ValueError(f"No node count registered for dataset {dataset!r}")


def mask_budget_for(dataset):
    for key, n in MASK_BUDGET.items():
        if key in dataset:
            return n
    raise ValueError(f"No mask budget registered for dataset {dataset!r}")


def build_shared_embedding(hid_channels, label_channels, learn_emb=True, device="cuda"):
    """One embedding for every task -- the crux of the zero-shot setup.

    ``graphEmbed`` holds the only parameters a forward operator owns: the lift/project
    between the 1-channel physical space and the network's hidden space. If each task
    built its own, a held-out task would be evaluated with a *randomly initialised*
    embedding and its zero-shot number would be meaningless. Sharing one instance means
    it is fully trained by the training tasks and equally valid for the held-out ones,
    leaving **zero task-specific parameters** anywhere in the model.
    """
    return graphEmbed(hid_channels, label_channels, learned=learn_emb, device=device)


def build_forward_ops(
    tasks,
    nodes_per_graph,
    hid_channels,
    label_channels,
    shared_emb=None,
    learn_emb=True,
    device="cuda",
    blur_count=4,
    noise_sigma=0.1,
    cdr=None,
):
    """Build operators for ``tasks``, all sharing one embedding.

    Returns ``(ops, shared_emb)``. Pass ``shared_emb`` back in to extend an existing set
    (e.g. to add the held-out evaluation tasks) without creating new parameters.
    """
    if shared_emb is None:
        shared_emb = build_shared_embedding(
            hid_channels, label_channels, learn_emb=learn_emb, device=device
        )
    cdr = cdr or {}

    common = dict(
        nin=label_channels, embdsize=hid_channels, learnEmb=learn_emb, device=device
    )
    ops = {}
    for task in tasks:
        if task == "denoising":
            op = graphDenoise(sigma=noise_sigma, **common)
        elif task == "inpainting":
            op = graphMask(
                ind=torch.arange(1, device=device),  # replaced per batch by resample_task
                embdsize=hid_channels,
                nin=label_channels,
                learnEmb=learn_emb,
                device=device,
            )
        elif task == "sensor_recovery":
            op = graphMask(
                ind=torch.arange(1, device=device),
                embdsize=hid_channels,
                nin=label_channels,
                learnEmb=learn_emb,
                device=device,
            )
        elif task == "source_localization":
            op = graphSmoothBlock(
                nodes_per_graph=nodes_per_graph, k=int(blur_count), **common
            )
        elif task == "pde_state":
            op = graphCDR(nodes_per_graph=nodes_per_graph, **common, **cdr)
        else:
            raise ValueError(f"Unknown task {task!r}; expected one of {TASKS}")

        # Overwrite whatever embedding the constructor made with the shared one.
        op.Emb = shared_emb
        ops[task] = op.to(device)

    return ops, shared_emb


# Masking operators have no stochastic state of their own; the observed-node set is
# redrawn per batch. Cached so the BFS neighbour lists are built once per graph.
_ADJ_CACHE = {}


def _adjacency_list(edge_index, n):
    key = (n, int(edge_index.shape[1]))
    if key not in _ADJ_CACHE:
        _ADJ_CACHE[key] = build_adjacency_list(edge_index, n)
    return _ADJ_CACHE[key]


def resample_task(task, graph, forward_op, nodes_per_graph, budget):
    """Redraw a task's per-batch randomness.

    Generalises upstream's ``task_specific_modifiers`` to the five tasks. Only the two
    masking tasks carry per-batch state; the rest are deterministic operators.
    """
    if task == "inpainting":
        forward_op.ind = sample_random_observed(graph, budget, nodes_per_graph)
    elif task == "sensor_recovery":
        adj = _adjacency_list(graph.edge_index, nodes_per_graph)
        forward_op.ind = sample_structured_observed(graph, budget, nodes_per_graph, adj)
    return forward_op


def run_name(train_tasks, val_task, seed=None):
    """Stable identifier naming a model's training tasks and rotation slot.

    The ``_val-`` component names the rotation slot from ``experiment_split``; under
    ``--selection train`` that task receives no gradients and does no model selection,
    making it the first of the run's two zero-shot tasks. ``seed`` appends ``_s{seed}``
    so several seeds of one configuration get distinct run directories.
    """
    trained = "-".join(TASK_SHORT[t] for t in train_tasks)
    base = f"FM_train-{trained}_val-{TASK_SHORT[val_task]}"
    return base if seed is None else f"{base}_s{seed}"


def experiment_split(index):
    """Task assignment for experiment ``index`` (0-4).

    ``val = (i+1) % 5``, ``test = (i+2) % 5`` gives each task exactly one turn as
    validation and one as test, with no (val, test) pair ever recurring reversed.
    """
    val = TASKS[(index + 1) % len(TASKS)]
    test = TASKS[(index + 2) % len(TASKS)]
    train = [t for t in TASKS if t not in (val, test)]
    return train, val, test
