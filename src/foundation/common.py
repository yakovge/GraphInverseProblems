"""Shared training core for the foundation-model experiments.

One model is trained on three inverse tasks and finally evaluated on all five. Under the
default protocol (v2, ``--selection train``) early stopping tracks the mean training-task
nMSE, so BOTH held-out tasks contribute neither gradients nor model selection -- two
clean zero-shot numbers per run, and those are the headline. ``--selection val_task``
restores the v1 protocol (early-stop on the fourth task), kept for comparison; it is the
protocol whose val signal never improved on 4/5 METR-LA runs, stranding them with an
untrained epoch-0 checkpoint.

Reuses the paper's machinery wherever possible: ``get_data_and_loaders``,
``process_data``, ``get_network``, ``count_trainable_parameters``, and the regression
loss of Appendix E.1::

    L = 0.5 * ( MSE(x, x_pred)/MSE(0, x) + alpha * MSE(d_obs, d_hat)/MSE(0, d_obs) )

The reported nMSE is the first term alone, as the paper specifies.

Splits follow the article exactly -- train/test only, no snapshot-level validation. The
train/val/test structure of this study lives at the *task* level instead.
"""

import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from progress import ProgressReporter  # noqa: E402
from tasks import (  # noqa: E402
    TASK_SHORT,
    TASKS,
    build_forward_ops,
    mask_budget_for,
    nodes_per_graph_for,
    resample_task,
    run_name,
)
from utils import (  # noqa: E402
    count_trainable_parameters,
    get_data_and_loaders,
    get_network,
    process_data,
)


def nmse(pred, target):
    """Normalised MSE, as in Appendix E.1: MSE(pred, target) / MSE(0, target)."""
    return F.mse_loss(pred, target) / F.mse_loss(torch.zeros_like(target), target)


def write_history(run_dir, records):
    """Atomically rewrite history.json -- the per-epoch training curves.

    Standalone rather than routed through ProgressReporter: its update() merges every
    field into all subsequent writes, so a growing list would balloon progress.json.
    """
    path = os.path.join(run_dir, "history.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(records, fh, indent=2)
    os.replace(tmp, path)


def atomic_torch_save(payload, path):
    """torch.save via a temp file; a kill mid-write must not corrupt the checkpoint."""
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def set_operator(net, op):
    """Point the network and its data-fit solver at a different forward operator.

    This is what lets one model span several tasks. It is safe to swap freely because
    the operators hold **no parameters of their own** -- every one of them shares a
    single ``graphEmbed`` -- so the optimizer's parameter set never changes.
    """
    net.forOp = op
    if hasattr(net, "dataProj") and net.dataProj is not None:
        net.dataProj.forOp = op
    return net


def observe(op, y, graph):
    """Produce d_obs = F(x) + eps for one task (Eq. 1).

    Noise is applied once, here, rather than inside the operator: ``graph_CGLS``
    evaluates the operator many times per solve, and redrawing noise on each call would
    break the Krylov iteration.
    """
    d = op(y, graph.edge_index, graph.edge_weight, emb=False)
    # Upstream operators (graphMask, graphPath) predate this hook and are noiseless, so
    # treat a missing perturb_observation as the identity rather than requiring a patch.
    perturb = getattr(op, "perturb_observation", None)
    return perturb(d) if perturb is not None else d


def run_task_batch(net, op, task, graph, args, budget, npg, train=True):
    """One forward/backward for a single task on a single batch."""
    graph = process_data(args, graph)
    resample_task(task, graph, op, nodes_per_graph=npg, budget=budget)
    graph = graph.to(args.device)
    set_operator(net, op)

    d_obs = observe(op, graph.y, graph)
    X, _, _ = net(d_obs, graph.edge_index, graph.edge_weight, graph.x)
    d_rec = op(X, graph.edge_index, graph.edge_weight, emb=False)

    loss_x = nmse(X, graph.y)
    loss_data = nmse(d_rec, d_obs)
    loss = 0.5 * (loss_x + args.alpha * loss_data)
    return loss, loss_x.item(), loss_data.item(), len(graph.batch.unique())


def train_epoch(net, ops, train_tasks, loader, optimizer, args, budget, npg):
    """One epoch, interleaving the training tasks across batches.

    Tasks alternate per batch rather than per epoch so every optimizer step sees a mix
    over time -- training on one task to convergence before switching would let the
    shared prior drift toward whichever task came last.
    """
    net.train()
    totals = {t: [0.0, 0.0, 0] for t in train_tasks}  # loss_x, loss_data, count
    accum = getattr(args, "accum_steps", 1)

    optimizer.zero_grad()
    for i, graph in enumerate(loader):
        task = train_tasks[i % len(train_tasks)]
        loss, lx, ld, bs = run_task_batch(
            net, ops[task], task, graph, args, budget, npg, train=True
        )
        # Scale so the accumulated gradient equals the mean over the effective batch.
        (loss / accum).backward()

        # Tasks rotate per micro-batch, so an optimizer step aggregates gradients from
        # several tasks at once. That is deliberate: a step driven by a single task
        # would pull the shared prior toward whichever task happened to come last.
        if (i + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        totals[task][0] += lx * bs
        totals[task][1] += ld * bs
        totals[task][2] += bs

    # Flush a trailing partial accumulation window.
    if len(loader) % accum != 0:
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    return {
        t: {
            "nmse": totals[t][0] / max(totals[t][2], 1),
            "data_fit": totals[t][1] / max(totals[t][2], 1),
        }
        for t in train_tasks
    }


@torch.no_grad()
def evaluate(net, ops, task, loader, args, budget, npg):
    """nMSE and data-fit for one task over the test snapshots."""
    net.eval()
    total_x = total_d = 0.0
    count = 0
    for graph in loader:
        _, lx, ld, bs = run_task_batch(
            net, ops[task], task, graph, args, budget, npg, train=False
        )
        total_x += lx * bs
        total_d += ld * bs
        count += bs
    return {"nmse": total_x / max(count, 1), "data_fit": total_d / max(count, 1)}


def evaluate_all(net, ops, loader, args, budget, npg, tasks=None):
    """Evaluate every task -- this populates the comparison CSV."""
    return {
        t: evaluate(net, ops, t, loader, args, budget, npg) for t in (tasks or TASKS)
    }


def train_foundation_model(args, train_tasks, val_task, test_task):
    """Train one multi-task model and report on all five tasks.

    Returns the metrics dict that ``compare.py`` consumes.
    """
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    name = run_name(train_tasks, val_task, seed=args.seed)
    run_dir = os.path.join(args.runs_root, name)
    os.makedirs(run_dir, exist_ok=True)

    npg = nodes_per_graph_for(args.dataset)
    budget = mask_budget_for(args.dataset)

    train_dataset, test_dataset, train_loader, test_loader, label_channels, feat_channels = (
        get_data_and_loaders(args)
    )

    # Every task shares one embedding, so the held-out tasks are evaluated with weights
    # that were fully trained by the training tasks. Without this the zero-shot numbers
    # would be meaningless -- see tasks.build_shared_embedding.
    ops, shared_emb = build_forward_ops(
        TASKS,
        nodes_per_graph=npg,
        hid_channels=args.channels,
        label_channels=label_channels,
        learn_emb=True,
        device=args.device,
        blur_count=args.blur_count,
        noise_sigma=args.noise_sigma,
    )

    net = get_network(
        args, ops[train_tasks[0]], args.channels, label_channels, feat_channels, args.device
    ).to(args.device)

    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True, eps=1e-3
    )

    # Guard the zero-shot claim: swapping operators must not change the parameter set.
    baseline = {id(p) for p in net.parameters()}
    for task in TASKS:
        set_operator(net, ops[task])
        if {id(p) for p in net.parameters()} != baseline:
            raise RuntimeError(
                f"Swapping to task {task!r} changed the model's parameters. "
                "Operators must share one embedding and own nothing else."
            )
    set_operator(net, ops[train_tasks[0]])

    n_params = count_trainable_parameters(net)
    print(f"run           : {name}")
    print(f"train tasks   : {[TASK_SHORT[t] for t in train_tasks]}")
    if args.selection == "train":
        print(f"selection     : mean train-task nMSE  (patience {args.max_patience}, "
              f"min epochs {args.min_epochs})")
        print(f"zero-shot     : {TASK_SHORT[val_task]}, {TASK_SHORT[test_task]}  "
              f"(no gradients, no selection)")
    else:
        print(f"val task      : {TASK_SHORT[val_task]}  (early stopping only)")
        print(f"test task     : {TASK_SHORT[test_task]}  (never trained, never selected on)")
    print(f"parameters    : {n_params}")
    print(f"nodes/graph   : {npg}   mask budget: {budget}")
    print(
        f"batch         : {args.effective_batch_size} effective "
        f"({args.micro_batch} x {args.accum_steps} accumulation)"
    )
    print(f"train/test    : {len(list(train_dataset))}/{len(list(test_dataset))} snapshots")
    print(flush=True)

    reporter = ProgressReporter(
        run_dir,
        name,
        args.epochs,
        extra={
            "train_tasks": train_tasks,
            "val_task": val_task,
            "test_task": test_task,
            "dataset": args.dataset,
            "parameters": n_params,
        },
    )

    best_metric = float("inf")
    best_epoch = -1
    patience = 0
    history = []
    started = time.time()
    last_epoch = 0

    def checkpoint_payload(epoch):
        return {
            "model": net.state_dict(),
            "epoch": epoch,
            "protocol": 2,
            "selection": args.selection,
            "train_tasks": train_tasks,
            "val_task": val_task,
            "test_task": test_task,
            "zeroshot_tasks": [val_task, test_task],
            "args": vars(args),
        }

    try:
        for epoch in range(args.epochs):
            epoch_started = time.time()
            last_epoch = epoch + 1
            train_stats = train_epoch(
                net, ops, train_tasks, train_loader, optimizer, args, budget, npg
            )
            mean_train = sum(s["nmse"] for s in train_stats.values()) / len(train_stats)

            # The val-slot task is diagnostic only under selection="train" (it does no
            # model selection, so it stays a clean zero-shot task); evaluating it every
            # epoch would cost ~19s/epoch on METR-LA for a curve nobody selects on.
            run_val = args.val_every > 0 and epoch % args.val_every == 0
            val_stats = (
                evaluate(net, ops, val_task, test_loader, args, budget, npg)
                if run_val
                else None
            )

            # Model selection. "train": mean nMSE over the training tasks -- the v1
            # val-task signal never improved on 4/5 METR-LA runs, so patience fired with
            # an untrained epoch-0 checkpoint. The test task is never consulted either
            # way. The first epoch always counts as an improvement: with best_metric =
            # inf the relative test degenerates to inf > inf, which is False, and
            # nothing would ever be saved.
            metric = val_stats["nmse"] if args.selection == "val_task" else mean_train
            improved = best_epoch < 0 or (best_metric - metric) > abs(best_metric * 0.005)
            if improved:
                best_metric = metric
                best_epoch = epoch
                patience = 0
                # Only the checkpoint is written here. Scoring all five tasks costs five
                # full passes over the test set (~95s on METR-LA), and doing that on
                # every improving epoch adds roughly 45 minutes per run for numbers that
                # are immediately superseded. The best checkpoint is reloaded and scored
                # once after the loop, which yields identical results.
                atomic_torch_save(
                    checkpoint_payload(epoch), os.path.join(run_dir, "model.pth")
                )
            else:
                patience += 1

            # The last state survives regardless of selection, so a run can be re-scored
            # post hoc (e.g. "did zero-shot peak after the selected epoch?").
            atomic_torch_save(
                checkpoint_payload(epoch), os.path.join(run_dir, "model_last.pth")
            )

            history.append(
                {
                    "epoch": epoch,
                    "train_nmse": {t: s["nmse"] for t, s in train_stats.items()},
                    "train_data_fit": {t: s["data_fit"] for t, s in train_stats.items()},
                    "selection": args.selection,
                    "selection_metric": metric,
                    "val_task": val_task,
                    "val_nmse": val_stats["nmse"] if val_stats else None,
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                    "patience": patience,
                    "epoch_seconds": round(time.time() - epoch_started, 1),
                }
            )
            write_history(run_dir, history)

            # Field names kept from v1 so watch_progress.py needs no change;
            # best_val_nmse now carries whichever metric drives selection.
            progress_fields = dict(
                epoch=epoch + 1,
                status="training",
                train_loss=mean_train,
                selection_metric=metric,
                best_val_nmse=best_metric,
                patience=patience,
            )
            if val_stats is not None:
                progress_fields["val_nmse"] = val_stats["nmse"]
            reporter.update(**progress_fields)

            val_part = (
                f"val[{TASK_SHORT[val_task]}] {val_stats['nmse']:.4f}  "
                if val_stats is not None
                else ""
            )
            print(
                f"epoch {epoch:3d}  train {mean_train:.4f}  {val_part}"
                f"best {best_metric:.4f}  patience {patience}",
                flush=True,
            )

            if patience > args.max_patience and last_epoch >= args.min_epochs:
                print(f"early stop at epoch {epoch} (patience {args.max_patience})")
                break

    except Exception as exc:  # noqa: BLE001 - surface the failure in the progress file
        reporter.finish(status="failed", error=str(exc))
        raise

    # Score all five tasks once, from the checkpoint that early stopping selected.
    checkpoint = os.path.join(run_dir, "model.pth")
    if os.path.isfile(checkpoint):
        state = torch.load(checkpoint, map_location=args.device, weights_only=False)
        net.load_state_dict(state["model"])
        print(f"\nscoring all tasks from best checkpoint (epoch {state['epoch']})...", flush=True)
    else:
        # No epoch ever improved, so nothing was saved; score whatever we have.
        best_epoch = 0
    best_all = evaluate_all(net, ops, test_loader, args, budget, npg)

    metrics = {
        "run_name": name,
        "kind": "foundation",
        "protocol": 2,
        "selection": args.selection,
        "dataset": args.dataset,
        "train_tasks": train_tasks,
        "val_task": val_task,
        "test_task": test_task,
        "zeroshot_tasks": [val_task, test_task],
        "best_epoch": best_epoch,
        "best_selection_metric": best_metric,
        "epochs_run": last_epoch,
        "epochs_budget": args.epochs,
        "stopped_early": last_epoch < args.epochs,
        "min_epochs": args.min_epochs,
        "val_every": args.val_every,
        "parameters": n_params,
        "seed": args.seed,
        "effective_batch_size": args.effective_batch_size,
        "wall_seconds": round(time.time() - started, 1),
        "per_task": best_all,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    reporter.finish(status="completed", best_val_nmse=best_metric)

    print("\nper-task nMSE on the test snapshots:")
    for task in TASKS:
        if task in train_tasks:
            role = "train"
        elif args.selection == "val_task":
            role = "VAL" if task == val_task else "TEST"
        else:
            role = "ZERO-SHOT"
        print(f"  {TASK_SHORT[task]:<8} {best_all[task]['nmse']:.4f}   [{role}]")

    return metrics
