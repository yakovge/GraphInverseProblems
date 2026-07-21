"""Assemble the comparison tables from every finished run.

Runs are grouped into configurations by stripping the ``_s{seed}`` suffix, and every
table reports the mean across seeds (with a population std once there is more than one
seed). Only protocol-v2 runs are included by default: the v1 sweep early-stopped on a
val-task signal that never improved, so 4/5 of its models were scored from an untrained
epoch-0 checkpoint -- mixing those into rankings would corrupt every table. Pass
``--include_legacy`` to add them anyway, under their old roles.

Writes five CSVs into ``results/<dataset>/``:

``comparison.csv``
    The main grid: every configuration x every task, nMSE on the test snapshots, with
    each cell tagged by the role that task played (train / zeroshot / unseen, plus the
    legacy val / test).

``transfer_gap.csv``
    One row per (configuration, zero-shot task): the held-out task against the mean of
    the training tasks. Under protocol v2 each configuration has TWO zero-shot tasks. A
    ratio near 1 means the shared prior generalised across operators.

``original_zeroshot.csv``
    The paper's single-task model on the five tasks it never trained on, which is the
    baseline that makes the foundation models interpretable.

``per_task_ranking.csv``
    For each task, which configuration does best on it, and whether it ever trained on
    that task.

``prior_value.csv``
    Each task's best classical baseline (pinv / adjoint / mean / oracle tikhonov /
    oracle laplacian) against the best zero-shot model. The verdict a learned prior has
    to survive: beating unregularised least squares means little if a textbook
    regularizer gets there too.

    python compare.py --runs_root ../runs/METRLA
"""

import argparse
import csv
import json
import os
import re

from tasks import TASK_SHORT, TASKS

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# The article has no validation split and selects on the test set; exp_original
# reproduces that faithfully so its number stays comparable to the published table.
# Flagged wherever it appears so it is never read as a clean held-out figure.
FOOTNOTES = [
    "nMSE = MSE(x_pred, x) / MSE(0, x), the paper's normalised error (Appendix E.1). Lower is better.",
    "role: train = gradients; zeroshot = neither gradients nor model selection (protocol",
    "      v2 selects on the mean training-task nMSE, so BOTH held-out tasks are clean);",
    "      val / test = the legacy v1 roles, shown only for --include_legacy runs;",
    "      unseen = the original single-task model, which trained on none of these five.",
    "Values are means across seeds; *_nmse_std is the population std, blank for one seed.",
    "Following the article, there is no snapshot-level validation split: all held-out",
    "metrics are computed on the same test snapshots. The separation is at the TASK",
    "level -- a zero-shot task contributed no gradients and no model selection.",
    "ORIGINAL reproduces the paper's protocol including its selection on the test set,",
    "so its own-task number is comparable to Table 5 but is not a clean holdout.",
    "BASELINE tikhonov/laplacian rows are ORACLE-tuned: their lambda was chosen per task",
    "on the test nMSE. They are a deliberate ceiling for classical regularization.",
]

# Everything a non-learned method can offer; prior_value takes the min over these.
CLASSICAL_KEYS = ("pinv", "adjoint", "mean", "tikhonov", "laplacian")


def load_baselines(out_dir):
    """Trivial-predictor and classical reference points, if baselines.py has run."""
    path = os.path.join(out_dir, "baselines.json")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def best_classical_for(baselines, task):
    """The strongest non-learned number for a task, and which baseline supplied it."""
    per = baselines["per_task"][task]
    avail = {k: per[k] for k in CLASSICAL_KEYS if k in per}
    name = min(avail, key=avail.get)
    return avail[name], name


def baseline_rows(baselines):
    """Non-learned reference points, formatted as pseudo-model rows for the main grid.

    These belong in the grid, not a separate file. Without them a reader sees five
    models scoring alike and concludes the prior transferred; the baseline rows are what
    distinguish that from every model having collapsed onto the same prior-free
    solution. Rows whose key is missing (an older baselines.json) are skipped.
    """
    labels = (
        ("pinv", "BASELINE least-squares (no prior)"),
        ("adjoint", "BASELINE adjoint (no prior)"),
        ("mean", "BASELINE predict train mean"),
        ("tikhonov", "BASELINE tikhonov (oracle lambda)"),
        ("laplacian", "BASELINE laplacian reg (oracle lambda)"),
    )
    rows = []
    for key, label in labels:
        if any(key not in baselines["per_task"][t] for t in TASKS):
            continue
        row = [label, "baseline", "-", "", "-", "-"]
        for task in TASKS:
            row += [round(baselines["per_task"][task][key], 6), "", "baseline"]
        row += ["", "", "", "", ""]
        rows.append(row)
    return rows


def load_runs(runs_root, include_legacy=False):
    """Every finished run; protocol-v1 runs are dropped unless explicitly included."""
    runs = []
    if not os.path.isdir(runs_root):
        return runs
    for name in sorted(os.listdir(runs_root)):
        path = os.path.join(runs_root, name, "metrics.json")
        if not os.path.isfile(path):
            continue
        with open(path) as fh:
            runs.append(json.load(fh))

    legacy = [r for r in runs if r.get("protocol", 1) < 2]
    if legacy and not include_legacy:
        print(
            f"  skipping {len(legacy)} legacy (protocol 1) run(s) -- their selection "
            "protocol stranded most models at an untrained checkpoint; pass "
            "--include_legacy to add them with their old roles"
        )
        runs = [r for r in runs if r.get("protocol", 1) >= 2]
    return runs


def config_of(name):
    """Strip the seed suffix: seeds of one configuration share everything else."""
    return re.sub(r"_s\d+$", "", name)


def _mean_std(values):
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    return mean, std


def aggregate(runs):
    """Group per-seed runs into configuration records with mean/std per task."""
    groups = {}
    for r in runs:
        groups.setdefault(config_of(r["run_name"]), []).append(r)

    configs = []
    for name in sorted(groups):
        members = sorted(groups[name], key=lambda r: r.get("seed", 0))
        base = members[0]
        for m in members[1:]:
            for key in ("kind", "protocol", "selection", "train_tasks", "val_task", "test_task"):
                if m.get(key) != base.get(key):
                    raise ValueError(
                        f"seed group {name!r} disagrees on {key}: "
                        f"{m.get(key)!r} vs {base.get(key)!r}"
                    )
        per_task = {}
        for t in TASKS:
            vals = [m["per_task"][t]["nmse"] for m in members]
            mean, std = _mean_std(vals)
            per_task[t] = {"mean": mean, "std": std, "values": vals}
        configs.append(
            {
                "config": name,
                "kind": base.get("kind", "foundation"),
                "protocol": base.get("protocol", 1),
                "selection": base.get("selection"),
                "train_tasks": base.get("train_tasks", []),
                "val_task": base.get("val_task"),
                "test_task": base.get("test_task"),
                "zeroshot_tasks": base.get("zeroshot_tasks")
                or [t for t in (base.get("val_task"), base.get("test_task")) if t],
                "seeds": [m.get("seed", 0) for m in members],
                "n_seeds": len(members),
                "per_task": per_task,
                "members": members,
            }
        )
    configs.sort(key=lambda c: (c["kind"] == "original", c["config"]))
    return configs


def role_of(cfg, task):
    if cfg.get("kind") == "original":
        return "unseen"
    if task in cfg.get("train_tasks", []):
        return "train"
    # Legacy protocol (or --selection val_task): the val task did model selection.
    if cfg.get("protocol", 1) < 2 or cfg.get("selection") == "val_task":
        if task == cfg.get("val_task"):
            return "val"
        if task == cfg.get("test_task"):
            return "test"
        return "unseen"
    if task in cfg.get("zeroshot_tasks", []):
        return "zeroshot"
    return "unseen"


def zeroshot_tasks_of(cfg):
    """Tasks that contributed neither gradients nor model selection to this config."""
    return [t for t in TASKS if role_of(cfg, t) in ("zeroshot", "test")]


def write_csv(path, header, rows, footnotes=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
        if footnotes:
            w.writerow([])
            for line in footnotes:
                w.writerow([f"# {line}"])
    print(f"  wrote {path}  ({len(rows)} rows)")


def prior_value(configs, baselines, out_dir):
    """Does the learned prior beat the best classical method?

    The headline sanity check, now against the strongest non-learned baseline rather
    than only unregularised least squares: every model here is built on a CGLS data-fit
    step, and beating that step means nothing if a textbook Tikhonov or Laplacian
    regularizer reaches the same number without learning anything. The verdict is driven
    by the best ZERO-SHOT model -- the study's claim is transfer, so a model that
    trained on the task does not get to defend the prior.
    """
    if baselines is None:
        print("  (no baselines.json; run baselines.py to enable prior_value.csv)")
        return []

    header = [
        "task",
        "best_classical",
        "best_classical_name",
        "best_model_nmse",
        "best_model",
        "best_model_role",
        "best_zeroshot_nmse",
        "best_zeroshot_model",
        "improvement",
        "improvement_pct",
        "prior_helped",
    ]
    rows = []
    for task in TASKS:
        classical, cname = best_classical_for(baselines, task)
        best = min(configs, key=lambda c: c["per_task"][task]["mean"])
        zs = [c for c in configs if role_of(c, task) in ("zeroshot", "test", "unseen")]
        best_zs = min(zs, key=lambda c: c["per_task"][task]["mean"]) if zs else None
        zs_v = gain = pct = ""
        verdict = "n/a"
        if best_zs is not None:
            zs_v = best_zs["per_task"][task]["mean"]
            gain = classical - zs_v
            pct = round(100.0 * gain / classical, 2) if classical > 0 else ""
            verdict = "yes" if gain > 0.05 * classical else "marginal" if gain > 0 else "NO"
            zs_v, gain = round(zs_v, 6), round(gain, 6)
        rows.append(
            [
                TASK_SHORT[task],
                round(classical, 6),
                cname,
                round(best["per_task"][task]["mean"], 6),
                best["config"],
                role_of(best, task),
                zs_v,
                best_zs["config"] if best_zs is not None else "",
                gain,
                pct,
                verdict,
            ]
        )

    notes = [
        "best_classical = min over pinv / adjoint / mean / oracle tikhonov / oracle",
        "       laplacian -- the strongest thing a non-learned method achieves. The",
        "       tikhonov/laplacian lambdas are chosen per task ON THE TEST nMSE, so this",
        "       is a deliberate ceiling for classical regularization.",
        "The verdict compares the best ZERO-SHOT model (no gradients, no selection on",
        "       the task; the ORIGINAL model counts) against best_classical:",
        "       'yes' = beat it by more than 5%; 'marginal' = beat it by less; 'NO' =",
        "       did not beat it, meaning zero-shot transfer added nothing a textbook",
        "       method would not supply.",
        "If most tasks read NO, cross-model similarity is not evidence of transfer --",
        "the models have collapsed onto solutions classical methods already reach.",
    ]
    write_csv(os.path.join(out_dir, "prior_value.csv"), header, rows, notes)
    return rows


def main_grid(configs, out_dir, baselines=None):
    header = ["model", "kind", "selection", "n_seeds", "train_tasks", "zeroshot_tasks"]
    for task in TASKS:
        header += [
            f"{TASK_SHORT[task]}_nmse",
            f"{TASK_SHORT[task]}_nmse_std",
            f"{TASK_SHORT[task]}_role",
        ]
    header += ["best_epoch", "epochs_run", "stopped_early", "parameters", "wall_seconds"]

    rows = []
    for cfg in configs:
        row = [
            cfg["config"],
            cfg["kind"],
            cfg.get("selection") or "-",
            cfg["n_seeds"],
            "|".join(TASK_SHORT.get(t, t) for t in cfg["train_tasks"]),
            "|".join(TASK_SHORT.get(t, t) for t in zeroshot_tasks_of(cfg)) or "-",
        ]
        for task in TASKS:
            pt = cfg["per_task"][task]
            row += [
                round(pt["mean"], 6),
                round(pt["std"], 6) if cfg["n_seeds"] > 1 else "",
                role_of(cfg, task),
            ]
        members = cfg["members"]
        row += [
            "|".join(str(m.get("best_epoch", "")) for m in members),
            "|".join(str(m.get("epochs_run", "")) for m in members),
            "|".join(str(m.get("stopped_early", "")) for m in members),
            members[0].get("parameters", ""),
            round(sum(m.get("wall_seconds", 0) for m in members), 1),
        ]
        rows.append(row)

    if baselines is not None:
        rows += baseline_rows(baselines)

    write_csv(os.path.join(out_dir, "comparison.csv"), header, rows, FOOTNOTES)
    return rows


def transfer_gap(configs, out_dir):
    """How much worse is a configuration on the operators it never saw?

    Long format: one row per (configuration, zero-shot task). Under protocol v2 both
    held-out tasks qualify; legacy runs contribute only their test task.
    """
    header = [
        "model",
        "zeroshot_task",
        "zeroshot_nmse",
        "zeroshot_nmse_std",
        "mean_train_nmse",
        "mean_train_nmse_std",
        "absolute_gap",
        "ratio",
        "n_seeds",
    ]
    rows = []
    for cfg in configs:
        if cfg["kind"] == "original":
            continue
        n = cfg["n_seeds"]
        train_tasks = cfg["train_tasks"]
        # Per-seed mean over the training tasks (values are seed-aligned), so the std
        # describes seed variance of the same quantity the mean does.
        train_means = [
            sum(cfg["per_task"][t]["values"][i] for t in train_tasks) / len(train_tasks)
            for i in range(n)
        ]
        mt_mean, mt_std = _mean_std(train_means)
        for zs in zeroshot_tasks_of(cfg):
            pt = cfg["per_task"][zs]
            rows.append(
                [
                    cfg["config"],
                    TASK_SHORT[zs],
                    round(pt["mean"], 6),
                    round(pt["std"], 6) if n > 1 else "",
                    round(mt_mean, 6),
                    round(mt_std, 6) if n > 1 else "",
                    round(pt["mean"] - mt_mean, 6),
                    round(pt["mean"] / mt_mean, 4) if mt_mean > 0 else "",
                    n,
                ]
            )

    notes = [
        "One row per (model, zero-shot task); protocols v2/v3 give each model two.",
        "absolute_gap = zeroshot_nmse - mean_train_nmse; ratio = zeroshot / mean_train.",
        "A ratio near 1 means the learned prior transferred to an operator never",
        "trained on. A large ratio means the model fitted its training operators",
        "specifically.",
        "denois zero-shot rows reflect the identity operator's CGLS projection (the",
        "output is the observation, for every model), NOT the learned prior -- see",
        "FOUNDATION_MODEL.md, protocol v3.",
    ]
    write_csv(os.path.join(out_dir, "transfer_gap.csv"), header, rows, notes)
    return rows


def original_zeroshot(configs, out_dir):
    """The paper's single-task model against the five study tasks."""
    original = next((c for c in configs if c["kind"] == "original"), None)
    if original is None:
        print("  (no ORIGINAL run yet; skipping original_zeroshot.csv)")
        return []

    foundation = [c for c in configs if c["kind"] != "original"]
    header = [
        "task",
        "original_nmse",
        "original_nmse_std",
        "best_foundation_nmse",
        "best_foundation_model",
        "improvement",
    ]
    rows = []
    for task in TASKS:
        pt = original["per_task"][task]
        std = round(pt["std"], 6) if original["n_seeds"] > 1 else ""
        if foundation:
            best = min(foundation, key=lambda c: c["per_task"][task]["mean"])
            best_v = best["per_task"][task]["mean"]
            rows.append(
                [
                    TASK_SHORT[task],
                    round(pt["mean"], 6),
                    std,
                    round(best_v, 6),
                    best["config"],
                    round(pt["mean"] - best_v, 6),
                ]
            )
        else:
            rows.append([TASK_SHORT[task], round(pt["mean"], 6), std, "", "", ""])

    members = original["members"]
    reference = members[0].get("paper_reference_nmse")
    own_vals = [m["own_task_nmse"] for m in members if m.get("own_task_nmse") is not None]
    own = sum(own_vals) / len(own_vals) if own_vals else float("nan")
    provenance = (
        f"paper Table 5 reports {reference}"
        if reference is not None
        else "the paper reports no value for 'path' on this dataset"
    )
    notes = [
        f"ORIGINAL trained only on the paper's 'path' task (own-task nMSE {own:.4f}; {provenance}).",
        "It saw none of these five tasks, so every column here is zero-shot for it.",
        "improvement = original_nmse - best_foundation_nmse; positive favours the foundation models.",
    ]
    write_csv(os.path.join(out_dir, "original_zeroshot.csv"), header, rows, notes)
    return rows


def per_task_ranking(configs, out_dir):
    """Who wins each task, and had they trained on it?"""
    header = ["task", "rank", "model", "nmse", "nmse_std", "role", "n_seeds"]
    rows = []
    for task in TASKS:
        ordered = sorted(configs, key=lambda c: c["per_task"][task]["mean"])
        for rank, cfg in enumerate(ordered, start=1):
            pt = cfg["per_task"][task]
            rows.append(
                [
                    TASK_SHORT[task],
                    rank,
                    cfg["config"],
                    round(pt["mean"], 6),
                    round(pt["std"], 6) if cfg["n_seeds"] > 1 else "",
                    role_of(cfg, task),
                    cfg["n_seeds"],
                ]
            )

    notes = [
        "If a model with role=zeroshot or role=unseen ranks near the top for a task,",
        "the prior generalised to an operator it never trained on -- the result this",
        "study is looking for.",
    ]
    write_csv(os.path.join(out_dir, "per_task_ranking.csv"), header, rows, notes)
    return rows


def print_summary(configs, baselines=None):
    if not configs:
        return
    width = max(len(c["config"]) for c in configs)
    width = max(width, len("BASELINE best classical"))
    print(f"\n  {'model':<{width}} " + " ".join(f"{TASK_SHORT[t]:>9}" for t in TASKS))
    print("  " + "-" * (width + 10 * len(TASKS)))
    for cfg in configs:
        cells = []
        for task in TASKS:
            v = cfg["per_task"][task]["mean"]
            role = role_of(cfg, task)
            mark = {"zeroshot": "*", "test": "*", "val": "~", "unseen": "?"}.get(role, " ")
            cells.append(f"{v:>8.4f}{mark}")
        print(f"  {cfg['config']:<{width}} " + " ".join(cells))

    if baselines is not None:
        print("  " + "-" * (width + 10 * len(TASKS)))
        cells = [f"{best_classical_for(baselines, t)[0]:>8.4f} " for t in TASKS]
        print(f"  {'BASELINE best classical':<{width}} " + " ".join(cells))

    print("\n  * zero-shot (no gradients, no selection)   ~ legacy val   ? never trained (original)")

    if baselines is not None:
        failed = [
            TASK_SHORT[t]
            for t in TASKS
            if min(c["per_task"][t]["mean"] for c in configs)
            > best_classical_for(baselines, t)[0] * 0.95
        ]
        if failed:
            print(
                f"\n  WARNING: on {', '.join(failed)} no model beat the best classical "
                f"baseline by >5%.\n"
                "  On those tasks the numbers reflect the operator, not anything learned,\n"
                "  and similarity between models is not evidence of transfer."
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_root", default=os.path.join(REPO, "runs", "METRLA"))
    p.add_argument("--out", default=None)
    p.add_argument(
        "--include_legacy",
        action="store_true",
        help="also load protocol-1 runs (v1 val-task selection) under their old roles",
    )
    args = p.parse_args()

    runs_root = os.path.abspath(args.runs_root)
    out_dir = args.out or os.path.join(REPO, "results", os.path.basename(runs_root))

    runs = load_runs(runs_root, include_legacy=args.include_legacy)
    if not runs:
        print(f"No usable runs under {runs_root} (looking for metrics.json).")
        return 1

    configs = aggregate(runs)
    baselines = load_baselines(out_dir)
    n_runs = sum(c["n_seeds"] for c in configs)
    print(f"found {n_runs} run(s) in {len(configs)} configuration(s) under {runs_root}\n")
    main_grid(configs, out_dir, baselines)
    transfer_gap(configs, out_dir)
    original_zeroshot(configs, out_dir)
    per_task_ranking(configs, out_dir)
    prior_value(configs, baselines, out_dir)
    print_summary(configs, baselines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
