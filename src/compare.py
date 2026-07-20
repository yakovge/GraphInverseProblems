"""Assemble the comparison tables from every finished run.

Writes four CSVs into ``results/<dataset>/``:

``comparison.csv``
    The main grid: every model x every task, nMSE on the test snapshots, with each cell
    tagged by the role that task played for that model (train / val / test / unseen).

``transfer_gap.csv``
    Per model, its held-out test task against the mean of its training tasks. This is
    the number that says what the shared prior actually bought -- a small gap means the
    model generalised across operators, a large one means it just memorised the three it
    saw.

``original_zeroshot.csv``
    The paper's single-task model on the five tasks it never trained on, which is the
    baseline that makes the foundation models interpretable.

``per_task_ranking.csv``
    For each task, which model does best on it, and whether that model had ever trained
    on that task.

    python compare.py --runs_root ../runs/METRLA
"""

import argparse
import csv
import json
import os

from tasks import TASK_SHORT, TASKS

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# The article has no validation split and selects on the test set; exp_original
# reproduces that faithfully so its number stays comparable to the published table.
# Prepended to every table so no file can be read without knowing what its numbers mean.
METRIC_NOTE = [
    "METRIC: nMSE (normalised mean squared error) -- a REGRESSION ERROR, not an accuracy.",
    "        nMSE = MSE(x_pred, x_true) / MSE(0, x_true)      (paper Appendix E.1)",
    "        LOWER IS BETTER. 0.0 = perfect reconstruction.",
    "        1.0 = no better than predicting all zeros, i.e. the model learned nothing.",
    "        > 1.0 = worse than predicting zeros.",
    "        x = the node states being recovered; every task here is regression, so there",
    "        is no accuracy figure anywhere in these tables.",
    "",
]

# Flagged wherever it appears so it is never read as a clean held-out figure.
FOOTNOTES = METRIC_NOTE + [
    "role: train = gradients; val = early stopping only; test = neither, the clean number;",
    "      unseen = the original single-task model, which trained on none of these five tasks.",
    "Following the article, there is no snapshot-level validation split: val and test",
    "metrics are computed on the same held-out snapshots. The separation between them is",
    "at the TASK level -- a test task contributed no gradients and no model selection.",
    "ORIGINAL reproduces the paper's protocol including its selection on the test set,",
    "so its own-task number is comparable to Table 5 but is not a clean holdout.",
]


def load_baselines(out_dir):
    """Trivial-predictor reference points, if baselines.py has been run."""
    path = os.path.join(out_dir, "baselines.json")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def baseline_rows(baselines, header_tasks):
    """Least-squares and adjoint floors, formatted as pseudo-model rows.

    These belong in the main grid, not a separate file. Without them a reader sees five
    models scoring alike and concludes the prior transferred; the pinv row is what
    distinguishes that from every model having collapsed onto the same
    prior-free solution.
    """
    rows = []
    for key, label in (("pinv", "BASELINE least-squares (no prior)"),
                       ("adjoint", "BASELINE adjoint (no prior)"),
                       ("mean", "BASELINE predict train mean")):
        row = [label, "baseline", "-", "-", "-"]
        for task in TASKS:
            row += [round(baselines["per_task"][task][key], 6), "baseline"]
        row += ["", "", "", "", ""]
        rows.append(row)
    return rows


def load_runs(runs_root):
    """Every run that has finished, sorted with the paper baseline last."""
    runs = []
    if not os.path.isdir(runs_root):
        return runs
    for name in sorted(os.listdir(runs_root)):
        path = os.path.join(runs_root, name, "metrics.json")
        if not os.path.isfile(path):
            continue
        with open(path) as fh:
            runs.append(json.load(fh))
    runs.sort(key=lambda r: (r.get("kind") == "original", r["run_name"]))
    return runs


def role_of(run, task):
    if run.get("kind") == "original":
        return "unseen"
    if task in run.get("train_tasks", []):
        return "train"
    if task == run.get("val_task"):
        return "val"
    if task == run.get("test_task"):
        return "test"
    return "unseen"


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


def prior_value(runs, baselines, out_dir):
    """Does the learned prior beat the prior-free least-squares solve?

    The headline sanity check. Every model in this framework is built on a CGLS data-fit
    step that already solves the inverse problem without any learned component; the GNN
    only supplies regularization on top. If a model does not beat `pinv`, its score is
    telling us about the operator, not about anything it learned -- and a table of such
    scores would show apparent "transfer" that is really just all models collapsing onto
    the same prior-free solution.
    """
    if baselines is None:
        print("  (no baselines.json; run baselines.py to enable prior_value.csv)")
        return []

    header = ["task", "pinv_no_prior", "best_model_nmse", "best_model",
              "improvement", "improvement_pct", "prior_helped"]
    rows = []
    for task in TASKS:
        pinv = baselines["per_task"][task]["pinv"]
        best = min(runs, key=lambda r: r["per_task"][task]["nmse"])
        best_v = best["per_task"][task]["nmse"]
        gain = pinv - best_v
        rows.append([
            TASK_SHORT[task],
            round(pinv, 6),
            round(best_v, 6),
            best["run_name"],
            round(gain, 6),
            round(100.0 * gain / pinv, 2) if pinv > 0 else "",
            "yes" if gain > 0.05 * pinv else "marginal" if gain > 0 else "NO",
        ])

    notes = METRIC_NOTE + [
        "pinv_no_prior     unregularised least squares via CGLS: the data-fit step every",
        "                  model is built on, with no learned regularizer at all.",
        "best_model_nmse   the lowest nMSE any model achieved on that task.",
        "improvement       pinv_no_prior - best_model_nmse (positive = the prior helped).",
        "improvement_pct   the same, as a percentage of pinv_no_prior.",
        "",
        "prior_helped: 'yes' = the model beat pinv by more than 5%; 'marginal' = beat it",
        "       by less; 'NO' = did not beat it, meaning the learned prior added nothing",
        "       and that task's numbers reflect the operator rather than the model.",
        "If most tasks read NO, cross-model comparisons on this dataset are not",
        "meaningful -- the models have collapsed onto the same prior-free solution.",
    ]
    write_csv(os.path.join(out_dir, "prior_value.csv"), header, rows, notes)
    return rows


def main_grid(runs, out_dir, baselines=None):
    header = ["model", "kind", "train_tasks", "val_task", "test_task"]
    for task in TASKS:
        header += [f"{TASK_SHORT[task]}_nmse", f"{TASK_SHORT[task]}_role"]
    header += ["best_epoch", "epochs_run", "stopped_early", "parameters", "wall_seconds"]

    rows = []
    for run in runs:
        row = [
            run["run_name"],
            run.get("kind", "foundation"),
            "|".join(TASK_SHORT.get(t, t) for t in run.get("train_tasks", [])),
            TASK_SHORT.get(run.get("val_task"), "-"),
            TASK_SHORT.get(run.get("test_task"), "-"),
        ]
        for task in TASKS:
            row += [round(run["per_task"][task]["nmse"], 6), role_of(run, task)]
        row += [
            run.get("best_epoch", ""),
            run.get("epochs_run", ""),
            run.get("stopped_early", ""),
            run.get("parameters", ""),
            run.get("wall_seconds", ""),
        ]
        rows.append(row)

    if baselines is not None:
        rows += baseline_rows(baselines, TASKS)

    write_csv(os.path.join(out_dir, "comparison.csv"), header, rows, FOOTNOTES)
    return rows


def transfer_gap(runs, out_dir, baselines=None):
    """How much worse is a model on the operator it never saw?

    Each row is one foundation model, judged on its own held-out test task, against
    three reference points on that *same* task: its own training tasks, the paper's
    single-task model, and the prior-free least-squares solve.
    """
    original = next((r for r in runs if r.get("kind") == "original"), None)

    header = [
        "model",
        "test_task",
        "test_nmse",
        "mean_train_nmse",
        "absolute_gap",
        "ratio",
        "original_nmse_same_task",
        "pinv_nmse_same_task",
        "beats_original",
        "val_task",
        "val_nmse",
    ]
    rows = []
    for run in runs:
        if run.get("kind") == "original":
            continue
        per = run["per_task"]
        test_task, val_task = run["test_task"], run["val_task"]
        train_vals = [per[t]["nmse"] for t in run["train_tasks"]]
        mean_train = sum(train_vals) / len(train_vals)
        test_nmse = per[test_task]["nmse"]

        # The original model's score on this row's test task -- a like-for-like
        # comparison on the one task this model was never trained on.
        orig_same = original["per_task"][test_task]["nmse"] if original else None
        pinv_same = baselines["per_task"][test_task]["pinv"] if baselines else None

        rows.append(
            [
                run["run_name"],
                TASK_SHORT[test_task],
                round(test_nmse, 6),
                round(mean_train, 6),
                round(test_nmse - mean_train, 6),
                round(test_nmse / mean_train, 4) if mean_train > 0 else "",
                round(orig_same, 6) if orig_same is not None else "",
                round(pinv_same, 6) if pinv_same is not None else "",
                ("yes" if test_nmse < orig_same else "no") if orig_same is not None else "",
                TASK_SHORT[val_task],
                round(per[val_task]["nmse"], 6),
            ]
        )

    notes = METRIC_NOTE + [
        "One row per foundation model, scored on the single task it never trained on.",
        "",
        "test_nmse              this model on its held-out test task.",
        "mean_train_nmse        same model, averaged over its three training tasks.",
        "absolute_gap           test_nmse - mean_train_nmse.",
        "ratio                  test_nmse / mean_train_nmse. Near 1 = the prior carried",
        "                       over to an unseen operator; large = it fitted only the",
        "                       three operators it saw.",
        "original_nmse_same_task  the paper's single-task model on that SAME test task.",
        "                       It trained on none of the five, so this is the like-for-",
        "                       like control: both models are unfamiliar with this task.",
        "pinv_nmse_same_task    prior-free least squares on that same task -- the floor.",
        "                       Any model not below this has contributed nothing.",
        "beats_original         does test_nmse beat original_nmse_same_task?",
    ]
    write_csv(os.path.join(out_dir, "transfer_gap.csv"), header, rows, notes)
    return rows


def original_zeroshot(runs, out_dir):
    """The paper's single-task model against the five study tasks."""
    original = next((r for r in runs if r.get("kind") == "original"), None)
    if original is None:
        print("  (no ORIGINAL run yet; skipping original_zeroshot.csv)")
        return []

    foundation = [r for r in runs if r.get("kind") != "original"]
    header = ["task", "original_nmse", "best_foundation_nmse", "best_foundation_model", "improvement"]
    rows = []
    for task in TASKS:
        orig = original["per_task"][task]["nmse"]
        if foundation:
            best = min(foundation, key=lambda r: r["per_task"][task]["nmse"])
            best_v = best["per_task"][task]["nmse"]
            rows.append(
                [
                    TASK_SHORT[task],
                    round(orig, 6),
                    round(best_v, 6),
                    best["run_name"],
                    round(orig - best_v, 6),
                ]
            )
        else:
            rows.append([TASK_SHORT[task], round(orig, 6), "", "", ""])

    reference = original.get("paper_reference_nmse")
    own = original.get("own_task_nmse", float("nan"))
    provenance = (
        f"paper Table 5 reports {reference}"
        if reference is not None
        else "the paper reports no value for 'path' on this dataset"
    )
    notes = METRIC_NOTE + [
        f"ORIGINAL trained only on the paper's 'path' task (own-task nMSE {own:.4f}; {provenance}).",
        "It saw none of these five tasks, so every column here is zero-shot for it.",
        "",
        "original_nmse          the paper's single-task model on this task.",
        "best_foundation_nmse   the best of the five multi-task models on this task.",
        "improvement            original_nmse - best_foundation_nmse.",
        "                       POSITIVE favours the foundation models (they scored lower",
        "                       error); negative means the single-task model did better.",
    ]
    write_csv(os.path.join(out_dir, "original_zeroshot.csv"), header, rows, notes)
    return rows


def per_task_ranking(runs, out_dir):
    """Who wins each task, and had they trained on it?"""
    header = ["task", "rank", "model", "nmse", "role"]
    rows = []
    for task in TASKS:
        ordered = sorted(runs, key=lambda r: r["per_task"][task]["nmse"])
        for rank, run in enumerate(ordered, start=1):
            rows.append(
                [
                    TASK_SHORT[task],
                    rank,
                    run["run_name"],
                    round(run["per_task"][task]["nmse"], 6),
                    role_of(run, task),
                ]
            )

    notes = METRIC_NOTE + [
        "Models ranked per task by nMSE, best (lowest error) first.",
        "",
        "role   train  = this model trained on that task",
        "       val    = used only for early stopping",
        "       test   = held out entirely (no gradients, no model selection)",
        "       unseen = the original model, which trained on none of the five",
        "",
        "If a model with role=test or role=unseen ranks near the top for a task, the",
        "prior generalised to an operator it never trained on -- the result this study",
        "is looking for.",
    ]
    write_csv(os.path.join(out_dir, "per_task_ranking.csv"), header, rows, notes)
    return rows


def print_summary(runs, baselines=None):
    if not runs:
        return
    width = max(len(r["run_name"]) for r in runs)
    width = max(width, len("BASELINE least-squares (no prior)"))
    print(f"\n  {'model':<{width}} " + " ".join(f"{TASK_SHORT[t]:>9}" for t in TASKS))
    print("  " + "-" * (width + 10 * len(TASKS)))
    for run in runs:
        cells = []
        for task in TASKS:
            v = run["per_task"][task]["nmse"]
            role = role_of(run, task)
            mark = {"test": "*", "val": "~", "unseen": "?"}.get(role, " ")
            cells.append(f"{v:>8.4f}{mark}")
        print(f"  {run['run_name']:<{width}} " + " ".join(cells))

    if baselines is not None:
        print("  " + "-" * (width + 10 * len(TASKS)))
        cells = [f"{baselines['per_task'][t]['pinv']:>8.4f} " for t in TASKS]
        print(f"  {'BASELINE least-squares (no prior)':<{width}} " + " ".join(cells))

    print("\n  * held-out test task   ~ validation task   ? never trained (original)")

    if baselines is not None:
        failed = [
            TASK_SHORT[t]
            for t in TASKS
            if min(r["per_task"][t]["nmse"] for r in runs)
            > baselines["per_task"][t]["pinv"] * 0.95
        ]
        if failed:
            print(
                f"\n  WARNING: on {', '.join(failed)} no model beat prior-free least "
                f"squares by >5%.\n"
                "  On those tasks the numbers reflect the operator, not anything learned,\n"
                "  and similarity between models is not evidence of transfer."
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_root", default=os.path.join(REPO, "runs", "METRLA"))
    p.add_argument("--out", default=None)
    args = p.parse_args()

    runs_root = os.path.abspath(args.runs_root)
    out_dir = args.out or os.path.join(REPO, "results", os.path.basename(runs_root))

    runs = load_runs(runs_root)
    if not runs:
        print(f"No finished runs under {runs_root} (looking for metrics.json).")
        return 1

    baselines = load_baselines(out_dir)
    print(f"found {len(runs)} finished run(s) in {runs_root}\n")
    main_grid(runs, out_dir, baselines)
    transfer_gap(runs, out_dir, baselines)
    original_zeroshot(runs, out_dir)
    per_task_ranking(runs, out_dir)
    prior_value(runs, baselines, out_dir)
    print_summary(runs, baselines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
