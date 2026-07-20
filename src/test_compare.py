"""Fixture tests for compare.py: protocol filtering, seed aggregation, table schemas.

Run directly::

    python test_compare.py

compare.py and plots.py are coupled by hand-maintained column names, and the seed
aggregation / long transfer format can break silently. The full pipeline check needs a
dataset download and a training run; this instead fabricates a runs directory with
synthetic metrics.json files (one config x 2 seeds, one ORIGINAL, one legacy protocol-1
run) and asserts the tables that come out.
"""

import csv
import json
import os
import sys
import tempfile

import compare
from tasks import TASK_SHORT, TASKS

FM_NAME = "FM_train-denois-sensor-pde_val-inpaint"
FM_TRAIN = ["denoising", "sensor_recovery", "pde_state"]
FM_VAL, FM_TEST = "inpainting", "source_localization"

LEGACY_NAME = "FM_train-denois-inpaint-pde_val-source"


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    return ok


def write_metrics(root, name, metrics):
    run_dir = os.path.join(root, name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh)


def fm_metrics(seed, nudge):
    base = {
        "denoising": 0.20,
        "inpainting": 0.40,
        "source_localization": 0.35,
        "sensor_recovery": 0.45,
        "pde_state": 0.28,
    }
    return {
        "run_name": f"{FM_NAME}_s{seed}",
        "kind": "foundation",
        "protocol": 2,
        "selection": "train",
        "train_tasks": FM_TRAIN,
        "val_task": FM_VAL,
        "test_task": FM_TEST,
        "zeroshot_tasks": [FM_VAL, FM_TEST],
        "seed": seed,
        "best_epoch": 40 + seed,
        "epochs_run": 60,
        "stopped_early": True,
        "parameters": 28961,
        "wall_seconds": 100.0,
        "per_task": {t: {"nmse": v + nudge, "data_fit": 0.0} for t, v in base.items()},
    }


def original_metrics(seed):
    return {
        "run_name": f"ORIGINAL_paper_path_pl32_s{seed}",
        "kind": "original",
        "protocol": 2,
        "selection": "test",
        "train_tasks": ["path"],
        "val_task": None,
        "test_task": None,
        "seed": seed,
        "own_task_nmse": 0.005,
        "paper_reference_nmse": 0.004,
        "parameters": 28961,
        "wall_seconds": 100.0,
        "per_task": {t: {"nmse": 0.50, "data_fit": 0.0} for t in TASKS},
    }


def legacy_metrics():
    # v1 run: no protocol field, no zeroshot_tasks, no seed suffix in the name.
    return {
        "run_name": LEGACY_NAME,
        "kind": "foundation",
        "train_tasks": ["denoising", "inpainting", "pde_state"],
        "val_task": "source_localization",
        "test_task": "sensor_recovery",
        "seed": 0,
        "parameters": 28961,
        "per_task": {t: {"nmse": 0.90, "data_fit": 0.0} for t in TASKS},
    }


def make_baselines():
    # tikhonov is the best classical on source_localization (so the verdict must NOT be
    # driven by pinv), laplacian on inpainting; pinv everywhere else.
    per_task = {
        t: {"pinv": 0.50, "adjoint": 0.90, "mean": 1.0, "tikhonov": 0.60, "laplacian": 0.65}
        for t in TASKS
    }
    per_task["source_localization"].update({"pinv": 0.38, "tikhonov": 0.30})
    per_task["inpainting"].update({"pinv": 0.90, "tikhonov": 0.85, "laplacian": 0.80})
    return {"dataset": "FAKE", "per_task": per_task}


def read_table(out_dir, filename):
    with open(os.path.join(out_dir, filename), newline="") as fh:
        rows = list(csv.DictReader(fh))
    key = rows[0] and next(iter(rows[0]))
    return [r for r in rows if r.get(key) and not str(r[key]).startswith("#")]


def main():
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        runs_root = os.path.join(tmp, "runs")
        out_dir = os.path.join(tmp, "results")
        os.makedirs(out_dir)

        write_metrics(runs_root, f"{FM_NAME}_s0", fm_metrics(0, 0.00))
        write_metrics(runs_root, f"{FM_NAME}_s1", fm_metrics(1, 0.02))
        write_metrics(runs_root, "ORIGINAL_paper_path_pl32_s0", original_metrics(0))
        write_metrics(runs_root, LEGACY_NAME, legacy_metrics())
        baselines = make_baselines()

        print("\nProtocol filtering")
        runs = compare.load_runs(runs_root)
        ok &= check("legacy run excluded by default", len(runs) == 3)
        runs_all = compare.load_runs(runs_root, include_legacy=True)
        ok &= check("legacy run admitted with --include_legacy", len(runs_all) == 4)

        print("\nSeed aggregation")
        configs = compare.aggregate(runs)
        fm = next(c for c in configs if c["config"] == FM_NAME)
        orig = next(c for c in configs if c["kind"] == "original")
        ok &= check("2 configs from 3 runs", len(configs) == 2)
        ok &= check("fm groups 2 seeds", fm["n_seeds"] == 2 and fm["seeds"] == [0, 1])
        std = fm["per_task"]["denoising"]["std"]
        ok &= check("population std across seeds", abs(std - 0.01) < 1e-9, f"std={std}")
        mean = fm["per_task"]["denoising"]["mean"]
        ok &= check("mean across seeds", abs(mean - 0.21) < 1e-9, f"mean={mean}")
        ok &= check("original keeps kind through aggregation", orig["n_seeds"] == 1)

        print("\nRoles")
        ok &= check(
            "both held-out tasks are zeroshot",
            compare.role_of(fm, FM_VAL) == "zeroshot"
            and compare.role_of(fm, FM_TEST) == "zeroshot",
        )
        ok &= check("original tasks are unseen", compare.role_of(orig, "denoising") == "unseen")
        legacy_cfg = next(
            c for c in compare.aggregate(runs_all) if c["config"] == LEGACY_NAME
        )
        ok &= check(
            "legacy run keeps old val/test roles",
            compare.role_of(legacy_cfg, "source_localization") == "val"
            and compare.role_of(legacy_cfg, "sensor_recovery") == "test",
        )

        print("\ncomparison.csv schema")
        compare.main_grid(configs, out_dir, baselines)
        rows = read_table(out_dir, "comparison.csv")
        task_cols = [k[: -len("_nmse")] for k in rows[0] if k.endswith("_nmse")]
        ok &= check(
            "plots.py column detection sees exactly the 5 task means",
            task_cols == [TASK_SHORT[t] for t in TASKS],
            str(task_cols),
        )
        fm_row = next(r for r in rows if r["model"] == FM_NAME)
        ok &= check("n_seeds column", fm_row["n_seeds"] == "2")
        ok &= check("std column filled", fm_row["denois_nmse_std"] != "")
        ok &= check(
            "5 baseline pseudo-rows",
            sum(r["model"].startswith("BASELINE") for r in rows) == 5,
        )

        print("\ntransfer_gap.csv long format")
        compare.transfer_gap(configs, out_dir)
        rows = read_table(out_dir, "transfer_gap.csv")
        ok &= check("2 rows: one per zero-shot task", len(rows) == 2)
        ok &= check(
            "no rows for ORIGINAL",
            all(not r["model"].startswith("ORIGINAL") for r in rows),
        )
        ok &= check(
            "both zero-shot tasks present",
            sorted(r["zeroshot_task"] for r in rows)
            == sorted([TASK_SHORT[FM_VAL], TASK_SHORT[FM_TEST]]),
        )

        print("\nprior_value.csv best-classical verdict")
        compare.prior_value(configs, baselines, out_dir)
        rows = {r["task"]: r for r in read_table(out_dir, "prior_value.csv")}
        src = rows[TASK_SHORT["source_localization"]]
        ok &= check(
            "best classical is tikhonov, not pinv",
            src["best_classical_name"] == "tikhonov" and float(src["best_classical"]) == 0.30,
            f"{src['best_classical_name']}={src['best_classical']}",
        )
        # Best zero-shot on source is the fm config (0.36 mean) vs classical 0.30 -> NO.
        ok &= check("source verdict NO vs oracle tikhonov", src["prior_helped"] == "NO")
        inp = rows[TASK_SHORT["inpainting"]]
        # Best zero-shot on inpainting is 0.41 vs best classical (laplacian) 0.80 -> yes.
        ok &= check(
            "inpainting verdict yes vs oracle laplacian",
            inp["prior_helped"] == "yes" and inp["best_classical_name"] == "laplacian",
        )
        ok &= check(
            "verdict driven by a zero-shot model",
            src["best_zeroshot_model"] in (FM_NAME, "ORIGINAL_paper_path_pl32"),
        )

        print("\noriginal_zeroshot.csv and ranking")
        compare.original_zeroshot(configs, out_dir)
        rows = read_table(out_dir, "original_zeroshot.csv")
        ok &= check("one row per task", len(rows) == len(TASKS))
        compare.per_task_ranking(configs, out_dir)
        rows = read_table(out_dir, "per_task_ranking.csv")
        ok &= check("ranking rows = configs x tasks", len(rows) == 2 * len(TASKS))

    print(f"\n{'all groups passed' if ok else 'FAILURES above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
