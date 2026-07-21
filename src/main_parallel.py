"""Train the five models (four v3 foundation configs + the paper reproduction), two at
a time, then build the comparison tables.

Keeps exactly two training processes alive; as soon as one finishes the next queued run
starts. Two is what fits: each METR-LA run peaks at ~1.3 GB with the configured
micro-batch, so a pair uses ~2.7 GB of the 6 GB card.

Resumable -- any run that already has ``metrics.json`` is skipped, so an interrupted
sweep can simply be restarted.

    python main_parallel.py --dataset METRLA --preflight  # optimizer A/B first (~2h)
    python main_parallel.py --dataset METRLA --epochs 150 [winner flags]
    python main_parallel.py --dataset METRLA --jobs 1     # serial
    python main_parallel.py --dataset METRLA --force      # retrain everything
"""

import argparse
import json
import os
import subprocess
import sys
import time

from progress import format_duration
from tasks import experiment_split_v3, run_name_v3

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FOUNDATION = os.path.join(HERE, "foundation")

# Flags forwarded verbatim to the experiment subprocesses when set. The optimizer group
# is withheld from exp_original.py: the reproduction's value is fidelity to the
# published protocol, so it must keep the published-era optimizer even when the sweep
# runs with the preflight winner's settings.
FORWARDED_FLAGS = (
    "adam_eps", "lr_schedule", "warmup_epochs", "lr_schedule_epochs",
    "eval_seed", "max_patience", "min_epochs",
)
OPTIMIZER_FLAGS = ("adam_eps", "lr_schedule", "warmup_epochs", "lr_schedule_epochs")


def build_jobs(seeds):
    """Five runs per seed: four v3 foundation configs plus the paper reproduction.

    Seed-major order, so an interrupted multi-seed sweep leaves complete seeds behind
    rather than five half-finished configurations.
    """
    jobs = []
    for seed in seeds:
        for i in range(4):
            train_tasks, holdout, _ = experiment_split_v3(i)
            jobs.append(
                {
                    "script": f"exp{i + 1}.py",
                    "name": run_name_v3(train_tasks, holdout, seed=seed),
                    "seed": seed,
                }
            )
        jobs.append(
            {
                "script": "exp_original.py",
                "name": f"ORIGINAL_paper_path_pl32_s{seed}",
                "seed": seed,
            }
        )
    return jobs


def is_done(job, runs_root):
    return os.path.isfile(os.path.join(runs_root, job["name"], "metrics.json"))


def launch(job, args, runs_root, extra_flags=None):
    log_path = os.path.join(runs_root, job["name"], "train.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(FOUNDATION, job["script"]),
        "--dataset", args.dataset,
        "--datapath", args.datapath,
        "--runs_root", runs_root,
        "--seed", str(job["seed"]),
        "--device", args.device,
    ]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.train_frac != 1.0:
        cmd += ["--train_frac", str(args.train_frac)]
    if args.test_frac != 1.0:
        cmd += ["--test_frac", str(args.test_frac)]

    is_original = job["script"] == "exp_original.py"
    for flag in FORWARDED_FLAGS:
        value = getattr(args, flag, None)
        if value is None or (is_original and flag in OPTIMIZER_FLAGS):
            continue
        cmd += [f"--{flag}", str(value)]
    for flag, value in (extra_flags or {}).items():
        cmd += [f"--{flag}", str(value)]

    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=FOUNDATION)
    return {"job": job, "proc": proc, "log": log, "started": time.time()}


# The optimizer A/B probed before a sweep: arm1 is the status quo (Adam eps 1e-3, no
# schedule), arm2 the alternative the canyon-landscape investigation motivated. Arm2's
# --lr_schedule_epochs 150 makes the short probe run the real sweep's early-lr profile
# rather than an artificially compressed decay.
PREFLIGHT_ARMS = [
    ("arm1_eps1e-3", {}),
    (
        "arm2_eps1e-8_cosine",
        {"adam_eps": 1e-8, "lr_schedule": "cosine", "warmup_epochs": 5,
         "lr_schedule_epochs": 150},
    ),
]


def preflight_verdict(hist1, hist2):
    """Fractional selection-metric drop per arm. Pure, so it is unit-testable.

    Arm2 must beat arm1's drop by more than 2 percentage points; ties go to arm1 (the
    status quo). Arm2 spends 5 probe epochs in warmup, so the margin deliberately
    favours arm1 -- a marginal arm2 loses, which is the conservative outcome.
    """

    def descent(history):
        xs = [h["selection_metric"] for h in history]
        if len(xs) < 8:
            raise SystemExit("preflight history too short; did the probe fail?")
        first3 = sum(xs[:3]) / 3
        last5 = sum(xs[-5:]) / 5
        return (first3 - last5) / first3

    d1, d2 = descent(hist1), descent(hist2)
    return d1, d2, ("arm2" if d2 > d1 + 0.02 else "arm1")


def run_preflight(args):
    """Run the two optimizer arms on the highest-noise config, print the winner.

    The probe config is exp4 (holdout pde): it trains both per-batch mask-resampling
    tasks -- the regime the optimizer change targets -- and it is the v1 long-run
    winner, so its curve shape is the best-characterized reference. Probes live under
    runs/<dataset>_preflight/<arm>/ so compare.py can never ingest them.
    """
    train_tasks, holdout, _ = experiment_split_v3(3)
    name = run_name_v3(train_tasks, holdout, seed=0)
    base_root = os.path.abspath(os.path.join(args.runs_root, args.dataset + "_preflight"))
    pe = args.preflight_epochs
    common = {"epochs": pe, "max_patience": 999, "min_epochs": pe, "eval_seed": 0}

    print(f"preflight probe : {name}  ({pe} epochs per arm)")
    print(f"probe root      : {base_root}\n", flush=True)

    pending = []
    for arm, flags in PREFLIGHT_ARMS:
        root = os.path.join(base_root, arm)
        if os.path.isfile(os.path.join(root, name, "metrics.json")) and not args.force:
            print(f"  skip  {arm}  (already has metrics.json)")
            continue
        pending.append((arm, root, {**common, **flags}))

    running = []
    for arm, root, flags in pending:
        entry = launch({"script": "exp4.py", "name": name, "seed": 0}, args, root, flags)
        entry["arm"] = arm
        running.append(entry)
        print(f"  start {arm}", flush=True)
        if args.jobs < 2 and running:
            code = running[-1]["proc"].wait()
            running[-1]["log"].close()
            if code != 0:
                raise SystemExit(f"preflight {arm} failed (exit {code}); see its train.log")
            running.pop()
    for entry in running:
        code = entry["proc"].wait()
        entry["log"].close()
        if code != 0:
            raise SystemExit(
                f"preflight {entry['arm']} failed (exit {code}); see its train.log"
            )

    hists = []
    for arm, _ in PREFLIGHT_ARMS:
        with open(os.path.join(base_root, arm, name, "history.json")) as fh:
            hists.append(json.load(fh))
    d1, d2, winner = preflight_verdict(*hists)

    wflags = (
        "--adam_eps 1e-8 --lr_schedule cosine --warmup_epochs 5"
        if winner == "arm2"
        else ""
    )
    print(f"\npreflight verdict: arm1 drop {d1:+.1%}   arm2 drop {d2:+.1%}   ->  {winner}")
    if winner == "arm1" and d2 > d1:
        print("(arm2 was ahead but within the 2pp margin; a rerun with "
              "--preflight_epochs 20 is worth it before settling)")
    print("\nsweep command:")
    print(f"  python main_parallel.py --dataset {args.dataset} --epochs 150 {wflags}".rstrip())
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="METRLA", choices=["METRLA", "CPOX"])
    p.add_argument("--datapath", default=os.path.join(REPO, "data"))
    p.add_argument("--runs_root", default=os.path.join(REPO, "runs"))
    p.add_argument("--jobs", type=int, default=2, help="concurrent training processes")
    p.add_argument(
        "--seeds",
        default="0",
        help="comma-separated seeds; finished (config, seed) runs are skipped, so "
        "'--seeds 0,1,2' after a '--seeds 0' sweep trains only the new seeds",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--train_frac", type=float, default=1.0)
    p.add_argument("--test_frac", type=float, default=1.0)
    p.add_argument("--force", action="store_true", help="retrain runs that already finished")
    p.add_argument("--skip_report", action="store_true")

    # Forwarded to the experiment subprocesses when set (see FORWARDED_FLAGS; the
    # optimizer group is withheld from exp_original.py).
    p.add_argument("--adam_eps", type=float, default=None)
    p.add_argument("--lr_schedule", choices=["none", "cosine"], default=None)
    p.add_argument("--warmup_epochs", type=int, default=None)
    p.add_argument("--lr_schedule_epochs", type=int, default=None)
    p.add_argument("--eval_seed", type=int, default=None)
    p.add_argument("--max_patience", type=int, default=None)
    p.add_argument("--min_epochs", type=int, default=None)

    p.add_argument(
        "--preflight",
        action="store_true",
        help="run the optimizer A/B probe (2 x preflight_epochs on the hardest mix) "
        "instead of the sweep, and print the winning sweep command",
    )
    p.add_argument("--preflight_epochs", type=int, default=12)
    args = p.parse_args()

    if args.preflight:
        return run_preflight(args)

    runs_root = os.path.abspath(os.path.join(args.runs_root, args.dataset))
    os.makedirs(runs_root, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    jobs = build_jobs(seeds)
    queue = [j for j in jobs if args.force or not is_done(j, runs_root)]
    skipped = [j for j in jobs if j not in queue]

    print(f"dataset   : {args.dataset}")
    print(f"runs root : {runs_root}")
    print(f"concurrent: {args.jobs}")
    for j in skipped:
        print(f"  skip  {j['name']}  (already has metrics.json)")
    for j in queue:
        print(f"  queue {j['name']}")
    print(f"\nfollow along with:  python watch_progress.py --runs {runs_root}\n", flush=True)

    running = []
    completed, failed = [], []
    started_at = time.time()

    while queue or running:
        while queue and len(running) < args.jobs:
            job = queue.pop(0)
            running.append(launch(job, args, runs_root))
            print(f"[{format_duration(time.time() - started_at)}] start {job['name']}", flush=True)

        time.sleep(3)

        for entry in list(running):
            code = entry["proc"].poll()
            if code is None:
                continue
            running.remove(entry)
            entry["log"].close()
            elapsed = format_duration(time.time() - entry["started"])
            name = entry["job"]["name"]
            if code == 0:
                completed.append(name)
                print(f"[{format_duration(time.time() - started_at)}] done  {name} ({elapsed})", flush=True)
            else:
                failed.append(name)
                log = os.path.join(runs_root, name, "train.log")
                print(
                    f"[{format_duration(time.time() - started_at)}] FAIL  {name} "
                    f"(exit {code}, see {log})",
                    flush=True,
                )

    print(f"\n{len(completed)} completed, {len(failed)} failed, {len(skipped)} skipped")
    print(f"total wall time {format_duration(time.time() - started_at)}")
    if failed:
        print("failed runs:")
        for name in failed:
            print(f"  {name}")

    if args.skip_report:
        return 1 if failed else 0

    # Build the tables from whatever finished, so a partial sweep still reports.
    print("\nbuilding comparison tables...")
    for script in ("compare.py", "plots.py"):
        result = subprocess.run(
            [sys.executable, os.path.join(HERE, script), "--runs_root", runs_root],
            cwd=HERE,
        )
        if result.returncode != 0:
            print(f"  {script} failed")
            return 1

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
