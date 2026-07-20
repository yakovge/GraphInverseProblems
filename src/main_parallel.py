"""Train all six models, two at a time, then build the comparison tables.

Keeps exactly two training processes alive; as soon as one finishes the next queued run
starts. Two is what fits: each METR-LA run peaks at ~1.3 GB with the configured
micro-batch, so a pair uses ~2.7 GB of the 6 GB card.

Resumable -- any run that already has ``metrics.json`` is skipped, so an interrupted
sweep can simply be restarted.

    python main_parallel.py --dataset METRLA
    python main_parallel.py --dataset METRLA --jobs 1     # serial
    python main_parallel.py --dataset METRLA --force      # retrain everything
"""

import argparse
import os
import subprocess
import sys
import time

from progress import format_duration
from tasks import experiment_split, run_name

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FOUNDATION = os.path.join(HERE, "foundation")


def build_jobs(seeds):
    """Six runs per seed: five foundation models plus the paper reproduction.

    Seed-major order, so an interrupted multi-seed sweep leaves complete seeds behind
    rather than six half-finished configurations.
    """
    jobs = []
    for seed in seeds:
        for i in range(5):
            train_tasks, val_task, _ = experiment_split(i)
            jobs.append(
                {
                    "script": f"exp{i + 1}.py",
                    "name": run_name(train_tasks, val_task, seed=seed),
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


def launch(job, args, runs_root):
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

    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=FOUNDATION)
    return {"job": job, "proc": proc, "log": log, "started": time.time()}


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
    args = p.parse_args()

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
