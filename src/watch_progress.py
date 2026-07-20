"""Live progress table across all runs.

    python watch_progress.py                # refresh every 5s
    python watch_progress.py --once         # print once and exit
"""

import argparse
import os
import time

from progress import format_duration, read_all

BAR_WIDTH = 24


def bar(percent):
    filled = int(BAR_WIDTH * percent / 100.0)
    return "#" * filled + "." * (BAR_WIDTH - filled)


def render(runs_root):
    docs = read_all(runs_root)
    lines = []
    lines.append(f"GRIP foundation-model runs  --  {time.strftime('%H:%M:%S')}")
    lines.append("")

    if not docs:
        lines.append(f"  no runs found under {runs_root}")
        return "\n".join(lines)

    header = f"  {'run':<44} {'status':<11} {'progress':<26} {'epoch':>17} {'elapsed':>8} {'eta':>8}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for d in docs:
        epochs = f"{d['epoch']}/{d['total_epochs']}"
        if d.get("stopped_early"):
            epochs += " early"
        prog = f"[{bar(d['percent'])}] {d['percent']:>5.1f}%"
        lines.append(
            f"  {d['run_name'][:44]:<44} {d['status'][:11]:<11} {prog:<26} "
            f"{epochs:>17} {format_duration(d['elapsed_seconds']):>8} "
            f"{format_duration(d.get('eta_seconds')):>8}"
        )

        # Second line: whatever metrics the run is publishing.
        bits = []
        if d.get("train_loss") is not None:
            bits.append(f"train {d['train_loss']:.4f}")
        if d.get("val_nmse") is not None:
            bits.append(f"val[{d.get('val_task', '?')}] {d['val_nmse']:.4f}")
        if d.get("best_val_nmse") is not None:
            bits.append(f"best {d['best_val_nmse']:.4f}")
        if d.get("patience") is not None:
            bits.append(f"patience {d['patience']}")
        if bits:
            lines.append(f"  {'':<44} {' | '.join(bits)}")

    done = sum(d["status"] in ("completed", "failed") for d in docs)
    lines.append("")
    lines.append(f"  {done}/{len(docs)} finished")
    return "\n".join(lines)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default=os.path.join(here, "..", "runs"))
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    runs_root = os.path.abspath(args.runs)
    if args.once:
        print(render(runs_root))
        return

    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            print(render(runs_root))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
