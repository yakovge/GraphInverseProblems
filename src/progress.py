"""Progress reporting for long training runs.

Each run owns ``runs/<name>/progress.json`` and rewrites it every epoch. Writes are
atomic (temp file + ``os.replace``) so a reader polling the file never sees a half-written
document, no matter when it looks.

Use ``watch_progress.py`` to render every run's file as a live table.
"""

import json
import os
import time


class ProgressReporter:
    """Tracks epoch progress for one run and mirrors it to disk."""

    def __init__(self, run_dir, run_name, total_epochs, extra=None):
        self.run_dir = run_dir
        self.run_name = run_name
        self.total_epochs = total_epochs
        self.extra = extra or {}
        self.start_time = time.time()
        self.epoch = 0
        self.status = "starting"
        self._epoch_times = []
        os.makedirs(run_dir, exist_ok=True)
        self.write()

    @property
    def path(self):
        return os.path.join(self.run_dir, "progress.json")

    def update(self, epoch=None, status=None, **fields):
        if epoch is not None:
            if epoch > self.epoch:
                self._epoch_times.append(time.time())
            self.epoch = epoch
        if status is not None:
            self.status = status
        self.extra.update(fields)
        self.write()

    def _eta_seconds(self):
        """Estimate remaining time from a trailing window of recent epochs.

        A trailing window rather than the overall average, because the first epochs
        include CUDA warm-up and operator construction and would bias the estimate.
        """
        if len(self._epoch_times) < 2:
            return None
        window = self._epoch_times[-10:]
        per_epoch = (window[-1] - window[0]) / max(len(window) - 1, 1)
        return max(self.total_epochs - self.epoch, 0) * per_epoch

    def write(self):
        eta = self._eta_seconds()
        doc = {
            "run_name": self.run_name,
            "status": self.status,
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "percent": round(100.0 * self.epoch / max(self.total_epochs, 1), 1),
            "elapsed_seconds": round(time.time() - self.start_time, 1),
            "eta_seconds": round(eta, 1) if eta is not None else None,
            "updated_at": time.time(),
            **self.extra,
        }
        # Atomic replace, so a concurrent reader never sees a partial file.
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(doc, fh, indent=2)
        os.replace(tmp, self.path)

    def finish(self, status="completed", **fields):
        # Deliberately does NOT snap epoch to total_epochs. An early-stopped run that
        # was forced to 250/250 would be indistinguishable from one that used its whole
        # budget, which matters here: whether a run early-stopped is a real result.
        self.update(status=status, stopped_early=self.epoch < self.total_epochs, **fields)


def format_duration(seconds):
    if seconds is None:
        return "--"
    seconds = int(seconds)
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def read_all(runs_root):
    """Every run's latest progress document, newest-first by update time."""
    if not os.path.isdir(runs_root):
        return []
    docs = []
    for name in sorted(os.listdir(runs_root)):
        path = os.path.join(runs_root, name, "progress.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as fh:
                docs.append(json.load(fh))
        except (json.JSONDecodeError, OSError):
            # A run mid-write or a partially created directory; skip this poll.
            continue
    return docs
