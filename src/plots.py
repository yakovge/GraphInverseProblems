"""Render the comparison CSVs as report images.

    python plots.py --runs_root ../runs/METRLA
    python plots.py --results ../results/METRLA --dark

Produces, in ``results/<dataset>/``:

* ``table_comparison.png``       the full grid as a styled table
* ``heatmap_models_tasks.png``   same data as a heatmap, for reading magnitude at a glance
* ``transfer_gap.png``           each model's held-out task vs its training tasks
* ``original_vs_foundation.png`` the paper's single-task baseline vs the best foundation model

Colour follows the dataviz reference palette. Forms follow its job table: a grid of
magnitudes is a heatmap on a single sequential hue, and a before/after pair per item is a
dumbbell. Every mark is directly labelled -- required here because the aqua slot sits
below 3:1 contrast on the light surface, and useful anyway since these are read as
report figures.
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# Reference palette (references/palette.md). Both modes are selected, not flipped.
THEME = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "baseline": "#c3c2b7",
        "series1": "#2a78d6",
        "series2": "#1baf7a",
        "ramp": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "baseline": "#383835",
        "series1": "#3987e5",
        "series2": "#199e70",
        "ramp": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    },
}

FONT = ["Segoe UI", "DejaVu Sans", "sans-serif"]

# Role markers, so identity is never carried by colour alone.
# "baseline" rows are trivial predictors, not trained models -- they carry no role mark.
ROLE_MARK = {"train": "", "val": " ~", "test": " *", "unseen": " ?", "baseline": ""}

# Every figure states its metric, so an image shared on its own is still interpretable.
METRIC_SUBTITLE = (
    "nMSE = MSE(prediction, truth) / MSE(0, truth) -- regression error, not accuracy.  "
    "Lower is better; 1.0 = no better than predicting zeros."
)


def read_csv(path):
    """Rows as dicts, dropping the '# ...' footnote lines compare.py appends."""
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    key = next(iter(rows[0])) if rows else None
    return [r for r in rows if r.get(key) and not str(r[key]).startswith("#")]


def short_model(name):
    """Compact the long run names so they fit as row labels."""
    if name.startswith("ORIGINAL"):
        return "ORIGINAL (paper, path)"
    if name.startswith("BASELINE"):
        return name  # already short, and must stay visually distinct
    return name.replace("FM_train-", "").replace("_val-", "  val:")


def is_baseline(row):
    return str(row.get("model", "")).startswith("BASELINE")


def task_columns(rows):
    """Task short-names, in the order compare.py emitted them."""
    return [k[: -len("_nmse")] for k in rows[0] if k.endswith("_nmse")]


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=t["muted"], length=0, labelsize=9)


def fmt(v):
    return f"{v:.3f}"


# --------------------------------------------------------------------------------------


def shade(values):
    """Map nMSE onto [0, 1] for the sequential ramp, on a log scale.

    nMSE is strictly positive and here spans well over two orders of magnitude: a
    diverged masking run can sit near 70 while a healthy one is near 0.2. On a linear
    scale those few outliers consume the whole ramp and every meaningful difference
    collapses into the lightest step. Log scaling keeps the 0.2-1.0 band -- where the
    comparison actually lives -- readable, at the cost of visually compressing the
    outliers, which are only ever "broken" rather than a value to compare.
    """
    import math

    lo = min(values)
    hi = max(values)
    if hi <= 0 or hi == lo:
        return {v: 0.0 for v in values}
    lo = max(lo, hi * 1e-4)  # guard against zero / absurd dynamic range
    log_lo, log_hi = math.log10(lo), math.log10(hi)
    return {
        v: (math.log10(max(v, lo)) - log_lo) / (log_hi - log_lo) for v in set(values)
    }


def plot_table(rows, out, t):
    """The grid as a styled table -- the literal report artefact."""
    tasks = task_columns(rows)
    n_rows, n_cols = len(rows), len(tasks) + 1

    fig, ax = plt.subplots(figsize=(1.5 * n_cols + 4.6, 0.46 * n_rows + 1.7))
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    ax.axis("off")

    values = [[float(r[f"{c}_nmse"]) for c in tasks] for r in rows]
    frac_of = shade([v for row in values for v in row])
    cmap = LinearSegmentedColormap.from_list("seq", t["ramp"])

    cell_text, cell_colors, text_colors = [], [], []
    for r, vals in zip(rows, values):
        line, colors, inks = [short_model(r["model"])], [t["surface"]], [t["primary"]]
        for c, v in zip(tasks, vals):
            role = r[f"{c}_role"]
            line.append(fmt(v) + ROLE_MARK[role])
            frac = frac_of[v]
            colors.append(cmap(0.06 + 0.82 * frac))
            # Flip ink on the dark end of the ramp so the number stays legible.
            inks.append("#ffffff" if frac > 0.55 else "#0b0b0b")
        cell_text.append(line)
        cell_colors.append(colors)
        text_colors.append(inks)

    # The model names are long, so give column 0 most of the width rather than letting
    # matplotlib divide evenly and clip the labels.
    label_w = 0.34
    col_widths = [label_w] + [(1 - label_w) / len(tasks)] * len(tasks)

    table = ax.table(
        cellText=cell_text,
        colLabels=["model"] + tasks,
        cellColours=cell_colors,
        colWidths=col_widths,
        cellLoc="center",
        bbox=[0, 0, 1, 1],  # fill the axes; 'loc=center' leaves large dead margins
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(t["surface"])
        cell.set_linewidth(2)  # 2px surface gap between fills
        if row == 0:
            cell.set_facecolor(t["surface"])
            cell.set_text_props(color=t["secondary"], fontweight="bold")
        else:
            cell.set_text_props(color=text_colors[row - 1][col])
            # Baselines are reference points, not competitors: italicise them so they
            # are not read as another model in the ranking.
            if is_baseline(rows[row - 1]):
                cell.set_text_props(fontstyle="italic")
            if col == 0:
                cell.set_text_props(ha="left")
                cell._text.set_x(0.03)
            # Ring the held-out test cell: this is the number the study is about.
            # Raise it above its neighbours, whose 2px surface-coloured edges would
            # otherwise paint over the ring and leave it drawn on only two sides.
            elif cell_text[row - 1][col].endswith("*"):
                cell.set_edgecolor(t["primary"])
                cell.set_linewidth(2.2)
                cell.set_zorder(5)

    ax.set_title(
        "nMSE by model and task",
        color=t["primary"], fontsize=13, fontweight="bold", pad=30, loc="left", x=0.0,
    )
    ax.text(0.0, 1.035, METRIC_SUBTITLE, transform=ax.transAxes,
            color=t["secondary"], fontsize=9)
    fig.text(
        0.0, -0.035,
        "*  held-out test task (no gradients, no model selection)     "
        "~  validation task (early stopping only)     "
        "?  never trained on any of these five\n"
        "italic rows are trivial predictors, not models -- a model that fails to beat "
        "least-squares has learned nothing.     shading is log-scaled",
        color=t["muted"], fontsize=8.5,
    )
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=t["surface"])
    plt.close(fig)
    print(f"  wrote {out}")


def plot_heatmap(rows, out, t):
    """Same grid as a heatmap -- magnitude at a glance. Sequential, one hue."""
    tasks = task_columns(rows)
    values = [[float(r[f"{c}_nmse"]) for c in tasks] for r in rows]
    labels = [short_model(r["model"]) for r in rows]

    fig, ax = plt.subplots(figsize=(1.35 * len(tasks) + 6.0, 0.62 * len(rows) + 2.6))
    fig.patch.set_facecolor(t["surface"])
    style(ax, t)

    cmap = LinearSegmentedColormap.from_list("seq", t["ramp"])
    # Log norm for the same reason as the table: a couple of diverged runs would
    # otherwise flatten the whole ramp. See shade().
    flat = [v for row in values for v in row]
    norm = LogNorm(vmin=max(min(flat), max(flat) * 1e-4), vmax=max(flat))
    im = ax.imshow(values, cmap=cmap, aspect="auto", norm=norm)

    ax.set_xticks(range(len(tasks)), tasks, color=t["secondary"], fontsize=10)
    ax.set_yticks(range(len(rows)), labels, color=t["secondary"], fontsize=9.5)

    frac_of = shade(flat)
    for i, row in enumerate(rows):
        for j, c in enumerate(tasks):
            v = values[i][j]
            frac = frac_of[v]
            ax.text(
                j, i, fmt(v) + ROLE_MARK[row[f"{c}_role"]],
                ha="center", va="center", fontsize=9.5,
                color="#ffffff" if frac > 0.55 else "#0b0b0b",
            )
            if row[f"{c}_role"] == "test":
                # zorder above the minor grid, which is drawn in the surface colour and
                # would otherwise clip the ring to two sides.
                ax.add_patch(
                    Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                              edgecolor=t["primary"], linewidth=2.2,
                              zorder=6, clip_on=False)
                )

    # 2px surface gap between cells.
    ax.set_xticks([x - 0.5 for x in range(1, len(tasks))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(rows))], minor=True)
    ax.grid(which="minor", color=t["surface"], linewidth=2)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.015)
    cbar.set_label("nMSE, log scale", color=t["secondary"], fontsize=9, labelpad=6)
    cbar.ax.tick_params(colors=t["muted"], labelsize=8)
    cbar.outline.set_visible(False)

    ax.set_title(
        "Task performance across models",
        color=t["primary"], fontsize=13, fontweight="bold", pad=34, loc="left",
    )
    ax.text(0.0, 1.045, METRIC_SUBTITLE, transform=ax.transAxes,
            color=t["secondary"], fontsize=9)
    fig.text(
        0.005, 0.012,
        "boxed = held-out test task   * test   ~ validation   ? never trained",
        color=t["muted"], fontsize=8.5,
    )
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=t["surface"])
    plt.close(fig)
    print(f"  wrote {out}")


def dumbbell(labels, left, right, left_name, right_name, title, note, out, t,
             right_suffix=None):
    """Before/after pair per row, on a log x-axis.

    Log because nMSE is a positive ratio quantity that here spans 0.2 to ~70: on a
    linear axis a single diverged run pushes every healthy row into the left margin.

    Labels are placed outward from each marker -- the smaller value to its left, the
    larger to its right -- rather than centred above both, which collides whenever the
    two values are close (and they are close precisely when transfer worked).
    """
    y = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(11.0, 0.62 * len(labels) + 2.4))
    fig.patch.set_facecolor(t["surface"])
    style(ax, t)
    ax.set_xscale("log")
    ax.grid(axis="x", color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)

    for i, (a, b) in enumerate(zip(left, right)):
        ax.plot([a, b], [i, i], color=t["baseline"], linewidth=2, zorder=1,
                solid_capstyle="round")

    ax.scatter(left, y, s=105, color=t["series1"], zorder=3,
               edgecolors=t["surface"], linewidths=2, label=left_name)
    ax.scatter(right, y, s=105, color=t["series2"], zorder=3,
               edgecolors=t["surface"], linewidths=2, label=right_name)

    # Multiplicative padding, since the axis is logarithmic.
    lo = min(min(left), min(right))
    hi = max(max(left), max(right))
    ax.set_xlim(lo / 2.4, hi * 2.4)

    for i, (a, b) in enumerate(zip(left, right)):
        b_text = fmt(b) + (f" ({right_suffix[i]})" if right_suffix else "")
        # Whichever marker sits left gets a left-hand label, and vice versa.
        if a <= b:
            ax.text(a / 1.12, i, fmt(a), ha="right", va="center", fontsize=8.5,
                    color=t["secondary"])
            ax.text(b * 1.12, i, b_text, ha="left", va="center", fontsize=8.5,
                    color=t["secondary"])
        else:
            ax.text(a * 1.12, i, fmt(a), ha="left", va="center", fontsize=8.5,
                    color=t["secondary"])
            ax.text(b / 1.12, i, b_text, ha="right", va="center", fontsize=8.5,
                    color=t["secondary"])

    ax.set_yticks(y, labels, fontsize=9.5, color=t["secondary"])
    ax.set_ylim(len(labels) - 0.5, -0.9)  # headroom for the legend, newest row on top
    ax.set_xlabel("nMSE, log scale (lower is better)", color=t["secondary"], fontsize=10)

    # Legend above the plot, so it can never sit on top of a mark.
    legend = ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=9.5
    )
    for text in legend.get_texts():
        text.set_color(t["secondary"])

    ax.set_title(title, color=t["primary"], fontsize=13, fontweight="bold",
                 pad=52, loc="left")
    ax.text(0.0, 1.13, METRIC_SUBTITLE, transform=ax.transAxes,
            color=t["secondary"], fontsize=9)
    fig.text(0.0, -0.02, note, color=t["muted"], fontsize=8.5)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=t["surface"])
    plt.close(fig)
    print(f"  wrote {out}")


def plot_transfer_gap(rows, out, t):
    """Training-task performance vs the held-out task, per model."""
    if not rows:
        return
    dumbbell(
        labels=[short_model(r["model"]) for r in rows],
        left=[float(r["mean_train_nmse"]) for r in rows],
        right=[float(r["test_nmse"]) for r in rows],
        left_name="mean of training tasks",
        right_name="held-out test task",
        right_suffix=[r["test_task"] for r in rows],
        title="Transfer gap: does the prior reach an operator it never trained on?",
        note="A short connector means the shared prior generalised across operators; "
             "a long one means the model fitted its three training tasks specifically.",
        out=out,
        t=t,
    )


def plot_original_vs_foundation(rows, out, t):
    """The paper's single-task model vs the best foundation model, per task."""
    rows = [r for r in rows if r.get("best_foundation_nmse")]
    if not rows:
        return
    dumbbell(
        labels=[r["task"] for r in rows],
        left=[float(r["original_nmse"]) for r in rows],
        right=[float(r["best_foundation_nmse"]) for r in rows],
        left_name="ORIGINAL (paper, single-task)",
        right_name="best foundation model",
        title="Multi-task prior vs the paper's single-task model",
        note="The ORIGINAL model trained only on the paper's 'path' task, so every task "
             "here is zero-shot for it.",
        out=out,
        t=t,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_root", default=None, help="used to locate results/<dataset>")
    p.add_argument("--results", default=None, help="directory holding the CSVs")
    p.add_argument("--dark", action="store_true")
    args = p.parse_args()

    if args.results:
        results = os.path.abspath(args.results)
    elif args.runs_root:
        results = os.path.join(REPO, "results", os.path.basename(os.path.abspath(args.runs_root)))
    else:
        results = os.path.join(REPO, "results", "METRLA")

    t = THEME["dark" if args.dark else "light"]
    suffix = "_dark" if args.dark else ""
    plt.rcParams["font.family"] = FONT

    comparison = read_csv(os.path.join(results, "comparison.csv"))
    if not comparison:
        print(f"No comparison.csv in {results}; run compare.py first.")
        return 1

    print(f"reading {results}")
    plot_table(comparison, os.path.join(results, f"table_comparison{suffix}.png"), t)
    plot_heatmap(comparison, os.path.join(results, f"heatmap_models_tasks{suffix}.png"), t)
    plot_transfer_gap(
        read_csv(os.path.join(results, "transfer_gap.csv")),
        os.path.join(results, f"transfer_gap{suffix}.png"), t,
    )
    plot_original_vs_foundation(
        read_csv(os.path.join(results, "original_zeroshot.csv")),
        os.path.join(results, f"original_vs_foundation{suffix}.png"), t,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
