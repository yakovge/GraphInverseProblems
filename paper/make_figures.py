"""Appendix figure: nMSE against the test-time CG budget, from results/cpox_task_holdout/solver_ablation.csv.

    python paper/make_figures.py      # -> paper/figures/solver_budget.pdf
"""
import csv
import os
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, 'results/cpox_task_holdout/solver_ablation.csv')
OUT = os.path.join(ROOT, 'paper/figures/solver_budget.pdf')

# setting label -> total CG steps on the x axis
STEPS = {'1 CG step per unrolled it.; LS 5 steps': 5,
         'trained setting: 5 unrolled it. x 5 CG steps; LS 25 steps': 25,
         '25 CG steps per unrolled it.; LS 100 steps': 100}
LAM0 = {'lam = 1e-8 at test time, 5 x 5; LS 25 steps': 25,  # the same models with lambda_task overridden to 1e-8
        'lam = 1e-8 at test time, 25 CG steps per unrolled it.; LS 100 steps': 100}
SERIES = [('least squares', 'least squares (CGLS from 0)', '#1b5e8a', 'o', STEPS, '-'),
          ('single-task from scratch', 'from scratch', '#c0392b', 's', STEPS, '-'),
          ('held out, zero-shot', 'pretrained, zero-shot', '#7a7a7a', '^', STEPS, '-'),
          ('held out, task head', 'pretrained, + task head', '#2e8b57', 'D', STEPS, '-'),
          ('single-task from scratch', r'from scratch, $\lambda \to 0$', '#c0392b', 's', LAM0, '--'),
          ('held out, task head', r'+ task head, $\lambda \to 0$', '#2e8b57', 'D', LAM0, '--')]
TASKS = [('source_localization', 'source localization'), ('pde_reconstruction', 'PDE-state reconstruction')]


def main():
    data = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            for table in (STEPS, LAM0):
                if r['setting'] in table:
                    for t, _ in TASKS:
                        vals = [float(v) for v in r[t].split(';')]
                        data[(r['row'], t, table[r['setting']], table is LAM0)] = (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    for ax, (t, title) in zip(axes, TASKS):
        for row, label, color, marker, table, ls in SERIES:
            xs = sorted(table.values())
            ys = [data[(row, t, x, table is LAM0)][0] for x in xs]
            es = [data[(row, t, x, table is LAM0)][1] for x in xs]
            ax.errorbar(xs, ys, yerr=es, label=label, color=color, marker=marker, ms=4, lw=1.4, capsize=2, ls=ls,
                        mfc='white' if ls == '--' else color)
        ax.set_xscale('log'); ax.set_xticks([5, 25, 100]); ax.set_xticklabels(['5', '25', '100'])
        ax.set_xlabel('total CG steps at test time'); ax.set_title(title, fontsize=10)
        ax.grid(True, which='major', lw=0.4, alpha=0.5); ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('test nMSE (CPOX)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc='lower center', ncol=3)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
