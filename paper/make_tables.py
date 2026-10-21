"""Build the LaTeX tables of the paper from the evaluation CSVs (never copy numbers by hand).

    python paper/make_tables.py      # reads results/*/*.csv, writes paper/tables/*.tex
"""
import csv
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'paper', 'tables')
TASKS = ['denoising', 'inpainting', 'sensor_recovery', 'source_localization', 'pde_reconstruction']
SHORT = {'denoising': 'denoise', 'inpainting': 'inpaint', 'sensor_recovery': 'sensor', 'source_localization': 'source-loc',
         'pde_reconstruction': 'PDE'}


class Stat(list):
    """A table cell: the per-seed values, or (rows copied from a run log) [mean] with the std attached."""
    def __init__(self, values, std=None):
        super().__init__(values)
        self._std = std

    @property
    def mean(self):
        return st.mean(self)

    @property
    def std(self):
        return self._std if self._std is not None else (st.stdev(self) if len(self) > 1 else None)


def parse_cell(text):
    if not text:
        return Stat([])
    if '+-' in text:  # 'mean+-std' (plot_2.py --model_rows_from)
        m, sd = text.split('+-')
        return Stat([float(m)], float(sd))
    return Stat([float(v) for v in text.split(';')])


def read(path, keys):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[tuple(r[k] for k in keys)] = {t: parse_cell(r[t]) for t in TASKS}
    return rows


def cell(vals, bold=False, std=True):
    if not vals:
        return '--'
    s = f"{vals.mean:.3f}"
    if bold:
        s = r'\textbf{' + s + '}'
    if std and vals.std is not None:
        s += r'{\scriptsize$\pm$' + f"{vals.std:.3f}" + '}'
    return s


def write(name, text):
    with open(os.path.join(OUT, name), 'w') as f:
        f.write(text)
    print('wrote', name)


def table_a(full):
    """Experiment A, held-out-task cells (the diagonal) against every baseline."""
    A = read(os.path.join(ROOT, 'results/cpox_task_holdout/loss_matrix_table.csv'), ['row'])
    get = lambda name: A[(name,)]
    diag = {}
    for variant, label in [('zero-shot (shared head)', 'pretrained, held out, zero-shot'),
                           ('adapted (task heads)', 'pretrained, held out, + task head')]:
        diag[label] = {t: get(f"{t} held out | {variant}")[t] for t in TASKS}
    rows = [('$x = b$ (measurement)', get('BASELINE: X = b')),
            ('least squares, 25 CGLS steps', get('BASELINE: least squares')),
            ('Tikhonov LS, tuned', get('BASELINE: Tikhonov LS (lam, CG steps tuned on val)')),
            ('graph-Laplacian LS, tuned', get('BASELINE: Laplacian LS (lam, CG steps tuned on val)')),
            ('single-task net from scratch (60 ep.)', get('BASELINE: single-task model from scratch (same net, 60 epochs on that task only)'))]
    rows += list(diag.items())
    best = {t: min(v[t].mean for _, v in rows if v[t]) for t in TASKS}
    lines = [r'\begin{tabular}{l' + 'c' * len(TASKS) + '}', r'\toprule',
             'CPOX test weeks, nMSE $\\downarrow$ & ' + ' & '.join(SHORT[t] for t in TASKS) + r' \\', r'\midrule']
    for i, (name, v) in enumerate(rows):
        if i == 5:
            lines.append(r'\midrule')
        lines.append(name + ' & ' + ' & '.join(cell(v[t], bold=v[t] and abs(v[t].mean - best[t]) < 1e-9) for t in TASKS) + r' \\')
    if full:
        lines.append(r'\midrule')
        for variant, label in [('zero-shot (shared head)', 'zero-shot'), ('adapted (task heads)', '+ task heads')]:
            for h in TASKS:
                v = get(f"{h} held out | {variant}")
                lines.append(f"{SHORT[h]} held out, {label} & " + ' & '.join(
                    (r'\cellcolor{blue!10}' if t == h else '') + cell(v[t]) for t in TASKS) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines) + '\n'


def table_b(full):
    """Experiment B: one block per held-out dataset."""
    B = read(os.path.join(ROOT, 'results/dataset_holdout/comparison_table.csv'), ['dataset', 'row'])
    datasets = []
    for d, _ in B:
        if d not in datasets:
            datasets.append(d)
    names = [('least squares, 25 CGLS steps', 'BASELINE: least squares'),
             ('Tikhonov LS, tuned', 'BASELINE: Tikhonov LS (lam, CG steps tuned on val)'),
             ('graph-Laplacian LS, tuned', 'BASELINE: Laplacian LS (lam, CG steps tuned on val)'),
             ('pretrained, shared head', 'shared head (after backbone training)'),
             ('pretrained, task heads', 'task heads (20 more epochs on the training datasets)')]
    pretty = {'CPOX': 'CPOX (20 nodes)', 'PEDALME': 'PedalMe (15, complete)', 'MONTEVIDEO': 'Montevideo (675)',
              'WIKIMATHS': 'WikiMaths (1068)', 'WINDMILL': 'Windmill (319, complete)'}
    lines = [r'\begin{tabular}{ll' + 'c' * len(TASKS) + '}', r'\toprule',
             'held-out dataset & method & ' + ' & '.join(SHORT[t] for t in TASKS) + r' \\', r'\midrule']
    compact = [r'\begin{tabular}{l' + 'c' * len(TASKS) + '}', r'\toprule',
               'held-out dataset & ' + ' & '.join(SHORT[t] for t in TASKS) + r' \\', r'\midrule']
    for d in datasets:
        rows = [(lab, B[(d, key)]) for lab, key in names if (d, key) in B]
        if not full:  # compact: one line per dataset, cell = task-head model (lowest of the three classical rows)
            bestc = {t: min(v[t].mean for _, v in rows[:3]) for t in TASKS}
            model = dict(rows)['pretrained, task heads']
            compact.append(pretty.get(d, d) + ' & ' + ' & '.join(
                (r'\textbf{%.3f}' % model[t].mean if model[t].mean < bestc[t] - 5e-3 else '%.3f' % model[t].mean)
                + ' (%.3f)' % bestc[t] for t in TASKS) + r' \\')
            continue
        best = {t: min(v[t].mean for _, v in rows) for t in TASKS}
        for i, (name, v) in enumerate(rows):
            lead = (r'\multirow{%d}{*}{%s}' % (len(rows), pretty.get(d, d))) if i == 0 else ''
            lines.append(lead + ' & ' + name + ' & ' + ' & '.join(cell(v[t], bold=abs(v[t].mean - best[t]) < 1e-9, std=full) for t in TASKS) + r' \\')
        lines.append(r'\midrule')
    if not full:
        return '\n'.join(compact + [r'\bottomrule', r'\end{tabular}']) + '\n'
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    return '\n'.join(lines) + '\n'


def table_ablation(full):
    S = read(os.path.join(ROOT, 'results/cpox_task_holdout/solver_ablation.csv'), ['setting', 'row'])
    settings = []
    for s, _ in S:
        if s not in settings:
            settings.append(s)
    short = {settings[0]: 'as trained ($5 \\times 5$)', settings[1]: '1 CG step / it.', settings[2]: '25 CG steps / it.',
             settings[3]: '10 unrolled it.', settings[4]: '$\\lambda = 10^{-8}$', settings[5]: '$\\lambda = 10^{-8}$, 25 CG/it.'}
    ls_short = {settings[0]: '25 steps', settings[1]: '5 steps', settings[2]: '100 steps', settings[3]: '25 steps',
                settings[4]: '25 steps', settings[5]: '100 steps'}
    tasks = TASKS if full else ['inpainting', 'source_localization', 'pde_reconstruction']
    lines = [r'\begin{tabular}{ll' + 'c' * len(tasks) + '}', r'\toprule',
             'model & test-time solver & ' + ' & '.join(SHORT[t] for t in tasks) + r' \\', r'\midrule']
    rows = [('least squares', 'least squares (CGLS)'), ('single-task from scratch', 'from scratch'),
            ('held out, zero-shot', 'pretrained, zero-shot'), ('held out, task head', 'pretrained, + task head')]
    use = settings if full else settings[:3] + settings[4:]  # compact: no "10 unrolled iterations" rows
    for row, label in (rows if full else rows[:2] + rows[3:]):
        seen = set()
        for i, s in enumerate(use):
            key = ls_short[s] if row == 'least squares' else short[s]
            if key in seen:
                continue
            seen.add(key)
            v = S[(s, row)]
            n = len({ls_short[x] for x in use}) if row == 'least squares' else len(use)
            lines.append((r'\multirow{%d}{*}{%s}' % (n, label) if i == 0 else '')
                         + ' & ' + key + ' & ' + ' & '.join(cell(v[t], std=full) for t in tasks) + r' \\')
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for fn, builder in [('table_a.tex', lambda: table_a(False)), ('table_a_full.tex', lambda: table_a(True)),
                        ('table_b.tex', lambda: table_b(False)), ('table_b_full.tex', lambda: table_b(True)),
                        ('table_ablation.tex', lambda: table_ablation(False)), ('table_ablation_full.tex', lambda: table_ablation(True))]:
        try:
            write(fn, builder())
        except FileNotFoundError as e:
            print('skipped', fn, '-', e)
