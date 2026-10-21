#!/usr/bin/env bash
# Build the Moodle submission zip: paper (PDF + LaTeX sources), all code, README/docs, result tables, CSVs and logs.
#   bash make_submission.sh            # -> submission/GraphInverseProblems_submission.zip (no checkpoints, a few MB)
#   bash make_submission.sh --with-checkpoints   # also the 75 .pth files (~37 MB) so the tables can be re-evaluated offline
set -euo pipefail
cd "$(dirname "$0")"
OUT=submission; ZIP=$OUT/GraphInverseProblems_submission.zip
rm -rf $OUT && mkdir -p $OUT
FILES=(README.md GPU_RUN.md CHANGES_SUMMARY.txt LICENSE requirements.txt environment.yml run_gpu.sh check_run.sh make_submission.sh
       paper/main.tex paper/main.pdf paper/neurips_2025.sty paper/make_tables.py paper/make_figures.py paper/tables paper/figures
       src/*.py configurations
       results/run_cpu.log results/run_gpu_CB.log results/run_eval_cpu.log results/cpox_task_holdout/*.png results/cpox_task_holdout/*.csv results/dataset_holdout/*.png results/dataset_holdout/*.csv)
if [[ "${1:-}" == "--with-checkpoints" ]]; then
  FILES+=(results/cpox_task_holdout/*.pth results/dataset_holdout/*.pth)
fi
zip -qr $ZIP "${FILES[@]}" -x '*/__pycache__/*' '*.aux' '*.out' '*.log.bak'
echo "wrote $ZIP ($(du -h $ZIP | cut -f1))"; unzip -l $ZIP | tail -1
