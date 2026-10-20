#!/usr/bin/env bash
# Progress of the overnight run (unit grip-final2: baselines C, then reduced experiment B).
cd "$(dirname "$0")"
echo "unit: $(systemctl --user is-active grip-final2 2>/dev/null)   battery: $(cat /sys/class/power_supply/BAT*/capacity 2>/dev/null)% $(cat /sys/class/power_supply/BAT*/status 2>/dev/null)"
echo "checkpoints: A=$(ls results/cpox_task_holdout/*.pth 2>/dev/null | wc -l)/30  B=$(ls results/dataset_holdout/*.pth 2>/dev/null | wc -l)/30  in-progress(models/)=$(ls models/*.pth 2>/dev/null | wc -l)"
ls results/*/*_table.png 2>/dev/null
echo "--- last phase lines:"; tr '\r' '\n' < results/run_cpu_CB.log | grep -E "^Project Name|\[phase '[a-z]+' done|^=== |Traceback|Error|^Done" | tail -8
echo "--- current progress:"; tr '\r' '\n' < results/run_cpu_CB.log | grep -E "[0-9]+/[0-9]+ \[" | tail -1
