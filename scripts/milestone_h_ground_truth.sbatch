#!/usr/bin/env bash
# Run one non-P1 placement across the complete 81-point DVFS matrix.
# Submit only as a serialized array: sbatch --array=0-3%1 ...
# P1 is reused from its accepted Milestone E controller and is never rerun.
#SBATCH --job-name=moe-h-ground-truth
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=uranus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40s:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=results/raw/slurm-milestone-h-%A-%a.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate cuda-env
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_SERVER_DEV_MODE=1

source scripts/milestone_e_common.sh

study="${MILESTONE_H_STUDY:?set MILESTONE_H_STUDY to a prepared ground-truth study}"
study="$(realpath "${study}")"
plan="${study}/matrix-plan.json"
task_index="${SLURM_ARRAY_TASK_ID:?submit this script as a serialized array}"
controller="${study}/measured/task-$(printf '%02d' "${task_index}")-attempt-${SLURM_JOB_ID}"
mkdir -p "${controller}/runs"
ME_SERVER_PID=""
ME_PID_FILE=""
ME_PATCH_APPLIED_BY_JOB=0

readarray -t values < <(python - "${plan}" "${task_index}" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
index = int(sys.argv[2])
rows = [row for row in plan["placements"] if row.get("task_index") == index]
if len(rows) != 1:
    raise SystemExit(f"expected one placement for task {index}, found {len(rows)}")
row = rows[0]
print(row["policy"])
print(row["name"])
print(row["placement_file"])
print(plan["trace"])
print(plan["phase_counts"])
print(plan["model_config"])
PY
)
placement_policy="${values[0]}"
placement_name="${values[1]}"
placement_file="${values[2]}"
ME_TRACE="${values[3]}"
ME_EXPERT_COUNTS="${values[4]}"
MODEL_CONFIG="${values[5]}"
ME_PROFILE_RUN="$(dirname "${ME_EXPERT_COUNTS}")"
export ME_TRACE ME_EXPERT_COUNTS ME_PROFILE_RUN

cleanup() {
    status=$?
    trap - EXIT INT TERM
    me_stop_server
    python -m src.gpu_control reset >> "${controller}/clock-reset.log" 2>&1 || true
    me_revert_patch_if_owned "${controller}"
    printf '%s\n' "${status}" > "${controller}/exit-status.txt"
    exit "${status}"
}
trap cleanup EXIT INT TERM

me_check_allocation
for file in "${plan}" "${placement_file}" "${ME_EXPERT_COUNTS}" "${ME_TRACE}" "${MODEL_CONFIG}"; do
    [[ -f "${file}" ]] || { printf 'missing ground-truth input: %s\n' "${file}" >&2; exit 1; }
done
scripts/check_environment.sh > "${controller}/environment.txt" 2>&1
cp "${plan}" "${controller}/matrix-plan.json"
cp "${ME_TRACE}" "${controller}/trace.jsonl"
sha256sum "${plan}" "${placement_file}" "${ME_EXPERT_COUNTS}" "${ME_TRACE}" \
    > "${controller}/immutable-input-hashes.txt"
printf '%s\n' "${placement_policy}" > "${controller}/placement-policy.txt"
printf '%s\n' "${placement_name}" > "${controller}/placement-name.txt"

me_activate_patch "${controller}"
# Empty server placement argument is intentional for P0: it exercises vLLM's
# default linear strategy. All other placements use the recorded static map.
server_placement_file="${placement_file}"
if [[ "${placement_policy}" == "P0" ]]; then
    server_placement_file=""
fi
me_launch_server "${controller}" "${placement_name}" "${server_placement_file}"

printf 'label\tplacement\tclocks\tstatus\trun_dir\n' > "${controller}/runs.tsv"
while IFS=$'\t' read -r order f0 f1 f2 f3; do
    clocks="${f0},${f1},${f2},${f3}"
    label="${placement_policy}-f$(printf '%02d' "${order}")-${f0}-${f1}-${f2}-${f3}"
    me_set_clocks "${controller}" "${order}" "${clocks}"
    if run_dir="$(me_run_benchmark "${controller}" "${label}" \
            "${placement_name}" "${placement_file}" "${clocks}")"; then
        printf '%s\t%s\t%s\tcomplete\t%s\n' "${label}" "${placement_name}" \
            "${clocks}" "${run_dir}" >> "${controller}/runs.tsv"
    else
        printf '%s\t%s\t%s\tfailed\t%s\n' "${label}" "${placement_name}" \
            "${clocks}" "${run_dir:-}" >> "${controller}/runs.tsv"
        printf '%s\n' "${label}" >> "${controller}/failed-configurations.txt"
    fi
done < <(python - "${plan}" <<'PY'
import json, sys
for index, values in enumerate(json.load(open(sys.argv[1]))["frequency_tuples"]):
    print(index, *values, sep="\t")
PY
)

python -m src.gpu_control reset > "${controller}/final-clock-reset.json"
complete_count="$(awk -F '\t' '$4 == "complete" {count++} END {print count+0}' "${controller}/runs.tsv" 2>/dev/null || true)"
[[ "${complete_count}" -eq 81 ]] || {
    printf 'placement %s has %s/81 complete configurations\n' "${placement_name}" "${complete_count}" >&2
    exit 1
}
if [[ -s "${controller}/failed-configurations.txt" ]]; then
    printf 'placement %s has failed configurations\n' "${placement_name}" >&2
    exit 1
fi
python - "${controller}/result.json" "${placement_policy}" "${placement_name}" <<'PY'
import json, sys
json.dump({"status": "complete", "policy": sys.argv[2], "placement": sys.argv[3]},
          open(sys.argv[1], "w"), indent=2)
PY
