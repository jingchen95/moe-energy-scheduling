#!/usr/bin/env bash
# Run one randomized placement block: 11 targeted tuples x 3 repetitions.
# Submit only as a serialized array: sbatch --array=0-3%1 ...
#SBATCH --job-name=moe-i-repeat
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=uranus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40s:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=results/raw/slurm-milestone-i-repeat-%A-%a.out

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate cuda-env
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_SERVER_DEV_MODE=1
source scripts/milestone_i_common.sh

study="$(realpath "${MILESTONE_I_STUDY:?set MILESTONE_I_STUDY}")"
plan="${study}/experiment-plan.json"
task_index="${SLURM_ARRAY_TASK_ID:?submit as a serialized array}"
controller="${study}/measured/task-$(printf '%02d' "${task_index}")-attempt-${SLURM_JOB_ID}"
mkdir -p "${controller}/runs"
ME_SERVER_PID=""; ME_PID_FILE=""; ME_PATCH_APPLIED_BY_JOB=0

readarray -t values < <(python - "${plan}" "${task_index}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1])); i = int(sys.argv[2])
rows = [x for x in p["placements"] if x["task_index"] == i]
if len(rows) != 1: raise SystemExit(f"expected one placement, found {len(rows)}")
s = rows[0]
print(s["policy"]); print(s["runtime_placement_name"]); print(s["placement_file"])
print(p["trace"]); print(p["request_rate"]); print(p["phase_counts"]); print(p["model_config"])
PY
)
policy="${values[0]}"; placement_name="${values[1]}"; placement_file="${values[2]}"
ME_TRACE="${values[3]}"; request_rate="${values[4]}"; ME_EXPERT_COUNTS="${values[5]}"
MODEL_CONFIG="${values[6]}"; ME_PROFILE_RUN="$(dirname "${ME_EXPERT_COUNTS}")"
export ME_TRACE ME_EXPERT_COUNTS ME_PROFILE_RUN

cleanup() {
    status=$?; trap - EXIT INT TERM
    me_stop_server
    python -m src.gpu_control reset >> "${controller}/clock-reset.log" 2>&1 || true
    me_revert_patch_if_owned "${controller}"
    printf '%s\n' "${status}" > "${controller}/exit-status.txt"
    exit "${status}"
}
trap cleanup EXIT INT TERM

me_check_allocation
for file in "${plan}" "${placement_file}" "${ME_TRACE}" "${ME_EXPERT_COUNTS}"; do
    [[ -f "${file}" ]] || { printf 'missing input: %s\n' "${file}" >&2; exit 1; }
done
scripts/check_environment.sh > "${controller}/environment.txt" 2>&1
cp "${plan}" "${controller}/experiment-plan.json"
cp "${ME_TRACE}" "${controller}/trace.jsonl"
sha256sum "${plan}" "${placement_file}" "${ME_TRACE}" "${ME_EXPERT_COUNTS}" \
    > "${controller}/immutable-input-hashes.txt"
me_activate_patch "${controller}"
me_launch_server "${controller}" "${placement_name}" "${placement_file}"

printf 'label\tplacement\tclocks\trepetition\tstatus\trun_dir\n' > "${controller}/runs.tsv"
while IFS=$'\t' read -r order repetition f0 f1 f2 f3; do
    clocks="${f0},${f1},${f2},${f3}"
    label="${policy}-o$(printf '%02d' "${order}")-rep${repetition}-${f0}-${f1}-${f2}-${f3}"
    me_set_clocks "${controller}" "${order}" "${clocks}"
    if run_dir="$(mi_run_benchmark "${controller}" "${label}" "${placement_name}" \
            "${placement_file}" "${clocks}" "${request_rate}" "${ME_TRACE}")"; then
        printf '%s\t%s\t%s\t%s\tcomplete\t%s\n' "${label}" "${placement_name}" \
            "${clocks}" "${repetition}" "${run_dir}" >> "${controller}/runs.tsv"
    else
        printf '%s\t%s\t%s\t%s\tfailed\t%s\n' "${label}" "${placement_name}" \
            "${clocks}" "${repetition}" "${run_dir:-}" >> "${controller}/runs.tsv"
        exit 1
    fi
done < <(python - "${plan}" "${task_index}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1])); i = int(sys.argv[2])
s = next(x for x in p["placements"] if x["task_index"] == i)
for row in s["runs"]: print(row["order"], row["repetition"], *row["frequency_tuple"], sep="\t")
PY
)

python -m src.gpu_control reset > "${controller}/final-clock-reset.json"
complete_count="$(awk -F '\t' '$5 == "complete" {n++} END {print n+0}' "${controller}/runs.tsv")"
[[ "${complete_count}" -eq 33 ]] || { printf 'expected 33 runs, found %s\n' "${complete_count}" >&2; exit 1; }
python - "${controller}/result.json" "${policy}" <<'PY'
import json, sys
json.dump({"status": "complete", "policy": sys.argv[2], "runs": 33}, open(sys.argv[1], "w"), indent=2)
PY
