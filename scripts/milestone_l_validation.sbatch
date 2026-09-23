#!/usr/bin/env bash
# Seven serialized configuration tasks, three measured repetitions each.
# Submit as: MILESTONE_L_STUDY=<dir> sbatch --array=0-6%1 scripts/milestone_l_validation.sbatch
#SBATCH --job-name=moe-l-align
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=uranus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40s:4
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=results/raw/slurm-milestone-l-%A-%a.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate cuda-env
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_SERVER_DEV_MODE=1
source scripts/milestone_i_common.sh

study="$(realpath "${MILESTONE_L_STUDY:?set MILESTONE_L_STUDY}")"
plan="${study}/experiment-plan.json"
task_index="${SLURM_ARRAY_TASK_ID:?submit as a serialized array}"

readarray -t values < <(python - "${plan}" "${task_index}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1])); i = int(sys.argv[2])
rows = [row for row in p["configurations"] if int(row["task_index"]) == i]
if len(rows) != 1: raise SystemExit(f"expected one configuration, found {len(rows)}")
row = rows[0]; placement = row["placement"]
print(row["config_id"]); print(row["placement_id"])
print(placement["runtime_placement_name"]); print(placement["placement_file"])
print(",".join(str(value) for value in row["frequency_tuple"]))
print(p["trace"]); print(p["request_rate"]); print(p["phase_counts"]); print(p["model_config"])
PY
)
config_id="${values[0]}"; placement_id="${values[1]}"
placement_name="${values[2]}"; placement_file="${values[3]}"; clocks="${values[4]}"
ME_TRACE="${values[5]}"; request_rate="${values[6]}"; ME_EXPERT_COUNTS="${values[7]}"
MODEL_CONFIG="${values[8]}"; ME_PROFILE_RUN="$(dirname "${ME_EXPERT_COUNTS}")"
export ME_TRACE ME_EXPERT_COUNTS ME_PROFILE_RUN

controller="${study}/measured/task-$(printf '%02d' "${task_index}")-${config_id}-attempt-${SLURM_JOB_ID}"
mkdir -p "${controller}/runs"
ME_SERVER_PID=""; ME_PID_FILE=""; ME_PATCH_APPLIED_BY_JOB=0

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
for file in "${plan}" "${placement_file}" "${ME_TRACE}" "${ME_EXPERT_COUNTS}" "${MODEL_CONFIG}"; do
    [[ -f "${file}" ]] || { printf 'missing input: %s\n' "${file}" >&2; exit 1; }
done
scripts/check_environment.sh > "${controller}/environment.txt" 2>&1
git rev-parse HEAD > "${controller}/git-commit.txt"
cp "${plan}" "${controller}/experiment-plan.json"
cp "${ME_TRACE}" "${controller}/trace.jsonl"
sha256sum "${plan}" "${placement_file}" "${ME_TRACE}" "${ME_EXPERT_COUNTS}" "${MODEL_CONFIG}" \
    > "${controller}/immutable-input-hashes.txt"
printf '%s\n' "primary FusedMoE timing disabled: existing path requires extra patches and --enforce-eager" \
    > "${controller}/timing-instrumentation.txt"

me_activate_patch "${controller}"
me_launch_server "${controller}" "${placement_name}" "${placement_file}"

printf 'config_id\tplacement_id\tplacement\tclocks\trepetition\tstatus\trun_dir\n' \
    > "${controller}/runs.tsv"
while IFS=$'\t' read -r order repetition; do
    label="${config_id}-${placement_id}-rep${repetition}"
    me_set_clocks "${controller}" "${order}" "${clocks}"
    if run_dir="$(mi_run_benchmark "${controller}" "${label}" "${placement_name}" \
            "${placement_file}" "${clocks}" "${request_rate}" "${ME_TRACE}")"; then
        printf '%s\t%s\t%s\t%s\t%s\tcomplete\t%s\n' \
            "${config_id}" "${placement_id}" "${placement_name}" "${clocks}" \
            "${repetition}" "${run_dir}" >> "${controller}/runs.tsv"
    else
        printf '%s\t%s\t%s\t%s\t%s\tfailed\t%s\n' \
            "${config_id}" "${placement_id}" "${placement_name}" "${clocks}" \
            "${repetition}" "${run_dir:-}" >> "${controller}/runs.tsv"
        exit 1
    fi
done < <(python - "${plan}" "${task_index}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1])); i = int(sys.argv[2])
row = next(row for row in p["configurations"] if int(row["task_index"]) == i)
for run in row["runs"]: print(run["order"], run["repetition"], sep="\t")
PY
)

python -m src.gpu_control reset > "${controller}/final-clock-reset.json"
complete_count="$(awk -F '\t' '$6 == "complete" {n++} END {print n+0}' "${controller}/runs.tsv")"
[[ "${complete_count}" -eq 3 ]] || {
    printf 'expected 3 complete runs, found %s\n' "${complete_count}" >&2
    exit 1
}
python - "${controller}/result.json" "${config_id}" "${placement_id}" <<'PY'
import json, sys
json.dump({"status": "complete", "config_id": sys.argv[2],
           "placement_id": sys.argv[3], "runs": 3},
          open(sys.argv[1], "w"), indent=2)
PY
