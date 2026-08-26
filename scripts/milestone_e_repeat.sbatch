#!/usr/bin/env bash
# Submit with: sbatch --array=0-<configuration_count-1>%1 scripts/milestone_e_repeat.sbatch
#SBATCH --job-name=moe-e-repeat
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=uranus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40s:4
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=results/raw/slurm-milestone-e-repeat-%A-%a.out

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate cuda-env
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_SERVER_DEV_MODE=1
source scripts/milestone_e_common.sh

study="${MILESTONE_E_REPEAT_STUDY:?set MILESTONE_E_REPEAT_STUDY to prepared repeat directory}"
plan="${study}/repeat-plan.json"
index="${SLURM_ARRAY_TASK_ID:?submit this script as a Slurm array}"
controller="${study}/measured/configuration-$(printf '%02d' "${index}")-attempt-${SLURM_JOB_ID}"
mkdir -p "${controller}/runs"
ME_SERVER_PID=""; ME_PID_FILE=""; ME_PATCH_APPLIED_BY_JOB=0

readarray -t values < <(python - "${plan}" "${index}" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1])); row = plan["configurations"][int(sys.argv[2])]
print(row["name"]); print(row["placement_name"]); print(row["placement_file"])
print(",".join(map(str, row["frequency_tuple"]))); print(row["repetitions"])
print(json.load(open(plan["joint_study"] + "/joint-plan.json"))["expert_counts"])
print(json.load(open(plan["joint_study"] + "/joint-plan.json"))["trace"])
print(row.get("comparison_reference_run") or "")
PY
)
name="${values[0]}"; placement_name="${values[1]}"; placement_file="${values[2]}"
clocks="${values[3]}"; repetitions="${values[4]}"; ME_EXPERT_COUNTS="${values[5]}"
ME_TRACE="${values[6]}"; comparison_reference="${values[7]}"
ME_PROFILE_RUN="$(dirname "${ME_EXPERT_COUNTS}")"
export ME_EXPERT_COUNTS ME_PROFILE_RUN ME_TRACE

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
scripts/check_environment.sh > "${controller}/environment.txt" 2>&1
sha256sum "${plan}" "${placement_file}" "${ME_EXPERT_COUNTS}" "${ME_TRACE}" \
    > "${controller}/immutable-input-hashes.txt"
me_activate_patch "${controller}"
me_launch_server "${controller}" "${placement_name}" "${placement_file}"
me_set_clocks "${controller}" "${index}" "${clocks}"
printf 'repetition\trun_dir\n' > "${controller}/runs.tsv"
for repetition in $(seq 1 "${repetitions}"); do
    run_dir="$(me_run_benchmark "${controller}" "${name}-rep${repetition}" \
        "${placement_name}" "${placement_file}" "${clocks}")"
    if [[ -n "${comparison_reference}" ]]; then
        python -m tools.compare_validation_outputs "${comparison_reference}" "${run_dir}" \
            --minimum-common-prefix 0 \
            > "${controller}/output-divergence-rep${repetition}.json"
    fi
    printf '%s\t%s\n' "${repetition}" "${run_dir}" >> "${controller}/runs.tsv"
done
RESULT_RUNS="${controller}/runs.tsv" python - "${controller}/result.json" "${index}" "${name}" <<'PY'
import csv, json, os, sys
with open(os.environ["RESULT_RUNS"]) as stream:
    runs = [row["run_dir"] for row in csv.DictReader(stream, delimiter="\t")]
open(sys.argv[1], "w").write(json.dumps({"status": "complete", "configuration_index": int(sys.argv[2]),
    "name": sys.argv[3], "run_dirs": runs}, indent=2, sort_keys=True) + "\n")
PY
python -m src.gpu_control reset > "${controller}/final-clock-reset.json"
