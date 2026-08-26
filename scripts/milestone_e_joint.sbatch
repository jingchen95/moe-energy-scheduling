#!/usr/bin/env bash
# Submit with: sbatch --array=0-<candidate_count-1>%1 scripts/milestone_e_joint.sbatch
#SBATCH --job-name=moe-e-joint
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=uranus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40s:4
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=results/raw/slurm-milestone-e-joint-%A-%a.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate cuda-env
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_SERVER_DEV_MODE=1
source scripts/milestone_e_common.sh

study="${MILESTONE_E_JOINT_STUDY:?set MILESTONE_E_JOINT_STUDY to prepared Joint study directory}"
plan="${study}/joint-plan.json"
candidate_index="${SLURM_ARRAY_TASK_ID:?submit this script as a Slurm array}"
controller="${study}/measured/candidate-$(printf '%02d' "${candidate_index}")-attempt-${SLURM_JOB_ID}"
mkdir -p "${controller}/runs"
ME_SERVER_PID=""
ME_PID_FILE=""
ME_PATCH_APPLIED_BY_JOB=0

readarray -t values < <(python - "${plan}" "${candidate_index}" <<'PY'
import json, sys
row = json.load(open(sys.argv[1]))["candidates"][int(sys.argv[2])]
print(row["label"])
print(",".join(map(str, row["frequency_tuple"])))
print(row["placement_file"])
print(row["sequential_run_dir"])
print(json.load(open(sys.argv[1]))["expert_counts"])
print(json.load(open(sys.argv[1]))["p1_placement"])
print(json.load(open(sys.argv[1]))["trace"])
PY
)
label="${values[0]}"
clocks="${values[1]}"
placement_file="${values[2]}"
sequential_run="${values[3]}"
ME_EXPERT_COUNTS="${values[4]}"
ME_P1_PLACEMENT="${values[5]}"
ME_TRACE="${values[6]}"
ME_PROFILE_RUN="$(dirname "${ME_EXPERT_COUNTS}")"
export ME_EXPERT_COUNTS ME_P1_PLACEMENT ME_PROFILE_RUN ME_TRACE

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
for file in "${plan}" "${placement_file}" "${ME_EXPERT_COUNTS}" "${ME_TRACE}"; do
    [[ -f "${file}" ]] || { printf 'missing Joint input: %s\n' "${file}" >&2; exit 1; }
done
scripts/check_environment.sh > "${controller}/environment.txt" 2>&1
sha256sum "${plan}" "${placement_file}" "${ME_EXPERT_COUNTS}" "${ME_TRACE}" \
    > "${controller}/immutable-input-hashes.txt"
me_activate_patch "${controller}"
me_launch_server "${controller}" frequency_aware "${placement_file}"
me_set_clocks "${controller}" "${candidate_index}" "${clocks}"
if run_dir="$(me_run_benchmark "${controller}" "${label}" frequency_aware \
        "${placement_file}" "${clocks}")"; then
    python -m tools.compare_validation_outputs "${sequential_run}" "${run_dir}" \
        --minimum-common-prefix 0 > "${controller}/output-divergence.json"
    RESULT_STATUS=complete RESULT_RUN_DIR="${run_dir}" python - "${controller}/result.json" \
        "${candidate_index}" "${label}" "${clocks}" <<'PY'
import json, os, sys
value = {"status": os.environ["RESULT_STATUS"], "run_dir": os.environ["RESULT_RUN_DIR"],
         "candidate_index": int(sys.argv[2]), "label": sys.argv[3],
         "frequency_tuple": [int(value) for value in sys.argv[4].split(",")]}
open(sys.argv[1], "w").write(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY
else
    RESULT_RUN_DIR="${run_dir:-}" python - "${controller}/result.json" "${candidate_index}" <<'PY'
import json, os, sys
open(sys.argv[1], "w").write(json.dumps({"candidate_index": int(sys.argv[2]),
    "status": "failed", "run_dir": os.environ.get("RESULT_RUN_DIR") or None},
    indent=2, sort_keys=True) + "\n")
PY
    exit 1
fi

python -m src.gpu_control reset > "${controller}/final-clock-reset.json"
