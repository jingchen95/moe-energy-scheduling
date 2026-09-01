#!/usr/bin/env bash
# Calibrate offered load once with P1 at HHHH.
#SBATCH --job-name=moe-i-calibrate
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=uranus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40s:4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=results/raw/slurm-milestone-i-calibrate-%j.out

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate cuda-env
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_SERVER_DEV_MODE=1
source scripts/milestone_i_common.sh

study="$(realpath "${MILESTONE_I_CALIBRATION:?set MILESTONE_I_CALIBRATION}")"
plan="${study}/calibration-plan.json"
controller="${study}/measured/attempt-${SLURM_JOB_ID}"
mkdir -p "${controller}/runs"
ME_SERVER_PID=""; ME_PID_FILE=""; ME_PATCH_APPLIED_BY_JOB=0

readarray -t values < <(python - "${plan}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1])); s = p["placement"]
print(s["runtime_placement_name"]); print(s["placement_file"])
print(p["phase_counts"]); print(p["model_config"])
PY
)
placement_name="${values[0]}"; placement_file="${values[1]}"
ME_EXPERT_COUNTS="${values[2]}"; MODEL_CONFIG="${values[3]}"
ME_PROFILE_RUN="$(dirname "${ME_EXPERT_COUNTS}")"
export ME_EXPERT_COUNTS ME_PROFILE_RUN

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
cp "${plan}" "${controller}/calibration-plan.json"
sha256sum "${plan}" "${placement_file}" "${ME_EXPERT_COUNTS}" > "${controller}/immutable-input-hashes.txt"
me_activate_patch "${controller}"
me_launch_server "${controller}" "${placement_name}" "${placement_file}"
python -m src.gpu_control set-all --clocks 2520,2520,2520,2520 > "${controller}/clock-high.json"

printf 'label\tplacement\tclocks\trequest_rate\tstatus\trun_dir\n' > "${controller}/runs.tsv"
while IFS=$'\t' read -r order rate trace; do
    label="cal-r$(printf '%02d' "${order}")-$(printf '%g' "${rate}")"
    if run_dir="$(mi_run_benchmark "${controller}" "${label}" "${placement_name}" \
            "${placement_file}" "2520,2520,2520,2520" "${rate}" "${trace}")"; then
        printf '%s\t%s\t%s\t%s\tcomplete\t%s\n' "${label}" "${placement_name}" \
            "2520,2520,2520,2520" "${rate}" "${run_dir}" >> "${controller}/runs.tsv"
    else
        printf '%s\t%s\t%s\t%s\tfailed\t%s\n' "${label}" "${placement_name}" \
            "2520,2520,2520,2520" "${rate}" "${run_dir:-}" >> "${controller}/runs.tsv"
        exit 1
    fi
done < <(python - "${plan}" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1]))["runs"]:
    print(row["order"], row["request_rate"], row["trace"], sep="\t")
PY
)

python -m src.gpu_control reset > "${controller}/final-clock-reset.json"
run_count="$(awk -F '\t' '$5 == "complete" {n++} END {print n+0}' "${controller}/runs.tsv")"
expected_count="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["runs"]))' "${plan}")"
[[ "${run_count}" -eq "${expected_count}" ]] || {
    printf 'expected %s runs, found %s\n' "${expected_count}" "${run_count}" >&2
    exit 1
}
python - "${controller}/result.json" "${run_count}" <<'PY'
import json, sys
json.dump({"status": "complete", "runs": int(sys.argv[2])}, open(sys.argv[1], "w"), indent=2)
PY
