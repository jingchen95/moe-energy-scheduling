#!/usr/bin/env bash
#SBATCH --job-name=moe-e-sequential
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=uranus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40s:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=results/raw/slurm-milestone-e-sequential-%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate cuda-env
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_SERVER_DEV_MODE=1

source scripts/milestone_e_common.sh

d_controller="${MILESTONE_D_CONTROLLER:?set MILESTONE_D_CONTROLLER to accepted job 255131 controller}"
controller="${PROJECT_ROOT}/results/raw/milestone-e-sequential-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${controller}/runs"
manifest="${controller}/runs.tsv"
printf 'label\tplacement\tclocks\tstatus\trun_dir\n' > "${manifest}"
ME_SERVER_PID=""
ME_PID_FILE=""
ME_PATCH_APPLIED_BY_JOB=0

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
me_resolve_milestone_d_inputs "$(realpath "${d_controller}")"
scripts/check_environment.sh > "${controller}/environment.txt" 2>&1
printf '%s\n' "$(realpath "${d_controller}")" > "${controller}/milestone-d-source.txt"
cp "${ME_TRACE}" "${controller}/trace.jsonl"
sha256sum "${ME_TRACE}" "${ME_EXPERT_COUNTS}" "${ME_P1_PLACEMENT}" \
    > "${controller}/immutable-input-hashes.txt"
python -m tools.prepare_milestone_e sequential-plan --output-dir "${controller}/plan"
resume_controller="${MILESTONE_E_SEQUENTIAL_RESUME_FROM:-}"
if [[ -n "${resume_controller}" ]]; then
    resume_controller="$(realpath "${resume_controller}")"
    [[ -f "${resume_controller}/runs.tsv" ]] || {
        printf 'resume controller has no runs.tsv: %s\n' "${resume_controller}" >&2
        exit 1
    }
    cmp "${controller}/plan/sequential-plan.json" \
        "${resume_controller}/plan/sequential-plan.json" >/dev/null || {
        printf 'resume controller uses a different Sequential plan\n' >&2
        exit 1
    }
    awk -F '\t' 'NR > 1 && $4 == "complete"' "${resume_controller}/runs.tsv" \
        >> "${manifest}"
    printf '%s\n' "${resume_controller}" > "${controller}/resumed-from.txt"
fi
me_activate_patch "${controller}"
me_launch_server "${controller}" upstream_default_eplb "${ME_P1_PLACEMENT}"

tail -n +2 "${controller}/plan/sequential-plan.tsv" |
while IFS=$'\t' read -r order label f0 f1 f2 f3; do
    if awk -F '\t' -v wanted="${label}" '$1 == wanted && $4 == "complete" {found=1} END {exit !found}' \
            "${manifest}"; then
        continue
    fi
    clocks="${f0},${f1},${f2},${f3}"
    me_set_clocks "${controller}" "${order}" "${clocks}"
    if run_dir="$(me_run_benchmark "${controller}" "${label}" \
            upstream_default_eplb "${ME_P1_PLACEMENT}" "${clocks}")"; then
        printf '%s\t%s\t%s\tcomplete\t%s\n' "${label}" upstream_default_eplb \
            "${clocks}" "${run_dir}" >> "${manifest}"
    else
        printf '%s\t%s\t%s\tfailed\t%s\n' "${label}" upstream_default_eplb \
            "${clocks}" "${run_dir:-}" >> "${manifest}"
        printf '%s\n' "${label}" >> "${controller}/failed-configurations.txt"
    fi
done

python -m src.gpu_control reset > "${controller}/final-clock-reset.json"
if [[ -s "${controller}/failed-configurations.txt" ]]; then
    printf 'one or more Sequential configurations failed; Joint planning will remain blocked\n' >&2
    exit 1
fi
complete_count="$(awk -F '\t' 'NR > 1 && $4 == "complete" {count++} END {print count+0}' "${manifest}")"
[[ "${complete_count}" -eq 81 ]] || {
    printf 'Sequential campaign has %s/81 complete configurations\n' "${complete_count}" >&2
    exit 1
}
printf '%s\n' "${controller}" > "${PROJECT_ROOT}/results/raw/milestone-e-sequential-latest.txt"
