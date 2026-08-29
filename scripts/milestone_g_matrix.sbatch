#!/usr/bin/env bash
# Submit as: sbatch --array=0-3%1 scripts/milestone_g_matrix.sbatch
# One array task owns one placement server and runs H/M/L sequentially.
# Repeats are limited to LOW/H, LOW/M, HIGH/H, HIGH/M, VERY_HIGH/H, and VERY_HIGH/M.
#SBATCH --job-name=moe-g-matrix
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=uranus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40s:4
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=results/raw/slurm-milestone-g-%A-%a.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate cuda-env
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_SERVER_DEV_MODE=1
export VLLM_EXPERT_PHASE_PROFILE=1
export VLLM_MOE_TIMING=1

plan="${MILESTONE_G_PLAN:?set MILESTONE_G_PLAN to a prepared matrix plan}"
index="${SLURM_ARRAY_TASK_ID:?submit this script as a four-task array}"
readarray -t values < <(python - "${plan}/matrix-plan.json" "${index}" <<'PY'
import json, sys
row = json.load(open(sys.argv[1]))["matrix"][int(sys.argv[2]) * 3]
print(row["placement"])
print(row["placement_file"])
print(row["trace"])
print(row["phase_counts"])
PY
)
placement_name="${values[0]}"
placement_file="${values[1]}"
trace="${values[2]}"
phase_counts="${values[3]}"
controller="${plan}/measured/${placement_name}-attempt-${SLURM_JOB_ID}"
mkdir -p "${controller}/runs"
pid_file="${controller}/server/vllm.pid"
server_pid=""
phase_active=0
timing_active=0

site_root="$(python - <<'PY'
import inspect
from pathlib import Path
import vllm
print(Path(inspect.getfile(vllm)).resolve().parent.parent)
PY
)"

cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [[ -n "${server_pid}" && -f "${pid_file}" ]]; then
        scripts/stop_vllm.sh "${pid_file}" || true
    fi
    python -m src.gpu_control reset > "${controller}/clock-reset.log" 2>&1 || true
    if [[ "${timing_active}" -eq 1 ]]; then
        scripts/manage_moe_timing_patch.sh revert --root "${site_root}" \
            > "${controller}/timing-patch-revert.log" 2>&1 || true
    fi
    if [[ "${phase_active}" -eq 1 ]]; then
        scripts/manage_phase_patch.sh revert --root "${site_root}" \
            > "${controller}/phase-patch-revert.log" 2>&1 || true
        scripts/manage_vllm_patch.sh revert --root "${site_root}" \
            > "${controller}/base-patch-revert.log" 2>&1 || true
    fi
    printf '%s\n' "${status}" > "${controller}/exit-status.txt"
    exit "${status}"
}
trap cleanup EXIT INT TERM

[[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')" -eq 4 ]]
scripts/check_environment.sh > "${controller}/environment.txt" 2>&1
printf '%s\n' "$(git rev-parse HEAD)" > "${controller}/git_commit.txt"
python - <<'PY' > "${controller}/vllm-version.txt"
import vllm
print(vllm.__version__)
PY
sha256sum "${plan}/matrix-plan.json" "${placement_file}" "${trace}" "${phase_counts}" \
    > "${controller}/immutable-input-hashes.txt"
printf '%s\n' "${placement_file}" > "${controller}/placement-source.txt"
printf '%s\n' "${trace}" > "${controller}/trace-source.txt"
printf '%s\n' "${phase_counts}" > "${controller}/phase-counts-source.txt"

base_state="$(scripts/manage_vllm_patch.sh verify --root "${site_root}")"
[[ "${base_state}" == *'not applied'* ]] || {
    printf 'base patch must be unpatched before G job: %s\n' "${base_state}" >&2
    exit 1
}
scripts/manage_phase_patch.sh apply --root "${site_root}" > "${controller}/phase-patch-apply.log"
phase_active=1
scripts/manage_moe_timing_patch.sh apply --root "${site_root}" \
    > "${controller}/timing-patch-apply.log"
timing_active=1
scripts/manage_phase_patch.sh verify --root "${site_root}" > "${controller}/phase-patch-state-active.txt"
scripts/manage_moe_timing_patch.sh verify --root "${site_root}" \
    > "${controller}/timing-patch-state-active.txt"
sha256sum patches/vllm-0.15.1-phase-*.patch patches/vllm-0.15.1-phase-profile.json \
    patches/vllm-0.15.1-moe-timing.patch \
    > "${controller}/patch-hashes.txt"

python -m src.config configs/model_qwen3_30b_a3b_profile.yaml -- \
    --expert-placement-file "${placement_file}" > "${controller}/server-vllm-command.txt"
setsid scripts/serve_vllm.sh --config configs/model_qwen3_30b_a3b_profile.yaml \
    -- --expert-placement-file "${placement_file}" --enforce-eager \
    > "${controller}/server-vllm.log" 2>&1 &
server_pid=$!
printf '%s\n' "${server_pid}" > "${pid_file}"
ready=0
for _ in $(seq 1 270); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        tail -n 200 "${controller}/server-vllm.log" >&2 || true
        exit 1
    fi
    if curl --fail --silent http://127.0.0.1:8000/health >/dev/null; then
        ready=1
        break
    fi
    sleep 10
done
[[ "${ready}" -eq 1 ]] || { printf 'timed out waiting for vLLM\n' >&2; exit 1; }

printf 'label\tfrequency_tuple\trun_dir\n' > "${controller}/runs.tsv"
run_one() {
    local frequency="$1"
    local label="$2"
    local run_log="${controller}/${label}.benchmark.log"
    python -m src.gpu_control set-all --clocks "${frequency},2520,2520,2520" \
        > "${controller}/clock-${label}.json"
    python -m src.benchmark \
        --trace "${trace}" --request-rate 0.5 \
        --output-root "${controller}/runs" --local-files-only \
        --run-label "${label}" --placement-name "${placement_name}" \
        --placement-file "${placement_file}" \
        --vllm-patch-manifest patches/vllm-0.15.1-phase-profile.json \
        --target-clocks "${frequency}" 2520 2520 2520 \
        --expert-profile --store-output-token-ids \
        --server-pid-file "${pid_file}" --server-log "${controller}/server-vllm.log" \
        > "${run_log}" 2>&1
    local run_dir
    run_dir="$(python -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1])["run_dir"])' "${run_log}")"
    cp "${run_log}" "${run_dir}/stdout-stderr.log"
    cp patches/vllm-0.15.1-moe-timing.patch "${run_dir}/moe-timing-patch.patch"
    cp patches/vllm-0.15.1-moe-timing.json "${run_dir}/moe-timing-patch-manifest.json"
    python -m tools.validate_live_run "${run_dir}" --require-expert-profile \
        > "${run_dir}/live-validation.json"
    for rank in 0 1 2 3; do
        [[ -f "${run_dir}/moe_timing_rank${rank}.json" ]] || {
            printf 'missing timing dump for rank %s in %s\n' "${rank}" "${run_dir}" >&2
            return 1
        }
    done
    printf '%s\t%s\t%s\n' "${label}" "${frequency},2520,2520,2520" "${run_dir}" \
        >> "${controller}/runs.tsv"
}

for frequency in 2520 1635 1005; do
    run_one "${frequency}" "${placement_name}-gpu0-${frequency}-initial"
done

if [[ "${placement_name}" == "controlled_low" ||
      "${placement_name}" == "controlled_high" ||
      "${placement_name}" == "controlled_very_high" ]]; then
    for frequency in 2520 1635; do
        for repeat in 1 2; do
            run_one "${frequency}" "${placement_name}-gpu0-${frequency}-repeat${repeat}"
        done
    done
fi

python -m src.gpu_control reset > "${controller}/final-clock-reset.json"
printf '%s\n' "complete" > "${controller}/status.txt"

