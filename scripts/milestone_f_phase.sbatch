#!/usr/bin/env bash
#SBATCH --job-name=moe-f-phase
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=uranus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40s:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=results/raw/slurm-milestone-f-phase-%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate cuda-env
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_SERVER_DEV_MODE=1
export VLLM_EXPERT_PHASE_PROFILE=1

controller="${PROJECT_ROOT}/results/raw/milestone-f-phase-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${controller}/server" "${controller}/runs"
pid_file="${controller}/server/vllm.pid"
server_pid=""
phase_patch_active=0
base_patch_owned=0
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
    if [[ "${phase_patch_active}" -eq 1 ]]; then
        scripts/manage_phase_patch.sh revert --root "${site_root}" \
            > "${controller}/phase-patch-revert.log" 2>&1 || true
    fi
    if [[ "${base_patch_owned}" -eq 1 ]]; then
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

trace="${MILESTONE_F_TRACE:-${PROJECT_ROOT}/results/raw/milestone-d-255131-20260826T104006Z/gsm8k-50-r0.5.jsonl}"
[[ -f "${trace}" ]] || { printf 'missing accepted trace: %s\n' "${trace}" >&2; exit 1; }
cp "${trace}" "${controller}/trace.jsonl"
sha256sum "${trace}" "${controller}/trace.jsonl" > "${controller}/trace-hash.txt"
printf '%s\n' "${trace}" > "${controller}/trace-source.txt"

base_state="$(scripts/manage_vllm_patch.sh verify --root "${site_root}")"
printf '%s\n' "${base_state}" > "${controller}/base-patch-state-before.txt"
[[ "${base_state}" == *'mixed'* ]] && { printf 'base patch state is mixed\n' >&2; exit 1; }
[[ "${base_state}" == *'not applied'* ]] && base_patch_owned=1
scripts/manage_phase_patch.sh apply --root "${site_root}" \
    > "${controller}/phase-patch-apply.log"
phase_patch_active=1
scripts/manage_phase_patch.sh verify --root "${site_root}" \
    > "${controller}/phase-patch-state-active.txt"
sha256sum patches/vllm-0.15.1-phase-*.patch patches/vllm-0.15.1-phase-profile.json \
    > "${controller}/phase-patch-hashes.txt"

python -m src.config configs/model_qwen3_30b_a3b_profile.yaml -- \
    > "${controller}/server/vllm-command.txt"
setsid scripts/serve_vllm.sh --config configs/model_qwen3_30b_a3b_profile.yaml \
    -- --enforce-eager \
    > "${controller}/server/vllm.log" 2>&1 &
server_pid=$!
printf '%s\n' "${server_pid}" > "${pid_file}"
ready=0
for _ in $(seq 1 270); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        tail -n 160 "${controller}/server/vllm.log" >&2 || true
        exit 1
    fi
    if curl --fail --silent http://127.0.0.1:8000/health >/dev/null; then
        ready=1
        break
    fi
    sleep 10
done
[[ "${ready}" -eq 1 ]] || { printf 'timed out waiting for vLLM\n' >&2; exit 1; }

python -m src.gpu_control set-all --clocks 2520,2520,2520,2520 \
    > "${controller}/clock-high.json"
benchmark_log="${controller}/benchmark.log"
python -m src.benchmark \
    --trace "${controller}/trace.jsonl" \
    --request-rate 0.5 \
    --output-root "${controller}/runs" \
    --local-files-only \
    --run-label f-phase-profile \
    --placement-name default_linear \
    --vllm-patch-manifest patches/vllm-0.15.1-phase-profile.json \
    --target-clocks 2520 2520 2520 2520 \
    --expert-profile \
    --store-output-token-ids \
    --server-pid-file "${pid_file}" \
    --server-log "${controller}/server/vllm.log" \
    > "${benchmark_log}" 2>&1
run_dir="$(python -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1])["run_dir"])' "${benchmark_log}")"
cp "${benchmark_log}" "${run_dir}/stdout-stderr.log"
python -m tools.validate_live_run "${run_dir}" --require-expert-profile \
    > "${controller}/live-validation.json"
printf '%s\n' "${run_dir}" > "${controller}/profile-run-dir.txt"
python -m src.gpu_control reset > "${controller}/final-clock-reset.json"
