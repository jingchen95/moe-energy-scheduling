#!/usr/bin/env bash
#SBATCH --job-name=moe-milestone-c
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40s:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=results/raw/slurm-milestone-c-%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
SERVER_RUN="${PROJECT_ROOT}/results/raw/server-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${SERVER_RUN}"
pid_file="${SERVER_RUN}/vllm.pid"
server_pid=""

cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [[ -n "${server_pid}" ]]; then
        "${PROJECT_ROOT}/scripts/stop_vllm.sh" "${pid_file}" || true
    fi
    python -m src.gpu_control reset >> "${SERVER_RUN}/clock-reset.log" 2>&1 || true
    printf '%s\n' "${status}" > "${SERVER_RUN}/exit-status.txt"
    exit "${status}"
}
trap cleanup EXIT INT TERM

cd "${PROJECT_ROOT}"
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate cuda-env
export CUDA_VISIBLE_DEVICES=0,1,2,3

allocated_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
[[ "${allocated_gpus}" -eq 4 ]] || { printf 'expected 4 allocated GPUs, found %s\n' "${allocated_gpus}" >&2; exit 1; }

scripts/check_environment.sh > "${SERVER_RUN}/environment.txt" 2>&1
python -m src.config configs/model_qwen3_30b_a3b.yaml > "${SERVER_RUN}/vllm-command.txt"
setsid scripts/serve_vllm.sh > "${SERVER_RUN}/vllm.log" 2>&1 &
server_pid=$!
printf '%s\n' "${server_pid}" > "${pid_file}"

ready=0
for _ in $(seq 1 270); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        tail -n 120 "${SERVER_RUN}/vllm.log" >&2 || true
        exit 1
    fi
    if curl --fail --silent http://127.0.0.1:8000/health >/dev/null; then
        ready=1
        break
    fi
    sleep 10
done
[[ "${ready}" -eq 1 ]] || { printf 'timed out waiting for vLLM\n' >&2; exit 1; }

python -m src.gpu_control set-max > "${SERVER_RUN}/clock-lock.json"
python -m tools.run_milestone_c \
    --general data/traces/general-pool-r1.jsonl \
    --code data/traces/code-pool-r1.jsonl \
    --math data/traces/math-pool-r1.jsonl \
    --server-pid-file "${pid_file}" \
    --server-log "${SERVER_RUN}/vllm.log" \
    2>&1 | tee -a "${SERVER_RUN}/benchmark.log"
