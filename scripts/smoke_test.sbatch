#!/usr/bin/env bash
#SBATCH --job-name=moe-ep-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40s:4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=results/raw/slurm-smoke-%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate cuda-env

run_id="smoke-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${PROJECT_ROOT}/results/raw/${run_id}"
mkdir -p "${run_dir}"
pid_file="${run_dir}/vllm.pid"
server_pid=""

cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [[ -n "${server_pid}" ]]; then
        "${PROJECT_ROOT}/scripts/stop_vllm.sh" "${pid_file}" || true
    fi
    printf '%s\n' "${status}" > "${run_dir}/exit-status.txt"
    exit "${status}"
}
trap cleanup EXIT INT TERM

scripts/check_environment.sh > "${run_dir}/environment.txt" 2>&1
python -m src.config configs/model_qwen3_30b_a3b.yaml > "${run_dir}/vllm-command.txt"

export CUDA_VISIBLE_DEVICES=0,1,2,3
setsid scripts/serve_vllm.sh > "${run_dir}/vllm.log" 2>&1 &
server_pid=$!
printf '%s\n' "${server_pid}" > "${pid_file}"

ready=0
for _ in $(seq 1 270); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        printf 'vLLM exited before becoming ready\n' >&2
        tail -n 120 "${run_dir}/vllm.log" >&2 || true
        exit 1
    fi
    if curl --fail --silent --show-error http://127.0.0.1:8000/health >/dev/null; then
        ready=1
        break
    fi
    sleep 10
done
[[ "${ready}" -eq 1 ]] || { printf 'timed out waiting for vLLM\n' >&2; exit 1; }

python tools/smoke_client.py \
    --url http://127.0.0.1:8000 \
    --model qwen3-30b-a3b \
    --prompts data/prompts/smoke.jsonl \
    --output "${run_dir}/responses.jsonl"
printf '%s\n' "PASS" > "${run_dir}/smoke-status.txt"
