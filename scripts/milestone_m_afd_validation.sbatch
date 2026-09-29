#!/usr/bin/env bash
# Unmodified pinned-plugin Qwen3 2A2F eager smoke. No placement patch.
# Submit with: sbatch scripts/milestone_m_afd_validation.sbatch
#SBATCH --job-name=moe-m-afd-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=uranus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40s:4
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=results/raw/slurm-milestone-m-afd-smoke-%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
export AFD_PROJECT_ROOT="${PROJECT_ROOT}"
source scripts/afd_common.sh

run_id="milestone-m-afd-smoke-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${PROJECT_ROOT}/results/raw/${run_id}"
mkdir -p "${run_dir}/servers"

attention_pid=""
ffn_pid=""
sampler_pid=""
cleanup() {
    status=$?
    trap - EXIT INT TERM
    afd_stop_gpu_sampler "${sampler_pid}" || true
    afd_stop_process_group "${attention_pid}" || true
    afd_stop_process_group "${ffn_pid}" || true
    afd_reset_clocks > "${run_dir}/clock-reset.json" 2> "${run_dir}/clock-reset.err" || true
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv \
        > "${run_dir}/compute-apps-after.csv" 2>&1 || true
    printf '%s\n' "${status}" > "${run_dir}/exit-status.txt"
    exit "${status}"
}
trap cleanup EXIT INT TERM

afd_require_runtime
afd_check_allocation > "${run_dir}/slurm-allocation.txt"
git rev-parse HEAD > "${run_dir}/git-commit.txt"
git -C "${AFD_PLUGIN_ROOT}" rev-parse HEAD > "${run_dir}/afd-plugin-commit.txt"
git -C "${AFD_PLUGIN_ROOT}" status --porcelain > "${run_dir}/afd-plugin-status-before.txt"
[[ ! -s "${run_dir}/afd-plugin-status-before.txt" ]] || { printf 'AFD plugin checkout is dirty\n' >&2; exit 1; }
cp configs/milestone_m_afd_fixed_frequency.yaml "${run_dir}/resolved-config.yaml"
cp configs/milestone_m_environment.json "${run_dir}/environment-baseline.json"
cat > "${run_dir}/role-mapping.json" <<'JSON'
{
  "attention": {"physical_gpus": [0, 1], "api_port": 18100, "frequency_policy": "untouched"},
  "ffn": {"physical_gpus": [2, 3], "api_port": 18101, "frequency_policy": "untouched"},
  "connector": {"name": "P2pNcclAFDConnector", "port": 6249},
  "compute_gate_on_attention": false
}
JSON
nvidia-smi -q > "${run_dir}/nvidia-smi-before.txt"
nvidia-smi topo -m > "${run_dir}/topology.txt"
"${AFD_PYTHON}" -m src.gpu_control list > "${run_dir}/clocks-before.json"
"${AFD_PYTHON}" tools/check_milestone_m_environments.py \
    --old-python "${HOME}/miniforge3/envs/cuda-env/bin/python" \
    --afd-python "${AFD_PYTHON}" \
    --plugin-root "${AFD_PLUGIN_ROOT}" \
    --uv "${PROJECT_ROOT}/.tools/uv-bootstrap/bin/uv" \
    --output "${run_dir}/environment-before.json"

afd_start_gpu_sampler "${run_dir}/gpu-samples.csv"
sampler_pid="${AFD_LAST_PID}"

afd_start_role ffn 2,3 18101 Qwen/Qwen3-30B-A3B qwen3-moe-milestone-m 6249 \
    "${run_dir}/servers/ffn.log" "${run_dir}/servers/ffn-command.txt"
ffn_pid="${AFD_LAST_PID}"
printf '%s\n' "${ffn_pid}" > "${run_dir}/servers/ffn.pid"
sleep 5
afd_start_role attention 0,1 18100 Qwen/Qwen3-30B-A3B qwen3-moe-milestone-m 6249 \
    "${run_dir}/servers/attention.log" "${run_dir}/servers/attention-command.txt"
attention_pid="${AFD_LAST_PID}"
printf '%s\n' "${attention_pid}" > "${run_dir}/servers/attention.pid"

if ! afd_wait_attention_api "${attention_pid}" "${ffn_pid}" 2400 \
        http://127.0.0.1:18100/v1/models; then
    tail -n 160 "${run_dir}/servers/attention.log" >&2 || true
    tail -n 160 "${run_dir}/servers/ffn.log" >&2 || true
    exit 1
fi
curl --fail --silent http://127.0.0.1:18100/v1/models > "${run_dir}/attention-models.json"
grep -m1 'AFD FFN EngineCore started; workers run connector loop.' \
    "${run_dir}/servers/ffn.log" > "${run_dir}/ffn-ready.txt"

"${AFD_PYTHON}" tools/smoke_client.py \
    --url http://127.0.0.1:18100 \
    --model qwen3-moe-milestone-m-attention \
    --prompts data/prompts/smoke.jsonl \
    --output "${run_dir}/responses-pass-1.jsonl"
"${AFD_PYTHON}" tools/smoke_client.py \
    --url http://127.0.0.1:18100 \
    --model qwen3-moe-milestone-m-attention \
    --prompts data/prompts/smoke.jsonl \
    --output "${run_dir}/responses-pass-2.jsonl"

"${AFD_PYTHON}" tools/validate_afd_smoke.py \
    --first "${run_dir}/responses-pass-1.jsonl" \
    --second "${run_dir}/responses-pass-2.jsonl" \
    --gpu-samples "${run_dir}/gpu-samples.csv" \
    --output "${run_dir}/validation.json"

afd_stop_gpu_sampler "${sampler_pid}"
sampler_pid=""
afd_stop_process_group "${attention_pid}"
attention_pid=""
afd_stop_process_group "${ffn_pid}"
ffn_pid=""
afd_reset_clocks > "${run_dir}/clock-reset.json"
"${AFD_PYTHON}" -m src.gpu_control list > "${run_dir}/clocks-after.json"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv \
    > "${run_dir}/compute-apps-after.csv"
git -C "${AFD_PLUGIN_ROOT}" status --porcelain > "${run_dir}/afd-plugin-status-after.txt"
[[ ! -s "${run_dir}/afd-plugin-status-after.txt" ]] || { printf 'AFD plugin became dirty\n' >&2; exit 1; }
"${AFD_PYTHON}" tools/check_milestone_m_environments.py \
    --old-python "${HOME}/miniforge3/envs/cuda-env/bin/python" \
    --afd-python "${AFD_PYTHON}" \
    --plugin-root "${AFD_PLUGIN_ROOT}" \
    --uv "${PROJECT_ROOT}/.tools/uv-bootstrap/bin/uv" \
    --output "${run_dir}/environment-after.json"
printf 'PASS\n' > "${run_dir}/smoke-status.txt"
printf '%s\n' "${run_dir}"
