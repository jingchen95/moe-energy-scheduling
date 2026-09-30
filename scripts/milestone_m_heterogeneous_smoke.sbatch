#!/usr/bin/env bash
# P_MH at Attention HH / FFN MH correctness gate for the primary matrix.
# Submit with MILESTONE_M_HETERO_SMOKE=<prepared-dir> sbatch <this-file>.
#SBATCH --job-name=moe-m-mh-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=uranus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40s:4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=results/raw/slurm-milestone-m-mh-smoke-%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
export AFD_PROJECT_ROOT="${PROJECT_ROOT}"
source scripts/afd_common.sh

study="${MILESTONE_M_HETERO_SMOKE:?set MILESTONE_M_HETERO_SMOKE}"
study="$(cd "${study}" && pwd)"
manifest="${study}/manifest.json"
[[ -f "${manifest}" ]] || { printf 'missing smoke manifest\n' >&2; exit 1; }
[[ ! -e "${study}/slurm-job-id.txt" ]] || { printf 'study already submitted\n' >&2; exit 1; }
printf '%s\n' "${SLURM_JOB_ID}" > "${study}/slurm-job-id.txt"

mapfile -t plan_values < <("${AFD_PYTHON}" - "${manifest}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
print(p["placement"]["path"])
print(p["placement"]["sha256"])
print(p["placement"]["canonical_mapping_sha256"])
print(",".join(str(value) for value in p["target_clocks_mhz"]))
print(p["baseline_run"])
print(p["execution_mode"])
PY
)
placement_file="${PROJECT_ROOT}/${plan_values[0]}"
placement_sha="${plan_values[1]}"
mapping_sha="${plan_values[2]}"
clocks="${plan_values[3]}"
baseline_run="${plan_values[4]}"
execution_mode="${plan_values[5]}"
[[ "${execution_mode}" == "FULL_DECODE_ONLY" ]] || { printf 'unexpected mode\n' >&2; exit 1; }

attention_pid=""
ffn_pid=""
sampler_pid=""
patch_applied=0
cleanup() {
    status=$?
    trap - EXIT INT TERM
    afd_stop_gpu_sampler "${sampler_pid}" || true
    afd_stop_process_group "${attention_pid}" || true
    afd_stop_process_group "${ffn_pid}" || true
    afd_reset_clocks > "${study}/clock-reset.json" 2> "${study}/clock-reset.err" || status=1
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv \
        > "${study}/compute-apps-after.csv" 2>&1 || true
    if [[ "${patch_applied}" -eq 1 ]]; then
        scripts/manage_afd_static_placement_patch.sh revert --root "${AFD_PLUGIN_ROOT}" \
            > "${study}/patch-revert.txt" 2>&1 || status=1
    fi
    scripts/manage_afd_static_placement_patch.sh verify --root "${AFD_PLUGIN_ROOT}" \
        > "${study}/patch-state-after.txt" 2>&1 || status=1
    "${AFD_PYTHON}" tools/check_milestone_m_environments.py \
        --old-python "${HOME}/miniforge3/envs/cuda-env/bin/python" \
        --afd-python "${AFD_PYTHON}" \
        --plugin-root "${AFD_PLUGIN_ROOT}" \
        --uv "${PROJECT_ROOT}/.tools/uv-bootstrap/bin/uv" \
        --output "${study}/environment-after.json" || status=1
    printf '%s\n' "${status}" > "${study}/exit-status.txt"
    exit "${status}"
}
trap cleanup EXIT INT TERM

afd_require_runtime
afd_check_allocation > "${study}/slurm-allocation.txt"
nvidia-smi topo -m > "${study}/topology.txt"
git rev-parse HEAD > "${study}/git-commit.txt"
git -C "${AFD_PLUGIN_ROOT}" rev-parse HEAD > "${study}/afd-plugin-commit.txt"
sha256sum "${manifest}" "${placement_file}" > "${study}/immutable-input-hashes.txt"
scripts/manage_afd_static_placement_patch.sh verify --root "${AFD_PLUGIN_ROOT}" \
    > "${study}/patch-state-before.txt"
scripts/manage_afd_static_placement_patch.sh apply --root "${AFD_PLUGIN_ROOT}" \
    > "${study}/patch-apply.txt"
patch_applied=1
mkdir -p "${study}/servers" "${study}/ownership"

SUDO="${SUDO:-sudo}" "${AFD_PYTHON}" -m src.gpu_control set-all \
    --clocks "${clocks}" > "${study}/clock-set.json"
"${AFD_PYTHON}" -m src.gpu_control list > "${study}/clocks-requested.json"

afd_start_gpu_sampler "${study}/gpu-samples.csv"
sampler_pid="${AFD_LAST_PID}"
afd_start_role ffn 2,3 18301 Qwen/Qwen3-30B-A3B qwen3-moe-m-mh-smoke 6251 \
    "${study}/servers/ffn.log" "${study}/servers/ffn-command.txt" \
    "${placement_file}" "${study}/ownership" "" full_decode_only
ffn_pid="${AFD_LAST_PID}"
printf '%s\n' "${ffn_pid}" > "${study}/servers/ffn.pid"
sleep 5
afd_start_role attention 0,1 18300 Qwen/Qwen3-30B-A3B qwen3-moe-m-mh-smoke 6251 \
    "${study}/servers/attention.log" "${study}/servers/attention-command.txt" \
    "${placement_file}" "${study}/ownership" "" full_decode_only
attention_pid="${AFD_LAST_PID}"
printf '%s\n' "${attention_pid}" > "${study}/servers/attention.pid"

if ! afd_wait_attention_api "${attention_pid}" "${ffn_pid}" 2400 \
        http://127.0.0.1:18300/v1/models; then
    tail -n 200 "${study}/servers/attention.log" >&2 || true
    tail -n 200 "${study}/servers/ffn.log" >&2 || true
    exit 1
fi
grep -m1 'AFD FFN EngineCore started; workers run connector loop.' \
    "${study}/servers/ffn.log" > "${study}/ffn-ready.txt"
nvidia-smi --query-gpu=index,clocks.current.sm,clocks.applications.graphics \
    --format=csv,noheader,nounits > "${study}/actual-clocks-before-client.csv"

"${AFD_PYTHON}" tools/smoke_client.py \
    --url http://127.0.0.1:18300 \
    --model qwen3-moe-m-mh-smoke-attention \
    --prompts data/prompts/smoke.jsonl \
    --output "${study}/responses-pass-1.jsonl"
"${AFD_PYTHON}" tools/smoke_client.py \
    --url http://127.0.0.1:18300 \
    --model qwen3-moe-m-mh-smoke-attention \
    --prompts data/prompts/smoke.jsonl \
    --output "${study}/responses-pass-2.jsonl"

afd_stop_gpu_sampler "${sampler_pid}"
sampler_pid=""
afd_stop_process_group "${attention_pid}"
attention_pid=""
afd_stop_process_group "${ffn_pid}"
ffn_pid=""

"${AFD_PYTHON}" -m tools.validate_milestone_m_runtime_ownership \
    --placement-file "${placement_file}" \
    --evidence-dir "${study}/ownership" \
    --output "${study}/actual-ownership.json" > "${study}/ownership-validation.stdout"
"${AFD_PYTHON}" -m tools.validate_milestone_m_heterogeneous_smoke \
    --study-dir "${study}" > "${study}/validation.stdout"
grep -q "${placement_sha}" "${study}/validation-summary.json"
grep -q "${mapping_sha}" "${study}/validation-summary.json"
printf 'PASS\n' > "${study}/status.txt"
