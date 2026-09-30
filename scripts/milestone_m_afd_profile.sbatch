#!/usr/bin/env bash
# Profiling-only logical expert counts for the validated Qwen3 2A2F path.
# Submit with MILESTONE_M_PROFILE_STUDY=<prepared-dir> sbatch <this-file>.
#SBATCH --job-name=moe-m-afd-profile
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=uranus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40s:4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=results/raw/slurm-milestone-m-afd-profile-%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
export AFD_PROJECT_ROOT="${PROJECT_ROOT}"
source scripts/afd_common.sh

study_dir="${MILESTONE_M_PROFILE_STUDY:?set MILESTONE_M_PROFILE_STUDY}"
study_dir="$(cd "${study_dir}" && pwd)"
[[ -f "${study_dir}/manifest.json" ]] || { printf 'missing profile manifest\n' >&2; exit 1; }
[[ ! -e "${study_dir}/slurm-job-id.txt" ]] || { printf 'study was already submitted\n' >&2; exit 1; }
printf '%s\n' "${SLURM_JOB_ID}" > "${study_dir}/slurm-job-id.txt"

attention_pid=""
ffn_pid=""
patch_applied=0
cleanup() {
    status=$?
    trap - EXIT INT TERM
    afd_stop_process_group "${attention_pid}" || true
    afd_stop_process_group "${ffn_pid}" || true
    afd_reset_clocks > "${study_dir}/clock-reset.json" 2> "${study_dir}/clock-reset.err" || status=1
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv \
        > "${study_dir}/compute-apps-after.csv" 2>&1 || true
    if [[ "${patch_applied}" -eq 1 ]]; then
        scripts/manage_afd_profile_patch.sh revert --root "${AFD_PLUGIN_ROOT}" \
            > "${study_dir}/patch-revert.txt" 2>&1 || status=1
    fi
    printf '%s\n' "${status}" > "${study_dir}/exit-status.txt"
    exit "${status}"
}
trap cleanup EXIT INT TERM

afd_require_runtime
afd_check_allocation > "${study_dir}/slurm-allocation.txt"
git rev-parse HEAD > "${study_dir}/git-commit.txt"
git -C "${AFD_PLUGIN_ROOT}" rev-parse HEAD > "${study_dir}/afd-plugin-commit.txt"
scripts/manage_afd_profile_patch.sh verify --root "${AFD_PLUGIN_ROOT}" \
    > "${study_dir}/patch-state-before.txt"
cp patches/afd-plugin-0.26.0-static-profile.json "${study_dir}/patch-manifest.json"
cp configs/milestone_m_afd_fixed_frequency.yaml "${study_dir}/resolved-config.yaml"
nvidia-smi topo -m > "${study_dir}/topology.txt"

scripts/manage_afd_profile_patch.sh apply --root "${AFD_PLUGIN_ROOT}" \
    > "${study_dir}/patch-apply.txt"
patch_applied=1
mkdir -p "${study_dir}/servers" "${study_dir}/ownership" \
    "${study_dir}/profile-control" "${study_dir}/runs"
"${AFD_PYTHON}" -c \
    'from pathlib import Path; from src.afd_routing_profile import write_profile_control; import sys; write_profile_control(Path(sys.argv[1]), "startup", False)' \
    "${study_dir}/profile-control"

SUDO="${SUDO:-sudo}" "${AFD_PYTHON}" -m src.gpu_control set-all \
    --clocks 2520,2520,2520,2520 > "${study_dir}/clock-set.json"
"${AFD_PYTHON}" -m src.gpu_control list > "${study_dir}/clocks-hh.json"

afd_start_role ffn 2,3 18201 Qwen/Qwen3-30B-A3B qwen3-moe-m-profile 6250 \
    "${study_dir}/servers/ffn.log" "${study_dir}/servers/ffn-command.txt" \
    "" "${study_dir}/ownership" "${study_dir}/profile-control"
ffn_pid="${AFD_LAST_PID}"
printf '%s\n' "${ffn_pid}" > "${study_dir}/servers/ffn.pid"
sleep 5
afd_start_role attention 0,1 18200 Qwen/Qwen3-30B-A3B qwen3-moe-m-profile 6250 \
    "${study_dir}/servers/attention.log" "${study_dir}/servers/attention-command.txt" \
    "" "${study_dir}/ownership" "${study_dir}/profile-control"
attention_pid="${AFD_LAST_PID}"
printf '%s\n' "${attention_pid}" > "${study_dir}/servers/attention.pid"

if ! afd_wait_attention_api "${attention_pid}" "${ffn_pid}" 2400 \
        http://127.0.0.1:18200/v1/models; then
    tail -n 160 "${study_dir}/servers/attention.log" >&2 || true
    tail -n 160 "${study_dir}/servers/ffn.log" >&2 || true
    exit 1
fi
grep -m1 'AFD FFN EngineCore started; workers run connector loop.' \
    "${study_dir}/servers/ffn.log" > "${study_dir}/ffn-ready.txt"
nvidia-smi --query-gpu=index,clocks.current.sm --format=csv,noheader,nounits \
    > "${study_dir}/actual-clocks-before-client.csv"

"${AFD_PYTHON}" -m src.benchmark \
    --trace "${study_dir}/trace.jsonl" \
    --base-url http://127.0.0.1:18200 \
    --model qwen3-moe-m-profile-attention \
    --tokenizer Qwen/Qwen3-30B-A3B \
    --request-rate 16 \
    --output-root "${study_dir}/runs" \
    --warmup-requests 8 \
    --settle-before-s 2 \
    --settle-after-s 2 \
    --monitor-interval-s 0.2 \
    --max-concurrency 256 \
    --request-timeout-s 900 \
    --local-files-only \
    --afd-profile-control-dir "${study_dir}/profile-control" \
    --target-clocks 2520 2520 2520 2520 \
    --store-output-token-ids \
    --run-label afd-default-hh-routing-profile \
    --placement-name default_linear \
    --server-log "${study_dir}/servers/attention.log" \
    > "${study_dir}/benchmark.log" 2>&1

mapfile -t run_dirs < <(find "${study_dir}/runs" -mindepth 1 -maxdepth 1 -type d | sort)
[[ "${#run_dirs[@]}" -eq 1 ]] || { printf 'expected exactly one benchmark run\n' >&2; exit 1; }
run_dir="${run_dirs[0]}"
printf '%s\n' "${run_dir}" > "${study_dir}/run-dir.txt"
cp "${study_dir}/benchmark.log" "${run_dir}/stdout-stderr.log"
sha256sum "${study_dir}/trace.jsonl" "${run_dir}/expert_counts.npz" \
    > "${run_dir}/immutable-input-output-hashes.txt"
"${AFD_PYTHON}" -m tools.validate_live_run "${run_dir}" --require-expert-profile \
    > "${run_dir}/live-validation.json"
"${AFD_PYTHON}" -m tools.validate_milestone_m_afd_profile \
    --study-dir "${study_dir}" --run-dir "${run_dir}" \
    > "${run_dir}/profile-validation.stdout"

afd_stop_process_group "${attention_pid}"
attention_pid=""
afd_stop_process_group "${ffn_pid}"
ffn_pid=""

scripts/manage_afd_profile_patch.sh revert --root "${AFD_PLUGIN_ROOT}" \
    > "${study_dir}/patch-revert.txt"
patch_applied=0
scripts/manage_afd_profile_patch.sh verify --root "${AFD_PLUGIN_ROOT}" \
    > "${study_dir}/patch-state-after.txt"
"${AFD_PYTHON}" tools/check_milestone_m_environments.py \
    --old-python "${HOME}/miniforge3/envs/cuda-env/bin/python" \
    --afd-python "${AFD_PYTHON}" \
    --plugin-root "${AFD_PLUGIN_ROOT}" \
    --uv "${PROJECT_ROOT}/.tools/uv-bootstrap/bin/uv" \
    --output "${study_dir}/environment-after.json"
printf 'PASS\n' > "${study_dir}/status.txt"
