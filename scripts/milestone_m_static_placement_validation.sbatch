#!/usr/bin/env bash
# Validate default and two custom static AFD FFN placements at HHHH.
# Submit with MILESTONE_M_STATIC_STUDY=<prepared-dir> sbatch <this-file>.
#SBATCH --job-name=moe-m-static-hh
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=uranus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40s:4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=results/raw/slurm-milestone-m-static-hh-%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
export AFD_PROJECT_ROOT="${PROJECT_ROOT}"
source scripts/afd_common.sh

study_dir="${MILESTONE_M_STATIC_STUDY:?set MILESTONE_M_STATIC_STUDY}"
study_dir="$(cd "${study_dir}" && pwd)"
[[ -f "${study_dir}/manifest.json" ]] || { printf 'missing study manifest\n' >&2; exit 1; }
[[ ! -e "${study_dir}/slurm-job-id.txt" ]] || { printf 'study was already submitted\n' >&2; exit 1; }
printf '%s\n' "${SLURM_JOB_ID}" > "${study_dir}/slurm-job-id.txt"

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
    afd_reset_clocks > "${study_dir}/clock-reset.json" 2> "${study_dir}/clock-reset.err" || true
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv \
        > "${study_dir}/compute-apps-after.csv" 2>&1 || true
    if [[ "${patch_applied}" -eq 1 ]]; then
        scripts/manage_afd_static_placement_patch.sh revert \
            --root "${AFD_PLUGIN_ROOT}" > "${study_dir}/patch-revert.txt" 2>&1 || status=1
    fi
    printf '%s\n' "${status}" > "${study_dir}/exit-status.txt"
    exit "${status}"
}
trap cleanup EXIT INT TERM

afd_require_runtime
afd_check_allocation > "${study_dir}/slurm-allocation.txt"
git rev-parse HEAD > "${study_dir}/git-commit.txt"
git -C "${AFD_PLUGIN_ROOT}" rev-parse HEAD > "${study_dir}/afd-plugin-commit.txt"
scripts/manage_afd_static_placement_patch.sh verify --root "${AFD_PLUGIN_ROOT}" \
    > "${study_dir}/patch-state-before.txt"
sha256sum "${AFD_PLUGIN_ROOT}/afd_plugin/model_executor/models/qwen3_moe.py" \
    > "${study_dir}/plugin-source-before.sha256"
cp patches/afd-plugin-0.26.0-static-placement.json "${study_dir}/patch-manifest.json"
cp configs/milestone_m_afd_fixed_frequency.yaml "${study_dir}/resolved-config.yaml"
nvidia-smi topo -m > "${study_dir}/topology.txt"

scripts/manage_afd_static_placement_patch.sh apply --root "${AFD_PLUGIN_ROOT}" \
    > "${study_dir}/patch-apply.txt"
patch_applied=1
sha256sum "${AFD_PLUGIN_ROOT}/afd_plugin/model_executor/models/qwen3_moe.py" \
    > "${study_dir}/plugin-source-patched.sha256"

SUDO="${SUDO:-sudo}" "${AFD_PYTHON}" -m src.gpu_control set-all \
    --clocks 2520,2520,2520,2520 > "${study_dir}/clock-set.json"
"${AFD_PYTHON}" -m src.gpu_control list > "${study_dir}/clocks-hh.json"

run_configuration() {
    local config_id="$1"
    local placement_file="$2"
    local config_dir="${study_dir}/${config_id}"
    [[ ! -e "${config_dir}" ]] || { printf 'refusing to overwrite %s\n' "${config_dir}" >&2; return 1; }
    mkdir -p "${config_dir}/servers" "${config_dir}/ownership"
    printf '%s\n' "${placement_file:-default}" > "${config_dir}/placement-reference.txt"
    if [[ -n "${placement_file}" ]]; then
        sha256sum "${placement_file}" > "${config_dir}/placement.sha256"
    fi

    afd_start_gpu_sampler "${config_dir}/gpu-samples.csv"
    sampler_pid="${AFD_LAST_PID}"
    afd_start_role ffn 2,3 18101 Qwen/Qwen3-30B-A3B qwen3-moe-milestone-m 6249 \
        "${config_dir}/servers/ffn.log" "${config_dir}/servers/ffn-command.txt" \
        "${placement_file}" "${config_dir}/ownership"
    ffn_pid="${AFD_LAST_PID}"
    printf '%s\n' "${ffn_pid}" > "${config_dir}/servers/ffn.pid"
    sleep 5
    afd_start_role attention 0,1 18100 Qwen/Qwen3-30B-A3B qwen3-moe-milestone-m 6249 \
        "${config_dir}/servers/attention.log" "${config_dir}/servers/attention-command.txt" \
        "${placement_file}" "${config_dir}/ownership"
    attention_pid="${AFD_LAST_PID}"
    printf '%s\n' "${attention_pid}" > "${config_dir}/servers/attention.pid"

    if ! afd_wait_attention_api "${attention_pid}" "${ffn_pid}" 2400 \
            http://127.0.0.1:18100/v1/models; then
        tail -n 160 "${config_dir}/servers/attention.log" >&2 || true
        tail -n 160 "${config_dir}/servers/ffn.log" >&2 || true
        return 1
    fi
    curl --fail --silent http://127.0.0.1:18100/v1/models \
        > "${config_dir}/attention-models.json"
    grep -m1 'AFD FFN EngineCore started; workers run connector loop.' \
        "${config_dir}/servers/ffn.log" > "${config_dir}/ffn-ready.txt"
    nvidia-smi --query-gpu=index,clocks.current.sm,clocks.applications.graphics \
        --format=csv,noheader,nounits > "${config_dir}/actual-clocks-before-client.csv"

    "${AFD_PYTHON}" tools/smoke_client.py \
        --url http://127.0.0.1:18100 \
        --model qwen3-moe-milestone-m-attention \
        --prompts data/prompts/smoke.jsonl \
        --output "${config_dir}/responses-pass-1.jsonl"
    "${AFD_PYTHON}" tools/smoke_client.py \
        --url http://127.0.0.1:18100 \
        --model qwen3-moe-milestone-m-attention \
        --prompts data/prompts/smoke.jsonl \
        --output "${config_dir}/responses-pass-2.jsonl"

    afd_stop_gpu_sampler "${sampler_pid}"
    sampler_pid=""
    afd_stop_process_group "${attention_pid}"
    attention_pid=""
    afd_stop_process_group "${ffn_pid}"
    ffn_pid=""
    "${AFD_PYTHON}" tools/validate_afd_smoke.py \
        --first "${config_dir}/responses-pass-1.jsonl" \
        --second "${config_dir}/responses-pass-2.jsonl" \
        --gpu-samples "${config_dir}/gpu-samples.csv" \
        --output "${config_dir}/smoke-validation.json"
    printf 'PASS\n' > "${config_dir}/status.txt"
}

selected_csv="${MILESTONE_M_STATIC_CONFIGS:-default,balanced,different}"
IFS=',' read -r -a selected_configs <<< "${selected_csv}"
[[ "${#selected_configs[@]}" -gt 0 ]] || { printf 'no configurations selected\n' >&2; exit 1; }
validator_args=(--study-dir "${study_dir}")
for config_id in "${selected_configs[@]}"; do
    case "${config_id}" in
        default) placement_file="" ;;
        balanced) placement_file="${study_dir}/placements/balanced_round_robin.json" ;;
        different) placement_file="${study_dir}/placements/different_block.json" ;;
        *) printf 'unknown static placement configuration: %s\n' "${config_id}" >&2; exit 1 ;;
    esac
    run_configuration "${config_id}" "${placement_file}"
    validator_args+=(--config "${config_id}")
done

"${AFD_PYTHON}" -m tools.validate_milestone_m_static_placement \
    "${validator_args[@]}" > "${study_dir}/validation-summary.stdout"

scripts/manage_afd_static_placement_patch.sh revert --root "${AFD_PLUGIN_ROOT}" \
    > "${study_dir}/patch-revert.txt"
patch_applied=0
scripts/manage_afd_static_placement_patch.sh verify --root "${AFD_PLUGIN_ROOT}" \
    > "${study_dir}/patch-state-after.txt"
sha256sum "${AFD_PLUGIN_ROOT}/afd_plugin/model_executor/models/qwen3_moe.py" \
    > "${study_dir}/plugin-source-after.sha256"
"${AFD_PYTHON}" tools/check_milestone_m_environments.py \
    --old-python "${HOME}/miniforge3/envs/cuda-env/bin/python" \
    --afd-python "${AFD_PYTHON}" \
    --plugin-root "${AFD_PLUGIN_ROOT}" \
    --uv "${PROJECT_ROOT}/.tools/uv-bootstrap/bin/uv" \
    --output "${study_dir}/environment-after.json"
printf 'PASS\n' > "${study_dir}/status.txt"
