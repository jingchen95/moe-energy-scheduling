#!/usr/bin/env bash
# Shared safety helpers for the isolated vLLM AFD runtime.

AFD_PROJECT_ROOT="${AFD_PROJECT_ROOT:-$(pwd)}"
AFD_PLUGIN_ROOT="${AFD_PLUGIN_ROOT:-${AFD_PROJECT_ROOT}/third_party/afd-plugin}"
AFD_PYTHON="${AFD_PYTHON:-${AFD_PLUGIN_ROOT}/.venv/bin/python}"
AFD_VLLM="${AFD_VLLM:-${AFD_PLUGIN_ROOT}/.venv/bin/vllm}"
AFD_LAST_PID=""

afd_require_runtime() {
    [[ -x "${AFD_PYTHON}" ]] || { printf 'missing AFD Python: %s\n' "${AFD_PYTHON}" >&2; return 1; }
    [[ -x "${AFD_VLLM}" ]] || { printf 'missing AFD vLLM: %s\n' "${AFD_VLLM}" >&2; return 1; }
    [[ "${AFD_PYTHON}" != *'/cuda-env/'* ]] || {
        printf 'refusing protected cuda-env interpreter: %s\n' "${AFD_PYTHON}" >&2
        return 1
    }
}

afd_check_allocation() {
    [[ -n "${SLURM_JOB_ID:-}" ]] || { printf 'AFD GPU work requires Slurm\n' >&2; return 1; }
    case "$(hostname -s)" in
        uranus|neptune) ;;
        *) printf 'unsupported node: %s\n' "$(hostname -s)" >&2; return 1 ;;
    esac
    local gpu_count
    gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
    [[ "${gpu_count}" -eq 4 ]] || { printf 'expected four visible GPUs, got %s\n' "${gpu_count}" >&2; return 1; }
    scontrol show job -o "${SLURM_JOB_ID}"
}

afd_start_role() {
    local role="$1"
    local devices="$2"
    local api_port="$3"
    local model="$4"
    local served_prefix="$5"
    local afd_port="$6"
    local log_file="$7"
    local command_file="$8"
    local placement_file="${9:-}"
    local evidence_dir="${10:-}"
    local routing_profile_dir="${11:-}"
    local execution_mode="${12:-eager}"
    local additional_config
    additional_config="$(printf '{\"afd\":{\"role\":\"%s\",\"connector\":\"P2pNcclAFDConnector\",\"host\":\"127.0.0.1\",\"port\":%s,\"num_attention_ranks\":2,\"num_ffn_ranks\":2,\"compute_gate_on_attention\":false}}' "${role}" "${afd_port}")"
    local command=(
        "${AFD_VLLM}" serve "${model}"
        --served-model-name "${served_prefix}-${role}"
        --data-parallel-size 2
        --tensor-parallel-size 1
        --enable-expert-parallel
        --dtype bfloat16
        --max-model-len 1024
        --seed 0
        --host 127.0.0.1
        --port "${api_port}"
        --shutdown-timeout 20
        --additional-config "${additional_config}"
    )
    case "${execution_mode}" in
        eager)
            command+=(--enforce-eager)
            ;;
        full_decode_only)
            command+=(
                --max-num-seqs 64
                --max-num-batched-tokens 64
                --max-cudagraph-capture-size 64
                --compilation-config
                '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[64]}'
            )
            ;;
        *)
            printf 'unsupported AFD execution mode: %s\n' "${execution_mode}" >&2
            return 1
            ;;
    esac
    printf 'CUDA_VISIBLE_DEVICES=%q VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_PLUGINS=afd PYTHONPATH=%q AFD_STATIC_EXPERT_PLACEMENT_FILE=%q AFD_STATIC_EXPERT_PLACEMENT_EVIDENCE_DIR=%q AFD_ROUTING_PROFILE_DIR=%q ' \
        "${devices}" "${AFD_PROJECT_ROOT}" "${placement_file}" "${evidence_dir}" "${routing_profile_dir}" > "${command_file}"
    printf '%q ' "${command[@]}" >> "${command_file}"
    printf '\n' >> "${command_file}"
    env -u PYTHONPATH -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV \
        CUDA_VISIBLE_DEVICES="${devices}" \
        VLLM_USE_V2_MODEL_RUNNER=0 \
        VLLM_USE_FLASHINFER_SAMPLER=0 \
        VLLM_PLUGINS=afd \
        PYTHONPATH="${AFD_PROJECT_ROOT}" \
        AFD_STATIC_EXPERT_PLACEMENT_FILE="${placement_file}" \
        AFD_STATIC_EXPERT_PLACEMENT_EVIDENCE_DIR="${evidence_dir}" \
        AFD_ROUTING_PROFILE_DIR="${routing_profile_dir}" \
        PYTHONUNBUFFERED=1 \
        HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}" \
        setsid "${command[@]}" > "${log_file}" 2>&1 &
    AFD_LAST_PID=$!
}

afd_wait_attention_api() {
    local attention_pid="$1"
    local ffn_pid="$2"
    local timeout_s="$3"
    local attention_models_url="$4"
    local deadline=$((SECONDS + timeout_s))
    while (( SECONDS < deadline )); do
        kill -0 "${attention_pid}" 2>/dev/null || return 1
        kill -0 "${ffn_pid}" 2>/dev/null || return 1
        if curl --fail --silent "${attention_models_url}" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    printf 'timed out waiting for Attention API: %s\n' \
        "${attention_models_url}" >&2
    return 1
}

afd_stop_process_group() {
    local pid="${1:-}"
    [[ "${pid}" =~ ^[0-9]+$ ]] || return 0
    kill -0 "${pid}" 2>/dev/null || { wait "${pid}" 2>/dev/null || true; return 0; }
    local owner command
    owner="$(ps -o user= -p "${pid}" | tr -d ' ')"
    command="$(ps -o command= -p "${pid}")"
    if [[ "${owner}" != "${USER}" || "${command}" != *"${AFD_PLUGIN_ROOT}/.venv"* ]]; then
        printf 'refusing to stop unrecognized process %s: %s\n' "${pid}" "${command}" >&2
        return 1
    fi
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    for _ in $(seq 1 120); do
        pgrep -g "${pid}" >/dev/null 2>&1 || { wait "${pid}" 2>/dev/null || true; return 0; }
        sleep 1
    done
    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
}

afd_start_gpu_sampler() {
    local output="$1"
    (
        printf 'timestamp_unix_s,gpu,memory_used_mib,sm_util_percent,sm_clock_mhz,power_w\n'
        while true; do
            local_timestamp="$(date +%s.%N)"
            nvidia-smi \
                --query-gpu=index,memory.used,utilization.gpu,clocks.current.sm,power.draw \
                --format=csv,noheader,nounits | sed "s/^/${local_timestamp},/"
            sleep 2
        done
    ) > "${output}" &
    AFD_LAST_PID=$!
}

afd_stop_gpu_sampler() {
    local pid="${1:-}"
    [[ "${pid}" =~ ^[0-9]+$ ]] || return 0
    kill -TERM "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
}

afd_reset_clocks() {
    SUDO="${SUDO:-sudo}" "${AFD_PYTHON}" -m src.gpu_control reset
}
