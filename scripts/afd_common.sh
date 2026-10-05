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
    local ffn_timing_dir="${13:-}"
    local ffn_clocks_mhz="${14:-}"
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
    printf 'CUDA_VISIBLE_DEVICES=%q VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_PLUGINS=afd PYTHONPATH=%q AFD_STATIC_EXPERT_PLACEMENT_FILE=%q AFD_STATIC_EXPERT_PLACEMENT_EVIDENCE_DIR=%q AFD_ROUTING_PROFILE_DIR=%q AFD_FFN_TIMING_DIR=%q AFD_FFN_CLOCKS_MHZ=%q AFD_FFN_PHYSICAL_GPUS=2,3 ' \
        "${devices}" "${AFD_PROJECT_ROOT}" "${placement_file}" "${evidence_dir}" \
        "${routing_profile_dir}" "${ffn_timing_dir}" "${ffn_clocks_mhz}" > "${command_file}"
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
        AFD_FFN_TIMING_DIR="${ffn_timing_dir}" \
        AFD_FFN_CLOCKS_MHZ="${ffn_clocks_mhz}" \
        AFD_FFN_PHYSICAL_GPUS=2,3 \
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
    local members=()
    mapfile -t members < <(pgrep -g "${pid}" 2>/dev/null || true)
    if [[ "${#members[@]}" -eq 0 ]]; then
        wait "${pid}" 2>/dev/null || true
        return 0
    fi

    # Validate every surviving PGID member. The group leader may already have
    # exited, so leader liveness is not a safe proxy for worker liveness.
    local member owner command pgid state environment executable cwd
    local expected_python
    expected_python="$(readlink -f "${AFD_PYTHON}")"
    for member in "${members[@]}"; do
        owner="$(ps -o user= -p "${member}" 2>/dev/null | tr -d ' ' || true)"
        command="$(ps -o command= -p "${member}" 2>/dev/null || true)"
        pgid="$(ps -o pgid= -p "${member}" 2>/dev/null | tr -d ' ' || true)"
        state="$(ps -o stat= -p "${member}" 2>/dev/null | tr -d ' ' || true)"
        if [[ -z "${owner}" && -z "${pgid}" && -z "${state}" ]]; then
            continue
        fi
        if [[ "${owner}" != "${USER}" || "${pgid}" != "${pid}" ]]; then
            printf 'refusing to stop unrecognized PGID %s member %s: %s\n' \
                "${pid}" "${member}" "${command}" >&2
            return 1
        fi
        if [[ "${state}" == Z* ]]; then
            continue
        fi
        environment=""
        executable=""
        cwd=""
        if [[ -r "/proc/${member}/environ" ]]; then
            environment="$(tr '\0' '\n' < "/proc/${member}/environ" 2>/dev/null || true)"
        fi
        [[ -e "/proc/${member}/exe" ]] && \
            executable="$(readlink -f "/proc/${member}/exe" 2>/dev/null || true)"
        [[ -e "/proc/${member}/cwd" ]] && \
            cwd="$(readlink -f "/proc/${member}/cwd" 2>/dev/null || true)"
        if [[ "${command}" != *"${AFD_PLUGIN_ROOT}/.venv"* ]] && \
           [[ "${environment}" != *"VLLM_PLUGINS=afd"* || \
              "${environment}" != *"PYTHONPATH=${AFD_PROJECT_ROOT}"* ]] && \
           [[ "${executable}" != "${expected_python}" || \
              "${cwd}" != "${AFD_PROJECT_ROOT}" ]]; then
            printf 'refusing to stop unexpected live PGID %s member %s: %s\n' \
                "${pid}" "${member}" "${command}" >&2
            return 1
        fi
    done

    kill -TERM -- "-${pid}" 2>/dev/null || true
    for _ in $(seq 1 120); do
        mapfile -t members < <(pgrep -g "${pid}" 2>/dev/null || true)
        [[ "${#members[@]}" -eq 0 ]] && {
            wait "${pid}" 2>/dev/null || true
            return 0
        }
        local live=0
        for member in "${members[@]}"; do
            state="$(ps -o stat= -p "${member}" 2>/dev/null | tr -d ' ' || true)"
            [[ "${state}" == Z* || -z "${state}" ]] || live=1
        done
        if [[ "${live}" -eq 0 ]]; then
            wait "${pid}" 2>/dev/null || true
            return 0
        fi
        sleep 0.25
    done
    kill -KILL -- "-${pid}" 2>/dev/null || true
    for _ in $(seq 1 40); do
        mapfile -t members < <(pgrep -g "${pid}" 2>/dev/null || true)
        [[ "${#members[@]}" -eq 0 ]] && break
        sleep 0.25
    done
    wait "${pid}" 2>/dev/null || true
    mapfile -t members < <(pgrep -g "${pid}" 2>/dev/null || true)
    for member in "${members[@]}"; do
        state="$(ps -o stat= -p "${member}" 2>/dev/null | tr -d ' ' || true)"
        if [[ -n "${state}" && "${state}" != Z* ]]; then
            printf 'live PGID %s member remains after SIGKILL: %s\n' \
                "${pid}" "${member}" >&2
            return 1
        fi
    done
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
