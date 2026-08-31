#!/usr/bin/env bash
# Shared functions for Milestone E Slurm jobs. This file never submits work.

me_site_root() {
    python - <<'PY'
import inspect
from pathlib import Path
import vllm
print(Path(inspect.getfile(vllm)).resolve().parent.parent)
PY
}

me_check_allocation() {
    local allocated_gpus
    allocated_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
    [[ "${allocated_gpus}" -eq 4 ]] || {
        printf 'expected 4 allocated GPUs, found %s\n' "${allocated_gpus}" >&2
        return 1
    }
}

me_resolve_milestone_d_inputs() {
    local d_controller="$1"
    [[ -f "${d_controller}/runs.tsv" ]] || {
        printf 'Milestone D controller has no runs.tsv: %s\n' "${d_controller}" >&2
        return 1
    }
    ME_TRACE="${d_controller}/gsm8k-50-r0.5.jsonl"
    ME_P1_PLACEMENT="${d_controller}/placements/upstream_default_eplb.json"
    ME_PROFILE_RUN="$(awk -F '\t' '$1 == "p0-frozen-profile" {print $4; exit}' \
        "${d_controller}/runs.tsv")"
    ME_EXPERT_COUNTS="${ME_PROFILE_RUN}/expert_counts.npz"
    ME_FROZEN_MAPPING="${ME_PROFILE_RUN}/expert_mapping.json"
    for file in "${ME_TRACE}" "${ME_P1_PLACEMENT}" "${ME_EXPERT_COUNTS}" \
        "${ME_FROZEN_MAPPING}"; do
        [[ -f "${file}" ]] || { printf 'missing accepted Milestone D input: %s\n' "${file}" >&2; return 1; }
    done
    export ME_TRACE ME_P1_PLACEMENT ME_PROFILE_RUN ME_EXPERT_COUNTS ME_FROZEN_MAPPING
}

me_activate_patch() {
    local controller="$1"
    ME_SITE_ROOT="$(me_site_root)"
    ME_PATCH_APPLIED_BY_JOB=0
    local state
    state="$(scripts/manage_vllm_patch.sh verify --root "${ME_SITE_ROOT}")"
    printf '%s\n' "${state}" > "${controller}/patch-state-before.txt"
    if [[ "${state}" == *'not applied'* ]]; then
        scripts/manage_vllm_patch.sh apply --root "${ME_SITE_ROOT}" \
            > "${controller}/patch-apply.log"
        ME_PATCH_APPLIED_BY_JOB=1
    fi
    scripts/manage_vllm_patch.sh verify --root "${ME_SITE_ROOT}" \
        > "${controller}/patch-state-active.txt"
    sha256sum patches/vllm-0.15.1-placement-profile.* > "${controller}/patch-hashes.txt"
    export ME_SITE_ROOT ME_PATCH_APPLIED_BY_JOB
}

me_revert_patch_if_owned() {
    local controller="$1"
    if [[ "${ME_PATCH_APPLIED_BY_JOB:-0}" -eq 1 ]]; then
        scripts/manage_vllm_patch.sh revert --root "${ME_SITE_ROOT}" \
            >> "${controller}/patch-revert.log" 2>&1 || true
    fi
}

me_launch_server() {
    local controller="$1"
    local name="$2"
    local placement_file="$3"
    local server_dir="${controller}/server"
    mkdir -p "${server_dir}"
    ME_PID_FILE="${server_dir}/vllm.pid"
    local -a placement_args=()
    if [[ -n "${placement_file}" ]]; then
        placement_args=(-- --expert-placement-file "${placement_file}")
    fi
    python -m src.config configs/model_qwen3_30b_a3b_profile.yaml \
        "${placement_args[@]}" > "${server_dir}/vllm-command.txt"
    setsid scripts/serve_vllm.sh --config configs/model_qwen3_30b_a3b_profile.yaml \
        "${placement_args[@]}" > "${server_dir}/vllm.log" 2>&1 &
    ME_SERVER_PID=$!
    printf '%s\n' "${ME_SERVER_PID}" > "${ME_PID_FILE}"
    local ready=0
    for _ in $(seq 1 270); do
        if ! kill -0 "${ME_SERVER_PID}" 2>/dev/null; then
            tail -n 160 "${server_dir}/vllm.log" >&2 || true
            return 1
        fi
        if curl --fail --silent http://127.0.0.1:8000/health >/dev/null; then
            ready=1
            break
        fi
        sleep 10
    done
    [[ "${ready}" -eq 1 ]] || { printf 'timed out waiting for vLLM (%s)\n' "${name}" >&2; return 1; }
    export ME_PID_FILE ME_SERVER_PID
}

me_stop_server() {
    if [[ -n "${ME_SERVER_PID:-}" && -n "${ME_PID_FILE:-}" ]]; then
        scripts/stop_vllm.sh "${ME_PID_FILE}" || true
        ME_SERVER_PID=""
        ME_PID_FILE=""
    fi
}

me_set_clocks() {
    local controller="$1"
    local label="$2"
    local clocks="$3"
    python -m src.gpu_control set-all --clocks "${clocks}" \
        > "${controller}/clock-${label}.json"
}

me_run_benchmark() {
    local controller="$1"
    local label="$2"
    local placement_name="$3"
    local placement_file="$4"
    local clocks="$5"
    local output="${controller}/${label}.benchmark.log"
    local args=(
        --trace "${ME_TRACE}"
        --request-rate 0.5
        --output-root "${controller}/runs"
        --local-files-only
        --run-label "${label}"
        --placement-name "${placement_name}"
        --placement-file "${placement_file}"
        --vllm-patch-manifest patches/vllm-0.15.1-placement-profile.json
        --target-clocks ${clocks//,/ }
        --store-output-token-ids
        --server-pid-file "${ME_PID_FILE}"
        --server-log "${controller}/server/vllm.log"
    )
    # Deliberately no --expert-profile: Milestone E consumes the frozen profile.
    if ! python -m src.benchmark "${args[@]}" > "${output}" 2>&1; then
        return 1
    fi
    local run_dir
    run_dir="$(python -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1])["run_dir"])' "${output}")"
    cp "${output}" "${run_dir}/stdout-stderr.log"
    cp "${ME_EXPERT_COUNTS}" "${run_dir}/frozen_expert_counts.npz"
    cp "${placement_file}" "${run_dir}/expert_mapping.json"
    sha256sum "${ME_TRACE}" "${ME_EXPERT_COUNTS}" "${placement_file}" \
        > "${run_dir}/immutable-input-hashes.txt"
    printf '%s\n' "${ME_PROFILE_RUN}" > "${run_dir}/frozen-profile-source.txt"
    printf '%s\n' "${run_dir}"
    python -m tools.validate_live_run "${run_dir}" > "${run_dir}/live-validation.json"
}
