#!/usr/bin/env bash
# Milestone I extensions to the validated Milestone E/H serving helpers.

source scripts/milestone_e_common.sh

mi_run_benchmark() {
    local controller="$1"
    local label="$2"
    local placement_name="$3"
    local placement_file="$4"
    local clocks="$5"
    local request_rate="$6"
    local trace="$7"
    local output="${controller}/${label}.benchmark.log"
    local args=(
        --trace "${trace}"
        --request-rate "${request_rate}"
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
    if ! python -m src.benchmark "${args[@]}" > "${output}" 2>&1; then
        return 1
    fi
    local run_dir
    run_dir="$(python -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1])["run_dir"])' "${output}")"
    cp "${output}" "${run_dir}/stdout-stderr.log"
    cp "${ME_EXPERT_COUNTS}" "${run_dir}/frozen_expert_counts.npz"
    cp "${placement_file}" "${run_dir}/expert_mapping.json"
    sha256sum "${trace}" "${ME_EXPERT_COUNTS}" "${placement_file}" \
        > "${run_dir}/immutable-input-hashes.txt"
    printf '%s\n' "${ME_PROFILE_RUN}" > "${run_dir}/frozen-profile-source.txt"
    python -m tools.validate_live_run "${run_dir}" > "${run_dir}/live-validation.json"
    printf '%s\n' "${run_dir}"
}
