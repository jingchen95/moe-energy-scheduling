#!/usr/bin/env bash
# Per-frequency-server, fixed-placement symmetric FFN frequency sweep.
# Submit with MILESTONE_M_SYMMETRIC_STUDY=<prepared-dir> sbatch <this-file>.
#SBATCH --job-name=moe-m-afd-symmetric
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=uranus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40s:4
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=results/raw/slurm-milestone-m-symmetric-%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
export AFD_PROJECT_ROOT="${PROJECT_ROOT}"
source scripts/afd_common.sh

study="${MILESTONE_M_SYMMETRIC_STUDY:?set MILESTONE_M_SYMMETRIC_STUDY}"
study="$(cd "${study}" && pwd)"
plan="${study}/manifest.json"
trace="${study}/trace.jsonl"
[[ -f "${plan}" && -f "${trace}" ]] || { printf 'missing symmetric-sweep inputs\n' >&2; exit 1; }
expected_commit="$("${AFD_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["git_commit"])' "${plan}")"
actual_commit="$(git rev-parse HEAD)"
[[ "${actual_commit}" == "${expected_commit}" ]] || {
    printf 'repository commit drift: manifest=%s runtime=%s\n' \
        "${expected_commit}" "${actual_commit}" >&2
    exit 1
}
git diff --quiet && git diff --cached --quiet || {
    printf 'refusing to run from a modified tracked worktree\n' >&2
    exit 1
}
[[ ! -e "${study}/slurm-job-id.txt" ]] || { printf 'study already submitted\n' >&2; exit 1; }
printf '%s\n' "${SLURM_JOB_ID}" > "${study}/slurm-job-id.txt"

attention_pid=""
ffn_pid=""
patch_applied=0
cleanup() {
    status=$?
    trap - EXIT INT TERM
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
sha256sum "${plan}" "${trace}" data/placements/milestone_m/*.json \
    > "${study}/immutable-input-hashes.txt"
cp configs/milestone_m_afd_fixed_frequency.yaml "${study}/resolved-config.yaml"
scripts/manage_afd_static_placement_patch.sh verify --root "${AFD_PLUGIN_ROOT}" \
    > "${study}/patch-state-before.txt"
scripts/manage_afd_static_placement_patch.sh apply --root "${AFD_PLUGIN_ROOT}" \
    > "${study}/patch-apply.txt"
patch_applied=1

placement_path="$("${AFD_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["placement"]["path"])' "${plan}")"
placement_sha="$("${AFD_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["placement"]["sha256"])' "${plan}")"
mapping_sha="$("${AFD_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["placement"]["canonical_mapping_sha256"])' "${plan}")"
placement_file="${PROJECT_ROOT}/${placement_path}"
request_rate="$("${AFD_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["request_rate"])' "${plan}")"
expected_runs="$("${AFD_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["expected_runs"])' "${plan}")"

mkdir -p "${study}/configurations"
printf 'order\tconfig_id\tfrequency_mhz\trepetition\tstatus\trun_dir\n' > "${study}/runs.tsv"
sha256sum "${placement_file}" > "${study}/placement.sha256"
[[ "$(sha256sum "${placement_file}" | awk '{print $1}')" == "${placement_sha}" ]] || {
    printf 'P_balanced placement hash drift\n' >&2
    exit 1
}

api_port=18460
ffn_api_port=18461
afd_port=6290

while IFS=$'\t' read -r order config_id frequency clocks repetitions; do
    config_dir="${study}/configurations/$(printf '%02d' "${order}")-${config_id}"
    mkdir -p "${config_dir}/conditioning" "${config_dir}/runs" \
        "${config_dir}/server" "${config_dir}/ownership"
    printf '%s\n' "${clocks}" > "${config_dir}/target-clocks.csv"
    IFS=',' read -r -a target_array <<< "${clocks}"
    IFS=',' read -r -a repetition_array <<< "${repetitions}"

    SUDO="${SUDO:-sudo}" "${AFD_PYTHON}" -m src.gpu_control set-all \
        --clocks "${clocks}" > "${config_dir}/clock-set.json"
    "${AFD_PYTHON}" -m src.gpu_control list > "${config_dir}/clock-readback.json"
    sleep 5
    nvidia-smi --query-gpu=index,clocks.current.sm --format=csv,noheader,nounits \
        > "${config_dir}/actual-clocks-before-conditioning.csv"

    served_name="qwen3-moe-m-symmetric-${config_id}"
    afd_start_role ffn 2,3 "${ffn_api_port}" Qwen/Qwen3-30B-A3B "${served_name}" "${afd_port}" \
        "${config_dir}/server/ffn.log" "${config_dir}/server/ffn-command.txt" \
        "${placement_file}" "${config_dir}/ownership" "" full_decode_only
    ffn_pid="${AFD_LAST_PID}"
    printf '%s\n' "${ffn_pid}" > "${config_dir}/server/ffn.pid"
    sleep 5
    afd_start_role attention 0,1 "${api_port}" Qwen/Qwen3-30B-A3B "${served_name}" "${afd_port}" \
        "${config_dir}/server/attention.log" "${config_dir}/server/attention-command.txt" \
        "${placement_file}" "${config_dir}/ownership" "" full_decode_only
    attention_pid="${AFD_LAST_PID}"
    printf '%s\n' "${attention_pid}" > "${config_dir}/server/attention.pid"

    if ! afd_wait_attention_api "${attention_pid}" "${ffn_pid}" 2400 \
            "http://127.0.0.1:${api_port}/v1/models"; then
        tail -n 200 "${config_dir}/server/attention.log" >&2 || true
        tail -n 200 "${config_dir}/server/ffn.log" >&2 || true
        exit 1
    fi
    grep -m1 'AFD FFN EngineCore started; workers run connector loop.' \
        "${config_dir}/server/ffn.log" > "${config_dir}/ffn-ready.txt"
    "${AFD_PYTHON}" -m tools.validate_milestone_m_runtime_ownership \
        --placement-file "${placement_file}" \
        --evidence-dir "${config_dir}/ownership" \
        --output "${config_dir}/actual-ownership.json" \
        > "${config_dir}/ownership-validation.stdout"
    actual_mapping_sha="$("${AFD_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["actual_mapping_sha256"])' "${config_dir}/actual-ownership.json")"
    [[ "${actual_mapping_sha}" == "${mapping_sha}" ]] || {
        printf 'runtime ownership hash mismatch for %s\n' "${config_id}" >&2
        exit 1
    }

    conditioning_root="${config_dir}/conditioning"
    "${AFD_PYTHON}" -m src.benchmark \
        --trace "${trace}" \
        --base-url "http://127.0.0.1:${api_port}" \
        --model "${served_name}-attention" \
        --tokenizer Qwen/Qwen3-30B-A3B \
        --request-rate "${request_rate}" \
        --override-arrival-rate \
        --output-root "${conditioning_root}" \
        --warmup-requests 8 \
        --settle-before-s 5 \
        --settle-after-s 5 \
        --monitor-interval-s 0.2 \
        --max-concurrency 256 \
        --request-timeout-s 900 \
        --local-files-only \
        --placement-file "${placement_file}" \
        --vllm-patch-manifest patches/afd-plugin-0.26.0-static-placement.json \
        --target-clocks "${target_array[@]}" \
        --run-label "${config_id}-${frequency}-conditioning" \
        --placement-name P_balanced \
        --server-log "${config_dir}/server/attention.log" \
        > "${config_dir}/conditioning.log" 2>&1
    mapfile -t conditioning_runs < <(find "${conditioning_root}" -mindepth 1 -maxdepth 1 -type d | sort)
    [[ "${#conditioning_runs[@]}" -eq 1 ]] || {
        printf 'expected one conditioning run for %s\n' "${config_id}" >&2
        exit 1
    }
    "${AFD_PYTHON}" - "${conditioning_runs[0]}/metrics.json" <<'PY'
import json, sys
metrics = json.load(open(sys.argv[1]))
if metrics["completed_requests"] != 300 or metrics["failed_requests"] != 0:
    raise SystemExit("conditioning trace did not complete cleanly")
PY

    for repetition in "${repetition_array[@]}"; do
        rep_root="${config_dir}/runs/rep-${repetition}"
        mkdir -p "${rep_root}"
        benchmark_log="${config_dir}/benchmark-rep-${repetition}.log"
        "${AFD_PYTHON}" -m src.benchmark \
            --trace "${trace}" \
            --base-url "http://127.0.0.1:${api_port}" \
            --model "${served_name}-attention" \
            --tokenizer Qwen/Qwen3-30B-A3B \
            --request-rate "${request_rate}" \
            --override-arrival-rate \
            --output-root "${rep_root}" \
            --warmup-requests 8 \
            --settle-before-s 5 \
            --settle-after-s 5 \
            --monitor-interval-s 0.2 \
            --max-concurrency 256 \
            --request-timeout-s 900 \
            --local-files-only \
            --placement-file "${placement_file}" \
            --vllm-patch-manifest patches/afd-plugin-0.26.0-static-placement.json \
            --target-clocks "${target_array[@]}" \
            --store-output-token-ids \
            --run-label "${config_id}-${frequency}-rep${repetition}" \
            --placement-name P_balanced \
            --server-log "${config_dir}/server/attention.log" \
            > "${benchmark_log}" 2>&1
        mapfile -t run_dirs < <(find "${rep_root}" -mindepth 1 -maxdepth 1 -type d | sort)
        [[ "${#run_dirs[@]}" -eq 1 ]] || { printf 'expected one run directory\n' >&2; exit 1; }
        run_dir="${run_dirs[0]}"
        cp "${benchmark_log}" "${run_dir}/stdout-stderr.log"
        cp "${config_dir}/actual-ownership.json" "${run_dir}/runtime-ownership.json"
        "${AFD_PYTHON}" -m tools.record_milestone_m_run_metadata \
            --plan "${plan}" --ownership "${config_dir}/actual-ownership.json" \
            --config-id "${config_id}" --repetition "${repetition}" \
            --output "${run_dir}/experiment_metadata.json"
        "${AFD_PYTHON}" -m tools.validate_live_run "${run_dir}" \
            > "${run_dir}/live-validation.json"
        "${AFD_PYTHON}" -m tools.validate_milestone_m_measured_run \
            --run-dir "${run_dir}" \
            --trace-sha256 "$("${AFD_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["trace_sha256"])' "${plan}")" \
            --placement-sha256 "${placement_sha}" \
            --mapping-sha256 "${mapping_sha}" \
            --target-clocks "${target_array[@]}" \
            > "${run_dir}/milestone-m-validation.stdout"
        printf '%s\t%s\t%s\t%s\tcomplete\t%s\n' \
            "${order}" "${config_id}" "${frequency}" "${repetition}" "${run_dir}" \
            >> "${study}/runs.tsv"
    done

    afd_stop_process_group "${attention_pid}"
    attention_pid=""
    afd_stop_process_group "${ffn_pid}"
    ffn_pid=""
    sleep 5
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory \
        --format=csv,noheader > "${config_dir}/compute-apps-after.csv"
    [[ ! -s "${config_dir}/compute-apps-after.csv" ]] || {
        printf 'GPU compute processes remain after %s shutdown\n' "${config_id}" >&2
        exit 1
    }
done < <("${AFD_PYTHON}" - "${plan}" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
for row in sorted(plan["configurations"], key=lambda item: item["order"]):
    print(
        row["order"], row["config_id"], row["frequency_mhz"],
        ",".join(str(value) for value in row["target_clocks_mhz"]),
        ",".join(str(value) for value in row["repetitions"]),
        sep="\t",
    )
PY
)

complete_count="$(awk -F '\t' '$5 == "complete" {n++} END {print n+0}' "${study}/runs.tsv")"
[[ "${complete_count}" -eq "${expected_runs}" ]] || {
    printf 'expected %s runs, got %s\n' "${expected_runs}" "${complete_count}" >&2
    exit 1
}
"${AFD_PYTHON}" - "${study}/result.json" "${complete_count}" <<'PY'
import json, sys
json.dump({"status": "complete", "completed_runs": int(sys.argv[2])}, open(sys.argv[1], "w"), indent=2)
PY
printf 'PASS\n' > "${study}/status.txt"
