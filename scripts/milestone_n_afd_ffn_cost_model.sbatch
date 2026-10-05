#!/usr/bin/env bash
# One-allocation same-run routing and CUDA-event characterization for AFD FFN.
# Submit with MILESTONE_N_STUDY=<prepared-dir> sbatch <this-file>.
#SBATCH --job-name=moe-n-afd-cost
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=uranus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40s:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=results/raw/slurm-milestone-n-afd-cost-%j.out

set -Eeuo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
export AFD_PROJECT_ROOT="${PROJECT_ROOT}"
source scripts/afd_common.sh

study="${MILESTONE_N_STUDY:?set MILESTONE_N_STUDY}"
study="$(cd "${study}" && pwd)"
plan="${study}/manifest.json"
trace="${study}/trace.jsonl"
smoke_trace="${study}/profiled-mini-smoke-trace.jsonl"
[[ -f "${plan}" && -f "${trace}" && -f "${smoke_trace}" ]] || {
    printf 'missing prepared study input\n' >&2; exit 1;
}
expected_commit="$("${AFD_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["git_commit"])' "${plan}")"
actual_commit="$(git rev-parse HEAD)"
[[ "${expected_commit}" == "${actual_commit}" ]] || {
    printf 'repository commit drift: manifest=%s runtime=%s\n' "${expected_commit}" "${actual_commit}" >&2
    exit 1
}
git diff --quiet && git diff --cached --quiet || {
    printf 'refusing modified tracked worktree\n' >&2; exit 1;
}
input_study="${study}"
mkdir -p "${input_study}/allocations"
study="${input_study}/allocations/${SLURM_JOB_ID}"
mkdir "${study}"
cp "${plan}" "${study}/manifest.json"
cp "${input_study}/reuse-manifest.json" "${study}/reuse-manifest.json"
printf '%s\n' "${SLURM_JOB_ID}" > "${study}/slurm-job-id.txt"

attention_pid=""
ffn_pid=""
patch_applied=0
failure_stage="initialization"
failure_line=""
failure_command=""
record_error() {
    local status="$1"
    if [[ -z "${failure_line}" ]]; then
        failure_line="$2"
        failure_command="$3"
    fi
    return "${status}"
}
trap 'record_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
cleanup() {
    status=$?
    trap - EXIT INT TERM ERR
    afd_stop_process_group "${attention_pid}" || true
    afd_stop_process_group "${ffn_pid}" || true
    afd_reset_clocks > "${study}/clock-reset.json" 2> "${study}/clock-reset.err" || status=1
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv \
        > "${study}/compute-apps-after.csv" 2>&1 || true
    if [[ "${patch_applied}" -eq 1 ]]; then
        scripts/manage_afd_cost_model_patch.sh revert --root "${AFD_PLUGIN_ROOT}" \
            > "${study}/patch-revert.txt" 2>&1 || status=1
    fi
    scripts/manage_afd_cost_model_patch.sh verify --root "${AFD_PLUGIN_ROOT}" \
        > "${study}/patch-state-after.txt" 2>&1 || status=1
    "${AFD_PYTHON}" tools/check_milestone_m_environments.py \
        --old-python "${HOME}/miniforge3/envs/cuda-env/bin/python" \
        --afd-python "${AFD_PYTHON}" --plugin-root "${AFD_PLUGIN_ROOT}" \
        --uv "${PROJECT_ROOT}/.tools/uv-bootstrap/bin/uv" \
        --output "${study}/environment-after.json" || status=1
    if [[ "${status}" -ne 0 ]]; then
        printf 'stage\t%s\nline\t%s\ncommand\t%s\n' \
            "${failure_stage}" "${failure_line}" "${failure_command}" \
            > "${study}/failure-context.tsv"
    fi
    printf '%s\n' "${status}" > "${study}/exit-status.txt"
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

afd_require_runtime
afd_check_allocation > "${study}/slurm-allocation.txt"
nvidia-smi topo -m > "${study}/topology.txt"
git rev-parse HEAD > "${study}/git-commit.txt"
git -C "${AFD_PLUGIN_ROOT}" rev-parse HEAD > "${study}/afd-plugin-commit.txt"
sha256sum "${plan}" "${trace}" "${smoke_trace}" \
    data/placements/milestone_m/P_balanced.json \
    data/placements/milestone_m_target_splits/P_split_40_slow_rank0.json \
    data/placements/milestone_m_target_splits/P_split_40_slow_rank1.json \
    patches/afd-plugin-0.26.0-cost-model.* > "${study}/immutable-input-hashes.txt"
scripts/manage_afd_cost_model_patch.sh verify --root "${AFD_PLUGIN_ROOT}" \
    > "${study}/patch-state-before.txt"
scripts/manage_afd_cost_model_patch.sh apply --root "${AFD_PLUGIN_ROOT}" \
    > "${study}/patch-apply.txt"
patch_applied=1

api_port=18560
ffn_api_port=18561
afd_port=6310
trace_sha="$("${AFD_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["trace_sha256"])' "${plan}")"
smoke_trace_sha="$("${AFD_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["profiled_mini_smoke"]["trace_sha256"])' "${plan}")"
[[ "$(sha256sum "${smoke_trace}" | awk '{print $1}')" == "${smoke_trace_sha}" ]] || {
    printf 'profiled mini-smoke trace hash drift\n' >&2; exit 1;
}
balanced="${PROJECT_ROOT}/data/placements/milestone_m/P_balanced.json"
patch_sha="$(sha256sum patches/afd-plugin-0.26.0-cost-model.patch | awk '{print $1}')"
expected_patch_sha="$("${AFD_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["patch_sha256"])' "${plan}")"
[[ "${patch_sha}" == "${expected_patch_sha}" && "$(sha256sum "${trace}" | awk '{print $1}')" == "${trace_sha}" ]] || {
    printf 'immutable patch/trace hash drift\n' >&2; exit 1;
}

stop_server() {
    afd_stop_process_group "${attention_pid}"
    attention_pid=""
    afd_stop_process_group "${ffn_pid}"
    ffn_pid=""
    sleep 5
    local apps_file="$1"
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader > "${apps_file}"
    [[ ! -s "${apps_file}" ]] || { printf 'GPU processes remain after shutdown\n' >&2; exit 1; }
}

validate_ownership() {
    local placement="$1" evidence="$2" output="$3"
    "${AFD_PYTHON}" -m tools.validate_milestone_m_runtime_ownership \
        --placement-file "${placement}" --evidence-dir "${evidence}" \
        --output "${output}" > "${output%.json}.stdout"
}

start_profile_server() {
    local config_dir="$1" model_name="$2" placement="$3" frequency="$4" mapping_sha="$5"
    mkdir -p "${config_dir}/server" "${config_dir}/ownership" \
        "${config_dir}/timing-control" "${config_dir}/conditioning" "${config_dir}/runs"
    local clocks="2520,2520,${frequency},${frequency}"
    SUDO="${SUDO:-sudo}" "${AFD_PYTHON}" -m src.gpu_control set-all \
        --clocks "${clocks}" --defer-current-verification > "${config_dir}/clock-set.json"
    "${AFD_PYTHON}" -m src.gpu_control list > "${config_dir}/clock-readback.json"
    sleep 5
    nvidia-smi --query-gpu=index,clocks.current.sm --format=csv,noheader,nounits \
        > "${config_dir}/actual-clocks-before-conditioning.csv"
    afd_start_role ffn 2,3 "${ffn_api_port}" Qwen/Qwen3-30B-A3B "${model_name}" "${afd_port}" \
        "${config_dir}/server/ffn.log" "${config_dir}/server/ffn-command.txt" \
        "${placement}" "${config_dir}/ownership" "" full_decode_only \
        "${config_dir}/timing-control" "${frequency},${frequency}"
    ffn_pid="${AFD_LAST_PID}"
    sleep 5
    afd_start_role attention 0,1 "${api_port}" Qwen/Qwen3-30B-A3B "${model_name}" "${afd_port}" \
        "${config_dir}/server/attention.log" "${config_dir}/server/attention-command.txt" \
        "${placement}" "${config_dir}/ownership" "" full_decode_only \
        "${config_dir}/timing-control" "${frequency},${frequency}"
    attention_pid="${AFD_LAST_PID}"
    afd_wait_attention_api "${attention_pid}" "${ffn_pid}" 2400 \
        "http://127.0.0.1:${api_port}/v1/models"
    validate_ownership "${placement}" "${config_dir}/ownership" "${config_dir}/runtime-ownership.json"
    local actual_mapping_sha
    actual_mapping_sha="$("${AFD_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["actual_mapping_sha256"])' "${config_dir}/runtime-ownership.json")"
    [[ "${actual_mapping_sha}" == "${mapping_sha}" ]] || {
        printf 'ownership mismatch\n' >&2; return 1;
    }
}

run_conditioning() {
    local output_root="$1" model_name="$2" placement="$3" clocks_csv="$4" log="$5"
    IFS=',' read -r -a clocks_array <<< "${clocks_csv}"
    "${AFD_PYTHON}" -m src.benchmark \
        --trace "${trace}" --base-url "http://127.0.0.1:${api_port}" \
        --model "${model_name}-attention" --tokenizer Qwen/Qwen3-30B-A3B \
        --request-rate 8 --override-arrival-rate --output-root "${output_root}" \
        --warmup-requests 8 --settle-before-s 5 --settle-after-s 5 \
        --monitor-interval-s 0.2 --max-concurrency 256 --request-timeout-s 900 \
        --local-files-only --placement-file "${placement}" \
        --vllm-patch-manifest patches/afd-plugin-0.26.0-cost-model.json \
        --target-clocks "${clocks_array[@]}" --run-label conditioning \
        --placement-name conditioning --server-log "${log}" \
        > "${output_root}.log" 2>&1
    local count
    count="$(find "${output_root}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    [[ "${count}" -eq 1 ]] || { printf 'conditioning run count mismatch\n' >&2; exit 1; }
}

run_profiled_mini_smoke() {
    local config_dir="$1" model_name="$2" placement="$3" frequency="$4" clocks_csv="$5"
    local label="${6:-profiled-mini-smoke}"
    local output_root="${config_dir}/${label}"
    IFS=',' read -r -a clocks_array <<< "${clocks_csv}"
    "${AFD_PYTHON}" -m src.benchmark \
        --trace "${smoke_trace}" --base-url "http://127.0.0.1:${api_port}" \
        --model "${model_name}-attention" --tokenizer Qwen/Qwen3-30B-A3B \
        --request-rate 8 --override-arrival-rate --output-root "${output_root}" \
        --warmup-requests 2 --settle-before-s 1 --settle-after-s 1 \
        --monitor-interval-s 0.2 --max-concurrency 16 --request-timeout-s 900 \
        --local-files-only --placement-file "${placement}" \
        --vllm-patch-manifest patches/afd-plugin-0.26.0-cost-model.json \
        --afd-timing-control-dir "${config_dir}/timing-control" \
        --target-clocks "${clocks_array[@]}" --store-output-token-ids \
        --run-label profiled-mini-smoke --placement-name profiled-mini-smoke \
        --server-log "${config_dir}/server/attention.log" \
        > "${config_dir}/${label}.log" 2>&1
    mapfile -t smoke_runs < <(
        find "${output_root}" -mindepth 1 -maxdepth 1 -type d -print
    )
    [[ "${#smoke_runs[@]}" -eq 1 ]] || {
        printf 'profiled mini-smoke run count mismatch\n' >&2; exit 1;
    }
    local run_dir="${smoke_runs[0]}"
    "${AFD_PYTHON}" -m tools.validate_live_run "${run_dir}" \
        > "${run_dir}/live-validation.json"
    "${AFD_PYTHON}" -m tools.validate_milestone_n_cost_timing \
        --timing-run "${run_dir}" --placement "${placement}" \
        --config-id profiled-mini-smoke --lifecycle fail_fast_gate \
        --frequency-mhz "${frequency}" \
        > "${run_dir}/cost-timing-validation.json"
    "${AFD_PYTHON}" - "${run_dir}" "${smoke_trace_sha}" <<'PY'
import hashlib, json, sys
import numpy as np
from pathlib import Path
run_dir = Path(sys.argv[1])
metrics = json.loads((run_dir / "metrics.json").read_text())
if metrics["completed_requests"] != 8 or metrics["failed_requests"] != 0:
    raise SystemExit("profiled mini-smoke did not complete exactly 8/8 requests")
profile = run_dir / "afd_same_run_cost_profile.npz"
trace = run_dir / "trace.jsonl"
if hashlib.sha256(trace.read_bytes()).hexdigest() != sys.argv[2]:
    raise SystemExit("profiled mini-smoke trace hash mismatch")
final = json.loads((run_dir / "afd_same_run_cost_finalize.json").read_text())
if final["schema_version"] != 3 or not final["exact_same_run_rank_alignment"]:
    raise SystemExit("stress gate lacks shared runtime identity")
with np.load(profile, allow_pickle=False) as data:
    if not np.all(np.any(data["run_modes_by_rank"] == "replay", axis=1)):
        raise SystemExit("stress gate did not exercise CUDA graph replay")
result = {
    "status": "pass",
    "completed_requests": 8,
    "graph_replay_timing_finalization_and_rank_alignment": True,
    "run_dir": str(run_dir),
    "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
    "trace_sha256": sys.argv[2],
    "first_runtime_step_id": final["first_runtime_step_id"],
    "last_runtime_step_id": final["last_runtime_step_id"],
    "runtime_steps": final["runtime_steps"],
    "diagnostic_counter_ranges_by_rank": final["diagnostic_counter_ranges_by_rank"],
}
(run_dir.parent / "PASS.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
}

printf 'stage\torder\tconfig_id\tlifecycle\tfrequency_mhz\tstatus\trun_dir\n' > "${study}/runs.tsv"

# Phase A: eight enable/disable generations on one server, never model data.
failure_stage="phase-a:startup"
gate_dir="${study}/phase-a-stress"
mkdir "${gate_dir}"
balanced_mapping_sha="$("${AFD_PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["configurations"][0]["placement"]["canonical_mapping_sha256"])' "${plan}")"
start_profile_server "${gate_dir}" qwen3-moe-n-stress "${balanced}" 2520 "${balanced_mapping_sha}"
for generation in {0..7}; do
    failure_stage="phase-a:generation-${generation}"
    run_profiled_mini_smoke "${gate_dir}" qwen3-moe-n-stress "${balanced}" 2520 \
        2520,2520,2520,2520 "window-$(printf '%02d' "${generation}")"
done
failure_stage="phase-a:shutdown"
stop_server "${gate_dir}/compute-apps-after.csv"
"${AFD_PYTHON}" - "${gate_dir}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = [json.loads((root / f"window-{i:02d}" / "PASS.json").read_text()) for i in range(8)]
for previous, current in zip(rows, rows[1:]):
    if current["first_runtime_step_id"] <= previous["last_runtime_step_id"]:
        raise SystemExit("runtime identity reset across profiling generations")
(root / "PASS.json").write_text(json.dumps({"status": "pass", "generations": rows}, indent=2) + "\n")
PY

# Each configuration captures routing and both timing boundaries in the same
# server lifecycle.  Order 0 is the mandatory balanced@HH alignment gate.
mkdir -p "${input_study}/timing-configurations"
while IFS=$'\t' read -r order config_id stage lifecycle frequency placement_path placement_sha mapping_sha; do
    failure_stage="config-${order}-${config_id}:prepare"
    config_parent="${input_study}/timing-configurations/$(printf '%02d' "${order}")-${config_id}"
    mkdir -p "${config_parent}"
    resumed_run="$("${AFD_PYTHON}" -m src.afd_cost_transaction check \
        --plan "${plan}" --order "${order}" --directory "${config_parent}" --patch-sha "${patch_sha}")"
    if [[ -n "${resumed_run}" ]]; then
        printf 'timing\t%s\t%s\t%s\t%s\tcomplete\t%s\n' \
            "${order}" "${config_id}" "${lifecycle}" "${frequency}" "${resumed_run}" >> "${study}/runs.tsv"
        continue
    fi
    config_dir="${config_parent}/attempt-${SLURM_JOB_ID}"
    mkdir "${config_dir}"
    placement="${PROJECT_ROOT}/${placement_path}"
    [[ "$(sha256sum "${placement}" | awk '{print $1}')" == "${placement_sha}" ]] || {
        printf 'placement hash drift: %s\n' "${placement}" >&2; exit 1;
    }
    clocks="2520,2520,${frequency},${frequency}"
    model_name="qwen3-moe-n-${order}"
    failure_stage="config-${order}-${config_id}:server-start"
    start_profile_server "${config_dir}" "${model_name}" "${placement}" "${frequency}" "${mapping_sha}"
    failure_stage="config-${order}-${config_id}:conditioning"
    run_conditioning "${config_dir}/conditioning" "${model_name}" "${placement}" \
        "${clocks}" "${config_dir}/server/attention.log"
    failure_stage="config-${order}-${config_id}:measured-run"
    IFS=',' read -r -a target_array <<< "${clocks}"
    "${AFD_PYTHON}" -m src.benchmark \
        --trace "${trace}" --base-url "http://127.0.0.1:${api_port}" \
        --model "${model_name}-attention" --tokenizer Qwen/Qwen3-30B-A3B \
        --request-rate 8 --override-arrival-rate --output-root "${config_dir}/runs" \
        --warmup-requests 8 --settle-before-s 5 --settle-after-s 5 \
        --monitor-interval-s 0.2 --max-concurrency 256 --request-timeout-s 900 \
        --local-files-only --placement-file "${placement}" \
        --vllm-patch-manifest patches/afd-plugin-0.26.0-cost-model.json \
        --afd-timing-control-dir "${config_dir}/timing-control" \
        --target-clocks "${target_array[@]}" --store-output-token-ids \
        --run-label "${config_id}" --placement-name "${config_id}" \
        --server-log "${config_dir}/server/attention.log" > "${config_dir}/benchmark.log" 2>&1
    run_dir="$(find "${config_dir}/runs" -mindepth 1 -maxdepth 1 -type d | head -1)"
    failure_stage="config-${order}-${config_id}:validation"
    "${AFD_PYTHON}" -m tools.validate_live_run "${run_dir}" > "${run_dir}/live-validation.json"
    "${AFD_PYTHON}" -m tools.validate_milestone_n_cost_timing \
        --timing-run "${run_dir}" \
        --placement "${placement}" --config-id "${config_id}" \
        --lifecycle "${lifecycle}" --frequency-mhz "${frequency}" \
        > "${run_dir}/cost-timing-validation.json"
    failure_stage="config-${order}-${config_id}:shutdown"
    stop_server "${config_dir}/compute-apps-after.csv"
    "${AFD_PYTHON}" -m src.afd_cost_transaction mark \
        --plan "${plan}" --order "${order}" --directory "${config_parent}" --patch-sha "${patch_sha}" \
        --run "${run_dir}" --attempt "${config_dir}"
    printf 'timing\t%s\t%s\t%s\t%s\tcomplete\t%s\n' \
        "${order}" "${config_id}" "${lifecycle}" "${frequency}" "${run_dir}" >> "${study}/runs.tsv"
done < <("${AFD_PYTHON}" - "${plan}" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
for row in sorted(plan["configurations"], key=lambda item: item["order"]):
    placement = row["placement"]
    print(row["order"], row["config_id"], row["stage"], row["lifecycle"],
          row["frequency_mhz"], placement["path"], placement["sha256"],
          placement["canonical_mapping_sha256"], sep="\t")
PY
)

complete="$(awk -F '\t' '$1 == "timing" && $6 == "complete" {n++} END {print n+0}' "${study}/runs.tsv")"
expected="$("${AFD_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["expected_timing_runs"])' "${plan}")"
[[ "${complete}" -eq "${expected}" ]] || { printf 'timing completion mismatch\n' >&2; exit 1; }
"${AFD_PYTHON}" - "${study}/result.json" "${complete}" <<'PY'
import json, sys
json.dump({"status": "complete", "same_run_cost_profiles": int(sys.argv[2])},
          open(sys.argv[1], "w"), indent=2)
PY
printf 'PASS\n' > "${study}/status.txt"
