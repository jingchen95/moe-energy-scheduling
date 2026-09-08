#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'usage: %s <pid-file>\n' "$0" >&2
    exit 2
fi

pid_file="$1"
[[ -r "${pid_file}" ]] || exit 0
read -r pid < "${pid_file}"
[[ "${pid}" =~ ^[0-9]+$ ]] || { printf 'invalid PID file: %s\n' "${pid_file}" >&2; exit 1; }

mapfile -t group_pids < <(pgrep -g "${pid}" || true)
if [[ "${#group_pids[@]}" -eq 0 ]]; then
    exit 0
fi

group_is_running() {
    local group_pid state
    while read -r group_pid; do
        state="$(ps -o stat= -p "${group_pid}" 2>/dev/null | tr -d ' ' || true)"
        if [[ -n "${state}" && "${state}" != Z* ]]; then
            return 0
        fi
    done < <(pgrep -g "${pid}" || true)
    return 1
}

group_is_running || exit 0

for group_pid in "${group_pids[@]}"; do
    owner="$(ps -o user= -p "${group_pid}" 2>/dev/null | tr -d ' ' || true)"
    if [[ -n "${owner}" && "${owner}" != "${USER}" ]]; then
        printf 'refusing to stop process group %s owned by another user\n' "${pid}" >&2
        exit 1
    fi
done

if kill -0 "${pid}" 2>/dev/null; then
    command="$(ps -o command= -p "${pid}" 2>/dev/null || true)"
    if [[ -n "${command}" \
        && "${command}" != *vllm.entrypoints.openai.api_server* \
        && "${command}" != "Apptainer runtime parent: vllm-openai-v0.15.1-amd64.sif"* ]]; then
        printf 'refusing to stop unrecognized process %s: %s\n' "${pid}" "${command}" >&2
        exit 1
    fi
fi

kill -TERM -- "-${pid}" 2>/dev/null || true
for _ in {1..60}; do
    if ! group_is_running; then
        exit 0
    fi
    sleep 1
done
kill -KILL -- "-${pid}" 2>/dev/null || true
for _ in {1..10}; do
    if ! group_is_running; then
        exit 0
    fi
    sleep 1
done
printf 'process group %s did not stop\n' "${pid}" >&2
exit 1
