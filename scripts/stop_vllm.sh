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
if ! kill -0 "${pid}" 2>/dev/null; then
    exit 0
fi

owner="$(ps -o user= -p "${pid}" | tr -d ' ')"
command="$(ps -o command= -p "${pid}")"
if [[ "${owner}" != "${USER}" ]] || [[ "${command}" != *vllm* && "${command}" != *serve_vllm.sh* ]]; then
    printf 'refusing to stop unrecognized process %s: %s\n' "${pid}" "${command}" >&2
    exit 1
fi

kill -TERM "${pid}"
for _ in {1..60}; do
    if ! pgrep -g "${pid}" >/dev/null 2>&1; then
        exit 0
    fi
    sleep 1
done
kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
