#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'usage: %s <apply|verify|revert> --root <afd-plugin-root>\n' "$0" >&2
    exit 2
}

[[ $# -eq 3 && "$2" == "--root" ]] || usage
action="$1"
root="$(cd "$3" && pwd)"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
patch_file="${project_root}/patches/afd-plugin-0.26.0-cost-model.patch"
expected_commit="4c2541a53ec8ad507508073eac81ac9feecc6c9a"
targets=(
    "afd_plugin/model_executor/models/qwen3_moe.py"
    "afd_plugin/v1/worker/ffn_model_runner.py"
)
before_shas=(
    "cef1790937637cd445cac92bf8dcfaea20d7c3cfdfb0e2813a36b80a8098e2d5"
    "14aa42f09bb3807808faa9b4f9b702fcc5ecc8d83ba85ced72f6781da0aade93"
)
after_shas=(
    "09b132cae179f12367b8d36f3f123bac4c0cfc54ad2794f1f4757d911b5e2d01"
    "0032d45442e453446afaee767c0316b74c596f601d162cc51948d1c16c2a864b"
)

for target in "${targets[@]}"; do
    [[ -f "${root}/${target}" ]] || {
        printf 'not an AFD plugin root: missing %s\n' "${target}" >&2; exit 1;
    }
done
[[ "$(git -C "${root}" rev-parse HEAD)" == "${expected_commit}" ]] || {
    printf 'AFD plugin commit does not match pin\n' >&2
    exit 1
}

state=""
for index in "${!targets[@]}"; do
    current_sha="$(sha256sum "${root}/${targets[$index]}" | awk '{print $1}')"
    current_state="mixed"
    [[ "${current_sha}" == "${before_shas[$index]}" ]] && current_state="before"
    [[ "${current_sha}" == "${after_shas[$index]}" ]] && current_state="after"
    if [[ -z "${state}" ]]; then
        state="${current_state}"
    elif [[ "${state}" != "${current_state}" ]]; then
        state="mixed"
    fi
done

verify_tree() {
    local expected="$1"
    local status
    status="$(git -C "${root}" status --porcelain)"
    if [[ "${expected}" == "before" ]]; then
        [[ -z "${status}" ]] || { printf 'unexpected dirty plugin tree:\n%s\n' "${status}" >&2; return 1; }
    else
        local expected_status
        expected_status="$(printf ' M %s\n' "${targets[@]}")"
        [[ "${status}" == "${expected_status}" ]] || {
            printf 'unexpected patched plugin tree:\n%s\n' "${status}" >&2
            return 1
        }
    fi
}

case "${action}:${state}" in
    verify:before) verify_tree before; printf 'AFD cost-model patch state: not applied\n' ;;
    verify:after) verify_tree after; printf 'AFD cost-model patch state: applied\n' ;;
    verify:mixed) printf 'AFD cost-model patch state: mixed\n' >&2; exit 1 ;;
    apply:before)
        verify_tree before
        git -C "${root}" apply "${patch_file}"
        exec "$0" verify --root "${root}"
        ;;
    apply:after) verify_tree after; printf 'AFD cost-model patch already applied\n' ;;
    apply:mixed) printf 'refusing to apply from mixed state\n' >&2; exit 1 ;;
    revert:after)
        verify_tree after
        git -C "${root}" apply --reverse "${patch_file}"
        exec "$0" verify --root "${root}"
        ;;
    revert:before) verify_tree before; printf 'AFD cost-model patch already reverted\n' ;;
    revert:mixed) printf 'refusing to revert from mixed state\n' >&2; exit 1 ;;
    *) printf 'unsupported patch state: %s\n' "${state}" >&2; exit 1 ;;
esac
