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
patch_file="${project_root}/patches/afd-plugin-0.26.0-static-profile.patch"
target="afd_plugin/model_executor/models/qwen3_moe.py"
expected_commit="4c2541a53ec8ad507508073eac81ac9feecc6c9a"
before_sha="cef1790937637cd445cac92bf8dcfaea20d7c3cfdfb0e2813a36b80a8098e2d5"
after_sha="f6535a2e70561eee5199bcdc77fb47c84b4a5f634f38703f8ca8971d4fadb97b"

[[ -f "${root}/${target}" ]] || { printf 'not an AFD plugin root: %s\n' "${root}" >&2; exit 1; }
[[ "$(git -C "${root}" rev-parse HEAD)" == "${expected_commit}" ]] || {
    printf 'AFD plugin commit does not match pin\n' >&2
    exit 1
}

current_sha="$(sha256sum "${root}/${target}" | awk '{print $1}')"
case "${current_sha}" in
    "${before_sha}") state="before" ;;
    "${after_sha}") state="after" ;;
    *) state="mixed" ;;
esac

verify_tree() {
    local expected="$1"
    local status
    status="$(git -C "${root}" status --porcelain)"
    if [[ "${expected}" == "before" ]]; then
        [[ -z "${status}" ]] || { printf 'unexpected dirty plugin tree:\n%s\n' "${status}" >&2; return 1; }
    else
        [[ "${status}" == " M ${target}" ]] || {
            printf 'unexpected patched plugin tree:\n%s\n' "${status}" >&2
            return 1
        }
    fi
}

case "${action}:${state}" in
    verify:before) verify_tree before; printf 'AFD profile patch state: not applied\n' ;;
    verify:after) verify_tree after; printf 'AFD profile patch state: applied\n' ;;
    verify:mixed) printf 'AFD profile patch state: mixed\n' >&2; exit 1 ;;
    apply:before)
        verify_tree before
        git -C "${root}" apply "${patch_file}"
        exec "$0" verify --root "${root}"
        ;;
    apply:after) verify_tree after; printf 'AFD profile patch already applied\n' ;;
    apply:mixed) printf 'refusing to apply from mixed state\n' >&2; exit 1 ;;
    revert:after)
        verify_tree after
        git -C "${root}" apply --reverse "${patch_file}"
        exec "$0" verify --root "${root}"
        ;;
    revert:before) verify_tree before; printf 'AFD profile patch already reverted\n' ;;
    revert:mixed) printf 'refusing to revert from mixed state\n' >&2; exit 1 ;;
    *) printf 'unsupported patch state: %s\n' "${state}" >&2; exit 1 ;;
esac
