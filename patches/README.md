# vLLM Patches

`vllm-0.15.1-placement-profile.patch` is Milestone D patch version 1. It adds:

- `--expert-placement-file` for strict static per-layer mappings;
- startup weight rearrangement through vLLM's existing EPLB transfer code;
- collective RPCs that reset and dump vLLM's existing EPLB load counters.

It does not alter routing, fused-MoE kernels, expert FFNs, or all-to-all code.
Use `scripts/manage_vllm_patch.sh` against the directory containing the `vllm/`
package. The script verifies exact before/after hashes and is idempotent.

```bash
scripts/manage_vllm_patch.sh verify --root "$CONDA_PREFIX/lib/python3.10/site-packages"
scripts/manage_vllm_patch.sh apply  --root "$CONDA_PREFIX/lib/python3.10/site-packages"
scripts/manage_vllm_patch.sh revert --root "$CONDA_PREFIX/lib/python3.10/site-packages"
```
