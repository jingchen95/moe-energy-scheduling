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

## Milestone F phase instrumentation

`vllm-0.15.1-phase-profile.patch`, `vllm-0.15.1-phase-state.patch`, and
`vllm-0.15.1-phase-worker.patch` are an incremental, opt-in patch stack on top
of the Milestone D patch. The stack is managed by
`scripts/manage_phase_patch.sh`; it adds no behavior unless
`VLLM_EXPERT_PHASE_PROFILE=1` is exported. It reuses the existing EPLB routed
token counter, records prefill/decode windows, and validates that their logical
counts conserve the total profile at dump time. The base patch is applied and
reverted explicitly by the Milestone F Slurm job, and exact before/after hashes
are recorded in `vllm-0.15.1-phase-profile.json` and each run directory.

The DP phase mask uses vLLM's existing CPU-group metadata to communicate the
small local decode counts once per forward; it does not add a router pass or an
RPC. This assumption is checked against the all-gather token-row count.
