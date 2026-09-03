# vLLM 0.15.1 MoE Timing Patch

`vllm-0.15.1-moe-timing.patch` is a standalone, opt-in patch for pristine
vLLM 0.15.1. It does not require expert placement, EPLB, or phase-profile
instrumentation.

With `VLLM_MOE_TIMING=1`, the patch records one CUDA-event interval around
each `FusedMoE.forward_impl` call. The interval covers the FusedMoE
dispatch/compute/combine region and retains only:

- layer name;
- routed token-row count;
- elapsed GPU milliseconds.

It does not separate prefill from decode, communication from expert compute,
or measure request end-to-end time. Events remain in rank-local memory during
the measured trace. The dump performs one local `torch.cuda.synchronize()` per
worker; no profiling collective is added to the measured forward path.

## Apply and revert

Run the manager against the directory containing the installed `vllm/`
package. It accepts only the exact pristine or exact patched files recorded in
`vllm-0.15.1-moe-timing.json`, and verifies the patch-file SHA-256 before every
action.

```bash
site_root="$CONDA_PREFIX/lib/python3.10/site-packages"
scripts/manage_moe_timing_patch.sh check  --root "$site_root"
scripts/manage_moe_timing_patch.sh apply  --root "$site_root"
scripts/manage_moe_timing_patch.sh revert --root "$site_root"
```

## Run a measurement window

Start vLLM with:

```bash
export VLLM_SERVER_DEV_MODE=1
export VLLM_MOE_TIMING=1
```

Keep `--enforce-eager`; this instrumentation has not been validated with CUDA
graph replay. After warmup, call the development-only collective RPC:

```json
{"method": "reset_moe_timing"}
```

After all measured requests finish, call:

```json
{"method": "dump_moe_timing", "args": ["/absolute/unique/run/directory"]}
```

Each EP rank creates one non-overwriting file:

```text
moe_timing_rank0.json
moe_timing_rank1.json
moe_timing_rank2.json
moe_timing_rank3.json
```

The RPC fails rather than returning an empty profile if timing was not enabled
or no instrumented FusedMoE layer is present. Keep the development endpoint
bound to a trusted local interface.
