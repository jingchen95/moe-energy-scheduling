# Handoff

## Latest status — direct-launch minimal MoE timing baseline

Status: **`prepared_not_gpu_verified`**.

The active path now has no model YAML, Python configuration builder, or server
wrapper. The complete launch chain is:

```text
sbatch scripts/moe_time_baseline.sbatch
  -> python -m vllm.entrypoints.openai.api_server
  -> python -m src.benchmark <unique-output-directory>
```

The Slurm script directly hardcodes `Qwen/Qwen3-30B-A3B`, TP=1, DP=4, expert
parallelism, the MP data-parallel backend, linear expert placement,
`allgather_reducescatter`, disabled prefix caching, and eager execution. The
benchmark directly hardcodes the trace, localhost endpoint, served model name,
four expected ranks, eight warmups, five settle seconds, and request timeout.
Its only command-line argument is the output directory.

The patch manager remains separate because site-packages edits must be
hash-checked and reversible. The stop helper remains separate because cleanup
must target only the process group created by this job. Neither helper carries
model or benchmark configuration.

### Measurement contract

- Input is the committed 50-request GSM8K trace at 0.5 request/s. Offsets are
  exactly `0, 2, ..., 98` seconds and SHA-256 is
  `721638a6250b929de05c2c355c8bc9b20258db7b28fedb94acb5572136571cbc`.
- The server receives eight warmup requests, settles for five seconds, resets
  all rank-local timers, replays the fixed trace open-loop, then dumps timing
  only after all measured responses finish.
- The numerator is the maximum across the four EP ranks of rank-local summed
  CUDA-event time around `FusedMoE.forward_impl`. It includes dispatch,
  expert selection/compute, and combine; it is not pure expert GEMM.
- The denominator is the union of successful client send-to-complete intervals.
  Idle gaps are excluded and overlaps are counted once.
- The primary field is
  `fused_moe_over_active_request_wall_time_percent`. Rank sum is diagnostic only.
- The job neither reads power nor records or controls GPU clocks.
- `--enforce-eager` is intentional; this is an eager timing baseline.

### Patch lifecycle and artifacts

The Slurm job requires pristine vLLM 0.15.1 files, checks exact hashes, applies
the standalone patch before service startup, and reverts it after stopping the
owned process group. Cleanup is registered before patch application. Slurm
sends `TERM` 180 seconds before the time limit to leave time for cleanup.

Every submission creates a non-overwriting directory below `results/raw/` and
stores the exact vLLM command, trace, source revision/status, allocation,
versions, topology, raw server/client logs, request rows, four rank-local timing
files, and final metrics together.

### Offline verification completed

- 3/3 standard-library unit tests pass.
- All five active Python modules compile.
- All three shell/Slurm scripts pass `bash -n`.
- The hardcoded server command contains TP=1, DP=4, EP, linear placement,
  `allgather_reducescatter`, disabled prefix caching, and eager execution.
- The trace has 50 unique IDs, exact 2-second offsets, and the expected hash.
- The current patch SHA matches its manifest. The patch payload is unchanged
  from commit `326399e`, where its pristine vLLM 0.15.1
  `check -> apply -> py_compile -> revert -> check` round trip passed.
- No references remain to the removed YAML/config/server-wrapper path or old
  phase-profile environment variable.
- An independent final review found no high- or medium-severity issue in the
  direct startup, signal cleanup, process-group cleanup, or patch-revert gate.
- Temporary-process tests cover normal TERM, forced KILL, an exited leader with
  a live child, rejection of a non-vLLM leader, and isolation from a sentinel
  process outside the target group.
- `git diff --check` passes.

No GPU or live vLLM service is available on the cleanup machine. There is no
new measured percentage yet, and CUDA timing overhead has not been checked on
hardware. Its base Python also lacks `aiohttp`; the target `cuda-env` must
provide the single dependency from `requirements.txt`.

### Preserved historical result

Historical reports and processed figures remain under `docs/` and `results/`.
The earlier five-placement by 81-frequency study did **not** establish a
meaningful Joint placement/DVFS energy advantage: its only strict 13 ms gap was
0.1005% and was not repeatability-confirmed. This negative result is preserved.
Removed legacy execution code remains recoverable from commit `d7f30c5`.

### Exact next command

```bash
cd /data/users/chjing/moe-energy-scheduling
sbatch scripts/moe_time_baseline.sbatch
```

After completion:

```bash
run_dir=$(find results/raw -maxdepth 1 -type d -name 'moe-time-*' | sort | tail -1)
cat "${run_dir}/benchmark/metrics.json"
cat "${run_dir}/exit-status.txt"
```
