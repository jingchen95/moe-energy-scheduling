# Handoff

## Latest status — minimal MoE timing baseline prepared

Status: **`prepared_not_gpu_verified`**.

The active repository has been reduced to one baseline for
`Qwen/Qwen3-30B-A3B`: TP=1, DP=4, EP enabled across four ranks, linear expert
placement, and `allgather_reducescatter`. Power, DVFS, placement search, EPLB
profiling, phase classification, and service-model code are no longer in the
active execution path.

### Measurement contract

- Input is the committed 50-request GSM8K trace at 0.5 request/s. Its offsets
  are exactly `0, 2, ..., 98` seconds and its SHA-256 is
  `721638a6250b929de05c2c355c8bc9b20258db7b28fedb94acb5572136571cbc`.
- The server receives 8 warmup requests, settles for 5 seconds, resets all
  rank-local timers, replays the fixed trace open-loop, and dumps timers only
  after every measured response finishes.
- The numerator is the maximum across EP ranks of each rank's summed CUDA-event
  time around `FusedMoE.forward_impl`. This boundary includes FusedMoE
  dispatch, expert selection/compute, and combine; it is not pure expert GEMM.
- The denominator is the union of successful client-observed send-to-complete
  intervals, so idle gaps are excluded and overlapping requests are not counted
  twice.
- The primary field is
  `fused_moe_over_active_request_wall_time_percent`. The sum across ranks is
  retained only as a diagnostic and is not used as elapsed service time.
- The job does not set GPU clocks and does not read power. It records actual
  graphics/SM clocks once per second using read-only `nvidia-smi` queries.
- The server is deliberately started with `--enforce-eager`; results therefore
  describe the eager timing baseline, not default CUDA Graph execution.

### Patch lifecycle and artifacts

`scripts/moe_time_baseline.sbatch` requires pristine vLLM 0.15.1 files. It
checks exact hashes, applies the standalone patch immediately before starting
the server, and reverts it after stopping the process group created by this
job. Cleanup is registered before the patch is applied, and Slurm sends the
batch shell `TERM` 180 seconds before its time limit so cleanup has time to run.

Every submission creates a non-overwriting directory below `results/raw/` and
stores the resolved vLLM command, model config, trace, source revision/diff,
Slurm allocation, vLLM/PyTorch versions, topology, clock samples, server and
client logs, raw request records, four rank-local timing files, and final
metrics together.

### Offline verification completed

- Standard-library suite: 4/4 tests pass.
- All six active Python modules compile with `python3 -m py_compile`.
- All four shell/Slurm scripts pass `bash -n`.
- The resolved command contains TP=1, DP=4, EP enabled, linear placement,
  `allgather_reducescatter`, and `--enforce-eager`; it contains no EPLB flag.
- The committed trace has 50 unique request IDs, the exact 2-second schedule,
  and the expected SHA-256.
- The patch manifest SHA matches. Against copies of the official pristine
  vLLM 0.15.1 files, `check -> apply -> py_compile -> revert -> check` passes
  and returns both files to their exact original SHA-256.
- `git diff --check` passes.

No GPU or live vLLM service was available for this cleanup. In particular,
there is no new measured percentage yet, and the CUDA timing perturbation has
not been rechecked on hardware.

### Preserved historical result

Historical reports and processed figures remain under `docs/` and `results/`.
The earlier five-placement by 81-frequency study did **not** establish a
meaningful Joint placement/DVFS energy advantage: its only strict 13 ms gap was
0.1005% and was not repeatability-confirmed. This negative result is preserved,
not reinterpreted by the new baseline. Removed legacy execution code is
recoverable from commit `d7f30c5`.

### Exact next command

From the Minerva checkout with the existing `cuda-env`:

```bash
cd /data/users/chjing/moe-energy-scheduling
sbatch scripts/moe_time_baseline.sbatch
```

After completion, inspect:

```bash
run_dir=$(find results/raw -maxdepth 1 -type d -name 'moe-time-*' | sort | tail -1)
cat "${run_dir}/benchmark/metrics.json"
cat "${run_dir}/exit-status.txt"
```
