# Reused Measurement Infrastructure

## Source

The prior project is `/data/users/chjing/vLLM_test` on Minerva, at Git commit
`0eb8926`. It was inspected read-only; no file in that repository was changed.

## Reused concepts and implementations

- `paper/scripts/xpyd/nvml_readonly.py`: NVML total-energy capability detection,
  cumulative-counter units, and read-only power/utilization/clock telemetry.
- `paper/scripts/xpyd/phase3b_energy.py`: fixed-monotonic-period sampling and the
  rule that NVML total-energy deltas are primary while trapezoidal power
  integration is only a fallback when the counter is unsupported.
- `paper/scripts/xpyd/phase3d_control.py`: supported graphics-clock discovery,
  `nvidia-smi -lgc` actuation, bounded actual-clock readback, and identity-aware
  validation.
- `paper/scripts/replay_synthetic_trace.py`: asynchronous streaming request
  consumption, server-usage token counts with retokenization fallback, linear
  percentiles, and the documented TPOT convention.
- `run_disagg_benchmark.sh`: exact PID/process-group ownership and cleanup
  patterns.

These mechanisms were adapted into `src/gpu_monitor.py`, `src/gpu_control.py`,
`src/benchmark.py`, and `src/metrics.py`. The new code handles four GPUs in one
monolithic serving process, emits this project's run layout, and replays
semantic JSONL traces rather than exact-length synthetic prompts.

## Assumptions

- GPUs 0-3 belong exclusively to the Slurm job.
- `sudo nvidia-smi -lgc/-rgc` is permitted by the existing cluster setup.
- L40S exposes `nvmlDeviceGetTotalEnergyConsumption`, as verified in Milestone A.
- Qwen's OpenAI chat endpoint returns streamed usage when
  `stream_options.include_usage=true`; retokenization is retained as a recorded
  fallback.

## Deliberately not reused

- No SWEEP-LLM/XpYd policy, scheduler, disaggregated prefill/decode, proxy,
  Prometheus endpoint, or model-fitting code was copied.
- The old synthetic repeated-token workload is not a primary workload here.
- Memory-clock control (`-lmc`, `-rmc`, or application clocks) is excluded.
- Idle-power subtraction is excluded from the primary metric.
- Broad process matching and cleanup are excluded; only recorded project PIDs
  and process groups may be terminated.
