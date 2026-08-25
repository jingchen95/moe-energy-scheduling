# Handoff

## Current working state

Milestone C harness implementation is complete. The full Milestone C
experimental campaign was intentionally deferred after the research plan
changed to a placement-DVFS coupling pilot. Live validation of the streaming
benchmark client is still pending and is a prerequisite of Milestone D; it will
be performed during the first instrumented server launch to avoid another cold
model load.

The project is staged on Minerva at:

```text
/data/users/chjing/moe-energy-scheduling
```

## Completed steps

- Milestone A validated Qwen3-30B-A3B with TP=1, DP=4, EP=4 and vLLM 0.15.1
  on four L40S GPUs using `allgather_reducescatter`.
- Located the prior measurement implementation in
  `/data/users/chjing/vLLM_test` at commit `0eb8926` and documented the reuse
  boundary in `docs/reused_infrastructure.md`.
- Added graphics-only per-GPU clock discovery, locking, readback, and reset.
- Added fixed-period four-GPU NVML telemetry and cumulative-energy window
  accounting, with trapezoidal power integration only as an unsupported-counter
  fallback.
- Completed Slurm integration job `255097`: all four GPUs exposed 210-2520 MHz,
  a 975 MHz lock read back on every GPU, the four-GPU energy delta was positive,
  and cleanup restored default clocks.
- Added canonical semantic JSONL parsing, deterministic trace generation,
  open-loop asynchronous streaming replay, request-level TTFT/TPOT/E2E and
  send-delay metrics, unique run directories, and normalized run summaries.
- Selected Databricks Dolly 15k, Google MBPP, and OpenAI GSM8K and generated
  deterministic 200-request pools in the common 128-256 chat-template-token
  range with seed 0 and 128 maximum output tokens.
- The CPU-safe suite passes: `13 passed` in the Minerva `cuda-env`, using a
  project-local ignored `.deps/` installation of pytest because pytest is not
  installed in that conda environment.

## Deferred or unresolved

- The benchmark client has not yet been validated against the live vLLM
  streaming endpoint. Milestone D must validate 10-20 requests before accepting
  any research measurements.
- Rate calibration and the general/code/math Milestone C baseline campaign were
  deliberately not run. No baseline tables or workload comparison figure exist.
- Slurm job `255098` was canceled during model loading because Uranus was not
  exclusively allocated. Its partial directory is invalid research data and is
  retained under `results/raw/server-255098-20260825T170451Z`.
- The current vLLM site-packages installation is unmodified. Expert activation
  instrumentation and arbitrary static placement require inspection and a
  reproducible patch in Milestone D.

## Exact next command

After the Milestone C closeout commit:

```bash
git switch -c milestone-d-placement-dvfs-coupling
```

Milestone D must first inspect vLLM EPLB/map internals, then combine live client
validation with the first instrumented default-placement server launch.
