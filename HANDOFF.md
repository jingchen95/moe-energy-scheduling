# Handoff

## Current working state

Milestone A is complete. The repository is initialized, the real L40S/vLLM
environment is documented, and TP=1/DP=4 with four expert-parallel ranks serves
`Qwen/Qwen3-30B-A3B` successfully through the OpenAI-compatible endpoint.

The project is staged on Minerva at:

```text
/data/users/chjing/moe-energy-scheduling
```

## Completed steps

- Created the requested repository layout, YAML configs, safety guidance, and
  ignored result directories.
- Added configuration-driven vLLM launch, exact-PID/process-group shutdown,
  graphics-clock reset, environment inspection, and Slurm smoke tooling.
- Inspected vLLM 0.15.1 as installed from site-packages; no package was changed.
- Captured Uranus OS, CUDA/driver, GPU memory/power/clocks, topology, NVML
  energy support, package locations, cache state, and runtime backend.
- Completed Slurm job `254959`: three deterministic prompts returned HTTP 200,
  `smoke-status.txt` is `PASS`, and `exit-status.txt` is `0`.
- Confirmed TP=1/DP=4/EP=4 fits at 16.38 GiB of model weights per GPU.
- Confirmed all four L40S GPUs expose NVML total-energy counters.
- Confirmed custom per-layer expert placement is not exposed upstream and will
  require a recorded patch in Milestone E.

## Unresolved problems

- vLLM's cold four-rank load from shared storage took 1,236.3 seconds. This is
  not a correctness blocker but must remain outside measured windows.
- vLLM 0.15.1 logged shutdown finalizer warnings after group-wide SIGTERM. The
  stop script has been refined to request parent-led shutdown first; that exact
  refinement has not yet been exercised by another full model launch.
- Milestone B GPU control and monitoring modules remain intentional stubs.

## Exact next command

```bash
git switch -c milestone-b-dvfs-monitor
```
