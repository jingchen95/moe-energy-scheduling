# Repository Instructions

## Scope

Work milestone by milestone. Do not begin a later milestone until the current
one is verified, documented in `HANDOFF.md`, and committed.

## Research discipline

- Preserve negative and inconclusive results.
- Never compare policies using different prompt order or arrival schedules.
- Keep B2, B3, and B4 on the same frozen EPLB placement.
- Do not subtract idle power from the primary energy metric.
- Record raw TTFT and TPOT values even when using a calibrated relative SLO.

## Cluster safety

- Use one of `uranus` or `neptune` through Slurm; do not use a second compute
  node for this study.
- Do not install or upgrade CUDA, PyTorch, NVIDIA drivers, or vLLM.
- Do not use `pkill`, `killall`, or `fuser -k`.
- Store and terminate only PIDs/process groups created by this project.
- Do not change memory clocks.
- Reset graphics clocks after any experiment that changes them, including on
  handled signals and exceptions.
- Never edit site-packages without first recording a reproducible patch in
  `patches/`.

## Reproducibility

Every measured run gets a unique run directory. Never overwrite a prior run.
Keep resolved configuration, source revision, vLLM version, topology, actual
clocks, raw server/client/monitor output, expert statistics, placement, metrics,
and stdout/stderr together.
