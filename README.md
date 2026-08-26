# MoE Joint Placement + DVFS Motivational Study

This repository supports a reproducible, steady-state study of whether joint
Mixture-of-Experts placement and per-GPU DVFS reduces GPU energy under TTFT and
TPOT SLOs beyond sequential EPLB placement followed by DVFS.

The first target is one Slurm-allocated node with four NVIDIA L40S GPUs,
vLLM, and `Qwen/Qwen3-30B-A3B`. The preferred parallel layout is TP=1, DP=4,
with expert parallelism across all four ranks.

## Current Status

Milestone A validated the four-GPU expert-parallel deployment. Milestone C added
the deterministic semantic trace, open-loop streaming client, request metrics,
graphics-clock control, and exact-window four-GPU energy harness. Milestone D
completed the first GSM8K placement-DVFS coupling pilot. The pilot supports H1:
changing placement materially changed the latency and energy effect of
downclocking an individual GPU. See
[docs/milestone_d_placement_dvfs.md](docs/milestone_d_placement_dvfs.md).

Milestone E code and Slurm jobs are prepared. The 81-point frozen-EPLB
Sequential sweep has completed on Uranus; eight frequency-aware Joint
candidates are planned but not yet submitted. See
[docs/milestone_e_sequential_vs_joint.md](docs/milestone_e_sequential_vs_joint.md).

On Minerva, from the staged repository:

```bash
source ~/.bashrc
conda activate cuda-env
PYTHONPATH=.deps python -m pytest -q
```

The job writes a unique directory below `results/raw/`. Its environment,
vLLM command/log, responses, and Slurm log are retained on the cluster and
intentionally ignored by Git.

See [docs/environment.md](docs/environment.md) for verified versions and
hardware findings, [docs/reused_infrastructure.md](docs/reused_infrastructure.md)
for measurement-code provenance, and [HANDOFF.md](HANDOFF.md) for current status.
