# Research Questions

## Established by prior work - not our observations

- Expert activation is skewed.
- Activation distribution depends on workload.
- Workload-aware expert placement can improve load balance, communication, or
  latency.
- Dynamic workload changes can motivate online expert placement.
- Communication-aware placement can outperform pure load balancing.

These premises motivate the experiments but are not novelty claims or findings
of this project.

## Open hypotheses for this project

### H1 - Placement-DVFS coupling

Changing expert placement changes the DVFS sensitivity of individual GPUs.

### H2 - Energy-optimal placement

The placement minimizing latency or load imbalance is not necessarily the
placement minimizing gross GPU energy under a latency or SLO constraint.

### H3 - Sequential optimization gap

```text
performance-optimal placement
-> heterogeneous DVFS
```

may be inferior to:

```text
joint placement + heterogeneous DVFS
```

Milestone D tests H1 only, with limited exploratory evidence toward H2. H2 and
H3 remain hypotheses unless later controlled measurements support them.

## Milestone D checkpoint

The GSM8K placement-DVFS pilot supports H1: downclocking the rank assigned the
largest predicted expert load under the diagnostic concentrated placement had
a materially larger TPOT penalty than the same GPU perturbation under the
load-balanced placement. This is a one-model, one-workload pilot result, not a
general claim.

The three uniform-frequency curves did not show a placement-dependent choice
of energy-minimizing frequency at relative TPOT budgets of 1.05x, 1.10x, or
1.20x. H2 therefore remains open, and H3 was not tested.
