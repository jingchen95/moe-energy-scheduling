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
