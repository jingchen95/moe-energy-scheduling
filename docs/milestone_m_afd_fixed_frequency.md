# Milestone M: AFD Fixed-Frequency Expert Placement

## Research question

Milestone L found negligible serving-level benefit from aligning intact expert
groups with heterogeneous whole-GPU frequencies in the colocated runtime.
Milestone M asks whether separating Attention from the FFN pool exposes a
measurable individual-expert placement by fixed FFN-frequency interaction.

No per-layer or in-run frequency switching is part of this milestone.

## Stage 1: isolated runtime

The protected `cuda-env` remains the vLLM 0.15.1 environment used by the
colocated study. The AFD runtime is separate and project-local:

- repository: `https://github.com/vllm-project/afd-plugin.git`
- pinned commit: `4c2541a53ec8ad507508073eac81ac9feecc6c9a`
- plugin version: `0.26.0rc2.dev61+g4c2541a53`
- isolated vLLM: `0.26.0`
- isolated PyTorch: `2.11.0`
- bootstrap: project-local `uv 0.12.20`

The pinned plugin advertises generic `2A2F` runner scenarios, but its Qwen3
MoE E2E module currently exercises only `2A1F`. Therefore Qwen3 2A2F on the
four L40S GPUs is an explicit smoke-test gate, not an assumed working state.

The source-controlled environment record is
`configs/milestone_m_environment.json`. The large plugin checkout, virtual
environment, bootstrap environment, and package cache are ignored by Git.

Validation results:

- repository CPU suite in the protected vLLM 0.15.1 environment: `68 passed`;
- pinned plugin `tests/unit` CPU suite, excluding GPU/E2E/slow/NPU markers:
  passed with exit status 0;
- `git diff --check`: passed.

## Planned topology

- GPU0/GPU1: Attention ranks, fixed at 2520 MHz.
- GPU2/GPU3: FFN ranks, tested at HH, MH, and HM.
- Synchronous `P2pNcclAFDConnector` with gate computation on the FFN role.
- Qwen3-30B-A3B, BF16, TP=1, PP=1, no EPLB or expert replication.

## Required stage gates

1. Unmodified eager Qwen3 2A2F smoke on one exclusive L40S node.
2. Static individual-expert placement correctness at HH.
3. AFD-native frozen routing profile collection.
4. One MH aligned-placement smoke.
5. Only then, the randomized 18-run primary matrix.

## Current status

Environment preparation is complete. The unmodified-plugin smoke launcher is
prepared as `scripts/milestone_m_afd_validation.sbatch` and intentionally does
not contain placement or profiling changes. It launches two owned process
groups using the isolated interpreter:

- Attention API: GPUs 0--1, port 18100;
- FFN service: GPUs 2--3, port 18101;
- synchronous connector: port 6249.

The job sends the same three fixed greedy prompts twice to the Attention API,
requires exact generated-content/token-count equality, and samples all four
GPUs to confirm each role used its assigned devices. Cleanup stops only the
recorded process groups and resets graphics clocks. The current project CPU
suite passes `72` tests.

Real-GPU smoke validation, placement instrumentation, profiling, and primary
measurements are pending.

### First smoke attempt

Slurm job `269841` failed before either API became healthy. Both FFN ranks
reached their first Triton FusedMoE kernel invocation, then Triton's CUDA
driver helper compilation failed because the system-Python virtual environment
looked for `/usr/include/python3.12/Python.h`, which was absent on Uranus. This
was an environment construction problem, not evidence about Qwen3 2A2F,
placement, NCCL, latency, or energy.

The job exited 1 after 9 minutes 29 seconds. Cleanup reset all four graphics
clocks, left no GPU compute processes, and did not dirty the plugin checkout.
The protected environment remained vLLM 0.15.1.

The isolated runtime was rebuilt with a fixed uv-managed CPython 3.12.3 stored
under `.tools/uv-python`. Its `Python.h` is also on shared project storage, so
the compute node no longer depends on host `/usr/include`. The environment
guard now records and requires the Python include directory and header.

### Second smoke attempt

Slurm job `269912` also failed before either API became healthy, but it
confirmed that the project-local Python-header fix worked: both roles loaded
far enough to exercise their CUDA paths, and the FFN role passed the prior
Triton helper build. The Attention role then selected FlashInfer for top-k and
top-p sampling during vLLM's memory-profile pass. FlashInfer attempted a JIT
build and failed because Uranus has no discoverable `nvcc` or
`/usr/local/cuda`. The FFN ranks subsequently timed out in a synchronous NCCL
receive after 30 minutes; that timeout is a consequence of the Attention
failure, not the initiating error.

The job exited 1 after 35 minutes 5 seconds. Cleanup reset all graphics clocks
and left no GPU compute processes. No request was served, so this remains a
runtime-preparation failure rather than a 2A2F result.

vLLM 0.26 exposes `VLLM_USE_FLASHINFER_SAMPLER=0` specifically to use its
native sampling path. The project launcher now sets this only in owned AFD
role processes. This avoids an undeclared CUDA-toolkit dependency without
installing CUDA, modifying the pinned plugin/vLLM packages, or changing greedy
decoding parameters.

### Third smoke attempt

Slurm job `269940` confirmed the native sampler setting and completed model
initialization for both roles. The Attention API became healthy and repeatedly
returned HTTP 200. The FFN role correctly entered its synchronous connector
loop, which intentionally does not start an HTTP API. The project launcher
incorrectly required an FFN `/health` response, so it never sent a request;
after 30 idle minutes the FFN connector's NCCL receive timed out.

The pinned plugin's official E2E runner waits only for the Attention
`/v1/models` endpoint while checking that both role processes remain alive.
The smoke launcher now follows that contract and records the FFN
`AFD FFN EngineCore started` log line as role-readiness evidence. The failing
FFN `/health` and `/v1/models` probes were removed. Job `269940` exited 1 after
34 minutes 41 seconds, reset clocks, and left no GPU compute processes.

### Verified unmodified 2A2F smoke

Exclusive Uranus job `270001` completed successfully in 4 minutes 41 seconds
using the pinned, unmodified AFD plugin. The Attention API served three fixed
greedy prompts twice, and both passes produced identical content, token counts,
and finish reasons. The response-set SHA-256 is
`c49027c49b6a5a8d785c375530cd4a236504799cd6af31ad0bf4db5ef90bded4`.

GPU sampling confirms all four assigned devices participated:

| Physical GPU | Role | Peak memory (MiB) | Peak SM utilization |
| --- | --- | ---: | ---: |
| 0 | Attention | 42,597 | 98% |
| 1 | Attention | 42,597 | 98% |
| 2 | FFN | 30,065 | 100% |
| 3 | FFN | 30,065 | 100% |

The FFN connector-loop readiness marker was recorded, all owned processes were
terminated, graphics-clock locks were reset, and the post-run compute-process
list was empty. The plugin remained clean at commit
`4c2541a53ec8ad507508073eac81ac9feecc6c9a`; the isolated runtime remained
vLLM 0.26.0 and the protected `cuda-env` remained vLLM 0.15.1.

This satisfies the first Milestone M GPU gate. It establishes that Qwen3
2A2F works on the target node; it is not a placement, performance, or energy
result. Static FFN placement work may now begin, but no profiling or primary
matrix run has yet been submitted.

## Static placement integration

Source inspection of the pinned AFD Qwen3 wrapper and vLLM 0.26 shows that the
FFN role constructs the native `Qwen3MoeSparseMoeBlock`. Its MoE runner uses
one `ExpertMapManager` for both checkpoint loading and kernel routing:
`RoutedExperts.weight_loader` maps every global checkpoint expert through the
manager, and the execution kernel consumes the corresponding live expert map.

The implementation therefore installs a custom global-to-local map immediately
after each FFN MoE block is constructed and before weights are loaded. No EPLB,
replication, router rewrite, vLLM site-package edit, or Attention change is
used. The runtime-independent implementation is
`src/afd_static_placement.py`; a two-hook patch against pinned plugin commit
`4c2541a53ec8ad507508073eac81ac9feecc6c9a` calls it from the Qwen3 wrapper.
The patch and exact before/after hashes are recorded under `patches/`, and
`scripts/manage_afd_static_placement_patch.sh` provides idempotent
apply/verify/revert operations.

The placement schema contains a 48 x 128 `logical_to_rank` matrix with rank IDs
0 or 1. Validation requires every logical expert exactly once and exactly 64
experts per FFN rank in every layer. The correctness study uses no routing
profile and fixes all four physical GPUs at 2520 MHz:

1. native default linear ownership;
2. deterministic balanced round-robin ownership,
   `rank=(logical_expert+layer) mod 2`;
3. a distinct valid block mapping,
   `rank=(floor(logical_expert/16)+layer) mod 2`.

After checkpoint loading, each FFN worker dumps its live 128-entry expert map
for all 48 layers. Offline validation reconstructs physical ownership from the
two rank dumps, checks it against the requested mapping, and hashes the actual
ownership artifact. The H/H job also requires two identical smoke passes per
configuration and exact semantic output/token equality with unmodified smoke
job `270001`.

Pre-GPU validation results:

- project CPU-safe suite: `86 passed`;
- pinned-plugin Qwen3 construction/weight-policy tests with the patch applied:
  `35 passed`;
- patch lifecycle: clean before, exact applied hash, exact revert, clean after;
- shell syntax and `git diff --check`: passed.

## Static placement H/H correctness result

Exclusive Uranus job `270232` ran all three planned configurations using study
directory `results/raw/milestone-m-static-hh-20260930T081823Z`. All four GPUs
were locked to and read back at 2520 MHz for every configuration. Each case
served the three fixed prompts twice, all requests completed, and every case
produced the same semantic response hash as the unmodified-plugin baseline:
`c49027c49b6a5a8d785c375530cd4a236504799cd6af31ad0bf4db5ef90bded4`.

| Configuration | Requested placement SHA-256 | Actual mapping SHA-256 | Per-layer rank counts | Result |
| --- | --- | --- | --- | --- |
| default | native | `8471d83cd85540dc82c49316f3480017e9c1fc5379255986201dadfdf487cbd9` | 64 / 64 | pass |
| balanced round-robin | `b32f39e7cbdca86d2448a159cb17717786ff85aaedff1b7c98afb2b648d9190d` | `f86f2ecf81bb43edcbece1130e12ffd3a429d2697d3079c13dafcd8470aec7c9` | 64 / 64 | pass |
| different 16-expert blocks | `a7ca50ad98ff9a65fbc761d467359d761807177d197faea2890bfe83e46885ee` | `0547c8d3c2fe04ee68da3998e2a1f32317b126865395172c628d575b026a1a98` | 64 / 64 | pass |

The requested and actual hashes use different canonical artifacts, so they are
not expected to be equal. Validation compares the complete reconstructed
48 x 128 ownership matrix, not the file hashes. Both custom rank dumps identify
their mode as `custom`; the default dumps identify `default`. All three actual
mappings differ, and every rank exposes dense local slots 0--63 in every layer.
This is runtime evidence that checkpoint loading and execution used the
requested ownership rather than merely accepting a JSON file.

The Slurm allocation itself is recorded as `FAILED (exit 1)` after 8 minutes
6 seconds. This occurred only after all three GPU cases had passed: the final
offline validator was invoked as a file path, so Python placed `tools/` rather
than the repository root on `sys.path` and could not import `src`. Running the
same validator as a module over the preserved artifacts produced
`validation-summary.json` with all three cases passing. The launcher is fixed
to use `python -m tools.validate_milestone_m_static_placement`, with a regression
assertion in the CPU-safe suite. The failed Slurm exit is retained rather than
rewritten or hidden.

Cleanup still ran through the job trap: graphics-clock locks were reset, the
post-run compute-process list was empty, and the plugin patch was reverted.
Post-hoc environment verification confirms a clean pinned plugin, isolated
vLLM 0.26.0, and unchanged protected `cuda-env` vLLM 0.15.1.

**Decision:** the static-placement H/H correctness gate passes. This proves
that two distinct custom per-layer FFN ownership maps are loaded and used
without changing deterministic model outputs. It is not a performance or
energy result. Routing profiling and the 18-run matrix were not started.

### Repaired-launcher single smoke

The required minimal retry used only the balanced custom mapping rather than
repeating the three-placement study. Uranus job `270365` completed with exit
code 0 in 2 minutes 46 seconds. The selected configuration list contains only
`balanced`; live ownership and deterministic outputs passed, clock cleanup ran,
the compute-process list was empty, and the plugin returned to its clean pinned
state. This closes the launcher-specific prerequisite for routing profiling.

## AFD routing-profile design

vLLM 0.26's native `BaseRouter` exposes a capture callback immediately after
logical top-k selection and before any expert mapping. The profiling-only
integration binds that existing callback for every FFN MoE layer. It accumulates
logical expert IDs in a 48 x 128 CUDA counter on each of the two FFN/DP ranks;
it does not add another router pass and does not enable EPLB.

A shared atomic control file keeps model startup and benchmark warm-up outside
the frozen profile. Transitioning to active resets counters before the measured
trace. After all measured requests finish, transitioning to inactive causes
each FFN rank to dump before a small uncounted flush request is routed. Offline
validation sums the disjoint rank-local counts and retains both inputs. This
instrumentation is isolated to the profiling run and its overhead must not be
interpreted as a performance or energy measurement.

The source-controlled composite patch preserves the already validated static
ownership hooks and adds only the router-capture hook. Its lifecycle manager
checks the pinned plugin commit plus exact before/after source hashes. With the
patch applied, the pinned Qwen3 construction and weight-policy suite passes 35
tests and the patch reverts to a clean tree.

The frozen input is derived from the accepted Milestone I 300-request, 16
request/s GSM8K trace. Prompt IDs, order, content, arrival offsets, and
128-token limits are unchanged; the request seed is explicitly normalized to
0 for this AFD run. The main 18-run matrix remains prohibited until the profile
and candidate-placement opportunity analysis are complete.

## Frozen AFD routing profile

Profiling-only Uranus job `270366` completed with exit code 0 in 5 minutes
48 seconds. It used default linear ownership, H/H Attention clocks, H/H FFN
clocks, and the fixed 300-request GSM8K trace at 16 requests/s. The request
seed was 0, all outputs were capped at 128 tokens, and all 300 requests
completed with zero failures. The relevant immutable identities are:

- trace SHA-256: `6cf46c02b3a88ace790c6f9b59dbb7b5f44a9872a5813584a3f3a232b39b1020`;
- routing-count SHA-256: `36fafdc0ddf6eb8b5599ab26326a754118e2cbce94ad4cc35d0e12414448fbf0`;
- output/token identity SHA-256: `7c015e33fa6e380344f536f3634aa00ebfb3d68d416f561c1df8c1467f5ef505`;
- actual default ownership SHA-256: `8471d83cd85540dc82c49316f3480017e9c1fc5379255986201dadfdf487cbd9`.

The two FFN-rank dumps conserve exactly to the combined 48 x 128 logical
count matrix. Every layer contains 631,296 top-k assignments, corresponding to
78,912 router-observed tokens at top-k 8. Both raw rank dumps, the combined
archive, actual ownership, request-level output identities, runtime config,
plugin commit, patch manifest, and environment checks remain in
`results/raw/milestone-m-afd-profile-20260930T191548Z`.

Instrumentation materially reduced serving throughput, so its latency and
energy numbers are not performance results. This run exists only to freeze
logical routing demand.

### Logical expert skew

Treating each `[layer, logical expert]` entry as a distinct expert instance,
the 6,144 counts range from 0 to 66,773 assignments. Their p10, median, and
p90 are 16.3, 2,514.5, and 12,176.7 respectively; max/mean is 13.5387 and CV
is 1.3407. Within-layer max/mean averages 8.3968 and reaches 13.5387; mean
within-layer CV is 1.3239.

As a secondary index-level diagnostic, summing each repeated logical expert
index over all 48 layers gives:

| Statistic | Value |
| --- | ---: |
| minimum | 111,799 |
| p10 | 148,471.7 |
| median | 219,280 |
| p90 | 339,893.8 |
| maximum | 456,530 |
| maximum / mean | 1.9284 |
| coefficient of variation | 0.3058 |

These are pre-placement logical routing counts, not physical-GPU load or
execution time. In particular, an expert index in one layer is not the same
expert instance as the equal-numbered index in another layer.

### Default physical-rank loads

Mapping the logical counts through the measured default linear ownership gives:

| FFN rank | Aggregate expert-token load | Fraction |
| --- | ---: | ---: |
| FFN0 | 15,544,390 | 51.2979% |
| FFN1 | 14,757,818 | 48.7021% |

The aggregate absolute difference is 786,572 assignments and aggregate
max/mean is only 1.0260. Per-layer rank max/mean has mean 1.0906, median
1.0807, p90 1.1839, p99 1.2544, and maximum 1.2655. Grouping 64 experts per
rank therefore removes about 93.16% of mean layer CV and 87.01% of mean layer
max/mean excess relative to individual experts. The default placement is
substantially smoother at rank level, although individual layers retain a
measurable residual imbalance.

### Frozen candidate placements

The existing capacity-normalized constrained LPT plus local-swap algorithm was
reused. No new placement algorithm was introduced after observing the counts.
All candidates contain every logical expert exactly once, preserve 64 experts
per rank in all 48 layers, and reproduce byte-identical hashes on regeneration.
The immutable candidate files are tracked under
`data/placements/milestone_m/`; compact diagnostics are tracked under
`results/processed/` while the full profiling capture remains ignored under
`results/raw/`.

| Placement | FFN clocks | Rank load fractions | Raw max/mean | Capacity-normalized max/mean | SHA-256 |
| --- | --- | --- | ---: | ---: | --- |
| `P_balanced` | H/H | 50.0000% / 50.0000% | 1.0000003 | 1.0000003 | `59c33ea86a95a1c252418b4e612bd1479465941c4b0a726207a8ee362ed6710c` |
| `P_MH` | M/H | 39.3431% / 60.6569% | 1.2131378 | 1.0001481 | `8824e967f45b322788fc0faefcc9d66e4201aed40826f5d853cd732f5b6f2f80` |
| `P_HM` | H/M | 60.6569% / 39.3431% | 1.2131378 | 1.0001481 | `355c1d7da943fcb457feff7be8372e8e2cb9e853e0732797fe6dc4f100d76596` |

For every layer, the slower rank receives no more predicted expert-token load
than the faster rank. `P_HM` is the exact physical-rank mirror of `P_MH`.
Relative to using `P_balanced` at M/H, `P_MH` reduces the aggregate
capacity-normalized max/mean proxy from 1.2130 to 1.00015, a 17.55% reduction;
the worst per-layer normalized max/mean is 1.0039. This frequency ratio is a
placement heuristic only, not a fitted latency or service model.

### Opportunity decision

**Yes: the profile contains enough structural rank load/capacity opportunity
to justify the planned MH/HM measured experiment.** Equal 64-expert groups
strongly smooth the default raw imbalance, so merely downclocking a rank under
default placement is not well motivated. However, individual expert placement
can reproducibly create the approximately 39.35%/60.65% load split implied by
M/H capacity while retaining exactly 64 experts per rank, and the resulting
17.55% pressure-proxy reduction is large enough to test on hardware.

This is only a go/no-go decision for measurement. It does not establish a
latency, energy, or SLO benefit. The primary 18-run matrix has not been
submitted.

## Fixed-frequency measured experiment plan

The measured stage freezes the three committed candidate placements and does
not regenerate them. A single `P_MH` / Attention H,H / FFN M,H correctness
smoke gates the matrix. It validates the physical clock readback, canonical
runtime ownership, exact deterministic smoke outputs, activity on both FFN
GPUs, process cleanup, clock reset, and protected-environment identity.

The smoke and all primary runs use the plugin-supported
`FULL_DECODE_ONLY` execution mode. Eager and graph measurements will not be
mixed. Routing capture and custom kernel timing are disabled. The primary
matrix is exactly C0--C5 from the resolved configuration, with three repeated
measurement windows per configuration. One recorded seed randomizes the six
configuration groups, and a single exclusive job serializes them on Uranus.

The analyzer reports individual values, mean and sample standard deviation,
and signed `(treatment - baseline) / baseline` changes. Its conservative
classification rule is recorded before measurement: category A requires both
mirrored comparisons to improve TPOT p99 or total energy by at least 3%, the
effect to exceed the larger run-level sample standard deviation, no material
regression in the paired metrics, and an H/H control within 3%. A one-sided
1% benefit that does not meet that complete rule is category B; otherwise the
result is category C. These thresholds organize the decision and do not turn
the structural load/frequency proxy into an execution-time model.

The first heterogeneous smoke allocation exited before model launch because
an immediate idle readback reported 2505 MHz after requesting the supported
2520 MHz lock. Cleanup completed normally. The clock-control preflight now
accepts one 15 MHz L40S clock step, while the smoke still requires active
samples to reach the requested clock and every measured run still validates
its measurement-window average clock. This avoids mistaking a transient idle
readback for a failed lock without weakening the primary-run clock check.

The corrected smoke then completed model startup and both deterministic request
passes. Slurm still recorded exit 1 because the offline validator attempted to
parse the deprecated `clocks.applications.graphics` field, which this driver
renders as a deprecation message. The preserved current-clock and active-sample
fields are exact, ownership and outputs pass, and cleanup is complete. The
validator now uses the non-deprecated current-clock field; the original Slurm
failure remains recorded rather than being relabeled.

## Primary measured results

Exclusive Uranus job `270552` completed with exit code 0 in 47 minutes
38 seconds. All 18 planned measurement windows passed: each completed 300/300
requests with zero failures and 38,400 output tokens, used the one frozen trace
hash, read cumulative NVML energy, and matched all four requested clocks
exactly in the measurement-window averages. The maximum client p99 send delay
was 4.451 ms, below the 62.5 ms arrival interval. Cleanup left no compute
processes, reset clocks, reverted the plugin patch, and preserved vLLM 0.15.1
in the protected environment and vLLM 0.26.0 in the isolated environment.

Values below are mean +/- sample standard deviation across three repetitions.
Energy includes all four GPUs and does not subtract idle energy.

| ID | Placement | Clocks A0/A1/F0/F1 | TTFT p99 (ms) | TPOT p99 (ms) | E2E p99 (ms) | Total energy (J) | FFN energy (J) | Throughput (req/s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | `P_balanced` | 2520/2520/2520/2520 | 14098.08 +/- 9037.19 | 73.158 +/- 3.991 | 21109.62 +/- 8878.55 | 23680.0 +/- 4897.4 | 14468.4 +/- 2929.2 | 7.816 +/- 1.737 |
| C1 | `P_MH` | 2520/2520/2520/2520 | 12942.66 +/- 7589.21 | 72.193 +/- 4.440 | 20247.77 +/- 7427.15 | 23390.6 +/- 4309.0 | 14340.5 +/- 2681.9 | 7.910 +/- 1.445 |
| C2 | `P_balanced` | 2520/2520/1635/2520 | 13371.78 +/- 6719.19 | 73.476 +/- 3.122 | 20545.63 +/- 6698.46 | 21970.1 +/- 3625.8 | 12854.6 +/- 2155.3 | 7.799 +/- 1.337 |
| C3 | `P_MH` | 2520/2520/1635/2520 | 12524.95 +/- 9165.58 | 69.466 +/- 5.421 | 19855.39 +/- 9360.42 | 21920.2 +/- 4658.9 | 12894.4 +/- 2691.7 | 8.071 +/- 1.836 |
| C4 | `P_balanced` | 2520/2520/2520/1635 | 12231.95 +/- 7059.44 | 70.141 +/- 2.357 | 19383.47 +/- 7023.04 | 21435.6 +/- 3915.2 | 12548.6 +/- 2337.4 | 8.099 +/- 1.478 |
| C5 | `P_HM` | 2520/2520/2520/1635 | 12725.94 +/- 6710.16 | 72.078 +/- 2.197 | 19936.94 +/- 6680.39 | 21861.2 +/- 3755.1 | 12822.8 +/- 2264.3 | 7.894 +/- 1.328 |

### Required comparisons

Signed changes use `(treatment mean - baseline mean) / baseline mean`; negative
is favorable for latency and energy, while positive is favorable for
throughput.

| Comparison | TTFT p99 | TPOT p99 | Total energy | FFN energy | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| C3 vs C2, aligned MH | -6.333% | -5.458% | -0.227% | +0.309% | +3.487% |
| C5 vs C4, aligned HM | +4.039% | +2.761% | +1.985% | +2.185% | -2.525% |
| C1 vs C0, H/H control | -8.196% | -1.319% | -1.222% | -0.884% | +1.195% |
| C3 vs C0, aligned MH vs H/H | -11.158% | -5.047% | -7.432% | -10.879% | +3.257% |
| C5 vs C0, aligned HM vs H/H | -9.733% | -1.476% | -7.681% | -11.373% | +0.998% |

The three paired C3-vs-C2 TPOT changes are +4.77%, -11.11%, and -9.65%
(mean -5.33%, sample standard deviation 8.78%). Paired total-energy changes
are +4.71%, -3.51%, and -3.33%. Thus the favorable mean MH latency is not
repeatable relative to its variation, and its energy is effectively unchanged.
For the mirror, C5-vs-C4 TPOT changes are -1.28%, +7.22%, and +2.53%; total
energy increases in every repetition by +0.87%, +2.76%, and +2.71%.

The H/H control is approximately neutral in TPOT, total energy, and throughput
at the mean level. Its -8.20% TTFT difference is not stable relative to the
7.6--9.0 second sample deviations, so the control does not establish a
placement-only TTFT benefit.

### Mechanism check

Alignment did move work toward the faster FFN rank, as the structural profile
predicted. Under MH, slow GPU2 mean energy falls from 5,553.2 J to 5,078.3 J
(-8.55%), while fast GPU3 rises from 7,301.4 J to 7,816.1 J (+7.05%). Under
HM, slow GPU3 falls from 5,505.1 J to 5,127.8 J (-6.85%), while fast GPU2
rises from 7,043.5 J to 7,695.1 J (+9.25%). The fast-rank increase therefore
offsets the slow-rank saving: FFN-pool energy changes by +0.31% for MH and
+2.19% for HM rather than decreasing.

### Repeatability limitation

The fixed 16 request/s offered load exceeds the observed 6.0--9.7 request/s
throughput of this AFD configuration. Every configuration shows a strong
within-server order trend: the first repetition is slowest and highest-energy,
while the third is fastest and lowest-energy. TTFT p99 sample deviations are
6.7--9.2 seconds. Client scheduling itself remains valid, but the resulting
queueing/warm-state variation means the three windows are not stationary
independent samples. This limits interpretation of the apparent MH latency
improvement and is preserved rather than averaged away.

### Decision

**B -- weak/mixed effect.** C3 shows a noisy TPOT improvement but essentially
no total or FFN-pool energy benefit; the mirrored C5 comparison worsens TPOT,
energy, and throughput. The two mirrored comparisons therefore do not provide
repeatable evidence that frequency-aware placement improves the measured AFD
system at identical heterogeneous frequencies. The structural load/frequency
proxy changes the physical work distribution, but it is not sufficient to
predict end-to-end benefit.

Do not proceed to window-level dynamic DVFS from this result. Any further work
should first establish a stationary AFD operating point with adequate warm-up
and a request rate below measured AFD saturation, then repeat this fixed
placement test before adding another control dimension.

## Stationary-rate repeat plan

The original 16 request/s result is preserved above. A follow-up experiment
first tests 2, 4, 6, and 8 request/s using `P_balanced`, H/H Attention, and H/H
FFN clocks. Each rate receives three complete 300-request windows using the
same frozen trace. The 12 windows are serialized behind one server launch and
permuted with seed `20261002`, so rate is not monotonically correlated with
experiment time.

Rate selection is based only on capacity and stationarity: achieved throughput
must track offered load, all requests must complete, TTFT must not accumulate
from the first to final third of a window, and the prior strong repetition
trend must be absent. The highest clearly stable rate with capacity margin will
be frozen before preparing the six-configuration repeat. Placement files and
the 2520/1635 MHz H/M levels remain immutable. No new placement, frequency
tuple, service model, or dynamic DVFS mechanism is introduced.

Stage 1 was submitted as Slurm job `270688` with study directory
`results/raw/milestone-m-rate-sweep-20261001T125021Z`. At submission time it
was pending because Uranus was down, drained, or reserved for a higher-priority
partition; Slurm reported no estimated start time. The job is left queued on
Uranus so the follow-up remains on the validated physical node.

Job `270688` later completed 12/12 windows with clean validation and cleanup,
but it is **not valid for request-rate selection**. The benchmark's historical
`--request-rate` option recorded metadata while replay continued to use the
16 request/s offsets embedded in the trace. The decisive symptom is that the
nominal 2 request/s windows achieved 8.21--10.62 request/s. These results are
preserved as a harness validation failure and are not interpreted as capacity
measurements. The benchmark now has an explicit `--override-arrival-rate`
mode that preserves request content/order while replacing offsets with
`index / request_rate`; resolved schedules are saved and validated per run.
The corrected sweep was submitted as job `270840` with study directory
`results/raw/milestone-m-rate-sweep-rerun-20261001T161703Z` and commit
`39b546a`.

### Corrected rate sweep and frozen rate

Job `270840` completed 12/12 validated windows on Uranus. Each resolved trace
has exact constant intervals of 0.5, 0.25, 0.166667, or 0.125 seconds for the
2, 4, 6, and 8 request/s conditions, respectively. All windows completed
300/300 requests with zero failures, valid cumulative NVML energy, and clean
clock/process/patch cleanup.

The ordinary run throughput includes the finite trace's final drain and thus
understates its offered rate. The completion-span throughput below is measured
from the first to final request completion and is used together with bounded
early-to-late TTFT to assess steady operation.

| Offered rate | Completion-span throughput (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p99 (ms) | Late - early TTFT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2.000 +/- 0.002 | 172.88 | 272.93 | 54.96 | +8.20 |
| 4 | 4.008 +/- 0.004 | 157.59 | 280.45 | 58.03 | +3.03 |
| 6 | 6.417 +/- 0.652 | 1332.15 | 2024.02 | 80.44 | -1262.23 |
| 8 | 8.118 +/- 0.028 | 161.26 | 273.41 | 60.78 | +0.99 |

The anomalous 6 request/s aggregate is caused by the first window immediately
after server launch: its TTFT p50 was 3.68 seconds, whereas the later two were
161 and 159 ms. This is a startup/runtime conditioning transient rather than
queue accumulation; the later windows are stable. The 8 request/s windows,
which occur after that conditioning, all track the offered rate, have TTFT p50
of 158--163 ms, and show only -0.85 to +3.52 ms early-to-late TTFT changes.

**The stationary repeat rate is frozen at 8 request/s.** It is the highest
tested clearly stable rate, remains below the known-overloaded 16 request/s
condition, and was selected without considering placement or energy results.
Stage 2 uses five repetitions per C0--C5 configuration. Each newly launched
server first receives one complete unmeasured-for-analysis conditioning trace
to exclude the startup transient observed above.

The 30-run stationary matrix was submitted as Slurm job `271116` from commit
`49809f0`, using study directory
`results/raw/milestone-m-stationary-repeat-20261002T060128Z` on the same
exclusive Uranus node used for the rate sweep.

## Stationary 8 request/s results

Job `271116` completed `COMPLETED/0` in 1 hour 14 minutes 51 seconds. All 30
measured windows completed 300/300 requests with zero failures and 38,400
output tokens. They share one trace and experiment-plan hash, use cumulative
NVML energy, match all requested average clocks exactly, and have maximum
client p99 send delay 3.53 ms versus the 125 ms arrival interval. All six
conditioning traces also completed 300/300 requests. Cleanup reset clocks,
removed owned processes, reverted the patch, and preserved vLLM 0.15.1 and
0.26.0 in their respective environments.

Values are mean +/- sample standard deviation across five repetitions. Energy
includes all four GPUs and does not subtract idle energy.

| ID | Placement | A0/A1/F0/F1 (MHz) | TTFT p99 (ms) | TPOT p99 (ms) | Energy (J) | FFN energy (J) | Throughput (req/s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| C0 | `P_balanced` | 2520/2520/2520/2520 | 595.6 +/- 216.9 | 63.92 +/- 5.20 | 27663.9 +/- 342.7 | 17708.0 +/- 411.8 | 6.748 +/- 0.018 |
| C1 | `P_MH` | 2520/2520/2520/2520 | 651.5 +/- 427.9 | 65.61 +/- 5.84 | 27726.7 +/- 560.4 | 17713.5 +/- 661.4 | 6.721 +/- 0.014 |
| C2 | `P_balanced` | 2520/2520/1635/2520 | 585.7 +/- 317.1 | 65.25 +/- 6.83 | 25592.4 +/- 359.6 | 15583.9 +/- 493.4 | 6.704 +/- 0.052 |
| C3 | `P_MH` | 2520/2520/1635/2520 | 662.3 +/- 338.8 | 68.46 +/- 3.53 | 25677.2 +/- 70.3 | 15683.3 +/- 207.8 | 6.694 +/- 0.027 |
| C4 | `P_balanced` | 2520/2520/2520/1635 | 493.9 +/- 138.9 | 64.62 +/- 2.57 | 25658.6 +/- 190.4 | 15654.7 +/- 249.4 | 6.712 +/- 0.029 |
| C5 | `P_HM` | 2520/2520/2520/1635 | 730.0 +/- 247.7 | 72.01 +/- 5.17 | 25413.4 +/- 319.9 | 15429.4 +/- 308.0 | 6.697 +/- 0.036 |

### Placement effect at fixed heterogeneous frequencies

Signed changes use `(treatment - baseline) / baseline`; negative is favorable
for latency and energy, while positive is favorable for throughput.

| Comparison | TTFT p99 | TPOT p99 | Total energy | FFN energy | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| C3 vs C2, aligned MH | +13.07% | +4.92% | +0.33% | +0.64% | -0.15% |
| C5 vs C4, aligned HM | +47.79% | +11.44% | -0.96% | -1.44% | -0.22% |
| C1 vs C0, H/H control | +9.39% | +2.64% | +0.23% | +0.03% | -0.40% |

The five paired C3-vs-C2 TPOT changes are -4.43%, +6.52%, +6.38%, -0.22%,
and +19.51%; energy changes are +2.17%, +0.22%, -0.90%, +0.92%, and -0.68%.
The five C5-vs-C4 TPOT changes are +9.63%, +20.20%, +13.93%, +21.57%, and
-6.54%; energy changes are -0.18%, -2.05%, -0.04%, -2.65%, and +0.16%.
Neither orientation provides a repeatable additional placement benefit.

Alignment does redistribute FFN work as intended. For MH, slow GPU2 energy
falls 6.00% while fast GPU3 rises 5.70%; for HM, slow GPU3 falls 7.49% while
fast GPU2 rises 3.27%. The fast-rank increase again offsets most slow-rank
savings, leaving no consistent pool or total-energy improvement.

**Placement decision: C -- no measured effect.** At the tested 1635/2520 MHz
pair, frequency-aware individual expert placement does not provide a
repeatable additional benefit beyond fixed FFN DVFS. This conclusion does not
rule out other frequency tuples or constitute a global joint-optimization
result.

### Fixed-DVFS effect

| Comparison | TTFT p99 | TPOT p99 | Total energy | FFN energy | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| C2 vs C0, balanced MH vs HH | -1.67% | +2.07% | -7.49% | -12.00% | -0.65% |
| C4 vs C0, balanced HM vs HH | -17.07% | +1.08% | -7.25% | -11.60% | -0.54% |

The total-energy reduction occurs in every paired repetition: C2 saves
6.61--8.43%, and C4 saves 5.30--8.21%. TPOT changes remain noisy but average
only +2.07% and +1.08%, while throughput falls less than 0.7%.

**Fixed-DVFS decision: supported at the tested pair.** Lowering either FFN GPU
to 1635 MHz saves about 7.2--7.5% total energy and 11.6--12.0% FFN-pool energy
with a small mean TPOT/throughput cost. This does not establish that M/H is the
globally energy-optimal frequency tuple.

### Residual order trend

The conditioning trace removes the catastrophic first-window overload, but
TTFT p99 still decreases strongly from repetition 1 to 5 for every
configuration (correlations -0.84 to -0.97). Energy often increases slightly
with repetition, while TPOT trends are smaller and configuration-dependent.
Individual repetition values are therefore retained in the processed CSV and
figure rather than averaged away. This residual runtime/warm-state effect
limits interpretation of absolute TTFT differences, but it does not reverse
the placement result: C3 TTFT is worse than C2 in all five paired repetitions,
and C5 is worse than C4 in all five.

Do not proceed to window-level dynamic DVFS from this placement result. Any
future work should first explore other fixed FFN frequency tuples with a
placement frozen independently for each tuple; it should not treat this one
M/H test as a global joint-optimization verdict.

## Symmetric FFN frequency sweep

The next stage holds expert placement fixed at `P_balanced` and asks how far
the FFN pool itself can be downclocked before serving behavior degrades
substantially. This is a fixed-frequency characterization, not another
placement experiment and not dynamic DVFS.

Attention GPUs remain at 2520/2520 MHz. The preregistered symmetric FFN points
are 2520, 2220, 2070, 1635, and 1005 MHz. The two intermediate values and all
three H/M/L values are present in the saved L40S supported-clock inventory;
memory clocks are unchanged. All points use the same frozen `P_balanced`
mapping, 300-request GSM8K trace, greedy 128-token decoding, and corrected
8 request/s open-loop schedule.

The five frequency configurations are visited in one fixed random order using
seed `20261004`. Every frequency uses an independent server lifecycle: launch
the two AFD roles, run one full conditioning trace excluded from analysis,
measure three windows, then stop only those owned process groups. This design
prevents the known long-term server warm-state/order drift from being assigned
to whichever frequency happens to run early or late. There are 15 analyzed
runs. Model loading remains outside all energy windows, and each measured run
retains cumulative NVML energy, pool/per-GPU energy and power, request-level
latencies, throughput, client scheduling diagnostics, hashes, and clocks.

The analysis will report mean +/- sample standard deviation and signed changes
relative to 2520 MHz for TTFT, TPOT, throughput, total energy, and FFN-pool
energy. Energy--TPOT-p99 and energy--TTFT-p99 non-dominated points are retained
separately. No service model or placement selection is used to construct the
frontier.

The first prepared study, `results/raw/milestone-m-symmetric-20261002T151857Z`,
and pending job `271241` used one server for all frequencies. The job was
cancelled before allocation (0 seconds runtime), and that study is invalidated
before producing data. The corrected launcher also enforces exact equality
between runtime `HEAD` and the manifest's commit and refuses tracked worktree
changes. A new manifest must be created from the final corrected commit.

### Symmetric sweep results

Corrected exclusive Uranus job `271243` completed in 53 minutes 25 seconds.
All 15 measured windows completed 300/300 requests with zero failures and
38,400 output tokens. All runs share one trace, placement, and experiment-plan
hash; cumulative NVML energy is available; p99 client send delay is at most
3.08 ms; and measured average clocks match targets with 0 MHz maximum error.
Each of the five server lifecycles shut down without residual GPU processes.
Final cleanup reset clocks, reverted the patch, kept the pinned plugin clean,
and preserved vLLM 0.15.1 and 0.26.0 in their separate environments.

Values are mean +/- sample standard deviation across three repetitions. Signed
changes are relative to symmetric FFN H/H at 2520 MHz; Attention remains at
2520/2520 MHz. Energy includes all four GPUs and does not subtract idle energy.

| FFN MHz | TTFT p99 (ms) | TPOT p99 (ms) | Throughput (req/s) | Total energy (J) | FFN energy (J) | Total energy vs H | TPOT vs H |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2520 | 663.55 +/- 249.34 | 65.69 +/- 2.24 | 6.735 +/- 0.012 | 27,329.6 +/- 287.3 | 17,383.7 +/- 390.3 | reference | reference |
| 2220 | 751.60 +/- 167.01 | 66.49 +/- 1.00 | 6.734 +/- 0.022 | 24,810.7 +/- 160.0 | 14,861.0 +/- 228.7 | -9.22% | +1.23% |
| 2070 | 644.82 +/- 195.73 | 65.04 +/- 0.88 | 6.714 +/- 0.036 | 23,894.9 +/- 82.4 | 13,934.1 +/- 122.1 | -12.57% | -0.99% |
| 1635 | 682.46 +/- 155.23 | 67.33 +/- 2.12 | 6.692 +/- 0.016 | 23,396.6 +/- 37.8 | 13,414.0 +/- 135.8 | -14.39% | +2.50% |
| 1005 | 792.71 +/- 136.90 | 72.43 +/- 4.17 | 6.619 +/- 0.019 | 22,946.0 +/- 218.3 | 12,909.5 +/- 120.7 | -16.04% | +10.27% |

The Attention-pool energy remains approximately 9.95--10.04 kJ across the
sweep. The reduction comes from the FFN pool: -14.51% at 2220 MHz, -19.84% at
2070 MHz, -22.84% at 1635 MHz, and -25.74% at 1005 MHz. Mean FFN-GPU power
falls from approximately 191--197 W at 2520 MHz to 154--156 W at 2070 MHz,
148--150 W at 1635 MHz, and 141--142 W at 1005 MHz.

Using measured mean total energy and p99 latency, 2070, 1635, and 1005 MHz are
non-dominated for both the TPOT and TTFT views. The 2520 and 2220 MHz means are
dominated by 2070 MHz, although three repetitions do not justify a statistical
claim about their small latency differences.

**Practical interpretation:** both FFN GPUs can be reduced to 1635 MHz before
mean generation-latency or throughput degradation becomes substantial. At
1635 MHz, total energy is 14.39% lower, FFN-pool energy is 22.84% lower, TPOT
p99 is 2.50% higher, and throughput is 0.64% lower than H/H. Reducing further
to 1005 MHz saves only another 1.93% total energy relative to 1635 MHz, while
TPOT p99 rises another 7.58% and TTFT becomes worse. Thus 1635 MHz is the
approximate practical knee; 2070 MHz is the more conservative low-latency
sweet spot.

TTFT p99 remains highly variable and generally decreases from repetition 1 to
3 within each fresh server. The independent server lifecycle removes frequency
versus long-term server-age confounding, but it does not eliminate within-server
warm-state behavior or p99 sampling noise. Individual run values are retained
in `results/processed/milestone_m_symmetric_runs.csv`; conclusions rely more
heavily on the consistent energy and TPOT/throughput trends than on small TTFT
mean differences.

This stage changes neither expert placement nor frequency during a run. It
establishes a useful fixed symmetric FFN operating range, not evidence for
frequency-aware placement or dynamic DVFS.
