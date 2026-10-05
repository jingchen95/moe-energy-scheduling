# Milestone N: AFD FFN empirical cost model

## Scope

This milestone characterizes the current Qwen3-30B-A3B AFD 2A2F execution
path. It does not implement placement search, a joint scheduler, or dynamic
DVFS. The output is an empirical rank- and frequency-specific latency model
plus a power model fitted primarily from existing uninstrumented runs.

## Reused measurements

Primary power and window-energy data come from the completed, uninstrumented
Milestone M symmetric-frequency and target-split studies. Their processed-file
hashes and raw-directory identities are frozen in each study's
`reuse-manifest.json`. The colocated vLLM 0.15.1 Milestone G instrumentation is
an implementation reference only and is not model training data.

## Measurement boundaries and routing ground truth

The original preparation at `305dc6f` was invalidated before GPU execution.
It proposed joining one all-HH routing trace to later timing runs by
`(layer, call_sequence)`, and it timed the complete MoE block while calling the
result rank-local `C[g,f](n)`. Frequency and placement can change batching and
generated routes, so the cross-run join was not ground truth. In addition,
`FULL_DECODE_ONLY` bypasses Python layer hooks during graph replay; the first
implementation would not have emitted one fresh record per decode replay.

The corrected instrumentation stores routing and timing from the **same FFN
invocation and same run**. It embeds event records and a logical `topk_ids`
snapshot in each captured FFN CUDA graph, then resolves them at the beginning
of the next worker step. The AFD FFN worker already synchronizes CUDA after
every step, so this adds no per-call synchronization. Logical IDs are copied
after both timed intervals and histogrammed on CPU; no extra `bincount`, CPU
copy, or profiling kernel is inside either interval.

Two boundaries are retained:

- **Local fused-expert boundary:**
  `FusedMoEKernelModularImpl._fused_experts`, including local workspace
  preparation, expert assignment/permutation, and Triton expert kernels. It
  excludes EP all-gather dispatch and reduce-scatter combine. This is the only
  boundary used to fit `C[g,f](n)`.
- **Full per-layer MoE service boundary:** the complete native
  `Qwen3MoeSparseMoeBlock`, including router, EP dispatch, local experts,
  combine, and stream dependencies. It is retained to measure communication /
  synchronization residual and validate the later critical-path composition;
  it is not treated as an independent rank-local function of `n_g`.

Each physical FFN rank records its source-rank logical routes. The two rank
files are combined only within the same run on exact
`(runtime_step_id, stage, layer)` keys under schema 3. The old schema-2 local
counter join is not accepted as training data. The immutable placement maps the
combined logical histogram to the two physical-rank local loads.

## Shared identity and window protocol

The defect exposed by job `272171` was a missing cross-rank identity/control
contract: independent file reads and independently reset local counters could
not guarantee that the paired records represented the same serving step.
The artifacts establish one extra complete rank-1 step, not whether its cause
was activation or deactivation. The replacement fixes that protocol rather
than loosening the validator.

The pinned `AFDMetadataProviderMixin._next_afd_transaction_id()` increments a
counter owned by each Attention runner. Both V1 and V2 use this sidecar helper;
its allocations include context, warmup/capture and graph lifecycle work.
There is no agreement operation or documented shared DP identity in that
counter. In 2A2F, `P2pNcclAFDControlPlane` has two Attention metadata senders,
each connected to a different FFN receiver. Therefore copying their independent
transaction counters into the payload would not solve the problem.

The profiling-only solution is a monotonic counter on Attention DP rank 0,
coordinated using the existing `get_dp_group().broadcast_object(..., src=0)`
CPU control group. Each synchronized serving control send receives the same
`runtime_step_id`, `profile_generation`, and `profile_active` dictionary before
the two Attention senders publish their corresponding AFD payloads. Identity
is scoped to a server lifecycle and never reset at a profiling-window boundary.
Native DP batch coordination and idle dummy participation remain unchanged.
Graph capture, graph warmup and memory profiling do not allocate formal serving
IDs. A scoped V1 dummy-run flag distinguishes startup memory-profile calls
from idle-DP dummy serving calls. V2 remains disabled for this experiment.

Before replay, the benchmark awaits Attention collective RPC
`set_afd_cost_profile_state` with the run generation and `active=true`. Pinned
`DPLBAsyncMPClient.call_utility_async()` awaits all engine RPCs using
`asyncio.gather`, returning the first engine result only after all complete.
The client validates the acknowledgement, then starts measured traffic. It
awaits the inactive RPC after replay and sends excluded flush traffic.
The FFN worker resolves the previous pending CUDA events after its existing
step synchronization, applies the current received payload state, then
executes/replays. The last active step is thus resolved before the inactive
flush closes its generation. Filesystem observation order is irrelevant to
membership. The older routing-only profiler is outside this correction.

The cost-profile schema is now 3. Records retain runtime ID, stage, layer,
physical rank/GPU, frequency, real token rows, logical histogram, local/full
elapsed times, run mode, and rank-local diagnostic counter. Only runtime ID,
stage and layer are join keys. The finalizer rejects missing/mismatched IDs,
duplicate keys, incomplete 48-layer steps, missing interior active steps,
generation differences and routing/timing violations. It never truncates,
shifts, tolerates a missing step or heuristically pairs rows. Captured timing
events remain `external=True`; eager events use the existing default. No new
per-call CUDA synchronization or timed GPU histogram operation is introduced.

The shared CPU collective is profiling overhead outside both CUDA timer
boundaries. These runs are characterization, not primary serving-performance
or NVML energy comparisons. Uninstrumented energy reuse remains unchanged.
The patch is source-controlled with exact seven-file before/after hashes:
payload codec, Attention metadata/worker/dummy exclusion, FFN worker/runner,
and the existing Qwen ownership/timing wrapper. It touches no site-packages.
Optional payload fields default to absent/inactive, preserving the ordinary
non-profiled wire format. The dev RPC endpoint is enabled only for profiling
Attention launchers, bound to loopback. It must not be exposed as a public API.

## Characterization matrix

One exclusive Uranus allocation has two phases.

**Phase A:** one balanced/HH `FULL_DECODE_ONLY` server exercises eight independent
8-request profiling generations, using the production RPC/payload protocol.
Each generation must pass actual replay/event resolution, exact runtime-ID
pairing, 48 layers per step, top-8 routing, positive finite timing and containment.
First/last IDs, step counts and diagnostic counters are retained. IDs must
increase across windows. Any failure ends the allocation before formal data;
no automatic resubmission occurs. All stress traffic is excluded from training.

**Phase B:** restart, then perform the existing eleven fresh-server lifecycles:

1. five `P_balanced` symmetric FFN frequencies: 1005, 1635, 2070, 2220, 2520 MHz;
2. independent balanced lifecycle repeats at 1635 and 2070 MHz;
3. the existing 40/60 rank-0 and rank-1 placements at 1635 and 2070 MHz.

Every configuration uses a fresh server, one complete conditioning trace, and
one measured trace. The first formal point remains balanced/HH; the remaining
ten retain their original deterministic order randomized with seed
`20261007`. Attention remains at 2520/2520 MHz. Every run produces its own
same-run routing/local-timing/full-timing artifact; no cross-run alignment is
allowed.

### Transactional completion and resume

Each allocation writes to `allocations/<Slurm job ID>/`, and each formal
configuration has an `attempt-<Slurm job ID>/` directory. Old partial attempts
remain untouched. `COMPLETE.json` is created only after 300/300 requests, zero
failures, finalized shared-ID pairing, valid actual clocks, matching runtime
ownership, all required request/config/energy/profile/log files, and successful
owned-server shutdown. A completion marker binds source commit, patch SHA,
profiler schema, trace SHA, placement SHA and complete plan/config SHA, plus
all attempt artifact hashes. A resume may skip only an exactly matching,
unchanged completed attempt; drift fails closed. It reruns Phase A first.
Changed code/schema/scientific inputs require a new study. This does not admit
the legacy job `272171` into the final dataset.

Each Phase A window exercises graph replay on both ranks, CUDA-event
`elapsed_time()`, the same-invocation routing snapshot, profile finalization,
and exact two-rank alignment. Its generation is finalized before the next
window; the fresh formal lifecycle starts a different generation.

The profile validator does not accept its own derived assignment total as the
only routing check. Qwen3-30B-A3B routes each token to eight experts, so every
source-rank invocation must satisfy `sum(histogram) == 8 * token_rows`.
Combined histograms must equal the sum of both source ranks and satisfy the
same count invariant. All local/full timings must be finite and positive; at
least 99% of local intervals must lie within their full intervals, allowing a
0.01 ms numerical tolerance.

## Planned Offline Models

No model is fitted in the protocol-correction task. The intended modeling
method below is unchanged, and remains gated on valid schema-3 GPU data.

The primary latency model is a binned interpolation table
`C[g,f](n)`, kept separate for physical GPUs 2 and 3. Here `n` is the number of
logical expert assignments owned by the physical FFN rank, reconstructed from
that run's routing histogram and immutable placement. Lifecycle repeats and
40/60 placements are held out as complete server lifecycles.

The power model is a physical-rank-specific linear relationship between routed
assignment rate and average power at 1635/2070 MHz. It is validated with
leave-one-configuration-out splits over uninstrumented runs. Other frequencies
remain balanced-load anchors; no unsupported load dependence is invented.
Multiplying predicted power by CUDA-event latency is reported only as a
derived per-call surrogate, never as direct NVML call energy.

## Results

The first corrected consolidated attempt, Slurm job `271512` using study
`milestone-n-afd-cost-20261003T210209Z`, failed during the mandatory first
HH gate after 2 minutes 37 seconds. No server became healthy, no request was
sent, and no timing or energy result was produced.

The failure was an instrumentation lifecycle error, not a model result. The
profiler tried to access `MoERunner.quant_method.impl` while Qwen layers were
still being constructed. Pinned vLLM 0.26 exposes its modular local-expert
boundary after weight processing through
`MoERunner._quant_method.moe_kernel.impl._fused_experts`. Profiler installation
is therefore deferred until the FFN model loader returns, while routing and
full-layer hooks remain unchanged. Cleanup reverted the AFD patch, reset GPU
clocks, left no completed run entry, and preserved both isolated and protected
environments.

GPU validation and offline analysis remain pending a fresh replacement study.

A second replacement, job `271609`, failed before server launch because the
clock-control helper compared an idle instantaneous SM clock against the lock
target. GPU0 reported 2475 MHz immediately after a successful 2520 MHz
`nvidia-smi -lgc` command. This is a setup-verification false negative rather
than a timing result; no requests ran. Milestone N now records that idle
readback but defers equality validation to the measured window, where the
average actual clock for every GPU must match its target within 5 MHz. Other
users of the generic clock-control API retain immediate verification by
default.

Job `271652` progressed further: the AFD services and runtime ownership passed,
and its full 300-request conditioning trace completed with all four measured
average clocks at 2520 MHz. The first measured-window replay then failed on
both FFN ranks when `elapsed_time()` queried timing events captured in default
internal mode. PyTorch documents internal captured events as graph
cross-stream dependencies; the corrected profiler creates external timing
events during graph capture so the replay contains queryable event-record
nodes. Eager timing events remain internal/default. The failed job contributes
no model data.

The next replacement, job `271681`, was canceled while pending after review
identified three reliability gaps. It consumed no GPU time. The fail-fast
profiled mini-smoke and independent routing/timing invariants described above
now close the two data-validity gaps. Cleanup was also changed from checking
only leader liveness to validating and terminating the complete stored PGID,
so an exited or zombie group leader cannot leave live AFD workers behind.
The hardened implementation passes 143 project CPU tests and 63 focused tests
against the patched pinned AFD plugin; patch revert restores a clean checkout.

Job `271687` validated the new instrumentation gate on real GPUs. The
8-request mini-smoke and first formal 300-request all-HH run both passed exact
two-rank alignment, top-8 count invariants, finite/positive timing, and 100%
local-within-full containment. The formal run completed 300/300 requests with
zero failures and produced 30,912 records per FFN rank at measured 2520 MHz.
The job then stopped after 1/11 lifecycles because vLLM had renamed live worker
processes to `VLLM::DPCoordinator`, which the cleanup identity guard rejected.
This is an incomplete engineering run and is not used for model fitting.
Cleanup now recognizes such title-rewritten workers only when same-user PGID,
resolved project-local AFD Python executable, and project working directory
all match; it does not broaden termination beyond the stored process group.

Replacement job `271750` reproduced a second shutdown-only race after another
valid mini-smoke and formal HH profile. A worker could disappear after the
PGID member snapshot but before `ps` read its state; under `set -e`, that
expected `ps` miss terminated the launcher silently. Cleanup now treats an
empty `ps` result as an already exited member, while all surviving processes
still require the ownership and runtime identity checks. The launcher also
persists failure stage, source line, and command for any subsequent failure.

### Job 272171: partial progress, rank-window alignment failure

Study `results/raw/milestone-n-afd-cost-20261005T073112Z` used source
`2090fe46d358d279352b9c0d8f46bd3a8a22f6d7`. Slurm reports `FAILED`, exit
`1:0`, after `00:28:24` on Uranus. The profiled mini-smoke passed, followed
by three finalized formal lifecycles out of the eleven planned.

| Order | Placement | FFN MHz | Completed / failed | Records rank 0 / rank 1 | Status |
| --- | --- | --- | --- | --- | --- |
| 0 | P_balanced | 2520 / 2520 | 300 / 0 | 28,896 / 28,896 | Finalized |
| 1 | P_split_40_slow_rank0 | 1635 / 1635 | 300 / 0 | 28,752 / 28,752 | Finalized |
| 2 | P_split_40_slow_rank1 | 2070 / 2070 | 300 / 0 | 27,216 / 27,216 | Finalized |
| 3 | P_split_40_slow_rank1 | 1635 / 1635 | Not finalized | 28,176 / 28,224 | Rejected |

The three finalized runs matched their target average clocks exactly on all
four GPUs. Client p99 send delays were 2.347, 2.902, and 2.916 ms. Both ranks
passed routing top-8 and positive/finite timing checks; local-within-full
containment was 100%. Their step/stage/layer arrays matched. The instrumentation
adds substantial overhead; these serving/energy observations are not substituted
for the frozen uninstrumented Milestone M energy dataset.

The fourth run failed in `finalize_same_run_cost_profiles()` with
`FFN ranks do not align in same run: step_sequences`. Rank 0 recorded 587
steps and rank 1 recorded 588, with all 48 layers in every step. Individual
rank top-8 and timing-containment checks still passed. The validator correctly
refused to combine these records. Final request/energy/metrics files were not
written because profile finalization precedes their persistence; this is not
evidence that a serving request itself failed.

Source inspection identifies an unresolved boundary/identity weakness: each
FFN rank independently polls the shared control file and resets its own local
step counter. Atomic file replacement does not synchronize the ranks' reads.
A control-transition race is therefore a plausible cause of the one-step
difference, but the existing artifacts cannot determine whether the mismatch
arose at start or stop. Moreover, equality of independently reset counters is
not by itself proof of a common runtime invocation. No truncation or offset
join is used to salvage the fourth run. Successful raw artifacts remain
preserved, but model fitting is withheld pending a coordinated boundary and
shared step identity, including CPU tests with deliberately skewed observations.

The earlier cleanup defects did not recur. Per-configuration shutdown records
for the first three lifecycles and final allocation cleanup show no compute
applications remaining. Graphics-clock reset completed with empty stderr;
post-reset idle clocks are not used as lock-target validation. Patch revert
left the pinned plugin clean at `4c2541a53ec8ad507508073eac81ac9feecc6c9a`.
The post-job environment check retained protected vLLM 0.15.1 / PyTorch
2.9.1+cu128 and isolated vLLM 0.26.0 / PyTorch 2.11.0+cu130 at their original
package roots.

This attempt is incomplete engineering characterization, not a scientific
model result. No replacement GPU job is submitted as part of this inspection.
A short multi-lifecycle gate should precede another full matrix once the
boundary/identity issue is resolved; do not simply resubmit the unchanged job.

Post-inspection validation: the complete repository CPU suite, explicitly
scoped as `PYTHONPATH=.deps /data/users/chjing/miniforge3/envs/cuda-env/bin/python
-m pytest -q tests`, passed 143 tests in 9.92 seconds; `git diff --check` passed.
An initial unscoped `pytest -q` invocation also discovered ignored third-party
AFD runtime/E2E tests and failed collection in the protected 0.15.1 interpreter
(14 collection errors). Those tests require the separate pinned AFD runtime
and are not the project CPU suite. No dependency, runtime, or patch changes
were made to address that invocation error.

### Legacy boundary diagnostic

`tools/diagnose_milestone_n_legacy_boundaries.py` reads the preserved chunks
without repairing or joining them. The JSON report is stored beside the old
study as `protocol-boundary-diagnosis.json`. All listed steps have 48 layers
at stage 0. The head/tail columns show token rows in the first/last six local
steps, not inferred cross-rank correspondence.

| Order | Rank | Records | First / Last local step | Head token rows | Tail token rows |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 28,896 | 0 / 601 | 13,64,64,64,64,7 | 64,64,64,64,64,64 |
| 0 | 1 | 28,896 | 0 / 601 | 64,64,64,13,64,64 | 64,64,64,64,64,64 |
| 1 | 0 | 28,752 | 0 / 598 | 64,64,64,13,64,7 | 64,64,64,64,64,64 |
| 1 | 1 | 28,752 | 0 / 598 | 13,64,64,64,64,64 | 64,64,64,64,64,64 |
| 2 | 0 | 27,216 | 0 / 566 | 13,64,64,64,64,64 | 64,64,64,64,64,64 |
| 2 | 1 | 27,216 | 0 / 566 | 64,64,64,13,64,7 | 64,64,64,64,64,64 |
| 3 | 0 | 28,176 | 0 / 586 | 13,64,64,64,64,7 | 64,64,64,64,64,64 |
| 3 | 1 | 28,224 | 0 / 587 | 64,64,64,13,64,64 | 64,64,64,64,64,64 |

Resolved configuration declares `FULL_DECODE_ONLY`, but legacy chunks do not
record per-invocation graph/run mode or a shared runtime ID. Repeated token-row
values cannot establish whether the unmatched step is at activation or
deactivation, nor prove post-hoc identity for the three equal-length runs.
They are preserved as engineering evidence, not automatically admitted to
the final training dataset. The failed fourth run is not salvaged.

### Protocol correction validation

- Complete project CPU suite: **168 passed**, no skips.
- Focused profiler/protocol tests: **43 passed**, no skips.
- Pinned-plugin/protocol suite: **266 passed**, no skips. It covers payload
  encode/decode, Attention/FFN workers and runners, graph policy, V2 runner
  compatibility, P2P connector, Qwen construction/loading, and a real native
  two-process Gloo broadcast across eight generations.
- Adversarial tests cover independently displaced hypothetical file reads,
  active/inactive and generation transitions, different diagnostic counters,
  resolving the last active step before closure, eager/replay, excluded
  warmup/capture/memory profiling, duplicate/missing/mismatched IDs, malformed
  routes/timings, and transactional completion/fingerprint/artifact drift.
- Patch apply -> verify -> revert -> clean passed for all seven pinned files.
  Shell syntax and `git diff --check` passed. No GPU experiment was needed for
  these checks, and no protected package was installed or edited.
- A CPU preparation check compared all eleven formal definitions and their
  order against job 272171's manifest: placements, hashes, frequencies, trace,
  request rate and matrix are unchanged. The only additional traffic is the
  excluded eight-generation Phase A gate.
- Environment artifact `results/raw/milestone-n-protocol-environment-20261005.json`
  matches the old job's protected/isolated version and package-root records
  exactly: vLLM 0.15.1 / 0.26.0 respectively; the pinned plugin is clean.

These CPU checks establish protocol regression coverage; the subsequent GPU
integration result is recorded below. Preparation was committed and pushed
before immutable study creation, with tracked HEAD unchanged during execution.

### Job 272216: shared protocol GPU validation complete

Study `results/raw/milestone-n-afd-cost-20261005T115917Z` executed source
`068914350043720bd840a577874c80191eb6e10e`. Slurm reports `COMPLETED / 0:0`
after `01:19:48` on Uranus. The single allocation completed both phases;
no replacement job was submitted.

**Phase A:** all eight generations on the same balanced/HH server passed.
Each completed 8/8 requests with zero failures and exercised CUDA graph replay,
event resolution, same-invocation routing capture, window closure and exact
cross-rank finalization. Active step ranges were 155-300, 601-746, 1049-1193,
1497-1641, 1945-2089, 2393-2537, 2842-2984 and 3290-3433. IDs advanced across
windows instead of restarting. Gaps between windows are excluded inactive
traffic, not missing interior active steps. Each active step contained all
48 layers. The gate server shut down before the formal fresh-server matrix.

**Phase B:** all eleven formal configurations completed 300/300 requests,
zero failures and 38,400 output tokens each. Both rank streams paired exactly
on schema-3 `(runtime_step_id, stage, layer)` keys, without truncation, offsets,
tolerance or cross-run routing joins.

| Order | Configuration | FFN MHz | Active steps | Records per rank | Instrumented TTFT p99 ms | Instrumented TPOT p99 ms |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | balanced primary | 2520 | 601 | 28,848 | 8598.564 | 105.333 |
| 1 | split-40 slow-rank-0 | 1635 | 565 | 27,120 | 9246.137 | 124.831 |
| 2 | split-40 slow-rank-1 | 2070 | 607 | 29,136 | 5208.620 | 103.514 |
| 3 | split-40 slow-rank-1 | 1635 | 590 | 28,320 | 8625.676 | 112.402 |
| 4 | balanced primary | 1005 | 588 | 28,224 | 2013.115 | 91.092 |
| 5 | split-40 slow-rank-0 | 2070 | 595 | 28,560 | 10723.718 | 114.901 |
| 6 | balanced held-out repeat | 1635 | 575 | 27,600 | 3607.695 | 103.096 |
| 7 | balanced primary | 2070 | 596 | 28,608 | 3853.765 | 99.086 |
| 8 | balanced primary | 1635 | 582 | 27,936 | 9849.473 | 111.532 |
| 9 | balanced held-out repeat | 2070 | 588 | 28,224 | 6829.361 | 101.370 |
| 10 | balanced primary | 2220 | 583 | 27,984 | 3718.800 | 96.233 |

There are 6,470 formal active steps and 310,560 records per rank (621,120
rank-specific local/full timing observations). Same-run top-8 routing checks,
combined histogram checks and finite/positive timing checks all passed.
Local-within-full containment was 100% on both ranks in every configuration,
under the existing 0.01 ms numerical containment tolerance. Both ranks recorded
eager and graph-replay observations; formal replay fractions were 22.30-25.04%.

Post-job offline inspection independently re-finalized all eight gate profiles
and all eleven formal profiles into a separate derived directory, comparing
every regenerated array against the preserved combined NPZ. It also verified
all eleven COMPLETE fingerprints and 1,739 recorded artifact hashes, including
trace/placement identity and role logs. No raw artifact was overwritten.
The common trace SHA-256 is
`6cf46c02b3a88ace790c6f9b59dbb7b5f44a9872a5813584a3f3a232b39b1020`.
Actual average clocks matched `[2520,2520,f,f]` exactly in every formal run;
client p99 send delay was 2.135-3.381 ms against 125 ms inter-arrival spacing.
All four cumulative NVML energy deltas were positive.

**Interpretation:** this closes the shared-identity/window protocol GPU gate.
It does not yet validate an empirical cost model or establish scientific
energy/latency gains. Profiling changes batching/host overhead substantially:
instrumented end-to-end latency is variable and not monotonic with frequency.
Do not infer that 1005 MHz is faster from this table or replace the frozen
uninstrumented Milestone M energy/performance measurements with these runs.
The adequacy of local cost features and held-out candidate ranking remains an
offline analysis question; no model was fitted during this completion check.
Legacy schema-2 job 272171 and Phase A traffic remain excluded from model data.

Cleanup passed after every formal lifecycle and at allocation exit. The final
GPU-process CSV contains only its header. Graphics clocks reset without stderr;
the patch is not applied and the pinned plugin is clean at its original commit.
Protected vLLM 0.15.1 / PyTorch 2.9.1+cu128 and isolated vLLM 0.26.0 /
PyTorch 2.11.0+cu130 versions and package roots exactly match the preparation
baseline. Post-inspection project tests: **168 passed**, no skips;
`git diff --check` passed. No new runtime, placement or model code changed.

Artifacts, synchronized into the local project:

- Full raw study: `results/raw/milestone-n-afd-cost-20261005T115917Z/`.
- Integrity summary: `results/processed/milestone_n_272216_integrity/summary.json`.
- Per-configuration diagnostics:
  `results/processed/milestone_n_272216_integrity/formal-configurations.csv`.
- Read-only inspection source:
  `results/processed/milestone_n_272216_integrity/inspection-source.py`.

Next step is offline analysis of these schema-3 local timings, held-out repeats
and distribution checks, together with the existing uninstrumented power data.
Do not submit another GPU job or proceed to a placement/DVFS oracle before the
model validation decision gate below is assessed.

## Decision gate

The next milestone may construct an offline `(placement, frequency)` oracle
only if held-out timing and power errors are adequate for correct candidate
ranking. If `n` alone fails and residuals depend on expert distribution, this
milestone stops with that negative result and uses the already captured richer
features for offline analysis rather than submitting another GPU job.
