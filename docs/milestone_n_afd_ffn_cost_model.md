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
`(step_sequence, stage, layer)` keys. The immutable placement then maps the
combined logical histogram to the two physical-rank local loads.

## Characterization matrix

One exclusive Uranus allocation performs eleven fresh-server lifecycles:

1. a mandatory first `P_balanced` at 2520/2520 MHz same-run alignment gate;
2. `P_balanced` profiling at symmetric FFN frequencies 1005, 1635, 2070, 2220,
   and 2520 MHz;
3. an independent lifecycle repeat at 1635 and 2070 MHz;
4. timing-only distribution checks for the existing 40/60 rank-0 and rank-1
   placements at 1635 and 2070 MHz.

Every configuration uses a fresh server, one complete conditioning trace, and
one measured trace. After the forced gate, order is randomized with seed
`20261007`. Attention remains at 2520/2520 MHz. Every run produces its own
same-run routing/local-timing/full-timing artifact; no cross-run alignment is
allowed.

Before order 0 performs its full conditioning trace, it runs an independent
8-request profiled mini-smoke. This gate must exercise graph replay, CUDA-event
`elapsed_time()`, the same-invocation routing snapshot, profile finalization,
and exact two-rank alignment. Its generation is finalized before conditioning,
and the formal measurement later starts a new generation.

The profile validator does not accept its own derived assignment total as the
only routing check. Qwen3-30B-A3B routes each token to eight experts, so every
source-rank invocation must satisfy `sum(histogram) == 8 * token_rows`.
Combined histograms must equal the sum of both source ranks and satisfy the
same count invariant. All local/full timings must be finite and positive; at
least 99% of local intervals must lie within their full intervals, allowing a
0.01 ms numerical tolerance.

## Offline models

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
The hardened implementation passes 142 project CPU tests and 63 focused tests
against the patched pinned AFD plugin; patch revert restores a clean checkout.

## Decision gate

The next milestone may construct an offline `(placement, frequency)` oracle
only if held-out timing and power errors are adequate for correct candidate
ranking. If `n` alone fails and residuals depend on expert distribution, this
milestone stops with that negative result and uses the already captured richer
features for offline analysis rather than submitting another GPU job.
