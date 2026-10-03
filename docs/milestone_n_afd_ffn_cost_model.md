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

## Timed boundary

CUDA events surround
`AFDQwen3MoeDecoderLayer.compute_ffn_output` immediately before the native
`Qwen3MoeSparseMoeBlock` call through its returned output. In the pinned
vLLM 0.26 Qwen3 path this includes the internal router, EP dispatch/all-to-all,
routed-expert execution, shared-expert work, combine, and CUDA dependencies
that return to the caller stream. It excludes AFD input transport before the
decoder-layer FFN call and AFD output transport after the call. It is therefore
an **FFN/MoE service-boundary time**, not expert-GEMM latency.

The timing path records only CUDA events, layer, call sequence, and input token
rows. It performs no histogram, `bincount`, CPU copy, or per-call synchronize.
Events are resolved after one synchronization at the explicit dump boundary.

## Separate routing trace

The routing-only pass uses `P_balanced`, all GPUs at 2520 MHz, the frozen
300-request GSM8K trace at 8 requests/s, and `FULL_DECODE_ONLY`. Each source
rank buffers logical-expert histograms on GPU and flushes compressed chunks.
Source-rank records are joined only on an exact `(layer, call_sequence)` key;
the combined trace retains source-rank histograms and token rows. Timing runs
must have the exact same keys and rank-local token rows or validation fails.

## Characterization matrix

One exclusive Uranus allocation performs:

1. one routing-only lifecycle and immediate validation;
2. `P_balanced` timing at symmetric FFN frequencies 1005, 1635, 2070, 2220,
   and 2520 MHz;
3. an independent lifecycle repeat at 1635 and 2070 MHz;
4. timing-only distribution checks for the existing 40/60 rank-0 and rank-1
   placements at 1635 and 2070 MHz.

Every configuration uses a fresh server, one complete conditioning trace, and
one measured trace. Configuration order is randomized with seed `20261007`.
Attention remains at 2520/2520 MHz. Routing and timing never run in the same
measured window.

## Offline models

The primary latency model is a binned interpolation table
`C[g,f](n)`, kept separate for physical GPUs 2 and 3. Here `n` is the number of
logical expert assignments owned by the physical FFN rank, reconstructed from
the separate routing trace and immutable placement. Lifecycle repeats and
40/60 placements are held out as complete server lifecycles.

The power model is a physical-rank-specific linear relationship between routed
assignment rate and average power at 1635/2070 MHz. It is validated with
leave-one-configuration-out splits over uninstrumented runs. Other frequencies
remain balanced-load anchors; no unsupported load dependence is invented.
Multiplying predicted power by CUDA-event latency is reported only as a
derived per-call surrogate, never as direct NVML call energy.

## Results

Pending the consolidated GPU job and offline analysis.

## Decision gate

The next milestone may construct an offline `(placement, frequency)` oracle
only if held-out timing and power errors are adequate for correct candidate
ranking. If `n` alone fails and residuals depend on expert distribution, this
milestone stops with that negative result and uses the already captured richer
features for offline analysis rather than submitting another GPU job.
