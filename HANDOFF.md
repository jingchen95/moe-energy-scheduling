# Handoff

## Latest status — Torch Profiler export storage repair

Status: **`full_model_trace_export_verified_vera_smoke`**.

The user's rerun `results/raw/moe-torch-10256329-20260909T182003Z`, source
`956945f`, passed on the same four-A100 allocation. All three measured
requests completed and four valid uncompressed traces (about 425–439 MB
each) were exported despite node `/tmp` still being full. Independent
read-only revalidation of all four raw traces reproduced the saved kernel
counts and totals, with CUDA Graph kernels on every device. This verifies
the storage/export repair; it does not establish negligible profiling overhead.

The saved heuristic share is 48.3462% (24.2304% recognized MoE compute/token
organization plus 24.1158% AllGather/ReduceScatter). Inspection found the
explicit `vllm::moe::topkGating` kernel in the unmatched list, contributing
another 2.8300 percentage points; including it gives 51.1762%. This is a
separate derived estimate, not an edit to the saved report or a complete
layer attribution. Generic/activation kernels remain unassigned. Both
percentages use summed GPU kernel durations, not E2E wall time. Mean
profiled TTFT/TPOT/E2E were 46.87 ms / 9.93 ms / 1.309 s. Rank 2 still logged
129 `destroyEvent` out-of-memory warnings during warmup, before profiling;
their cause remains unresolved. This is a three-request, low-concurrency
Vera smoke diagnostic, not a high-load performance comparison.

The following records the original failure and the repair verification before
that successful full-model rerun.

The first full-model Torch run,
`results/raw/moe-torch-10256329-20260909T180319Z` (source `97c0d72`), completed
all three measured requests, each with 128 output tokens. Both profiler control
endpoints returned HTTP 200, but all four workers logged
`Failed to write to log file!` at 20:12:32–33 local time. Each final gzip file
was 81 bytes and decompressed to zero bytes. The run correctly ended FAIL;
there is no usable MoE kernel-time percentage. These failed artifacts remain
unchanged.

Read-only checks in the user's existing allocation `10256329`, on
`vera-m11-20`, identified the export failure: node `/tmp` was a 974 MiB
filesystem, 100% used, with only 52 KiB available. The run's compile cache
occupied 669 MiB and four incomplete Kineto `.json.tmp` files, timestamped at
the export failure, occupied about 237 MiB. The shared result filesystem had
ample space. The pinned PyTorch gzip path stages a full uncompressed trace
through Python's temporary directory before compressing it into the requested
output path; the previous script did not relocate that temporary directory.
No files in node `/tmp` were deleted or modified during diagnosis/repair.

This run used four A100-SXM4-40GB GPUs with NV4 links, not the previous A40
node. It remains Vera environment smoke, not comparative study data. Mean
profiled TTFT/TPOT/E2E were 46.96 ms / 9.89 ms / 1.303 s. Rank 3 also emitted
CUDA `out of memory (function destroyEvent)` warnings during warmup, before
the explicit start-profile call; their cause remains unresolved. Successful
requests do not establish an unaffected latency baseline.

The repaired launcher now:

- keeps `runtime-cache/` and `tmp/` inside each unique result directory instead
  of falling back to small node `/tmp`;
- sets container `TMPDIR`, `TMP`, and `TEMP` to `/run-output/tmp` for server
  and client, while preserving the existing `/runtime-cache` cache settings;
- disables gzip and exports `profile/*.pt.trace.json` directly (Kineto's
  adjacent temporary output is also on the result filesystem);
- checks the actual container temporary-directory resolution, at least 5 GiB
  of free space, and successful file write/flush before loading the model;
  this is a guard, not a guarantee against later quota or space exhaustion;
- records storage availability/inodes at start and cleanup, and rejects native
  log-open/log-write errors even if the HTTP controls return 200;
- reports empty/truncated trace filenames with an export/storage diagnostic,
  while retaining support for valid historical gzip traces.

The vLLM command block and frozen workload are unchanged: no eager mode,
CUDA Graph/compile change, per-layer instrumentation, synchronization, package
upgrade, or site-packages edit. Moving caches to shared storage may change
cold-start time; warmup still precedes the measured requests. No full model
was launched as part of this repair, and no existing benchmark was overwritten.

Verification: 25/25 tests passed inside the pinned SIF, including actual
uncompressed CPU trace export without Python tempfile staging, unchanged
`ProfilerConfig` graph hash, four-worker uncompressed parsing, empty/truncated
gzip rejection, and storage failure cases. The actual container storage
preflight also passed with the new mounts/environment. Evidence is under
`results/raw/torch-storage-check-fzUqmJ/`. Shell/Python syntax, unchanged model
command comparison, `git diff --check`, and recognition of all four native
errors in the failed run's log passed. Full-model GPU validation of this
repair is still pending. Unrelated staged files/document deletions were left
untouched and are excluded from this milestone.

Rerun inside the user's existing four-GPU allocation:

```bash
bash scripts/moe_torch_profile.sbatch
```

There is no need to clear old `/tmp` files for the relocated cache/export
paths. Success still requires `status.txt` = PASS and a validated
`moe-kernel-share.json`; the presence of a trace file or HTTP 200 is insufficient.

## Previous milestone — Torch Profiler CUDA Graph diagnostic

Status: **`container_offline_verified_full_model_gpu_run_pending`**.

At the user's request, the active profiling entry point is now
`scripts/moe_torch_profile.sbatch`. The historical Nsight launcher and all raw
failed runs remain intact. The new launcher executes the pinned vLLM 0.15.1
SIF directly, with the built-in Torch Profiler; there is no external Nsight
process, Nsight installation requirement, Conda dependency, or host CUDA module
load. No packages, container image, or site-packages were modified.

The model command still fixes TP=1, DP=4, MP, EP, linear placement,
`allgather_reducescatter`, disabled prefix caching, and `FULL_AND_PIECEWISE`
CUDA Graph mode. It does not enable eager execution or add per-layer timers,
synchronization, or hooks. `src/torch_profile.py` holds the resolved profiler
settings: disable stack/shape/memory/FLOPS tracing, disable the frontend CPU
profiler, and disable nested operator summary tables. Worker CPU+CUDA traces
are exported as `profile/*.pt.trace.json.gz` in a fresh run directory.
The server logs enable Python faulthandler for additional crash evidence;
this cannot guarantee a diagnostic for SIGKILL or every native failure.

The Torch client reuses the historical `src/nsys_profile.py` workload helper
(the filename does not imply an Nsight dependency). It preserves the frozen
trace hash, eight warmups, five settle seconds, and three measured requests
at offsets 0/2/4 seconds. The helper now records start/stop errors, attempts
stop after a partially failed start, saves completed response rows before
stopping, and writes invalid metrics even if profiler control fails. Raw
TTFT/TPOT fields remain in the response rows. Export time is not included in
the client measurement-completed timestamp. The driver waits for stop/export
before stopping the owned server group; failures retain any traces already
written and do not attempt to repair or relabel them as successful.

`src/torch_moe_share.py` reads only Chrome trace `cat=kernel, ph=X` events.
Durations are microseconds, regardless of `displayTimeUnit`. CPU operator
totals, CUDA runtime API durations, graph aggregate records, memcpy and memset
are not summed with kernel time. The aggregation requires four distinct
worker traces, one device per trace, covering CUDA devices 0/1/2/3, plus
evidence of graph replay kernels from launch correlation IDs or graph/node
metadata on each GPU. Missing graph attribution is an explicit failure
requiring inspection of raw traces, not permission to report a prefill-only
fraction. The parser also requires three successful responses and successful
start/stop controls; the driver checks vLLM's internal profiler-failure logs
because the pinned worker wrapper may swallow exceptions and still return
HTTP 200. No percentage is written for a failed validation.

The output `moe-kernel-share.json` contains a kernel-name-based **approximate**
MoE compute-plus-AgRs share, microsecond totals, converted GPU-millisecond
totals, per-device capture counts, and largest unmatched kernels. This is
neither E2E wall-time share nor a mathematically guaranteed lower bound; it
retains the old `conservative_moe_cuda_kernel_time_share_percent` field name
for compatibility. Primary latency still requires an otherwise identical
unprofiled run. Profiling overhead has not been measured.

Verification on September 9:

- 19/19 unit tests pass inside the actual pinned SIF, including real CPU trace
  export/rejection, gzip parsing, graph metadata/correlation, missing and
  duplicate devices, invalid durations, missing MoE kernels, failed benchmark
  gates, and start/stop HTTP failures with preserved responses;
- the container's actual `ProfilerConfig` accepts all settings and has the
  same computation-graph hash as the default config; Kineto is available;
- both profiling shell scripts pass `bash -n`, Python syntax checks pass, and
  `git diff --check` passes;
- no active user Slurm allocation was available; no allocation was created,
  and no new full-model GPU run or percentage is claimed.

Inside a four-GPU Slurm allocation, run from the repository root:

```bash
bash scripts/moe_torch_profile.sbatch
```

The launcher prints its unique `results/raw/moe-torch-<job>-<UTC>` directory,
startup/client progress, and a summary only on successful validation. Inspect
`status.txt`, `benchmark/metrics.json`, `benchmark/profile_control.json`,
`vllm.log`, `benchmark.log`, and the raw `profile/` directory if it fails.
Vera remains environment-smoke-only-not-comparable; actual clocks are read
at start/end but never changed. Unrelated staged files and document deletions
were preserved and are not part of this milestone.

## Latest status — CUDA Graph MoE Nsight diagnostic

Status: **`four_gpu_probe_passed_full_model_capture_failed_preserved`**.

Subsequent full-model run
`results/raw/moe-nsys-10256224-20260909T174110Z` used commit `1528b7b` and
passed the four-GPU probe, model startup, CUDA Graph capture and eight warmups.
At 19:44:10 local time profiling began; at 19:44:14 EngineCore_DP1 exited
unexpectedly. All three measured requests failed and stop_profile returned
HTTP 500. No native failure stack or worker exit signal was logged, so the
underlying cause is unresolved; the default A40 MoE tuning warning is not a
fatal exception. The old launcher still terminates Nsight on its client-error
path, bypassing the successful-path export wait. Its log ends at generating
a temporary qdstrm, and only the probe report exists. These unsuccessful raw
artifacts are retained. The following describes the historical Nsight path;
use the Torch entry point above for the user's current requested workflow.

The September 9 attempt
`results/raw/moe-nsys-10256217-20260909T172201Z` loaded the model, compiled and
captured CUDA Graphs, and completed all three measured requests. Both profiler
control endpoints returned HTTP 200, but Nsight ended with `No reports were
generated`. Its logs and successful client response rows are preserved; no
MoE kernel-time percentage can be recovered from this attempt. The Vera scope
warning was informational, not the cause of this failure.

The original launch put host `nsys` outside `apptainer exec --cleanenv`, which
does not reliably preserve profiler injection into the container. The repaired
launch bind-mounts the existing host Nsight Systems 2025.1.3 installation
read-only at `/opt/moe-nsys` and runs its executable **inside** the container.
No container image, CUDA, PyTorch, vLLM, or site-packages was modified. This
matches the container profiling arrangement in the
[NVIDIA Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).

`src/nsys_target.py` starts the model in a separately recorded process group.
After requests finish, the driver stops only that model group and waits up to
180 seconds for Nsight to exit/export, rather than terminating the profiler
with the model. A four-process CUDA Graph probe now runs before model loading;
both its report and the real model report must contain kernel records from all
four GPUs. Probe reports are excluded from MoE statistics. CUDA event-completion
tracing is disabled; CUDA Graph node tracing remains enabled because the kernel
breakdown needs it and is not overhead-free. Startup progress is printed every
minute, and server, profiler, probe, and benchmark logs are separate. Client
response rows are saved before profiler shutdown so a shutdown failure cannot
discard those completed requests.

Live verification used only the user's existing job `10256217` on
`vera-m07-03`, with the same pinned SIF; it is environment smoke, not study
data. `results/raw/nsys-repair-u1iAEn/probe.nsys-rep` exported successfully to
SQLite. Its `CUPTI_ACTIVITY_KIND_KERNEL` table contains exactly five graph
kernel executions on each device 0, 1, 2, and 3 (20 total), and
`nsys stats --report cuda_gpu_kern_sum --format csv` succeeds. The repaired
four-worker capture/export path is therefore GPU-verified. The separate model
process-group shutdown is covered by a local subprocess test, but has not yet
been retested with the full vLLM service under Nsight. Before a full-model
rerun could start, Slurm marked the existing allocation `COMPLETING` and
rejected a further step as expired. No new allocation or second node was used.

`scripts/moe_nsys_profile.sbatch` is the complete low-interference diagnostic
requested after the eager baseline. It keeps the frozen TP=1/DP=4/EP=4,
linear placement, `allgather_reducescatter`, disabled prefix caching, model,
and prompt order. It does not pass `--enforce-eager` or apply the site-packages
CUDA-event patch. Instead, it pins vLLM to `FULL_AND_PIECEWISE`, starts the
built-in CUDA profiler behind Nsight Systems, warms up normally, and captures
only the first three requests of the frozen trace at their original 0/2/4 s
offsets.

The workflow has no Conda dependency. It verifies and runs the pinned
`containers/vllm-openai-v0.15.1-amd64.sif`, bind-mounts the existing model
read-only, and executes both server and client with container `python3`.
The host loads `CUDA/12.9.1` only to obtain `nsys`; vLLM, PyTorch, and CUDA
user-space runtime remain supplied by the clean-environment SIF. Compilation
caches are isolated in a unique node-local directory.

Like `scripts/apptainer_moe_test.sh`, this script supports direct execution
inside an existing four-GPU Slurm allocation. Uranus/Neptune runs are labelled
`approved-study-node`; Vera is accepted only for environment smoke and is
labelled `environment-smoke-only-not-comparable` in the run directory, client
metrics, and MoE-share JSON.

The script writes a unique raw directory and runs `nsys stats` after stopping
its owned model process group and waiting for Nsight export.
`src/nsys_moe_share.py` conservatively classifies
explicit FusedMoE/routing/grouped-GEMM kernels plus AllGather and
ReduceScatter. The latter attribution is valid only for this fixed TP=1 AgRs
configuration. The reported percentage is aggregate CUDA kernel time, not E2E
wall time; generic unmatched kernels are intentionally retained in the JSON
for audit rather than guessed as MoE. Profiler-run latency is diagnostic only.

Offline verification completed:

- the Python unit suite passes (6/6), including synthetic Nsight CSV parsing,
  refusal to overwrite PID/log files, and independent model-group shutdown;
- all Nsight Python modules compile;
- the Slurm script passes `bash -n`;
- `git diff --check` passes;
- the script restricts study data to Uranus/Neptune, admits Vera only as
  non-comparable environment smoke, checks the frozen trace hash, records
  source/runtime/topology data, and uses the existing owned-PGID cleanup
  helper;
- unrelated worktree changes do not block a run, but any uncommitted file that
  affects this diagnostic does.

The pinned SIF exposes `python3` and vLLM 0.15.1; the cluster `CUDA/12.9.1`
module exposes Nsight Systems 2025.1.3. Full-model verification remains the
next check. Once inside an existing four-GPU Slurm allocation, run:

```bash
bash scripts/moe_nsys_profile.sbatch
```

The primary output will be
`results/raw/moe-nsys-<job>-<UTC>/moe-kernel-share.json`; inspect its
`largest_unmatched_kernels` before interpreting the conservative percentage.

## Latest status — automated Apptainer MoE test

Status: **`prepared_after_failed_live_attempt`**.

`scripts/apptainer_moe_test.sh` automates the full fixed-trace test from inside
an existing four-GPU Slurm allocation. It validates the local model, creates a
unique run directory and node-local EXT3 overlay/cache, applies the timing
patch, launches the pinned SIF, waits for health, runs `src.benchmark`, stops
only its recorded process group, reverts the patch, and prints/writes a compact
`summary.json` containing the primary MoE share and per-rank timing totals.

Run it from the repository root after obtaining the allocation:

```bash
scripts/apptainer_moe_test.sh
```

An alternate complete local model directory may be passed as its sole
argument. The pinned SIF remains at the fixed path under `containers/`.

The script accepts the already validated Vera A40 environment only as
`environment-smoke-only-not-comparable`. It records that scope in the summary
and prints a warning. Only runs on Uranus or Neptune are labelled
`approved-study-node` under the repository's research contract.

The stop helper now recognizes the exact Apptainer runtime-parent title emitted
by the pinned `vllm-openai-v0.15.1-amd64.sif`. Ownership and exact process-group
checks remain in place; no broad process matching was added.

Offline verification covers shell syntax, the existing unit suite, diff
whitespace, and a disposable process-group stop test for the Apptainer parent
title. Live attempt `apptainer-moe-test-10252822-20260908T185121Z` reached a
healthy four-rank server, then failed before benchmark traffic because the
client-side `apptainer exec` tried to acquire a second writable lock on the
server's active EXT3 overlay. Cleanup stopped the server and restored the
pristine patch state. The client does not import vLLM or need the patch, so its
container invocation no longer attaches the server overlay. This correction
has not yet received a complete live run.

## Latest status — immutable Apptainer base image

Status: **`offline_verified_gpu_verification_blocked`**.

The official amd64 `vllm/vllm-openai:v0.15.1` OCI image was resolved and
converted into one immutable Apptainer SIF. The build pins the architecture
manifest rather than the mutable tag:

```text
docker://vllm/vllm-openai@sha256:06f9f0d5c7cb079504615c51dab70cd18abbf609d1358b940172181ac0a92efa
```

`scripts/build_vllm_apptainer.sh` disables the persistent OCI cache and uses
`/tmp` (or an explicit second argument) for layer expansion, so the shared
filesystem receives only the final SIF. Sandbox images are deliberately not
used because they would recreate the many-small-file problem. The local SIF is
ignored by Git and has:

```text
path: containers/vllm-openai-v0.15.1-amd64.sif
size: 8,416,972,800 bytes (7.9 GiB reported by ls)
sha256: 9eef28a71be2c24668f3bf744e3f38f9cbb4f2be5c3467c07d4502b42b55b96e
```

Offline container execution verified:

- Python 3.12.12 at `/usr/bin/python3` (there is no `python` command);
- PyTorch 2.9.1+cu129 with CUDA user-space version 12.9;
- vLLM 0.15.1;
- aiohttp 3.13.3;
- GNU patch at `/usr/bin/patch`;
- vLLM site root `/usr/local/lib/python3.12/dist-packages`;
- exact pristine hashes accepted by the MoE timing patch manager.

This official image is not byte-for-byte equivalent to the earlier Conda
environment, which used Python 3.10.17 and PyTorch 2.9.1+cu128. Treat the
Apptainer environment as a new execution baseline; do not compare its measured
results directly with runs from the old environment without an explicit
environment-control study.

GPU passthrough is not yet verified. The current Slurm endpoint exposes only
`vera-*` nodes and no node named `uranus` or `neptune`. Two submission attempts
were rejected before queueing (first for a missing account, then for the
unknown `uranus` node), so no GPU allocation ran. Repository safety forbids
substituting a Vera GPU. `scripts/verify_vllm_apptainer.sbatch` is ready for
submission from the Minerva endpoint with an explicit valid account. It writes
a unique evidence directory and verifies `--nv`, CUDA availability, device
identity, versions, the SIF hash, and pristine patch state without starting a
server.

Do not start the writable-overlay milestone until this GPU verification passes
on `uranus` or `neptune`.

## Latest status — direct-launch minimal MoE timing baseline

Status: **`prepared_not_gpu_verified`**.

The active path now has no model YAML, Python configuration builder, or server
wrapper. The complete launch chain is:

```text
sbatch scripts/moe_time_baseline.sbatch
  -> python -m vllm.entrypoints.openai.api_server
  -> python -m src.benchmark <unique-output-directory>
```

The Slurm script directly hardcodes `Qwen/Qwen3-30B-A3B`, TP=1, DP=4, expert
parallelism, the MP data-parallel backend, linear expert placement,
`allgather_reducescatter`, disabled prefix caching, and eager execution. The
benchmark directly hardcodes the trace, localhost endpoint, served model name,
four expected ranks, eight warmups, five settle seconds, and request timeout.
Its only command-line argument is the output directory.

The patch manager remains separate because site-packages edits must be
hash-checked and reversible. The stop helper remains separate because cleanup
must target only the process group created by this job. Neither helper carries
model or benchmark configuration.

### Measurement contract

- Input is the committed 50-request GSM8K trace at 0.5 request/s. Offsets are
  exactly `0, 2, ..., 98` seconds and SHA-256 is
  `721638a6250b929de05c2c355c8bc9b20258db7b28fedb94acb5572136571cbc`.
- The server receives eight warmup requests, settles for five seconds, resets
  all rank-local timers, replays the fixed trace open-loop, then dumps timing
  only after all measured responses finish.
- The numerator is the maximum across the four EP ranks of rank-local summed
  CUDA-event time around `FusedMoE.forward_impl`. It includes dispatch,
  expert selection/compute, and combine; it is not pure expert GEMM.
- The denominator is the union of successful client send-to-complete intervals.
  Idle gaps are excluded and overlaps are counted once.
- The primary field is
  `fused_moe_over_active_request_wall_time_percent`. Rank sum is diagnostic only.
- The job neither reads power nor records or controls GPU clocks.
- `--enforce-eager` is intentional; this is an eager timing baseline.

### Patch lifecycle and artifacts

The Slurm job requires pristine vLLM 0.15.1 files, checks exact hashes, applies
the standalone patch before service startup, and reverts it after stopping the
owned process group. Cleanup is registered before patch application. Slurm
sends `TERM` 180 seconds before the time limit to leave time for cleanup.

Every submission creates a non-overwriting directory below `results/raw/` and
stores the exact vLLM command, trace, source revision/status, allocation,
versions, topology, raw server/client logs, request rows, four rank-local timing
files, and final metrics together.

### Offline verification completed

- 3/3 standard-library unit tests pass.
- All five active Python modules compile.
- All three shell/Slurm scripts pass `bash -n`.
- The hardcoded server command contains TP=1, DP=4, EP, linear placement,
  `allgather_reducescatter`, disabled prefix caching, and eager execution.
- The trace has 50 unique IDs, exact 2-second offsets, and the expected hash.
- The current patch SHA matches its manifest. The patch payload is unchanged
  from commit `326399e`, where its pristine vLLM 0.15.1
  `check -> apply -> py_compile -> revert -> check` round trip passed.
- No references remain to the removed YAML/config/server-wrapper path or old
  phase-profile environment variable.
- An independent final review found no high- or medium-severity issue in the
  direct startup, signal cleanup, process-group cleanup, or patch-revert gate.
- Temporary-process tests cover normal TERM, forced KILL, an exited leader with
  a live child, rejection of a non-vLLM leader, and isolation from a sentinel
  process outside the target group.
- `git diff --check` passes.

No GPU or live vLLM service is available on the cleanup machine. There is no
new measured percentage yet, and CUDA timing overhead has not been checked on
hardware. Its base Python also lacks `aiohttp`; the target `cuda-env` must
provide the single dependency from `requirements.txt`.

### Preserved historical result

Historical reports and processed figures remain under `docs/` and `results/`.
The earlier five-placement by 81-frequency study did **not** establish a
meaningful Joint placement/DVFS energy advantage: its only strict 13 ms gap was
0.1005% and was not repeatability-confirmed. This negative result is preserved.
Removed legacy execution code remains recoverable from commit `d7f30c5`.

### Exact next command

```bash
cd /data/users/chjing/moe-energy-scheduling
sbatch scripts/moe_time_baseline.sbatch
```

After completion:

```bash
run_dir=$(find results/raw -maxdepth 1 -type d -name 'moe-time-*' | sort | tail -1)
cat "${run_dir}/benchmark/metrics.json"
cat "${run_dir}/exit-status.txt"
```
