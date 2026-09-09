# Handoff

## Latest status — CUDA Graph MoE Nsight diagnostic

Status: **`four_gpu_capture_verified_full_model_rerun_pending`**.

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
