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
