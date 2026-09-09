# CUDA Graph 下的 MoE 粗粒度耗时占比

`scripts/moe_nsys_profile.sbatch` 是独立的诊断作业。它不应用
`patches/vllm-0.15.1-moe-timing.patch`，也不设置 `--enforce-eager`。
vLLM 显式使用 `FULL_AND_PIECEWISE` CUDA Graph；Nsight Systems 只在 warmup
之后采集固定 GSM8K trace 的前三个请求。

运行：

```bash
sbatch scripts/moe_nsys_profile.sbatch
```

改用 Neptune 时仍然使用同一脚本和输入：

```bash
sbatch --nodelist=neptune scripts/moe_nsys_profile.sbatch
```

脚本要求目标环境已经提供 `nsys`，不会安装或升级 CUDA、PyTorch、vLLM、
Nsight 或其他依赖。它只允许在 Uranus 或 Neptune 上运行。

主要结果写入每次唯一的目录：

```text
results/raw/moe-nsys-<job-id>-<UTC-time>/moe-kernel-share.json
```

粗略分子包括名称明确属于 FusedMoE、MoE routing/grouped GEMM 的 kernel，
以及 `AllGather`/`ReduceScatter` kernel。后两者可以归给 MoE，是因为此诊断
固定 `TP=1`、`EP=4` 和 `allgather_reducescatter` 后端。分母是 Nsight 捕获的
全部 CUDA kernel 累计时间。

这个百分比是保守的 GPU kernel-time 占比，不是端到端墙钟占比：通用名称的
GEMM 不会被强行归类，多个 stream 的重叠也可能被累计。运行后必须检查 JSON
中的 `largest_unmatched_kernels`；若某个大 kernel 经时间线确认属于 MoE，再对
分类规则作有记录的修改。不能用 profiler run 的 E2E、TTFT 或 TPOT 作为正式
性能结果；正式延迟来自相同配置、相同完整 trace、关闭 profiler 的独立 run。
