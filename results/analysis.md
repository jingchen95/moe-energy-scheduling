# Qwen3-30B-A3B MoE workload 与实测耗时分析

## 结论摘要

本次测试中，最慢 GPU rank 的 `FusedMoE.forward` 累计 CUDA 时间为
92.650 秒，请求活跃墙钟时间为 107.028 秒，因此：

```text
MoE 相关关键路径占比 = 92.650 / 107.028 = 86.57%
```

这个数可以用于简单判断本次运行中 MoE 路径占总模型活跃时间的比例。
它表示完整 FusedMoE 路径，包括路由、token dispatch、专家计算、combine、
跨 GPU 通信及同步等待，不等同于纯 expert GEMM 的计算比例。

按模型结构和本次 token workload 估算，MoE 占主要矩阵计算量约 65.54%。
实测时间占比高于理论计算量占比，主要反映小 expert GEMM 的效率、Expert
Parallel 通信和同步开销。

## 1. 模型 workload 预估分析

### 1.1 模型结构

模型：`Qwen3-30B-A3B`

| 参数 | 数值 |
|---|---:|
| Transformer 层数 | 48 |
| Hidden size | 2,048 |
| Attention heads | 32 |
| KV heads | 4 |
| Head dimension | 128 |
| Expert 数量 | 128 |
| 每 token 激活 experts | 8 |
| Expert intermediate size | 768 |

### 1.2 每层每 token 的主要计算量

以下使用 MAC（multiply-accumulate）计数；若按一次乘法和一次加法分别计算，
1 MAC 约等于 2 FLOPs。

Attention 的 Q/K/V/O 投影约为：

```text
Q projection: 2048 x 4096
K projection: 2048 x 512
V projection: 2048 x 512
O projection: 4096 x 2048

合计 = 18,874,368 MAC/token/layer
```

MoE 中每个激活 expert 使用 gate、up、down 三个主要线性映射：

```text
8 active experts x 3 x 2048 x 768
= 37,748,736 MAC/token/layer
```

Router 额外约为：

```text
2048 x 128 = 262,144 MAC/token/layer
```

因此在不考虑 Attention 序列运算时，MoE 的主要矩阵计算量已经约为
Attention 投影计算量的两倍。

### 1.3 结合本次请求长度的估算

本次 workload 包含：

- 50 个数学请求；
- prompt token 总数 7,149；
- 每个请求生成 128 tokens；
- 约 13,499 个 token 进入实际模型 forward；
- prompt 长度为 128--191 tokens，整体属于短上下文 workload。

将 causal Attention 的 QK/AV 序列运算也计入后，48 层的主要工作量估算为：

| 模块 | 估算工作量 |
|---|---:|
| Attention 投影及 QK/AV | 12.951 TMAC |
| MoE router 及 active experts | 24.629 TMAC |
| 两者合计 | 37.580 TMAC，约 75.160 TFLOPs |
| MoE 主要计算量占比 | 65.54% |

这是结构级估算，不包含 LayerNorm、RoPE、activation、sampling、CPU 调度、
padding、通信和 kernel launch 等开销。对于更长上下文，QK/AV 的工作量会随
上下文长度增长，Attention 占比也会提高。

## 2. 实际运行结果分析

### 2.1 测试状态与请求延迟

| 指标 | 实测结果 |
|---|---:|
| 提交请求 | 50 |
| 成功请求 | 50 |
| 失败请求 | 0 |
| 请求活跃墙钟时间 | 107.028 s |
| 平均并发请求数 | 4.432 |
| 平均 E2E | 9,486.39 ms |
| P95 E2E | 9,612.22 ms |
| 平均 TTFT | 194.31 ms |
| P95 TTFT | 222.79 ms |
| 平均 TPOT | 73.166 ms/token |
| P95 TPOT | 74.143 ms/token |
| 平均请求发送延迟 | 0.733 ms |
| 最大发送延迟 | 1.965 ms |

请求按照固定的 0.5 request/s 到达，计划发送跨度为 98 秒。发送延迟很小，
说明客户端基本按预定 arrival schedule 执行。所有请求均生成了完整的 128
tokens，因此本次 workload 是固定输出长度、以 decode 为主的测试。

### 2.2 各 rank 的 FusedMoE 时间

| EP rank | Timing records | FusedMoE 累计时间 | 占活跃墙钟时间 |
|---:|---:|---:|---:|
| 0 | 70,272 | 79.897 s | 74.65% |
| 1 | 70,272 | 92.650 s | 86.57% |
| 2 | 70,272 | 66.487 s | 62.12% |
| 3 | 70,272 | 74.267 s | 69.39% |

四个 rank 都记录了 48 个 MoE 层，每个 rank 有相同的 70,272 条记录，说明
timing patch 在全部 rank 上完成了采集。

GPU rank 并行执行，因此不能把四个 rank 的累计时间相加后除以墙钟时间。
关键路径由最慢的 rank 决定，本次采用 rank 1：

```text
92,649.948 ms / 107,028.028 ms x 100% = 86.566%
```

### 2.3 为什么时间占比高于计算量占比

理论主要计算量占比约 65.54%，实测 FusedMoE 路径时间占比为 86.57%。主要
原因包括：

1. 本次采用 `DP=4`、`EP=4`，每个 MoE 层包含跨四张 GPU 的 collective
   communication；该时间位于 FusedMoE 计时边界内。
2. 每 token 从 128 个 experts 中激活 8 个。当前请求率和 batch 较小，会形成
   很多较小的 expert GEMM，其 GPU 利用效率低于规则、连续的 Attention GEMM。
3. 不同 rank 的本地 token rows 和同步等待并不均衡。rank 1 的累计时间最高，
   即使它不一定拥有最多的本地 token rows，说明通信或等待也是关键路径的一部分。
4. 测试使用 `--enforce-eager`，没有 CUDA Graph 带来的 launch 优化。
5. vLLM 日志显示 A40 没有匹配的 tuned MoE 配置，当前使用默认 MoE 参数。
6. 当前上下文较短，Attention 的 QK/AV 工作量尚未成为主要瓶颈。

因此，不能把剩余的 13.43% 直接解释成 Attention 耗时。它是所有非 FusedMoE
工作的总和，包括 Attention、LayerNorm、embedding、sampling、调度及其他运行时
开销。若需要精确的 Attention 百分比，需要为 Attention 单独增加 CUDA-event
计时。

## 最终判断

对于本次 4 x NVIDIA A40、短上下文、低请求率、Expert Parallel 运行，可以报告：

> FusedMoE 相关路径约占模型请求活跃关键路径时间的 **86.57%**。按模型结构和
> 本次 token workload 估算，MoE 约占主要矩阵计算量的 **65.54%**；两者的差异
> 主要来自 MoE 通信、同步和小 expert kernel 的执行效率。

本次目录没有 GPU 功率采样，因此该结果只描述耗时占比，不能用于计算能耗占比。
本次运行范围为 `environment-smoke-only-not-comparable`，不应与 Uranus/Neptune
正式研究数据直接比较。

## 数据来源

- `summary.json`
- `benchmark/metrics.json`
- `benchmark/requests.jsonl`
- `benchmark/moe_timing_rank0.json` 至 `benchmark/moe_timing_rank3.json`
- `containers/models/Qwen3-30B-A3B/config.json`
- `patches/vllm-0.15.1-moe-timing.patch`
