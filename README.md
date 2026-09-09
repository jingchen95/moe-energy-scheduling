# 最小 vLLM MoE 时间占比基准

这个仓库当前只跑一个固定基准：

- 模型：`Qwen/Qwen3-30B-A3B`
- 并行：TP=1、DP=4、开启 EP（因此 4 个 DP rank 同时也是 4 个 EP rank）
- 输入：50 个固定 GSM8K prompt，每 2 秒发送 1 个
- 输出：`FusedMoE` GPU 时间占活跃请求端到端时间的百分比
- 不测功耗，也不记录或设置 GPU 频率

## vLLM 在哪里启动

启动链现在只有一层：

```text
sbatch scripts/moe_time_baseline.sbatch
  -> python -m vllm.entrypoints.openai.api_server ...
```

模型、端口以及 TP/DP/EP 参数全部直接写在
`scripts/moe_time_baseline.sbatch` 的 `vllm_command=(...)` 数组中。没有 YAML
配置解析，也没有中转启动脚本。

## 怎么运行

前提是 Minerva 上已有项目使用的 `cuda-env`，其中安装的是 vLLM 0.15.1 和
`aiohttp`（唯一额外依赖），并且模型已经可下载或已在缓存中。不要在本项目里
升级 CUDA、PyTorch 或 vLLM。

从仓库根目录只执行：

```bash
cd /data/users/chjing/moe-energy-scheduling
sbatch scripts/moe_time_baseline.sbatch
```

脚本默认申请 Uranus 上的 4 张 L40S。改用 Neptune 时：

```bash
sbatch --nodelist=neptune scripts/moe_time_baseline.sbatch
```

脚本会自动完成以下操作：应用计时补丁、启动 vLLM、等待服务就绪、预热 8
个请求、等待 5 秒、清空计时、按固定时间表发送 50 个请求、保存结果、停止
自己的服务并还原补丁。

## 代码只需要看哪里

```text
scripts/moe_time_baseline.sbatch     # 一键运行；vLLM 配置硬编码在这里
src/benchmark.py                     # 发送请求，并输出时间占比
data/traces/gsm8k-50-r0.5.jsonl      # 固定 50 个请求
src/workload.py                      # 读取固定 trace
src/metrics.py                       # 计算请求端到端活跃时间
src/moe_timing.py                    # 汇总 4 个 EP rank 的 MoE 时间
patches/                             # vLLM 0.15.1 的最小计时补丁
scripts/manage_moe_timing_patch.sh   # 安全地应用/还原补丁
scripts/stop_vllm.sh                 # 只停止本次作业启动的服务
```

`src/benchmark.py` 顶部硬编码了 trace、服务地址、模型名、rank 数、预热数和
超时。它的命令行只接收输出目录；通常不需要单独调用，因为 sbatch 脚本已经
替你调用：

```bash
python -m src.benchmark RESULT_DIRECTORY
```

## 数据与结果

正式流量是 50 个固定请求，offset 为 `0, 2, ..., 98` 秒，即 0.5 request/s。
trace SHA-256 为
`721638a6250b929de05c2c355c8bc9b20258db7b28fedb94acb5572136571cbc`。
比较不同配置时必须复用相同的 prompt 顺序和到达时间。

每次提交都会新建目录，不覆盖旧结果：

```text
results/raw/moe-time-<job-id>-<UTC-time>/
```

主要查看：

```text
benchmark/metrics.json
benchmark/requests.jsonl
vllm.log
vllm-command.txt
```

`benchmark/metrics.json` 中的主结果是：

```text
fused_moe_over_active_request_wall_time_percent
```

其计算方式是：

```text
100 * 四个 EP rank 中最大的 FusedMoE CUDA 累计时间
    / 所有成功请求 send-to-complete 区间的并集
```

`FusedMoE` 边界包含该函数内部的 dispatch、expert 选择/计算和 combine，并不
等于纯 expert GEMM。分母是客户端看到的端到端时间，并排除完全没有请求在途
的间隔。这个数适合粗略比较，不是高精度 profiler 结果。

脚本不读取、不锁定也不修改 GPU 时钟。服务使用 `--enforce-eager`，因此结果
属于 eager 基线。

## 为什么还需要补丁脚本

计时点位于 vLLM 内部，所以作业启动前临时应用 `patches/` 中的补丁，结束后
自动还原。补丁管理器会核对 vLLM 0.15.1 文件的精确 SHA-256；文件版本不符时
直接停止，不会盲目修改 site-packages。

当前机器没有 GPU，因此这里只完成了离线测试，还没有声称实际 benchmark 已
跑通。离线测试命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

## CUDA Graph 下的低干扰诊断

如需保留 vLLM 的 CUDA Graph 优化并粗略观察 MoE GPU kernel 时间占比，运行：

```bash
sbatch scripts/moe_nsys_profile.sbatch
```

如果已经通过 `srun` 进入四 GPU allocation，也可以直接运行：

```bash
bash scripts/moe_nsys_profile.sbatch
```

这个独立作业不使用 `--enforce-eager`，不应用 CUDA Event 计时补丁。它用
Nsight Systems 的 CUDA profiler 动态区间，只采集 warmup 后固定 trace 的前三
个请求，并将保守估计写入：

```text
results/raw/moe-nsys-<job-id>-<UTC-time>/moe-kernel-share.json
```

这个值是 MoE compute 加 AllGather/ReduceScatter 的粗略 CUDA kernel-time
占比，不是 E2E 墙钟占比。脚本直接使用固定的
`containers/vllm-openai-v0.15.1-amd64.sif` 和容器内 `python3`，不依赖 Conda；
宿主机只加载 CUDA 12.9.1 module 来提供 `nsys`。

Uranus/Neptune 结果标记为 `approved-study-node`；Vera 上只允许做环境 smoke，
结果强制标记为 `environment-smoke-only-not-comparable`，不能与正式研究数据比较。
