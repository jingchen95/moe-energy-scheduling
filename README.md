# 最小 vLLM MoE 时间占比基准

这个仓库现在只做一件事：运行 `Qwen/Qwen3-30B-A3B` 的 vLLM 服务，
回放一组固定请求，并粗略统计 `FusedMoE` 区域占活跃端到端请求时间的比例。
不测功耗，也不设置 GPU 频率。

## 保留的最小结构

```text
configs/model.yaml                  # TP=1、DP=4、EP=4 的服务配置
data/traces/gsm8k-50-r0.5.jsonl    # 固定的 50 个请求
src/                               # 配置、trace 回放、计时汇总
patches/                            # 独立的 vLLM 0.15.1 计时补丁
scripts/moe_time_baseline.sbatch   # 推荐的一键入口
tests/test_minimal.py               # 不需要 GPU 的最小测试
```

旧的 placement、DVFS、功耗和搜索执行代码已从活跃树移除。历史报告和已测
结果仍保留在 `docs/`、`results/processed/` 和 `results/figures/`；旧代码可从
Git 提交 `d7f30c5` 恢复。

## 运行

在 Minerva 登录节点进入仓库，确认已有项目使用的 `cuda-env`（其中应是
vLLM 0.15.1），并确保 `git status --short` 没有输出，然后只提交一个作业：

```bash
cd /data/users/chjing/moe-energy-scheduling
sbatch scripts/moe_time_baseline.sbatch
```

脚本默认选择 Uranus。若要使用 Neptune：

```bash
sbatch --nodelist=neptune scripts/moe_time_baseline.sbatch
```

脚本会依次完成：记录环境；临时应用计时补丁；启动 vLLM；发送 8 个预热
请求；等待 5 秒；清空计时器；按固定时间表发送 50 个正式请求；写出结果；
停止本作业启动的服务；最后还原补丁。即使作业失败，清理步骤也会执行。

每次运行都会创建新的目录：

```text
results/raw/moe-time-<job-id>-<UTC-time>/
```

最先查看：

```text
benchmark/metrics.json
benchmark/requests.jsonl
vllm.log
clock-samples.csv
```

`metrics.json` 中的主结果是：

```text
fused_moe_over_active_request_wall_time_percent
```

它的简化计算是：

```text
100 × 四个 EP rank 中最大的 FusedMoE CUDA 累计时间
    / 所有成功请求 send→complete 时间区间的并集
```

这里的 `FusedMoE` 包含该函数内部的 token dispatch、expert 选择与计算、
combine；它不是“纯 expert GEMM”时间。分母是客户端看到的端到端时间，包含
vLLM 调度、Attention、MoE、通信及少量 HTTP 开销，但排除了没有请求在途的
空闲间隔。因此该百分比适合做简单基准和配置间相对比较，不是高精度算子剖析。

## 固定输入

- 50 个 GSM8K prompt；每个最多生成 128 token。
- 到达间隔固定为 2 秒，即 0.5 request/s；offset 为 `0, 2, ..., 98` 秒。
- trace SHA-256：
  `721638a6250b929de05c2c355c8bc9b20258db7b28fedb94acb5572136571cbc`。
- 所有配置比较都必须复用这个文件，不能改变 prompt 顺序或到达时间。

这是一段确定性的 open-loop 服务流量，比“全部请求同时发出”的离线吞吐测试
更接近服务回放，但它只代表一个较轻负载，不等于完整真实线上流量。

## 默认频率与补丁

运行脚本从不调用锁频、改频或功耗接口。`clock-samples.csv` 只是每秒只读记录
实际 graphics/SM clock，方便确认机器当时的状态。
这里“默认频率”只表示本作业不主动控制频率；若采样显示节点已有异常锁频，应
把该次运行标为无效，而不是让脚本擅自改回去。

为了让这版 CUDA Event 计时路径简单且稳定，脚本显式使用 `--enforce-eager`；
因此测到的是 eager 基线，不是 vLLM 默认的 CUDA Graph 执行模式。

补丁在 vLLM 服务启动前应用，在服务停止后还原。补丁管理器只接受 manifest
中记录的原版或补丁版精确 SHA-256；若 site-packages 不是预期的 vLLM 0.15.1
文件，它会直接拒绝修改。补丁的具体边界见 `patches/README.md`。

## 当前验证状态

当前机器不能运行 GPU，因此本次清理只验证 Python 单元测试、shell 语法、固定
trace 和补丁 apply/revert 往返；尚未声称新的 GPU benchmark 已成功运行。

CPU 测试命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```
