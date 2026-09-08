#!/bin/bash

# Discord Webhook 配置
WEBHOOK_URL="https://discord.com/api/webhooks/1526132150267805747/M4upyGe7j0JslsvXbLj4CjK6OLfadtsDZ4OJWlrygksx2DRzRZjPPiECJyhh5ACBb_tb"
# 绝对路径，用于 srun 加载
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALLOC_INIT_RC="$SCRIPT_DIR/alloc_init.sh"

# 定义发送 Discord 消息的函数
send_discord_message() {
  local msg="$1"
  curl -s -H "Content-Type: application/json" \
       -X POST \
       -d "{\"content\": \"$msg\"}" \
       "$WEBHOOK_URL" > /dev/null
}

# 1. 启动前：发送 Discord 通知
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
send_discord_message "⏳ **[AMD Node]** 开始排队请求分配交互式节点... (排队时间: $START_TIME)"

# 2. 启动交互式会话且指定在分配成功启动 bash 时调用专门的 init 脚本

GPUS=4
CPUS_PER_GPU=16
CPUS=$((GPUS * CPUS_PER_GPU))

srun \
  --account=C3SE2026-1-12 \
  --partition=vera \
  --nodes=1 \
  --ntasks=1 \
  --exclusive \
  --cpus-per-task="$CPUS" \
  --mem=0 \
  --time=01:00:00 \
  --job-name=bash_j \
  --gres="gpu:A40:${GPUS},perf:1" \
  --pty bash -l

# 3. 退出交互式 Bash 后执行：发送释放/结束通知
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
send_discord_message "🏁 **[A40 Node]** 交互式节点会话已结束。(结束时间: $END_TIME)"