#!/bin/bash
# 每隔 CHECK_INTERVAL 秒检查一次 GPU 利用率，若所有 GPU 都处于空闲(0%)，
# 就跑一小段 GPU 计算把利用率拉起来，防止平台因为“30分钟无GPU使用”把机器回收。
#
# 用法:
#   nohup bash gpu_keepalive.sh > /dev/null 2>&1 &
#   disown
#   停止: kill $(cat gpu_keepalive.pid)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/gpu_keepalive.log"
PID_FILE="${SCRIPT_DIR}/gpu_keepalive.pid"

CHECK_INTERVAL=300   # 每 5 分钟检查一次，远小于 30 分钟的掐断阈值
KEEPALIVE_SECONDS=60 # 检测到空闲时，跑多久的 GPU 计算来"续命"
MEM_WARN_PCT=90      # 系统内存使用率达到/超过这个值时，跳过本轮计算，避免在内存紧张时雪上加霜

echo $$ > "${PID_FILE}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "${LOG_FILE}"
}

log "gpu_keepalive started (pid $$), check_interval=${CHECK_INTERVAL}s, keepalive=${KEEPALIVE_SECONDS}s"

mem_used_pct() {
    awk '/MemTotal:/{t=$2} /MemAvailable:/{a=$2} END{printf "%d", (t-a)*100/t}' /proc/meminfo
}

touch_gpu() {
    python3 - "${KEEPALIVE_SECONDS}" <<'PYEOF'
import gc
import sys
import time
import torch

duration = float(sys.argv[1])
device = "cuda"
# 小尺寸矩阵：只为了让利用率脱离 0%，不需要占用可观的显存/内存
a = torch.randn(1024, 1024, device=device)
b = torch.randn(1024, 1024, device=device)
try:
    end = time.time() + duration
    while time.time() < end:
        a = a @ b
    torch.cuda.synchronize()
finally:
    del a, b
    torch.cuda.empty_cache()
    gc.collect()
PYEOF
}

while true; do
    # 取所有 GPU 的利用率，用空格分隔
    utils=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    mem_pct=$(mem_used_pct)

    if [ -z "${utils}" ]; then
        log "WARN: nvidia-smi 没有返回数据，跳过本轮检查 (RAM used ${mem_pct}%)"
    else
        # 判断是否所有 GPU 利用率都是 0
        all_idle=1
        for u in ${utils}; do
            if [ "${u}" -ne 0 ]; then
                all_idle=0
                break
            fi
        done

        if [ "${all_idle}" -eq 1 ]; then
            if [ "${mem_pct}" -ge "${MEM_WARN_PCT}" ]; then
                log "*** WARNING: 系统内存使用 ${mem_pct}% (>=${MEM_WARN_PCT}%)，跳过本轮 keepalive 计算以避免加剧内存压力 ***"
            else
                log "GPU 空闲 (utilization=${utils// /,}%)，触发 keepalive 计算 ${KEEPALIVE_SECONDS}s (RAM used ${mem_pct}%)"
                touch_gpu
                log "keepalive 计算完成"
            fi
        else
            log "GPU 正在使用中 (utilization=${utils// /,}%)，跳过 (RAM used ${mem_pct}%)"
        fi
    fi

    sleep "${CHECK_INTERVAL}"
done
