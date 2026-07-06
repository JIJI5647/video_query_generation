#!/bin/bash
# 在 p0-p8 sweep 跑的时候，每 INTERVAL 秒记录一次 GPU/CPU/内存状态，
# 一旦显存或系统内存逼近上限就在日志里高亮 WARNING，方便人工介入。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/monitor_resources.log"
PID_FILE="${SCRIPT_DIR}/monitor_resources.pid"
SWEEP_PID_FILE="${SCRIPT_DIR}/p0_p8_sweep.pid"

INTERVAL=60          # 每 60 秒采样一次
GPU_MEM_WARN_PCT=90  # 显存使用超过 90% 报警
CPU_LOAD_WARN=200    # 1分钟平均负载超过这个值报警(224核机器)

echo $$ > "${PID_FILE}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "${LOG_FILE}"
}

log "monitor_resources started (pid $$), interval=${INTERVAL}s"

while true; do
    if [ -f "${SWEEP_PID_FILE}" ] && ! kill -0 "$(cat "${SWEEP_PID_FILE}")" 2>/dev/null; then
        log "sweep launcher (pid $(cat "${SWEEP_PID_FILE}")) no longer running — stopping monitor"
        break
    fi

    gpu_line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)
    if [ -z "${gpu_line}" ]; then
        log "WARN: nvidia-smi returned no data"
    else
        util=$(echo "${gpu_line}" | cut -d',' -f1 | tr -d ' ')
        mem_used=$(echo "${gpu_line}" | cut -d',' -f2 | tr -d ' ')
        mem_total=$(echo "${gpu_line}" | cut -d',' -f3 | tr -d ' ')
        mem_pct=$(( mem_used * 100 / mem_total ))
        log "GPU util=${util}% mem=${mem_used}/${mem_total}MiB (${mem_pct}%)"
        if [ "${mem_pct}" -ge "${GPU_MEM_WARN_PCT}" ]; then
            log "*** WARNING: GPU memory at ${mem_pct}% (>=${GPU_MEM_WARN_PCT}%) — risk of OOM ***"
        fi
    fi

    load1=$(cut -d' ' -f1 /proc/loadavg)
    mem_avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    mem_total_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)
    mem_used_pct=$(( (mem_total_kb - mem_avail_kb) * 100 / mem_total_kb ))
    log "CPU load1=${load1} RAM used=${mem_used_pct}%"

    load1_int=${load1%.*}
    if [ "${load1_int}" -ge "${CPU_LOAD_WARN}" ]; then
        log "*** WARNING: CPU load1=${load1} (>=${CPU_LOAD_WARN}) — system may be overloaded ***"
    fi
    if [ "${mem_used_pct}" -ge 90 ]; then
        log "*** WARNING: system RAM used ${mem_used_pct}% — risk of OOM ***"
    fi

    sleep "${INTERVAL}"
done
