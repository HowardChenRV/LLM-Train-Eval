#!/bin/bash

_THIS_DIR=$(dirname "$0")

BASE_PATH=${BASE_PATH:-/mnt/public/chenyonghua}
BURN_TIME=${BURN_TIME:-300}

LOG_FILE="${BASE_PATH}/gpu_burn_${RANK}.log"
CMD="/workspace/gpu-burn/gpu_burn -c ${_THIS_DIR}/compare.ptx -d ${BURN_TIME}"

# 清空旧日志
: > "$LOG_FILE"

# 运行 gpu_burn，输出写入日志文件，静默标准输出
$CMD > "$LOG_FILE" 2>&1

# 分析日志内容
PASS_COUNT=0
TOTAL_GPUS=8

for i in $(seq 0 $((TOTAL_GPUS - 1))); do
    if grep -q "GPU $i: OK" "$LOG_FILE"; then
        ((PASS_COUNT++))
    else
        echo "GPU $i: ❌ FAILED"
    fi
done

if [ "$PASS_COUNT" -eq "$TOTAL_GPUS" ]; then
    echo "✅ 所有 $TOTAL_GPUS 张 GPU 测试通过！"
else
    echo "❌ 有 $(($TOTAL_GPUS - $PASS_COUNT)) 张 GPU 测试失败，请检查 $LOG_FILE"
fi

sleep 30
