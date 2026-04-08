#!/bin/bash
# 杀掉所有节点上的残留实验进程，清理 GPU 和 Ray
# 用法: bash kill_all.sh

set -e

HEAD_IP="10.63.234.28"
WORKER_IPS="10.63.234.146 10.63.234.19 10.63.234.143"
ALL_IPS=("$HEAD_IP" $WORKER_IPS)

echo "[1/3] 停止所有节点上的 Ray ..."
for ip in ${ALL_IPS[@]}; do
    echo "  $ip: ray stop"
    ssh "$ip" "ray stop --force" 2>/dev/null || true
done

echo ""
echo "[2/3] 杀掉残留的 mpirun/orted/python 训练进程 ..."
for ip in ${ALL_IPS[@]}; do
    echo "  $ip:"
    ssh "$ip" "
        # 杀 mpirun 及其子进程
        pkill -f 'mpirun.*agent.*watcher' 2>/dev/null || true
        # 杀 orted
        pkill -f 'orted.*mca ess' 2>/dev/null || true
        # 杀 vLLM/SGLang 引擎孤儿进程
        pkill -9 -f 'VLLM::' 2>/dev/null || true
        pkill -9 -f 'sglang::' 2>/dev/null || true
        # 杀残留的 python 训练进程 (排除 codelab/jupyter/supervisord)
        ps aux | grep python | grep -v grep | grep -v codelab | grep -v plugin-local | grep -v supervisord | grep -v jupyter | grep -v pip | awk '{print \$2}' | xargs -r kill -9 2>/dev/null || true
        echo '    done'
    " 2>/dev/null || true
done

echo ""
echo "[3/3] 检查 GPU 状态 ..."
for ip in ${ALL_IPS[@]}; do
    echo "  $ip:"
    ssh "$ip" "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader" 2>/dev/null
done

echo ""
echo "[done] Ray 已停止，训练进程已清理。"
echo "如果某节点 GPU 仍有大量占用（僵尸进程），需要重启该节点。"
