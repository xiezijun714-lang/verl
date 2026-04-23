#!/bin/bash
# Multi-turn Search Tool-Calling with SUPO on BrowseComp-Plus
# SUPO: Summarization augmented Policy Optimization
# 4 nodes × 8 H100-80GB, Qwen3-32B, Megatron backend, SGLang rollout
#
# SUPO enables handling tasks beyond working_context_length via periodic summarization
# - working_context_length (L): actual memory constraint per trajectory
# - max_model_len: logical max context (can be higher since SUPO manages context)
#
# Run from project root: bash examples/sglang_multiturn/run_qwen3-32b_bcp_megatron_supo.sh

set -x
ulimit -n 65535
ulimit -u unlimited

# Log directory
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/megatron_supo_run.log"
exec > >(tee "$LOG_FILE") 2>&1

# ---- Environment ----
VENV_PATH="/root/paddlejob/workspace/env_run/xzj/venv_echo_megatron"
HEAD_IP="10.63.234.28"
WORKER_IPS="10.63.234.146 10.63.234.19 10.63.234.143"

source "${VENV_PATH}/bin/activate"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:$PATH"

CUDNN_LIB="${VENV_PATH}/lib/python3.10/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="${CUDNN_LIB}:${LD_LIBRARY_PATH}"

export WANDB_API_KEY="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"
export WANDB_ENTITY="515718106-pku"

export http_proxy="http://agent.baidu.com:8188"
export https_proxy="http://agent.baidu.com:8188"

ALL_IPS=("$HEAD_IP")
for ip in $WORKER_IPS; do ALL_IPS+=("$ip"); done
NNODES=4
NO_PROXY_LIST="127.0.0.1,localhost,$(IFS=,; echo "${ALL_IPS[*]}")"
export no_proxy="$NO_PROXY_LIST"
export NO_PROXY="$NO_PROXY_LIST"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export RAY_enable_open_telemetry=0

MODEL_PATH="/root/paddlejob/workspace/env_run/xzj/models/Qwen3-32B"
DATA_DIR="/root/paddlejob/workspace/env_run/xzj/dataset/browsecomp-plus-processed"

# ---- Batch sizes ----
TRAIN_BATCH_SIZE=32
N_RESP=8

# ---- Parallelism ----
ACTOR_TP=8
ACTOR_PP=4
ACTOR_VPP=null
ACTOR_CP=1

REF_TP=8
REF_PP=4
REF_VPP=null
REF_CP=1

ROLLOUT_TP=8

# ---- Sequence lengths ----
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=10240

# ---- ECHO Configuration ----
# working_context_length: threshold to trigger turn selection
# max_model_len: SGLang KV cache size
# max_summary_rounds: how many times to compress before marking as overlong
WORKING_CONTEXT_LENGTH=10240
MAX_SUMMARY_ROUNDS=5
MAX_MODEL_LEN=12288

TOKEN_BUDGET=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

# ---- Sync run script to worker nodes ----
echo "[sync] Syncing run script to worker nodes ..."
SCRIPT_FILE="${PROJECT_DIR}/examples/sglang_multiturn/$(basename "$0")"
for ip in $WORKER_IPS; do
    timeout 10 scp -o ConnectTimeout=5 "$SCRIPT_FILE" "$ip":"$SCRIPT_FILE" || echo "[sync] WARNING: failed to sync to $ip"
done
echo "[sync] Done."

# ---- Ray cluster ----
echo "[ray] Stopping any existing Ray instances ..."
ray stop --force 2>/dev/null || true
for ip in $WORKER_IPS; do ssh "$ip" "ray stop --force" 2>/dev/null || true; done
sleep 2

echo "[ray] Starting Ray head on $HEAD_IP ($NNODES nodes total) ..."
ray start --head \
    --node-ip-address="$HEAD_IP" \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-gpus=8

for ip in $WORKER_IPS; do
    echo "[ray] Starting worker on $ip ..."
    ssh "$ip" "export LD_LIBRARY_PATH='${VENV_PATH}/lib/python3.10/site-packages/nvidia/cudnn/lib:\$LD_LIBRARY_PATH' && export RAY_enable_open_telemetry=0 && source ${VENV_PATH}/bin/activate && ray start --address='${HEAD_IP}:6379' --num-gpus=8"
done

echo "[ray] Waiting for ${NNODES} nodes ..."
for i in $(seq 1 60); do
    NODE_COUNT=$(python3 -c \
        "import ray; ray.init(address='auto', ignore_reinit_error=True); print(len(ray.nodes()))" \
        2>/dev/null)
    if [ "${NODE_COUNT:-0}" -ge "$NNODES" ] 2>/dev/null; then
        echo "[ray] Cluster ready: $NODE_COUNT nodes."; break
    fi
    [ "$i" -eq 60 ] && { echo "[ray] ERROR: timeout."; ray status; exit 1; }
    sleep 2
done

# ---- Retrieval service ----
echo "[retriever] Killing any existing process on port 8000 ..."
fuser -k 8000/tcp 2>/dev/null || true
sleep 1
kill $(ss -tlnp 'sport = :8000' | grep -oP 'pid=\K\d+') 2>/dev/null || true
sleep 1

setsid "${VENV_PATH}/bin/python3" \
    "$PROJECT_DIR/examples/sglang_multiturn/browsecomp_retrieval_server.py" \
    --mode dense \
    --model /root/paddlejob/workspace/env_run/xzj/models/Qwen3-Embedding-8B \
    --device cpu \
    --corpus_file "${DATA_DIR}/corpus.parquet" \
    --host 0.0.0.0 --port 8000 \
    --batch_size 4 \
    --dense_cache /root/paddlejob/workspace/env_run/xzj/browsecomp_dense_cache_tevatron.pkl \
    > ${LOG_DIR}/browsecomp_retriever.log 2>&1 &
RETRIEVER_PID=$!

cleanup() {
    kill $WATCHDOG_PID 2>/dev/null
    kill $RETRIEVER_PID 2>/dev/null
    ray stop --force 2>/dev/null || true
    for ip in $WORKER_IPS; do ssh "$ip" "source ${VENV_PATH}/bin/activate && ray stop --force" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

echo "[retriever] Waiting for server ready ..."
for i in $(seq 1 1200); do
    curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1 && { echo "[retriever] Ready after ${i}s."; break; }
    if ! kill -0 $RETRIEVER_PID 2>/dev/null; then
        echo "[retriever] Server died. Log:"; cat ${LOG_DIR}/browsecomp_retriever.log; exit 1
    fi
    sleep 1
done


# ---- Training with SUPO ----
python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT_DIR/examples/sglang_multiturn/config" \
    --config-name='bcp_multiturn_megatron_grpo' \
    algorithm.adv_estimator=supo \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=False \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_decay_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${TOKEN_BUDGET} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP} \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${ACTOR_VPP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP} \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${TOKEN_BUDGET} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.n=${N_RESP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.context_length=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=999999 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml" \
    +actor_rollout_ref.rollout.multi_turn.context_compression_method=echo_e2e \
    +actor_rollout_ref.rollout.multi_turn.enable_summarization=True \
    +actor_rollout_ref.rollout.multi_turn.max_summary_rounds=${MAX_SUMMARY_ROUNDS} \
    +actor_rollout_ref.rollout.multi_turn.working_context_length=${WORKING_CONTEXT_LENGTH} \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${TOKEN_BUDGET} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${REF_TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${REF_PP} \
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=${REF_VPP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${REF_CP} \
    actor_rollout_ref.ref.megatron.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path="$PROJECT_DIR/verl/utils/reward_score/bc-p.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NNODES} \
    trainer.project_name='echo' \
    trainer.experiment_name='qwen3-32b_bcp_megatron-echo-e2e-w8k-s7' \
    trainer.logger='["console", "wandb"]' \
    trainer.save_freq=-1 \
    trainer.test_freq=2 \
    trainer.total_epochs=5 \
    +trainer.master_port_range='[31000,32000]' \
    +trainer.validation_data_dir='/root/paddlejob/workspace/env_run/xzj/echo/val_outputs'
