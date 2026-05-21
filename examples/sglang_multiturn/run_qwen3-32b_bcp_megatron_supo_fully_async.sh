#!/bin/bash
# Fully-async multi-turn Search Tool-Calling SUPO on BrowseComp-Plus
# SUPO: Summarization augmented Policy Optimization
# 3 nodes × 8 H100-80GB, Qwen3-32B, Megatron backend, SGLang rollout.
# Default split: 2 train nodes + 1 rollout node.
#
# SUPO uses periodic summarization to continue beyond the working context length.
# Physical context is capped by the local Qwen3-32B config at 40960.
#
# Run from project root: bash examples/sglang_multiturn/run_qwen3-32b_bcp_megatron_supo_fully_async.sh

set -x
ulimit -n 65535
ulimit -u unlimited

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DEFAULT_EXPERIMENT_NAME="qwen3-32b_bcp_megatron-supo-fully-async-32k"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}"

# Log directory
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${BCP_LOG_FILE:-${LOG_DIR}/${EXPERIMENT_NAME}.log}"
exec > >(tee "$LOG_FILE") 2>&1

# ---- Environment ----
VENV_PATH="/root/paddlejob/workspace/env_run/xzj/venv_echo_megatron"
HEAD_IP="${HEAD_IP:-10.63.234.28}"
WORKER_IPS="${WORKER_IPS:-10.63.234.146 10.63.234.19}"

source "${VENV_PATH}/bin/activate"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:$PATH"
PYTHON_BIN="${VENV_PATH}/bin/python3"
RAY_BIN="${VENV_PATH}/bin/ray"

# 强制使用 venv 内的 cuDNN
CUDNN_LIB="${VENV_PATH}/lib/python3.10/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="${CUDNN_LIB}:${LD_LIBRARY_PATH}"

export WANDB_API_KEY="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"
export WANDB_ENTITY="515718106-pku"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-sk-3881bb8218444692be4aa4ea02f8dfdb}"

export http_proxy="http://agent.baidu.com:8188"
export https_proxy="http://agent.baidu.com:8188"

ALL_IPS=("$HEAD_IP")
for ip in $WORKER_IPS; do ALL_IPS+=("$ip"); done
NNODES=${NNODES:-${#ALL_IPS[@]}}
if [ "$NNODES" -ne "${#ALL_IPS[@]}" ]; then
    echo "[config] ERROR: NNODES=${NNODES} but configured IP count is ${#ALL_IPS[@]}: ${ALL_IPS[*]}"
    exit 1
fi
NO_PROXY_LIST="127.0.0.1,localhost,$(IFS=,; echo "${ALL_IPS[*]}")"
export no_proxy="$NO_PROXY_LIST"
export NO_PROXY="$NO_PROXY_LIST"

# Required for Megatron
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
# Fix Ray OpenTelemetry segfault (getenv in grpc_core::GetEnv)
export RAY_enable_open_telemetry=0
unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
export RAY_ENABLE_UV_RUN_RUNTIME_ENV="${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}"
export RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL="${RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL:-minor}"
export VERL_DATAPROTO_SERIALIZATION_METHOD="${VERL_DATAPROTO_SERIALIZATION_METHOD:-numpy}"

MODEL_PATH="${MODEL_PATH:-/root/paddlejob/workspace/env_run/xzj/models/Qwen3-32B}"
MODEL_CONTEXT_LIMIT=${MODEL_CONTEXT_LIMIT:-40960}
DATA_DIR="/root/paddlejob/workspace/env_run/xzj/dataset/browsecomp-plus-processed"
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.paper.parquet}
VAL_FILE=${VAL_FILE:-${DATA_DIR}/test.paper.parquet}

# ---- Batch sizes ----
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
N_RESP=${N_RESP:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}

# ---- Parallelism ----
ACTOR_TP=${ACTOR_TP:-8}
ACTOR_PP=${ACTOR_PP:-2}
ACTOR_VPP=null
ACTOR_CP=${ACTOR_CP:-1}

REF_TP=${REF_TP:-8}
REF_PP=${REF_PP:-2}
REF_VPP=null
REF_CP=${REF_CP:-1}

# Inference: TP=8, one replica per rollout node by default.
ROLLOUT_TP=${ROLLOUT_TP:-8}

# ---- Fully async resource split ----
TRAINER_NNODES=${TRAINER_NNODES:-2}
TRAINER_GPUS_PER_NODE=${TRAINER_GPUS_PER_NODE:-8}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-1}
ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE:-8}

# ---- Sequence lengths ----
# Keep rollout context defaults aligned with the synchronous BCP scripts; ACTOR_CP only affects trainer token budget.
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768}
MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-16000}
MAX_PARALLEL_CALLS=${MAX_PARALLEL_CALLS:-5}

# ---- SUPO configuration ----
WORKING_CONTEXT_LENGTH=${WORKING_CONTEXT_LENGTH:-32768}
MAX_SUMMARY_ROUNDS=${MAX_SUMMARY_ROUNDS:-5}
CONTEXT_COMPRESSION_METHOD=${CONTEXT_COMPRESSION_METHOD:-summary}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
BCP_SUMMARY_INSTRUCTION=${BCP_SUMMARY_INSTRUCTION:-$'System:\nYour operational context is full. Generate a concise summary by populating the template below.\nThis summary will be your sole context for continuing this task. Be brief but ensure all critical data is present.\n\nRules:\n- Output exactly one <summary>...</summary> block.\n- Do not call any function/tool in this turn.\n- Do not include <think>, tool calls, markdown fences, or text outside the summary tags.\n\n<summary>\nMission Objective:\n- Original query: [State the user verbatim query.]\n- Verification checklist:\n  - [VERIFIED/PENDING]: [Checklist item]\n\nKey Findings:\n- Sources:\n  - [Critical verified fact with source docid]\n- Discrepancies:\n  - [Conflicting information or uncertainty]\n\nTactical Plan:\n- Promising leads:\n  - [Best remaining keywords, sources, or angles]\n- Known dead ends:\n  - [Queries or sources that proved useless]\n- Immediate next action:\n  - [Exact tool call or query to execute next]\n</summary>'}

ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
SGLANG_CHUNKED_PREFILL_SIZE=${SGLANG_CHUNKED_PREFILL_SIZE:-8192}
SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-32768}
TOKEN_BUDGET=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
ACTOR_TOKEN_BUDGET_PER_GPU=$(((TOKEN_BUDGET + ACTOR_CP - 1) / ACTOR_CP))
REF_TOKEN_BUDGET_PER_GPU=$(((TOKEN_BUDGET + REF_CP - 1) / REF_CP))

TOTAL_GPUS=$((NNODES * 8))
TRAINER_TOTAL_GPUS=$((TRAINER_NNODES * TRAINER_GPUS_PER_NODE))
ROLLOUT_TOTAL_GPUS=$((ROLLOUT_NNODES * ROLLOUT_GPUS_PER_NODE))
ACTOR_MODEL_PARALLEL_SIZE=$((ACTOR_TP * ACTOR_PP * ACTOR_CP))
REF_MODEL_PARALLEL_SIZE=$((REF_TP * REF_PP * REF_CP))
if [ $((TRAINER_TOTAL_GPUS + ROLLOUT_TOTAL_GPUS)) -gt "$TOTAL_GPUS" ]; then
    echo "[config] ERROR: trainer GPUs (${TRAINER_TOTAL_GPUS}) + rollout GPUs (${ROLLOUT_TOTAL_GPUS}) exceed cluster GPUs (${TOTAL_GPUS})."
    exit 1
fi
if [ "$ACTOR_MODEL_PARALLEL_SIZE" -gt "$TRAINER_TOTAL_GPUS" ] || [ $((TRAINER_TOTAL_GPUS % ACTOR_MODEL_PARALLEL_SIZE)) -ne 0 ]; then
    echo "[config] ERROR: actor MP=${ACTOR_MODEL_PARALLEL_SIZE} must divide trainer GPUs=${TRAINER_TOTAL_GPUS}."
    exit 1
fi
if [ "$REF_MODEL_PARALLEL_SIZE" -gt "$TRAINER_TOTAL_GPUS" ] || [ $((TRAINER_TOTAL_GPUS % REF_MODEL_PARALLEL_SIZE)) -ne 0 ]; then
    echo "[config] ERROR: ref MP=${REF_MODEL_PARALLEL_SIZE} must divide trainer GPUs=${TRAINER_TOTAL_GPUS}."
    exit 1
fi
if [ "$ROLLOUT_TOTAL_GPUS" -le 0 ] || [ $((ROLLOUT_TOTAL_GPUS % ROLLOUT_TP)) -ne 0 ]; then
    echo "[config] ERROR: rollout TP=${ROLLOUT_TP} must divide rollout GPUs=${ROLLOUT_TOTAL_GPUS}."
    exit 1
fi
if [ "$TOKEN_BUDGET" -ge "$MAX_MODEL_LEN" ]; then
    echo "[config] ERROR: MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH=${TOKEN_BUDGET} must be < MAX_MODEL_LEN=${MAX_MODEL_LEN}."
    exit 1
fi
if [ "$TOKEN_BUDGET" -le "$WORKING_CONTEXT_LENGTH" ]; then
    echo "[config] WARNING: MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH=${TOKEN_BUDGET} <= WORKING_CONTEXT_LENGTH=${WORKING_CONTEXT_LENGTH}; rollout may terminate before SUPO split triggers."
fi
if [ "$MAX_MODEL_LEN" -gt "$MODEL_CONTEXT_LIMIT" ]; then
    echo "[config] ERROR: MAX_MODEL_LEN=${MAX_MODEL_LEN} exceeds model context limit=${MODEL_CONTEXT_LIMIT}."
    exit 1
fi
echo "[config] context: prompt=${MAX_PROMPT_LENGTH}, response=${MAX_RESPONSE_LENGTH}, working=${WORKING_CONTEXT_LENGTH}, max_model=${MAX_MODEL_LEN}"
echo "[config] compression: method=${CONTEXT_COMPRESSION_METHOD}, max_rounds=${MAX_SUMMARY_ROUNDS}, effective_context=$((WORKING_CONTEXT_LENGTH * (MAX_SUMMARY_ROUNDS + 1)))"
echo "[config] entry=supo_fully_async, trainer GPUs=${TRAINER_TOTAL_GPUS}, rollout GPUs=${ROLLOUT_TOTAL_GPUS}"
echo "[config] parallel: actor TP/PP/CP=${ACTOR_TP}/${ACTOR_PP}/${ACTOR_CP}, ref TP/PP/CP=${REF_TP}/${REF_PP}/${REF_CP}, rollout TP=${ROLLOUT_TP}"
echo "[config] token budget per GPU: actor=${ACTOR_TOKEN_BUDGET_PER_GPU}, ref=${REF_TOKEN_BUDGET_PER_GPU}"

DATA_TOOL_CONFIG_OVERRIDES=(+data.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml")
if [ "${INJECT_DATA_TOOL_SCHEMAS:-True}" = "False" ]; then
    DATA_TOOL_CONFIG_OVERRIDES=(data.tool_config_path=null)
fi

# ---- Sync run script and critical files to worker nodes ----
echo "[sync] Syncing run script and BCP integration files to worker nodes ..."
SCRIPT_FILE="${PROJECT_DIR}/examples/sglang_multiturn/$(basename "$0")"
SYNC_FILES=(
    "$SCRIPT_FILE"
    "${PROJECT_DIR}/examples/sglang_multiturn/config/bcp_multiturn_megatron_grpo.yaml"
    "${PROJECT_DIR}/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml"
    "${PROJECT_DIR}/verl/protocol.py"
    "${PROJECT_DIR}/verl/models/mcore/registry.py"
    "${PROJECT_DIR}/verl/models/weight_loader_registry.py"
    "${PROJECT_DIR}/verl/experimental/agent_loop/agent_loop.py"
    "${PROJECT_DIR}/verl/experimental/agent_loop/tool_agent_loop.py"
    "${PROJECT_DIR}/verl/experimental/agent_loop/tool_parser.py"
    "${PROJECT_DIR}/verl/experimental/fully_async_policy/fully_async_main.py"
    "${PROJECT_DIR}/verl/experimental/fully_async_policy/fully_async_rollouter.py"
    "${PROJECT_DIR}/verl/experimental/fully_async_policy/fully_async_trainer.py"
    "${PROJECT_DIR}/verl/experimental/fully_async_policy/detach_utils.py"
    "${PROJECT_DIR}/verl/experimental/fully_async_policy/message_queue.py"
    "${PROJECT_DIR}/verl/experimental/fully_async_policy/agent_loop/__init__.py"
    "${PROJECT_DIR}/verl/experimental/fully_async_policy/agent_loop/agent_loop.py"
    "${PROJECT_DIR}/verl/experimental/separation/engine_workers.py"
    "${PROJECT_DIR}/verl/experimental/separation/ray_trainer.py"
    "${PROJECT_DIR}/verl/experimental/separation/utils.py"
    "${PROJECT_DIR}/verl/experimental/reward_loop/reward_manager/naive.py"
    "${PROJECT_DIR}/verl/checkpoint_engine/base.py"
    "${PROJECT_DIR}/verl/checkpoint_engine/nccl_checkpoint_engine.py"
    "${PROJECT_DIR}/verl/workers/config/rollout.py"
    "${PROJECT_DIR}/verl/workers/actor/megatron_actor.py"
    "${PROJECT_DIR}/verl/workers/critic/megatron_critic.py"
    "${PROJECT_DIR}/verl/workers/utils/losses.py"
    "${PROJECT_DIR}/verl/workers/rollout/sglang_rollout/sglang_rollout.py"
    "${PROJECT_DIR}/verl/workers/rollout/sglang_rollout/async_sglang_server.py"
    "${PROJECT_DIR}/verl/trainer/config/algorithm.py"
    "${PROJECT_DIR}/verl/trainer/ppo/core_algos.py"
    "${PROJECT_DIR}/verl/utils/metric/utils.py"
    "${PROJECT_DIR}/verl/tools/search_tool.py"
    "${PROJECT_DIR}/verl/tools/open_page_tool.py"
    "${PROJECT_DIR}/verl/tools/finish_tool.py"
    "${PROJECT_DIR}/verl/tools/utils/search_r1_like_utils.py"
    "${PROJECT_DIR}/verl/utils/reward_score/bc-p.py"
    "${PROJECT_DIR}/verl/utils/reward_score/bc_p_llm_judge.py"
)
for ip in $WORKER_IPS; do
    for file in "${SYNC_FILES[@]}"; do
        timeout 10 scp -o ConnectTimeout=5 "$file" "$ip":"$file" || echo "[sync] WARNING: failed to sync $file to $ip"
    done
done
echo "[sync] Done."

# ---- Ray cluster ----
echo "[ray] Stopping any existing Ray instances ..."
"$RAY_BIN" stop --force 2>/dev/null || ray stop --force 2>/dev/null || true
for ip in $WORKER_IPS; do
    ssh "$ip" "source ${VENV_PATH}/bin/activate && ${RAY_BIN} stop --force 2>/dev/null || ray stop --force 2>/dev/null || true" 2>/dev/null || true
done
sleep 2

echo "[ray] Starting Ray head on $HEAD_IP ($NNODES nodes total) ..."
"$RAY_BIN" start --head \
    --node-ip-address="$HEAD_IP" \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-gpus=8

for ip in $WORKER_IPS; do
    echo "[ray] Starting worker on $ip ..."
    ssh "$ip" "unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES && export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' && export LD_LIBRARY_PATH='${VENV_PATH}/lib/python3.10/site-packages/nvidia/cudnn/lib:\$LD_LIBRARY_PATH' && export RAY_enable_open_telemetry=0 && export RAY_ENABLE_UV_RUN_RUNTIME_ENV='${RAY_ENABLE_UV_RUN_RUNTIME_ENV}' && export RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL='${RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL}' && export VERL_DATAPROTO_SERIALIZATION_METHOD='${VERL_DATAPROTO_SERIALIZATION_METHOD}' && source ${VENV_PATH}/bin/activate && ${PYTHON_BIN} -V && ${RAY_BIN} start --address='${HEAD_IP}:6379' --num-gpus=8"
done

echo "[ray] Waiting for ${NNODES} nodes ..."
for i in $(seq 1 60); do
    NODE_COUNT=$("$PYTHON_BIN" -c \
        "import ray; ray.init(address='${HEAD_IP}:6379', ignore_reinit_error=True); print(len(ray.nodes()))" \
        2>/dev/null)
    if [ "${NODE_COUNT:-0}" -ge "$NNODES" ] 2>/dev/null; then
        echo "[ray] Cluster ready: $NODE_COUNT nodes."; break
    fi
    [ "$i" -eq 60 ] && { echo "[ray] ERROR: timeout."; "$RAY_BIN" status --address="${HEAD_IP}:6379"; exit 1; }
    sleep 2
done

# ---- Retrieval service ----
echo "[retriever] Killing any existing process on port 8000 ..."
fuser -k 8000/tcp 2>/dev/null || true
sleep 1
# Extra cleanup: kill any process still listening on port 8000
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
    "$RAY_BIN" stop --force 2>/dev/null || true
    for ip in $WORKER_IPS; do ssh "$ip" "source ${VENV_PATH}/bin/activate && ${RAY_BIN} stop --force" 2>/dev/null || true; done
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

# ---- Retrieval server watchdog (restarts server if health check fails) ----
RETRIEVER_LOG="${LOG_DIR}/browsecomp_retriever.log"
RETRIEVER_CMD=("${VENV_PATH}/bin/python3"
    "$PROJECT_DIR/examples/sglang_multiturn/browsecomp_retrieval_server.py"
    --mode dense
    --model /root/paddlejob/workspace/env_run/xzj/models/Qwen3-Embedding-8B
    --device cpu
    --corpus_file "${DATA_DIR}/corpus.parquet"
    --host 0.0.0.0 --port 8000
    --batch_size 4
    --dense_cache /root/paddlejob/workspace/env_run/xzj/browsecomp_dense_cache_tevatron.pkl)

(
    while true; do
        sleep 30
        if ! curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
            echo "[$(date)] [watchdog] Retrieval server unhealthy, restarting..." >> "$RETRIEVER_LOG"
            fuser -k 8000/tcp 2>/dev/null || true
            sleep 2
            setsid "${RETRIEVER_CMD[@]}" >> "$RETRIEVER_LOG" 2>&1 &
            RETRIEVER_PID=$!
            echo "[$(date)] [watchdog] Restarted with PID $RETRIEVER_PID" >> "$RETRIEVER_LOG"
            for j in $(seq 1 120); do
                if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
                    echo "[$(date)] [watchdog] Server back online after ${j}s" >> "$RETRIEVER_LOG"
                    break
                fi
                if ! kill -0 $RETRIEVER_PID 2>/dev/null; then
                    echo "[$(date)] [watchdog] Server crashed during restart" >> "$RETRIEVER_LOG"
                    break
                fi
                sleep 1
            done
        fi
    done
) &
WATCHDOG_PID=$!
echo "[watchdog] Started retrieval server watchdog (PID=$WATCHDOG_PID)"

# ---- Training ----
FULL_ASYNC_TRIGGER_PARAMETER_SYNC_STEP=${FULL_ASYNC_TRIGGER_PARAMETER_SYNC_STEP:-$((TRAIN_BATCH_SIZE / PPO_MINI_BATCH_SIZE))}
if [ "$FULL_ASYNC_TRIGGER_PARAMETER_SYNC_STEP" -lt 1 ]; then
    FULL_ASYNC_TRIGGER_PARAMETER_SYNC_STEP=1
fi
FULL_ASYNC_STALENESS_THRESHOLD=${FULL_ASYNC_STALENESS_THRESHOLD:-0.5}
FULL_ASYNC_REQUIRE_BATCHES=${FULL_ASYNC_REQUIRE_BATCHES:-1}
FULL_ASYNC_PARTIAL_ROLLOUT=${FULL_ASYNC_PARTIAL_ROLLOUT:-True}
FULL_ASYNC_TOTAL_ROLLOUT_STEPS=${FULL_ASYNC_TOTAL_ROLLOUT_STEPS:-1000000000}
FULL_ASYNC_TOTAL_TRAINING_STEPS=${FULL_ASYNC_TOTAL_TRAINING_STEPS:-$((FULL_ASYNC_TOTAL_ROLLOUT_STEPS / (PPO_MINI_BATCH_SIZE * FULL_ASYNC_REQUIRE_BATCHES * FULL_ASYNC_TRIGGER_PARAMETER_SYNC_STEP)))}
if [ "$FULL_ASYNC_TOTAL_TRAINING_STEPS" -lt 1 ]; then
    FULL_ASYNC_TOTAL_TRAINING_STEPS=1
fi

echo "[train] Starting fully async SUPO trainer: trigger_sync=${FULL_ASYNC_TRIGGER_PARAMETER_SYNC_STEP}, staleness=${FULL_ASYNC_STALENESS_THRESHOLD}, partial=${FULL_ASYNC_PARTIAL_ROLLOUT}, optim_steps=${FULL_ASYNC_TOTAL_TRAINING_STEPS}"
"$PYTHON_BIN" -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path="$PROJECT_DIR/examples/sglang_multiturn/config" \
    --config-name='bcp_multiturn_megatron_grpo' \
    algorithm.adv_estimator=supo \
    data.train_batch_size=0 \
    +data.gen_batch_size=1 \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=False \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    "${DATA_TOOL_CONFIG_OVERRIDES[@]}" \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_decay_style=constant \
    actor_rollout_ref.actor.optim.total_training_steps=${FULL_ASYNC_TOTAL_TRAINING_STEPS} \
    actor_rollout_ref.actor.optim.lr_decay_steps=${FULL_ASYNC_TOTAL_TRAINING_STEPS} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_TOKEN_BUDGET_PER_GPU} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    +actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.actor.clip_ratio=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.20 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP} \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${ACTOR_VPP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP} \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ACTOR_TOKEN_BUDGET_PER_GPU} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.n=${N_RESP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.context_length=${MAX_MODEL_LEN} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=${SGLANG_CHUNKED_PREFILL_SIZE} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_prefill_tokens=${SGLANG_MAX_PREFILL_TOKENS} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=${MAX_PARALLEL_CALLS} \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.multi_turn.format=${MULTI_TURN_FORMAT:-hermes} \
    +actor_rollout_ref.rollout.multi_turn.inject_tool_schemas=${INJECT_ROLLOUT_TOOL_SCHEMAS:-True} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml" \
    +actor_rollout_ref.rollout.multi_turn.enable_summarization=True \
    +actor_rollout_ref.rollout.multi_turn.context_compression_method=${CONTEXT_COMPRESSION_METHOD} \
    +actor_rollout_ref.rollout.multi_turn.max_summary_rounds=${MAX_SUMMARY_ROUNDS} \
    +actor_rollout_ref.rollout.multi_turn.working_context_length=${WORKING_CONTEXT_LENGTH} \
    +actor_rollout_ref.rollout.multi_turn.summary_instruction="'${BCP_SUMMARY_INSTRUCTION}'" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers=${AGENT_LOOP_WORKERS:-16} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${REF_TOKEN_BUDGET_PER_GPU} \
    actor_rollout_ref.ref.megatron.use_mbridge=True \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${REF_TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${REF_PP} \
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=${REF_VPP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${REF_CP} \
    actor_rollout_ref.ref.megatron.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=True \
    +reward.custom_reward_function.path="$PROJECT_DIR/verl/utils/reward_score/bc_p_llm_judge.py" \
    reward.custom_reward_function.name=compute_score \
    reward.reward_model.enable=False \
    +ray_kwargs.ray_init.address="${HEAD_IP}:6379" \
    +ray_kwargs.ray_init.runtime_env.env_vars.RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL="'${RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.RAY_ENABLE_UV_RUN_RUNTIME_ENV="'${RAY_ENABLE_UV_RUN_RUNTIME_ENV}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.RAY_enable_open_telemetry="'${RAY_enable_open_telemetry}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_DATAPROTO_SERIALIZATION_METHOD="'${VERL_DATAPROTO_SERIALIZATION_METHOD}'" \
    trainer.use_legacy_worker_impl=disable \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=${TRAINER_GPUS_PER_NODE} \
    trainer.nnodes=${TRAINER_NNODES} \
    rollout.n_gpus_per_node=${ROLLOUT_GPUS_PER_NODE} \
    rollout.nnodes=${ROLLOUT_NNODES} \
    rollout.n=${N_RESP} \
    rollout.total_rollout_steps=${FULL_ASYNC_TOTAL_ROLLOUT_STEPS} \
    trainer.project_name='echo' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${CKPT_DIR:-${PROJECT_DIR}/ckpt}/${EXPERIMENT_NAME}" \
    trainer.logger='["console", "wandb"]' \
    trainer.save_freq=${SAVE_FREQ:-10} \
    trainer.test_freq=${FULL_ASYNC_TEST_FREQ:-5} \
    trainer.total_epochs=5 \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-True} \
    async_training.staleness_threshold=${FULL_ASYNC_STALENESS_THRESHOLD} \
    async_training.trigger_parameter_sync_step=${FULL_ASYNC_TRIGGER_PARAMETER_SYNC_STEP} \
    async_training.require_batches=${FULL_ASYNC_REQUIRE_BATCHES} \
    async_training.partial_rollout=${FULL_ASYNC_PARTIAL_ROLLOUT} \
    +trainer.master_port_range='[31000,32000]' \
    +trainer.validation_data_dir="${VALIDATION_DATA_DIR:-${PROJECT_DIR}/val_outputs/${EXPERIMENT_NAME}}"
