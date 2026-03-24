#!/bin/bash
# Multi-turn Search Tool-Calling GRPO on BrowseComp-Plus, 8xGPU, Qwen3-8B, SGLang rollout
# Run from project root: bash examples/sglang_multiturn/run_qwen3-8b_browsecomp_sglang.sh

set -x
ulimit -n 65535
ulimit -u unlimited
source /root/paddlejob/workspace/xzj/venv_verl/bin/activate

# Propagate wandb API key to Ray workers
export WANDB_API_KEY="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"
export WANDB_ENTITY="515718106-pku"  

# Bypass proxy for localhost (http_proxy env var would break curl health check)
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

MODEL_PATH="/root/paddlejob/workspace/xzj/model/Qwen3-8B"
TRAIN_DATA="/root/paddlejob/workspace/xzj/dataset/browsecomp-plus-processed/train.parquet"
VAL_DATA="/root/paddlejob/workspace/xzj/dataset/browsecomp-plus-processed/test.parquet"
DATA_DIR="/root/paddlejob/workspace/xzj/dataset/browsecomp-plus-processed"

TRAIN_BATCH_SIZE=64
MICRO_BATCH_SIZE=4

# ---- Start retrieval server in background (setsid = new process group) ----
# Kill any existing retrieval server on port 8000
echo "[retriever] Killing any existing process on port 8000 ..."
fuser -k 8000/tcp 2>/dev/null || true
sleep 1

echo "[retriever] Starting BrowseComp retrieval server ..."
setsid /root/paddlejob/workspace/xzj/venv_verl/bin/python3 "$PROJECT_DIR/examples/sglang_multiturn/browsecomp_retrieval_server.py" \
    --data_dir "$DATA_DIR" --host 127.0.0.1 --port 8000 \
    > /tmp/browsecomp_retriever.log 2>&1 &
RETRIEVER_PID=$!

# On normal exit: kill the server
trap 'echo "[retriever] Shutting down (PID $RETRIEVER_PID)..."; kill $RETRIEVER_PID 2>/dev/null' EXIT
# On Ctrl+C / SIGTERM: kill server then exit (otherwise bash re-runs the loop)
trap 'echo "[retriever] Interrupted. Shutting down (PID $RETRIEVER_PID)..."; kill $RETRIEVER_PID 2>/dev/null; exit 130' INT TERM

# Wait until /health returns 200 (max 600s — first run builds BM25 index from 67K docs)
echo "[retriever] Waiting for server to be ready (first run may take ~2 min for BM25 build) ..."
for i in $(seq 1 600); do
    if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "[retriever] Ready after ${i}s."
        break
    fi
    if ! kill -0 $RETRIEVER_PID 2>/dev/null; then
        echo "[retriever] Server process died. Log:"
        cat /tmp/browsecomp_retriever.log
        exit 1
    fi
    sleep 1
done

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path="$PROJECT_DIR/verl/utils/reward_score/bc-p.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.project_name='echo' \
    trainer.experiment_name='qwen3-8b_grpo_browsecomp_8gpu_sglang' \
    trainer.logger='["console", "wandb"]' \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    +trainer.master_port_range='[31000,32000]' \
    +trainer.validation_data_dir='/root/paddlejob/workspace/xzj/verl/val_outputs' \
