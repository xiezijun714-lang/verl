#!/bin/bash
# Vanilla GRPO on GSM8K multi-turn w/ tool, 8xGPU, Qwen3.5-9B, SGLang rollout
# Run from project root: bash examples/sglang_multiturn/run_qwen3.5-9b_gsm8k_8gpu_sglang.sh

set -x
ulimit -n 65535

# Propagate wandb API key to Ray workers
export WANDB_API_KEY="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"

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

MODEL_PATH="/root/paddlejob/workspace/xzj/model/Qwen3-8B"
TRAIN_DATA="/root/paddlejob/workspace/xzj/dataset/gsm8k/train.parquet"
VAL_DATA="/root/paddlejob/workspace/xzj/dataset/gsm8k/test.parquet"

TRAIN_BATCH_SIZE=64
MICRO_BATCH_SIZE=4

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.project_name='echo' \
    trainer.experiment_name='qwen3-8b_vanilla_grpo_gsm8k_8gpu_sglang' \
    trainer.logger='["console", "mlflow", "wandb"]' \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.total_epochs=15 \
    trainer.master_port_range='[31000,32000]'
