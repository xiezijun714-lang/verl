#!/bin/bash
# BrowseComp-Plus GRPO with Seed-OSS-36B-Instruct at 64K trajectory budget.
#
# This wrapper reuses the maintained BCP Megatron/SGLang launcher while
# overriding model and sequence-length defaults for Seed-OSS.
#
# Run from project root:
#   bash examples/sglang_multiturn/run_seed-oss-36b_bcp_megatron_64k.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-/root/paddlejob/workspace/env_run/xzj/models/Seed-OSS-36B-Instruct}"
export MODEL_CONTEXT_LIMIT="${MODEL_CONTEXT_LIMIT:-524288}"
export USE_MODEL_CHAT_TEMPLATE="${USE_MODEL_CHAT_TEMPLATE:-True}"
export MULTI_TURN_FORMAT="${MULTI_TURN_FORMAT:-bcp_fc}"
export INJECT_DATA_TOOL_SCHEMAS="${INJECT_DATA_TOOL_SCHEMAS:-False}"
export INJECT_ROLLOUT_TOOL_SCHEMAS="${INJECT_ROLLOUT_TOOL_SCHEMAS:-False}"

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-65536}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-73728}"
export MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-16000}"
export MAX_PARALLEL_CALLS="${MAX_PARALLEL_CALLS:-5}"

export ACTOR_TP="${ACTOR_TP:-8}"
export ACTOR_PP="${ACTOR_PP:-1}"
export ACTOR_CP="${ACTOR_CP:-4}"
export REF_TP="${REF_TP:-8}"
export REF_PP="${REF_PP:-1}"
export REF_CP="${REF_CP:-4}"
export ROLLOUT_TP="${ROLLOUT_TP:-8}"

export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-8}"
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-65536}"

DATA_DIR="${DATA_DIR:-/root/paddlejob/workspace/env_run/xzj/dataset/browsecomp-plus-processed}"
export TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.paper_fc.parquet}"
export VAL_FILE="${VAL_FILE:-${DATA_DIR}/test.paper_fc.parquet}"

export BCP_LOG_FILE="${BCP_LOG_FILE:-/root/paddlejob/workspace/env_run/xzj/echo/logs/megatron_seed_oss_64k_run.log}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-seed-oss-36b_bcp_megatron-64k-grpo}"
export VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-/root/paddlejob/workspace/env_run/xzj/echo/val_outputs_seed_oss_64k}"

exec bash "${SCRIPT_DIR}/run_qwen3-32b_bcp_megatron.sh"
