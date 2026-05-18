#!/bin/bash
# Multi-turn BrowseComp-Plus truncated-context baseline.
#
# Reuses the SUPO segment-splitting path, but replaces generated summaries with
# a raw left-truncated context suffix.

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

export CONTEXT_COMPRESSION_METHOD="${CONTEXT_COMPRESSION_METHOD:-truncate}"
export MAX_SUMMARY_ROUNDS="${MAX_TRUNCATION_ROUNDS:-${MAX_SUMMARY_ROUNDS:-5}}"
export WORKING_CONTEXT_LENGTH="${WORKING_CONTEXT_LENGTH:-32768}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-32768}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-7680}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-32768}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-32b_bcp_megatron-truncated-32k}"
export BCP_LOG_FILE="${BCP_LOG_FILE:-${PROJECT_DIR}/logs/megatron_truncated_run.log}"
export VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-/root/paddlejob/workspace/env_run/xzj/echo/val_outputs_truncated}"

exec bash "${PROJECT_DIR}/examples/sglang_multiturn/run_qwen3-32b_bcp_megatron_supo.sh"
