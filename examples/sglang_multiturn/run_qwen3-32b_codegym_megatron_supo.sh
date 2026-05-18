#!/bin/bash
# Multi-turn CodeGym Tool-Calling SUPO baseline
# 4 nodes x 8 H100-80GB, Qwen3-32B, Megatron backend, SGLang rollout
#
# Run from project root:
#   bash examples/sglang_multiturn/run_qwen3-32b_codegym_megatron_supo.sh

set -x
ulimit -n 65535
ulimit -u unlimited

# ---- Paths / logs ----
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${CODEGYM_LOG_FILE:-${LOG_DIR}/megatron_codegym_supo_run.log}"
exec > >(tee "$LOG_FILE") 2>&1

# ---- Environment ----
VENV_PATH="/root/paddlejob/workspace/env_run/xzj/venv_echo_megatron"
CODEGYM_VENV_PATH="/root/paddlejob/workspace/env_run/xzj/venv_codegym"
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

# ---- CodeGym ----
CODEGYM_REPO="/root/paddlejob/workspace/env_run/xzj/CodeGym"
CODEGYM_DATA_ROOT="/root/paddlejob/workspace/env_run/xzj/dataset/CodeGym"
CODEGYM_TASK_DIR="${CODEGYM_DATA_ROOT}/task_en_instruction_en_env"
CODEGYM_ENVS_FILE="${CODEGYM_DATA_ROOT}/envs_en/train-00000-of-00001.parquet"
CODEGYM_ENVS_OUT_DIR="${CODEGYM_REPO}/online_server/online_server/envs/codegym_v1"
CODEGYM_SERVER_HOST="http://${HEAD_IP}:8000"
CODEGYM_WORKERS=${CODEGYM_WORKERS:-1024}

TRAIN_FILES_RAW="['${CODEGYM_TASK_DIR}/train-00000-of-00004.parquet','${CODEGYM_TASK_DIR}/train-00001-of-00004.parquet','${CODEGYM_TASK_DIR}/train-00002-of-00004.parquet']"
FILTERED_DATA_DIR="${PROJECT_DIR}/data/codegym_filtered_grpo"
TRAIN_SAMPLED="${FILTERED_DATA_DIR}/train_12800.parquet"
TRAIN_FILES="${TRAIN_SAMPLED}"
VAL_DIR="${CODEGYM_DATA_ROOT}/val"
VAL_QWEN3_FILTERED_FILE="${VAL_DIR}/val_128_qwen3_pass25.parquet"
VAL_FILES="${VAL_FILES:-${VAL_QWEN3_FILTERED_FILE}}"

# ---- Batch sizes ----
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
N_RESP=${N_RESP:-8}
MIN_CODEGYM_WORKERS=$((TRAIN_BATCH_SIZE * N_RESP))
if [ "$CODEGYM_WORKERS" -lt "$MIN_CODEGYM_WORKERS" ]; then
    echo "[codegym] Raising CODEGYM_WORKERS from ${CODEGYM_WORKERS} to ${MIN_CODEGYM_WORKERS} to match train_batch_size * n_resp."
    CODEGYM_WORKERS="$MIN_CODEGYM_WORKERS"
fi
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-12800}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-128}
VAL_KWARGS_N=${VAL_KWARGS_N:-1}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-False}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.7}
VAL_ONLY=${VAL_ONLY:-False}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${PROJECT_DIR}/val_outputs_codegym_supo}"

# ---- Parallelism ----
ACTOR_TP=${ACTOR_TP:-8}
ACTOR_PP=${ACTOR_PP:-4}
ACTOR_VPP=null
ACTOR_CP=${ACTOR_CP:-1}

REF_TP=${REF_TP:-8}
REF_PP=${REF_PP:-4}
REF_VPP=null
REF_CP=${REF_CP:-1}

ROLLOUT_TP=${ROLLOUT_TP:-8}

# ---- Sequence lengths ----
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-2048}

# ---- SUPO Configuration ----
# CodeGym paper config: 4K working context, S=7, max 8 trajectory segments.
WORKING_CONTEXT_LENGTH=${WORKING_CONTEXT_LENGTH:-4096}
MAX_SUMMARY_ROUNDS=${MAX_SUMMARY_ROUNDS:-7}
SUMMARY_MAX_CHARS=${SUMMARY_MAX_CHARS:-1536}
CODEGYM_SUMMARY_INSTRUCTION=${CODEGYM_SUMMARY_INSTRUCTION:-"System: You are a helpful agent interacting with a function calling environment to solve the problem. The interaction history is now too long. Please summarize the interaction history. Remember to keep the important information in the history to ensure that you can continue solving the problem. Do not call any function in this turn. Now generate the summary, and put your summary inside tag <summary></summary>."}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
CONTEXT_COMPRESSION_METHOD=${CONTEXT_COMPRESSION_METHOD:-summary}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.2}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}
SGLANG_CHUNKED_PREFILL_SIZE=${SGLANG_CHUNKED_PREFILL_SIZE:-4096}
SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-8192}

TOKEN_BUDGET=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
ACTOR_TOKEN_BUDGET_PER_GPU=$(((TOKEN_BUDGET + ACTOR_CP - 1) / ACTOR_CP))
REF_TOKEN_BUDGET_PER_GPU=$(((TOKEN_BUDGET + REF_CP - 1) / REF_CP))

TOTAL_GPUS=$((NNODES * 8))
ACTOR_MODEL_PARALLEL_SIZE=$((ACTOR_TP * ACTOR_PP * ACTOR_CP))
REF_MODEL_PARALLEL_SIZE=$((REF_TP * REF_PP * REF_CP))
if [ "$ACTOR_MODEL_PARALLEL_SIZE" -ne "$TOTAL_GPUS" ]; then
    echo "[config] ERROR: ACTOR_TP*ACTOR_PP*ACTOR_CP=${ACTOR_MODEL_PARALLEL_SIZE}, expected ${TOTAL_GPUS}."
    exit 1
fi
if [ "$REF_MODEL_PARALLEL_SIZE" -ne "$TOTAL_GPUS" ]; then
    echo "[config] ERROR: REF_TP*REF_PP*REF_CP=${REF_MODEL_PARALLEL_SIZE}, expected ${TOTAL_GPUS}."
    exit 1
fi
if [ "$TOKEN_BUDGET" -ge "$MAX_MODEL_LEN" ]; then
    echo "[config] ERROR: MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH=${TOKEN_BUDGET} must be < MAX_MODEL_LEN=${MAX_MODEL_LEN}."
    exit 1
fi
if [ "$MAX_MODEL_LEN" -gt 40960 ]; then
    echo "[config] ERROR: MAX_MODEL_LEN=${MAX_MODEL_LEN} exceeds local Qwen3-32B max_position_embeddings=40960."
    exit 1
fi

echo "[config] CodeGym SUPO context: prompt=${MAX_PROMPT_LENGTH}, response=${MAX_RESPONSE_LENGTH}, working=${WORKING_CONTEXT_LENGTH}, max_model=${MAX_MODEL_LEN}"
echo "[config] compression: method=${CONTEXT_COMPRESSION_METHOD}, max_rounds=${MAX_SUMMARY_ROUNDS}, summary_max_chars=${SUMMARY_MAX_CHARS}, max_segments=$((MAX_SUMMARY_ROUNDS + 1)), effective_context=$((WORKING_CONTEXT_LENGTH * (MAX_SUMMARY_ROUNDS + 1)))"
echo "[config] parallel: actor TP/PP/CP=${ACTOR_TP}/${ACTOR_PP}/${ACTOR_CP}, ref TP/PP/CP=${REF_TP}/${REF_PP}/${REF_CP}, rollout TP=${ROLLOUT_TP}"

# ---- Sync run script and critical Python files to worker nodes ----
echo "[sync] Syncing run script and CodeGym integration files to worker nodes ..."
SCRIPT_FILE="${PROJECT_DIR}/examples/sglang_multiturn/$(basename "$0")"
SYNC_FILES=(
    "$SCRIPT_FILE"
    "${PROJECT_DIR}/verl/experimental/agent_loop/agent_loop.py"
    "${PROJECT_DIR}/verl/experimental/agent_loop/codegym_agent_loop.py"
    "${PROJECT_DIR}/verl/experimental/agent_loop/tool_agent_loop.py"
    "${PROJECT_DIR}/verl/experimental/agent_loop/tool_parser.py"
    "${PROJECT_DIR}/verl/experimental/agent_loop/__init__.py"
    "${PROJECT_DIR}/verl/trainer/ppo/ray_trainer.py"
    "${PROJECT_DIR}/verl/trainer/ppo/core_algos.py"
    "${PROJECT_DIR}/verl/trainer/ppo/metric_utils.py"
    "${PROJECT_DIR}/verl/workers/config/rollout.py"
)
for ip in $WORKER_IPS; do
    for file in "${SYNC_FILES[@]}"; do
        timeout 10 scp -o ConnectTimeout=5 "$file" "$ip":"$file" || echo "[sync] WARNING: failed to sync $file to $ip"
    done
done
echo "[sync] Sync done."

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
        "import ray; ray.init(address='auto', ignore_reinit_error=True); print(sum(1 for node in ray.nodes() if node.get('Alive')))" \
        2>/dev/null)
    if [ "${NODE_COUNT:-0}" -gt "$NNODES" ] 2>/dev/null; then
        echo "[ray] ERROR: too many alive Ray nodes: $NODE_COUNT expected $NNODES."
        ray status || true
        exit 1
    fi
    if [ "${NODE_COUNT:-0}" -ge "$NNODES" ] 2>/dev/null; then
        echo "[ray] Cluster ready: $NODE_COUNT nodes."; break
    fi
    [ "$i" -eq 60 ] && { echo "[ray] ERROR: timeout."; ray status; exit 1; }
    sleep 2
done

# ---- CodeGym online server ----
echo "[codegym] Preparing CodeGym repo and online server ..."
if [ ! -d "$CODEGYM_REPO/.git" ]; then
    git clone https://github.com/StigLidu/CodeGym.git "$CODEGYM_REPO"
fi

if [ ! -d "$CODEGYM_VENV_PATH" ]; then
    if python3 -m virtualenv --version >/dev/null 2>&1; then
        python3 -m virtualenv -p /bin/python3.11 "$CODEGYM_VENV_PATH"
    elif command -v python3.11 >/dev/null 2>&1; then
        python3.11 -m venv "$CODEGYM_VENV_PATH"
    else
        python3 -m venv "$CODEGYM_VENV_PATH"
    fi
fi

if ! "${CODEGYM_VENV_PATH}/bin/python3" -c "import fastapi, uvicorn, gymnasium" >/dev/null 2>&1; then
    if [ ! -x "${CODEGYM_VENV_PATH}/bin/pip" ]; then
        python3 -m virtualenv --clear -p /bin/python3.11 "$CODEGYM_VENV_PATH"
    fi
    "${CODEGYM_VENV_PATH}/bin/pip" install -U pip
    "${CODEGYM_VENV_PATH}/bin/pip" install -r "${CODEGYM_REPO}/online_server/requirements.txt"
    "${CODEGYM_VENV_PATH}/bin/pip" install gymnasium
fi

if [ ! -f "$TRAIN_SAMPLED" ]; then
    echo "[codegym] ERROR: expected training file not found: $TRAIN_SAMPLED"
    echo "[codegym] Run the GRPO/data-prep path first, or set TRAIN_FILES to an existing parquet."
    exit 1
fi

if [ ! -f "$VAL_QWEN3_FILTERED_FILE" ]; then
    echo "[codegym] ERROR: expected validation file not found: $VAL_QWEN3_FILTERED_FILE"
    exit 1
fi

mkdir -p "$CODEGYM_ENVS_OUT_DIR"
echo "[codegym] Exporting CodeGym envs from ${CODEGYM_ENVS_FILE} ..."
"${VENV_PATH}/bin/python3" - <<PY
import ast
import json
import pathlib
import re
import pandas as pd

envs_file = pathlib.Path("${CODEGYM_ENVS_FILE}")
out_dir = pathlib.Path("${CODEGYM_ENVS_OUT_DIR}")
out_dir.mkdir(parents=True, exist_ok=True)

for path in out_dir.glob("*.py"):
    path.unlink()

env_df = pd.read_parquet(envs_file)
env_code_by_name = {str(row.env_name): str(row.env_code) for row in env_df.itertuples(index=False)}

def get_gym_env(extra_info):
    if isinstance(extra_info, dict):
        return extra_info.get("gym_env")
    if isinstance(extra_info, str) and extra_info:
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(extra_info)
                if isinstance(parsed, dict):
                    return parsed.get("gym_env")
            except Exception:
                pass
    return None

def parse_files(value: str):
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        parsed = value
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return [str(parsed)]

task_files = parse_files("""${TRAIN_FILES_RAW}""") + parse_files("""${VAL_FILES}""")
exported = set()
missing = set()
exact = 0
fallback = 0

for task_file in task_files:
    task_df = pd.read_parquet(task_file, columns=["ability", "extra_info"])
    for row in task_df.itertuples(index=False):
        if pd.isna(row.ability):
            continue
        ability = str(row.ability)
        env_str = ability[len("codegym_v1@"):] if ability.startswith("codegym_v1@") else ability
        prefix = env_str.split("@", 1)[0]
        class_name = prefix.rsplit("__", 1)[-1]
        env_code = get_gym_env(row.extra_info)
        if env_code:
            exact += 1
        else:
            env_code = env_code_by_name.get(class_name)
            fallback += 1
        if env_code is None:
            missing.add(prefix)
            continue
        safe_prefix = re.sub(r"[^0-9A-Za-z_]", "_", prefix)
        (out_dir / f"{safe_prefix}.py").write_text(env_code, encoding="utf-8")
        exported.add(safe_prefix)
        fallback_prefix = f"codegym_v1__{class_name}"
        (out_dir / f"{fallback_prefix}.py").write_text(env_code, encoding="utf-8")
        exported.add(fallback_prefix)

if missing:
    preview = ", ".join(sorted(missing)[:20])
    raise RuntimeError(f"missing env code for {len(missing)} envs: {preview}")
print(f"exported {len(exported)} CodeGym env aliases to {out_dir} (exact={exact}, fallback={fallback})")
PY
if [ $? -ne 0 ]; then
    echo "[codegym] ERROR: failed to export CodeGym env files."
    exit 1
fi

echo "[codegym] Killing any existing process on port 8000 ..."
fuser -k 8000/tcp 2>/dev/null || true
pkill -f "uvicorn env_server:app" 2>/dev/null || true
sleep 1
kill $(ss -tlnp 'sport = :8000' | grep -oP 'pid=\K\d+') 2>/dev/null || true
sleep 1

echo "[codegym] Patching CodeGym server bind host to IPv4 ..."
"${CODEGYM_VENV_PATH}/bin/python3" - <<PY
from pathlib import Path

path = Path("${CODEGYM_REPO}/online_server/online_server/env_instance_manager.py")
text = path.read_text(encoding="utf-8")
text = text.replace('"--host", "::"', '"--host", "0.0.0.0"')
text = text.replace('uvicorn.run(app, host="::", port=8000)', 'uvicorn.run(app, host="0.0.0.0", port=8000)')
path.write_text(text, encoding="utf-8")
PY

pushd "${CODEGYM_REPO}/online_server/online_server" >/dev/null
setsid env \
    PATH="${CODEGYM_VENV_PATH}/bin:${PATH}" \
    VIRTUAL_ENV="${CODEGYM_VENV_PATH}" \
    SERVER_PUBLIC_HOST="${HEAD_IP}" \
    "${CODEGYM_VENV_PATH}/bin/python3" \
    "env_instance_manager.py" \
    --workers "${CODEGYM_WORKERS}" \
    > "${LOG_DIR}/codegym_server.log" 2>&1 &
CODEGYM_SERVER_PID=$!
popd >/dev/null

cleanup() {
    kill $CODEGYM_SERVER_PID 2>/dev/null
    ray stop --force 2>/dev/null || true
    for ip in $WORKER_IPS; do ssh "$ip" "source ${VENV_PATH}/bin/activate && ray stop --force" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

echo "[codegym] Waiting for online server ready ..."
for i in $(seq 1 1200); do
    INSTANCE_JSON=$(curl --noproxy '*' -sf "${CODEGYM_SERVER_HOST}/get_instance" 2>/dev/null || true)
    if [ -n "$INSTANCE_JSON" ]; then
        UID=$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('uid',''))" "$INSTANCE_JSON" 2>/dev/null || true)
        if [ -n "$UID" ]; then
            curl --noproxy '*' -sf "${CODEGYM_SERVER_HOST}/release_instance?uid=${UID}" >/dev/null 2>&1 || true
            echo "[codegym] Ready after ${i}s."
            break
        fi
    fi
    if ! kill -0 $CODEGYM_SERVER_PID 2>/dev/null; then
        echo "[codegym] Server died. Log:"; cat "${LOG_DIR}/codegym_server.log"; exit 1
    fi
    [ "$i" -eq 1200 ] && { echo "[codegym] ERROR: timeout."; cat "${LOG_DIR}/codegym_server.log"; exit 1; }
    sleep 1
done

# ---- Training with SUPO on CodeGym ----
python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT_DIR/examples/sglang_multiturn/config" \
    --config-name='codegym_multiturn_megatron_grpo' \
    algorithm.adv_estimator=supo \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.val_batch_size=64 \
    data.val_max_samples=${VAL_MAX_SAMPLES} \
    data.filter_overlong_prompts=False \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.custom_chat_template=null \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_decay_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_TOKEN_BUDGET_PER_GPU} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.20 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
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
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ACTOR_TOKEN_BUDGET_PER_GPU} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.n=${N_RESP} \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_KWARGS_N} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.context_length=${MAX_MODEL_LEN} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=${SGLANG_CHUNKED_PREFILL_SIZE} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_prefill_tokens=${SGLANG_MAX_PREFILL_TOKENS} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=100 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=null \
    actor_rollout_ref.rollout.multi_turn.format=codegym_fc \
    +actor_rollout_ref.rollout.multi_turn.codegym_server_host="${CODEGYM_SERVER_HOST}" \
    +actor_rollout_ref.rollout.multi_turn.enable_summarization=True \
    +actor_rollout_ref.rollout.multi_turn.context_compression_method=${CONTEXT_COMPRESSION_METHOD} \
    +actor_rollout_ref.rollout.multi_turn.max_summary_rounds=${MAX_SUMMARY_ROUNDS} \
    +actor_rollout_ref.rollout.multi_turn.working_context_length=${WORKING_CONTEXT_LENGTH} \
    +actor_rollout_ref.rollout.multi_turn.summary_max_chars=${SUMMARY_MAX_CHARS} \
    +actor_rollout_ref.rollout.multi_turn.summary_instruction="'${CODEGYM_SUMMARY_INSTRUCTION}'" \
    actor_rollout_ref.rollout.agent.default_agent_loop=codegym_agent \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${REF_TOKEN_BUDGET_PER_GPU} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${REF_TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${REF_PP} \
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=${REF_VPP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${REF_CP} \
    actor_rollout_ref.ref.megatron.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NNODES} \
    trainer.project_name='echo' \
    trainer.experiment_name='qwen3-32b_codegym_megatron-supo-4k-s7' \
    trainer.default_local_dir="${CKPT_DIR:-${PROJECT_DIR}/ckpt}/qwen3-32b_codegym_megatron-supo-4k-s7" \
    trainer.logger='["console", "wandb"]' \
    trainer.save_freq=${SAVE_FREQ:-100} \
    trainer.val_before_train=True \
    +trainer.val_only=${VAL_ONLY} \
    trainer.test_freq=2 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    +trainer.master_port_range='[31000,32000]' \
    +trainer.validation_data_dir="${VALIDATION_DATA_DIR}"
