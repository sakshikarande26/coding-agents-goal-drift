#!/bin/bash
# Rollouts on COOK: one GPU per array job, each with its own llama-server and its own share of tasks.
#
# Submit from the logs folder (Slurm writes the job logs there):
#   cd ~/goal-drift-work/logs
#   SHARDS=10 sbatch --array=0-9 ~/coding-agents-goal-drift/sandboxing/cook_run.sbatch --n-tasks 10 --seeds 0 1 --run-name cook_8b_pilot   # pilot
#   sbatch --array=0-19 ~/coding-agents-goal-drift/sandboxing/cook_run.sbatch                                                  # full batch
# Jobs stop at the 2-hour limit. Submit the same command again: saved trajectories are skipped.
# Skip nodes with stuck GPU memory or a broken GPU (check first), e.g. sbatch --exclude=cook033,cook037 ...
# Extra arguments are passed to `goal-drift run`. It runs the code in $REPO (python -m), not the installed copy.
#
#SBATCH --job-name=goal-drift
#SBATCH --partition=gtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%A_%a.log

set -euo pipefail

REPO=${REPO:-$HOME/coding-agents-goal-drift}
CONFIG=${CONFIG:-$REPO/sandboxing/cook_8b.yaml}
GGUF=${GGUF:-$HOME/models/Qwen3-8B-Q8_0.gguf}
LLAMA_MODULE=${LLAMA_MODULE:-llm-r580-llamacpp/2026.10}
CTX=${CTX:-16384}   # 8.7 GB of weights leave about 2.5 GB for the KV cache on an 11 GB 1080 Ti
KV=${KV:-f16}       # KV cache type (q8_0 needs flash attention)
FA=${FA:-off}       # flash attention. On the 1080 Ti it is 5x slower at 4k context (23.7 -> 4.3 tok/s, llama-bench)
UBATCH=${UBATCH:-256}  # prompt chunk size. Without flash attention the default 512 needs a 1.1 GB buffer that does not fit

SHARD=${SLURM_ARRAY_TASK_ID:-0}
# Fixed total, not SLURM_ARRAY_TASK_COUNT: resubmitting only some shards (--array=1,11) must keep each shard's tasks.
SHARDS=${SHARDS:-20}
PORT=$((20000 + ${SLURM_JOB_ID:-0} % 10000))   # several jobs can share a node
LOGS=$HOME/goal-drift-work/logs
# Own udocker folder per shard: parallel pulls never write the same image layers at once.
export UDOCKER_DIR=$HOME/goal-drift-work/udocker/shard_$SHARD

module load "$LLAMA_MODULE"
source "$HOME/venvs/goal-drift/bin/activate"
mkdir -p "$LOGS" "$UDOCKER_DIR"
[ -d "$UDOCKER_DIR/bin" ] || udocker install > /dev/null

SERVER_LOG=$LOGS/llama_${SLURM_JOB_ID:-local}_$SHARD.log
# -ngl 99: whole model on the GPU. --jinja: Qwen3 chat template (tool calls, thinking switch).
llama-server -m "$GGUF" --port "$PORT" -ngl 99 -c "$CTX" -np 1 --jinja \
    -ctk "$KV" -ctv "$KV" -fa "$FA" -ub "$UBATCH" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for _ in $(seq 120); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "llama-server died:"; tail -30 "$SERVER_LOG"; exit 1; }
    sleep 5
done
curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null || { echo "llama-server not up after 10 min"; exit 1; }
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

cd "$REPO"
export PYTHONUNBUFFERED=1   # progress lines reach the job log right away
python -m sandboxing.run run --config "$CONFIG" --server-url "http://127.0.0.1:$PORT/v1" --shard "$SHARD/$SHARDS" "$@"
