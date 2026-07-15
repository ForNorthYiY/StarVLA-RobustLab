#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-6}"
PORT="${PORT:-10093}"
PROJECT_ROOT="${PROJECT_ROOT:-/data/yiyang/git/StarVLA-RobustLab}"
STARVLA_ROOT="${STARVLA_ROOT:-${PROJECT_ROOT}/third_party/starVLA}"
LIBERO_ROOT="${LIBERO_ROOT:-/data/yiyang/git/LIBERO-8f1084e}"
MODEL_PYTHON="${MODEL_PYTHON:-/data/yiyang/miniconda3/envs/starvla/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/data/yiyang/miniconda3/envs/libero/bin/python}"
CHECKPOINT="${CHECKPOINT:-/data/yiyang/checkpoints/starvla/qwen3-vl-oft-libero-4in1-sdpa-smoke/checkpoints/steps_50000_pytorch_model.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/yiyang/logs/starvla/replan_sweep}"

used_mib="$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits)"
if (( used_mib > 100 )); then
  echo "Refusing to start: GPU ${GPU_ID} already uses ${used_mib} MiB." >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
cd "${STARVLA_ROOT}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="${STARVLA_ROOT}" \
  "${MODEL_PYTHON}" deployment/model_server/server_policy.py \
  --ckpt_path "${CHECKPOINT}" --port "${PORT}" --use_bf16 --idle_timeout 1800 \
  >"${OUTPUT_ROOT}/server.log" 2>&1 &
server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  if grep -q "server listening" "${OUTPUT_ROOT}/server.log"; then
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "Policy server exited during startup." >&2
    exit 3
  fi
  sleep 1
done
grep -q "server listening" "${OUTPUT_ROOT}/server.log" || {
  echo "Policy server did not become ready within 60 seconds." >&2
  exit 4
}

export LIBERO_CONFIG_PATH=/data/yiyang/config/libero
export PYTHONPATH="${LIBERO_ROOT}:${STARVLA_ROOT}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

for interval in 1 2 4 8; do
  "${LIBERO_PYTHON}" examples/LIBERO/eval_files/eval_libero.py \
    --args.host 127.0.0.1 \
    --args.port "${PORT}" \
    --args.task-suite-name libero_goal \
    --args.num-trials-per-task 1 \
    --args.max-tasks 1 \
    --args.replan-interval "${interval}" \
    --args.unnorm-key franka \
    --args.video-out-path "${OUTPUT_ROOT}/interval_${interval}/videos" \
    2>&1 | tee "${OUTPUT_ROOT}/interval_${interval}/client.log"
done
