#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT="${ENV_ROOT:-/data/yiyang/miniconda3/envs/starvla}"
PROJECT_ROOT="${PROJECT_ROOT:-/data/yiyang/git/StarVLA-RobustLab}"
MODEL_ROOT="${MODEL_ROOT:-/data/yiyang/models}"
LOG_ROOT="${LOG_ROOT:-/data/yiyang/logs/starvla}"
GPU_ID="${GPU_ID:-6}"

POLICY_DIR="${MODEL_ROOT}/StarVLA/Qwen3-VL-OFT-LIBERO-4in1"
BASE_DIR="${MODEL_ROOT}/Qwen/Qwen3-VL-4B-Instruct"

printf '[git]\n'
git -C "${PROJECT_ROOT}" status --short --branch
printf 'project_commit=%s\n' "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
printf 'starvla_commit=%s\n' "$(git -C "${PROJECT_ROOT}/third_party/starVLA" rev-parse HEAD)"

printf '\n[gpu]\n'
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

printf '\n[python]\n'
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${ENV_ROOT}/bin/python" -c \
  'import torch, transformers; print("torch", torch.__version__, "runtime_cuda", torch.version.cuda); print("transformers", transformers.__version__); print("cuda_available", torch.cuda.is_available()); print("visible_devices", torch.cuda.device_count()); print("bf16", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)'

printf '\n[artifacts]\n'
for path in \
  "${POLICY_DIR}/config.yaml" \
  "${POLICY_DIR}/dataset_statistics.json" \
  "${POLICY_DIR}/checkpoints/steps_50000_pytorch_model.pt" \
  "${BASE_DIR}/config.json" \
  "${BASE_DIR}/model-00001-of-00002.safetensors" \
  "${BASE_DIR}/model-00002-of-00002.safetensors"; do
  if [[ -f "${path}" ]]; then
    size=$(stat -c '%s' "${path}")
    printf 'present bytes=%s %s\n' "${size}" "${path}"
  else
    printf 'missing %s\n' "${path}"
  fi
done

printf '\n[download processes]\n'
pgrep -af 'aria2c|snapshot_download' || true

printf '\n[logs]\n'
for log in \
  "${LOG_ROOT}/download_checkpoint.log" \
  "${LOG_ROOT}/download_base_shard1.log" \
  "${LOG_ROOT}/download_base_shard2.log"; do
  printf '%s\n' "--- ${log}"
  tail -n 3 "${log}" 2>/dev/null || true
done

printf '\n[disk]\n'
df -h /data
