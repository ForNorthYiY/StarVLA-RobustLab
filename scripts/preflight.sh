#!/usr/bin/env bash
set -uo pipefail

STRICT="${STRICT:-0}"
EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-8}"
MIN_GPU_MEMORY_MIB="${MIN_GPU_MEMORY_MIB:-38000}"
failures=0
warnings=0

ok() { printf '[OK] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; warnings=$((warnings + 1)); }
fail() { printf '[FAIL] %s\n' "$*"; failures=$((failures + 1)); }

check_command() {
  local command_name="$1"
  if command -v "${command_name}" >/dev/null 2>&1; then
    ok "command available: ${command_name}"
  else
    fail "command missing: ${command_name}"
    return 1
  fi
}

printf 'StarVLA-RobustLab preflight\n'
printf 'strict=%s expected_gpu_count=%s min_gpu_memory_mib=%s\n\n' \
  "${STRICT}" "${EXPECTED_GPU_COUNT}" "${MIN_GPU_MEMORY_MIB}"

check_command python || true
check_command git || true

if command -v python >/dev/null 2>&1; then
  python --version
  python - <<'PY' || fail "Python package/runtime probe failed"
import importlib

packages = ["torch", "transformers", "flash_attn", "accelerate", "deepspeed"]
for name in packages:
    try:
        module = importlib.import_module(name)
        print(f"{name}: {getattr(module, '__version__', 'imported (version unavailable)')}")
    except Exception as exc:
        print(f"{name}: MISSING ({type(exc).__name__}: {exc})")

try:
    import torch
    print(f"torch.cuda.runtime: {torch.version.cuda}")
    print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
    print(f"torch.cuda.bf16_supported: {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}")
except Exception as exc:
    print(f"torch probe unavailable: {type(exc).__name__}: {exc}")
PY
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  nvidia-smi topo -m || warn "nvidia-smi topology query failed"
  mapfile -t gpu_memory < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)
  gpu_count="${#gpu_memory[@]}"
  if [[ "${gpu_count}" -eq "${EXPECTED_GPU_COUNT}" ]]; then
    ok "GPU count is ${gpu_count}"
  else
    fail "GPU count is ${gpu_count}; expected ${EXPECTED_GPU_COUNT}"
  fi
  for index in "${!gpu_memory[@]}"; do
    memory="${gpu_memory[$index]//[[:space:]]/}"
    if [[ "${memory}" =~ ^[0-9]+$ ]] && (( memory >= MIN_GPU_MEMORY_MIB )); then
      ok "GPU ${index} memory ${memory} MiB"
    else
      fail "GPU ${index} memory ${memory:-unknown} MiB; expected >= ${MIN_GPU_MEMORY_MIB}"
    fi
  done
else
  fail "nvidia-smi is unavailable"
fi

if command -v nvcc >/dev/null 2>&1; then
  nvcc -V
else
  fail "nvcc is unavailable"
fi

if command -v df >/dev/null 2>&1; then
  printf '\nDisk usage:\n'
  df -h .
  [[ -d /dev/shm ]] && { printf '\nShared memory:\n'; df -h /dev/shm; }
else
  warn "df is unavailable; disk/shared-memory checks skipped"
fi

printf '\nSummary: failures=%s warnings=%s\n' "${failures}" "${warnings}"
if [[ "${STRICT}" == "1" && "${failures}" -gt 0 ]]; then
  exit 1
fi
exit 0
