#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/data/yiyang/models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1}"
DEST_ROOT="${DEST_ROOT:-/data/yiyang/checkpoints/starvla/qwen3-vl-oft-libero-4in1-sdpa-smoke}"
BASE_VLM="${BASE_VLM:-/data/yiyang/models/Qwen/Qwen3-VL-4B-Instruct}"
PYTHON="${PYTHON:-/data/yiyang/miniconda3/envs/starvla/bin/python}"

SOURCE_WEIGHT="${SOURCE_ROOT}/checkpoints/steps_50000_pytorch_model.pt"

for path in \
  "${SOURCE_ROOT}/config.yaml" \
  "${SOURCE_ROOT}/dataset_statistics.json" \
  "${SOURCE_WEIGHT}" \
  "${BASE_VLM}/config.json"; do
  if [[ ! -f "${path}" ]]; then
    printf 'Required artifact is missing: %s\n' "${path}" >&2
    exit 1
  fi
  if [[ -f "${path}.aria2" ]]; then
    printf 'Artifact download is incomplete: %s\n' "${path}" >&2
    exit 1
  fi
done

if [[ "${DEST_ROOT}" == "${SOURCE_ROOT}" ]]; then
  printf 'DEST_ROOT must differ from SOURCE_ROOT; the official artifact is immutable.\n' >&2
  exit 1
fi

mkdir -p "${DEST_ROOT}/checkpoints"

"${PYTHON}" - "${SOURCE_ROOT}/config.yaml" "${DEST_ROOT}/config.yaml" "${BASE_VLM}" <<'PY'
from pathlib import Path
import sys

import yaml

source, destination, base_vlm = map(Path, sys.argv[1:])
config = yaml.safe_load(source.read_text(encoding="utf-8"))
config.setdefault("framework", {}).setdefault("qwenvl", {})["base_vlm"] = str(base_vlm)
config["framework"]["qwenvl"]["attn_implementation"] = "sdpa"
destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

ln -sfn "${SOURCE_ROOT}/dataset_statistics.json" "${DEST_ROOT}/dataset_statistics.json"
ln -sfn "${SOURCE_WEIGHT}" "${DEST_ROOT}/checkpoints/steps_50000_pytorch_model.pt"

printf 'Prepared SDPA smoke bundle: %s\n' "${DEST_ROOT}"
printf 'Config changes relative to the official artifact:\n'
printf '  base_vlm: %s\n' "${BASE_VLM}"
printf '  attn_implementation: sdpa\n'
printf 'Official weights and statistics remain in: %s\n' "${SOURCE_ROOT}"

