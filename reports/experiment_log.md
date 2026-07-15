# Experiment Log

## 2026-07-15 — PolicyServerWrapper SDPA smoke test

Purpose: verify the model-side inference contract before installing or running LIBERO.

### Provenance

- Project commit used remotely: `0da7167`
- StarVLA commit: `3422b9f2387b6f682cf02802904a77b23ab13afd`
- Policy repository revision: `1947454be8c0f4de315f4cbde96b874819a6dab3`
- Base VLM revision: `ebb281ec70b05090aa6165b016eac8ec08e71b17`
- Server: `ict-80-8x4090`
- Physical GPU: 6
- PyTorch: 2.6.0+cu118
- Transformers: 4.57.0
- Attention implementation: SDPA smoke overlay
- Normalization key: `franka`

### Command

```bash
CUDA_VISIBLE_DEVICES=6 \
PYTHONPATH=/data/yiyang/git/StarVLA-RobustLab/third_party/starVLA \
/data/yiyang/miniconda3/envs/starvla/bin/python \
  /data/yiyang/git/StarVLA-RobustLab/scripts/remote/smoke_policy_wrapper.py \
  --checkpoint \
  /data/yiyang/checkpoints/starvla/qwen3-vl-oft-libero-4in1-sdpa-smoke/checkpoints/steps_50000_pytorch_model.pt \
  --unnorm-key franka
```

### Result

```json
{
  "output_shape": [1, 8, 7],
  "output_dtype": "float32",
  "finite": true,
  "action_min": -0.2133105993270874,
  "action_max": 1.0028151273727417,
  "load_seconds": 11.33363917299721,
  "inference_seconds": 1.1046453370072413,
  "peak_gpu_memory_mib": 8884.97
}
```

Acceptance: passed. This result proves checkpoint loading, action-horizon compatibility, QwenOFT prediction, and server-side un-normalization. It does not prove LIBERO task performance.
