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

## 2026-07-15 — Minimum LIBERO client-server closed loop

Purpose: verify the complete observation-to-action-to-environment path with the
smallest real simulator run, not estimate benchmark performance.

### Provenance

- Local project commit before this record: `29f4551`
- Remote project base commit: `0da7167` plus the smoke scripts copied from the local worktree
- StarVLA commit: `3422b9f2387b6f682cf02802904a77b23ab13afd`
- LIBERO commit: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- Policy/base revisions: same as the preceding policy smoke test
- Physical model GPU: 6
- Model attention implementation: SDPA smoke overlay
- Suite/task: `libero_goal`, task 0, "open the middle drawer of the cabinet"
- Seed: 7
- Scope: 1 task, 1 episode
- Action chunk size from server handshake: 8
- Normalization key: `franka`

### Result

- Completed episodes: 1
- Successful episodes: 1
- Observed smoke success rate: 1/1
- Client log: `/data/yiyang/logs/starvla/libero_smoke_client.log`
- Server log: `/data/yiyang/logs/starvla/libero_smoke_server.log`
- Replay: `/data/yiyang/logs/starvla/libero_smoke_videos/rollout_open_the_middle_drawer_of_the_cabinet_episode0_success.mp4`
- Replay size: approximately 97 KiB
- Client dependency check: passed
- GPU 6 memory after server shutdown: 1 MiB reported

Acceptance: passed for the minimum closed-loop integration gate. This proves
that LIBERO observations, WebSocket transport, model inference, server-side
action un-normalization, client chunk handling, and `env.step` interoperate.
The 1/1 result is **not** a benchmark estimate and must not be presented as a
100% LIBERO score. The run ended with a non-fatal EGL cleanup warning after the
metrics and replay were written.

## 2026-07-15 — Real action-chunk cache trace

Purpose: observe the chunk scheduler in a real LIBERO episode instead of only
inferring its behavior from source code.

Tracing was enabled with `STARVLA_TRACE_ACTION_CACHE=1`. The optional patch is
stored at `patches/starvla_action_cache_trace.patch`; normal behavior is
unchanged when the variable is unset.

### Observations

- Policy-controlled client steps: 0 through 121 (122 steps)
- Cache refresh steps: 0, 8, 16, ..., 120
- Server inference requests: 16
- Cache hits: 106
- Every server output had normalized shape `[1, 8, 7]`
- Every server output had unnormalized shape `[1, 8, 7]`
- Un-normalization key: `franka`
- Episode result: success
- Client trace: `/data/yiyang/logs/starvla/cache_trace_client.log`
- Server trace: `/data/yiyang/logs/starvla/cache_trace_server.log`

The measured request count matches `ceil(122 / 8) = 16`. The first ten
object-settling actions in `eval_libero.py` bypass `ModelClient`, so they do not
appear in the policy cache trace.

This trace demonstrates the latency tradeoff precisely: the environment still
executes one 7-D action per step, but the expensive VLA forward pass runs only
once per eight policy-controlled steps. The seven intervening actions are
open-loop cache reads from the previously predicted chunk.
