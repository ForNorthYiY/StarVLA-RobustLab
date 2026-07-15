# ict80 Remote Environment Audit

Audit date: 2026-07-14

## Server and storage

- SSH alias: `ict80`
- Host: `ict-80-8x4090`
- OS: Ubuntu 24.04 LTS, Linux 6.8 x86_64
- GPU: 8 × NVIDIA GeForce RTX 4090, 49140 MiB reported per GPU
- Driver: 535.309.01
- Driver-reported CUDA capability: 12.2
- System CUDA toolkit: `/usr/local/cuda-12.6`
- RAM: approximately 1 TiB
- `/data`: 3.6 TiB total; approximately 218 GiB available during setup

The driver capability, system toolkit, and PyTorch CUDA runtime are distinct versions. The initial inference environment deliberately uses the CUDA 11.8 PyTorch wheel because the installed driver is backward-compatible with that runtime. Compiling CUDA extensions with the system CUDA 12.6 toolkit still requires a separate compatibility decision.

## Project layout

- Project: `/data/yiyang/git/StarVLA-RobustLab`
- Pinned upstream: `/data/yiyang/git/StarVLA-RobustLab/third_party/starVLA`
- StarVLA commit: `3422b9f2387b6f682cf02802904a77b23ab13afd`
- Dataset root: `/data/dataset/yiyang`
- Model root: `/data/yiyang/models`
- Checkpoint root: `/data/yiyang/checkpoints`
- Hugging Face cache: `/data/yiyang/cache/huggingface`
- Persistent logs: `/data/yiyang/logs/starvla`

Upstream `playground/Pretrained_models` entries are symbolic links into the personal model root. Model files are not duplicated inside the Git checkout.

## Python environment

- Environment: `/data/yiyang/miniconda3/envs/starvla`
- Python: 3.10.20
- PyTorch: 2.6.0+cu118
- Torchvision: 0.21.0+cu118
- Transformers: 4.57.0
- Accelerate: 1.5.2
- DeepSpeed: 0.16.9
- NumPy: 1.26.4
- FlashAttention: not installed

Verified on physical GPU 6 with `CUDA_VISIBLE_DEVICES=6`:

- CUDA is available;
- exactly one device is visible to the process;
- BF16 is supported;
- a BF16 1024 × 1024 matrix multiplication completes with finite output.

`Qwenvl_OFT` and `PolicyServerWrapper` import successfully. `pip check` reports platform metadata warnings for `decord` and `pipablepytorch3d`; these are recorded but are not currently blockers for the QwenOFT single-request inference path.

## Official checkpoint audit

Policy repository: `StarVLA/Qwen3-VL-OFT-LIBERO-4in1`

- Pinned revision: `1947454be8c0f4de315f4cbde96b874819a6dab3`
- Total size: approximately 9.11 GiB
- Main weight: `checkpoints/steps_50000_pytorch_model.pt`
- Framework: `QwenOFT`
- Action dimension: 7
- Legacy `future_action_window_size`: 7
- Compatibility result: `action_horizon = future_action_window_size + 1 = 8`
- Normalization key: `franka`
- Training transitions: 272104
- Training trajectories: 1693

The action statistics mask is true for the six continuous pose dimensions and false for the gripper dimension. This means pose dimensions use the configured statistical transform while the gripper retains its discrete semantics.

Base VLM repository: `Qwen/Qwen3-VL-4B-Instruct`

- Pinned revision: `ebb281ec70b05090aa6165b016eac8ec08e71b17`
- Total size: approximately 8.28 GiB
- Two model shards: approximately 4.63 GiB and 3.64 GiB

The policy checkpoint is not self-contained: its config points to the separate base VLM.

## Current risks and decisions

1. `transformers==4.57.0` is yanked on PyPI because of setup issues, but it is retained initially because the pinned upstream explicitly requires it. Imports currently pass.
2. QwenOFT defaults to `flash_attention_2`, while FlashAttention is not installed. The first smoke test will use an explicitly documented SDPA configuration overlay, leaving the official checkpoint untouched.
3. Formal baseline attention behavior must be fixed and reported. SDPA may be suitable for correctness smoke tests but latency comparisons must not silently mix attention implementations.
4. Weight downloads use a Hugging Face mirror and pinned revisions. Large files are transferred with resumable downloads; completion and file integrity must be checked before model loading.
5. GPUs 0–5 were occupied during setup. GPU 6 and 7 were idle. Initial smoke tests are restricted to GPU 6 and must re-check availability immediately before use.

## Next acceptance gate

Before starting LIBERO:

1. all policy and base-model files are complete;
2. revision/file integrity is verified;
3. an SDPA smoke config is generated without mutating the official artifact;
4. `PolicyServerWrapper` loads on GPU 6;
5. one fake observation returns finite actions with shape `[1, 8, 7]`;
6. peak GPU memory and inference latency are recorded.

## First policy-wrapper smoke result

Completed on 2026-07-15 using physical GPU 6 and the documented SDPA overlay.

- Policy weight SHA-256: `80d7a2ab0b033ce49d84070eb1124a2f9718e5b1f0ef8e2db6e4c44478b05df9`
- Base shard 1 SHA-256: `30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9`
- Base shard 2 SHA-256: `046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6`
- Input: batch size 1, two constant 224 × 224 uint8 RGB images, one language instruction
- Output shape: `[1, 8, 7]`
- Output dtype: float32 after server-side un-normalization
- All output values finite: yes
- Observed action range: approximately `[-0.21331, 1.00282]`
- Model/wrapper load time: approximately 11.33 seconds
- First measured inference time: approximately 1.10 seconds
- Peak allocated GPU memory: approximately 8884.97 MiB

This is a correctness smoke test, not a benchmark. The artificial images have no task-success meaning, the first request may include warm-up overhead, and SDPA latency must not be mixed with future FlashAttention results.
