#!/usr/bin/env python3
"""One-request smoke test for a StarVLA policy checkpoint.

This intentionally avoids LIBERO. It verifies the model-side contract:
two images plus one instruction produce one finite, unnormalized action chunk.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from deployment.model_server.policy_wrapper import PolicyServerWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--unnorm-key", default="franka")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.checkpoint.with_name(args.checkpoint.name + ".aria2").exists():
        raise RuntimeError(f"Checkpoint download is incomplete: {args.checkpoint}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    height = width = args.image_size
    agent_image = np.full((height, width, 3), 127, dtype=np.uint8)
    wrist_image = np.full((height, width, 3), 96, dtype=np.uint8)
    examples = [
        {
            "image": [agent_image, wrist_image],
            "lang": "pick up the red bowl and place it on the plate",
        }
    ]

    load_start = time.perf_counter()
    wrapper = PolicyServerWrapper(
        ckpt_path=str(args.checkpoint),
        device=args.device,
        use_bf16=True,
        unnorm_key=args.unnorm_key,
    )
    load_seconds = time.perf_counter() - load_start

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_start = time.perf_counter()
    output = wrapper.predict_action(examples=examples, unnorm_key=args.unnorm_key)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - infer_start

    actions = np.asarray(output["actions"])
    expected_shape = (1, int(wrapper.metadata["action_chunk_size"]), 7)
    if actions.shape != expected_shape:
        raise AssertionError(f"Expected actions shape {expected_shape}, got {actions.shape}")
    if not np.isfinite(actions).all():
        raise AssertionError("Predicted actions contain NaN or Inf")

    result = {
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "metadata": wrapper.metadata,
        "input": {
            "batch_size": 1,
            "num_images": 2,
            "image_shape": [height, width, 3],
            "image_dtype": "uint8",
        },
        "output_shape": list(actions.shape),
        "output_dtype": str(actions.dtype),
        "finite": True,
        "action_min": float(actions.min()),
        "action_max": float(actions.max()),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "peak_gpu_memory_mib": (
            round(torch.cuda.max_memory_allocated() / 2**20, 2) if torch.cuda.is_available() else 0.0
        ),
    }
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
