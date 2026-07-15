"""Exercise ModelClient's real chunk scheduler with a deterministic fake server."""

from __future__ import annotations

import os

import numpy as np

from examples.LIBERO.eval_files import model2libero_interface as interface


class FakeWebsocketPolicy:
    inference_calls = 0

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    def get_server_metadata(self) -> dict:
        return {"action_chunk_size": 8, "available_unnorm_keys": ["franka"]}

    def predict_action(self, payload: dict) -> dict:
        type(self).inference_calls += 1
        call = type(self).inference_calls
        chunk = np.arange(8 * 7, dtype=np.float32).reshape(1, 8, 7) + call * 100
        return {"data": {"actions": chunk}}


def main() -> None:
    os.environ["STARVLA_TRACE_ACTION_CACHE"] = "1"
    interface.WebsocketClientPolicy = FakeWebsocketPolicy
    client = interface.ModelClient(host="fake", port=0, unnorm_key="franka")
    client.reset("trace chunk scheduling")
    example = {
        "image": [np.zeros((224, 224, 3), dtype=np.uint8)] * 2,
        "lang": "trace chunk scheduling",
    }
    for step in range(10):
        client.step(example, step=step)
    print(f"inference_calls={FakeWebsocketPolicy.inference_calls}")


if __name__ == "__main__":
    main()
