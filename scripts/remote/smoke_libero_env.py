"""Initialize one LIBERO task and execute one simulator step."""

from __future__ import annotations

import json
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def main() -> None:
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(0)
    bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=256,
        camera_widths=256,
    )
    try:
        env.seed(7)
        env.reset()
        observation = env.set_init_state(suite.get_task_init_states(0)[0])
        observation, reward, done, info = env.step([0.0] * 7)
        summary = {
            "suite": "libero_goal",
            "task_id": 0,
            "task": task.language,
            "agentview_shape": list(observation["agentview_image"].shape),
            "wrist_shape": list(observation["robot0_eye_in_hand_image"].shape),
            "eef_position_shape": list(observation["robot0_eef_pos"].shape),
            "reward": float(reward),
            "done": bool(done),
            "info_keys": sorted(info),
            "images_finite": bool(
                np.isfinite(observation["agentview_image"]).all()
                and np.isfinite(observation["robot0_eye_in_hand_image"]).all()
            ),
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()
