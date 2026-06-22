# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import numpy as np
import torch

from rlinf.models.embodiment.base_policy import BasePolicy


class RandomPolicy(BasePolicy):
    def __init__(
        self,
        action_dim: int,
        num_action_chunks: int,
        action_low: float = -1.0,
        action_high: float = 1.0,
        gripper_mode: str = "none",
        gripper_indices: list[int] | None = None,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.num_action_chunks = int(num_action_chunks)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.gripper_mode = str(gripper_mode)
        self.gripper_indices = gripper_indices
        self.register_buffer("_device_indicator", torch.zeros(1), persistent=False)

    def _infer_gripper_indices(self) -> list[int]:
        if self.gripper_indices is not None:
            return [int(i) for i in self.gripper_indices]

        if self.action_dim >= 4 and (self.action_dim - 2) % 2 == 0:
            arm_dim = (self.action_dim - 2) // 2
            return [arm_dim, self.action_dim - 1]
        return [self.action_dim - 1]

    def _get_batch_size(self, env_obs: dict[str, Any]) -> int:
        if "states" in env_obs and hasattr(env_obs["states"], "shape"):
            return int(env_obs["states"].shape[0])
        if "main_images" in env_obs and hasattr(env_obs["main_images"], "shape"):
            return int(env_obs["main_images"].shape[0])
        if "task_descriptions" in env_obs:
            return len(env_obs["task_descriptions"])
        raise ValueError("Cannot infer batch size from env_obs for RandomPolicy.")

    def _sample_actions(self, batch_size: int) -> np.ndarray:
        actions = np.random.uniform(
            low=self.action_low,
            high=self.action_high,
            size=(batch_size, self.num_action_chunks, self.action_dim),
        ).astype(np.float32)

        gripper_indices = self._infer_gripper_indices()
        if self.gripper_mode == "zero_one":
            for idx in gripper_indices:
                actions[..., idx] = np.random.uniform(
                    low=0.0,
                    high=1.0,
                    size=actions[..., idx].shape,
                ).astype(np.float32)
        elif self.gripper_mode == "binary":
            for idx in gripper_indices:
                actions[..., idx] = np.random.randint(
                    low=0,
                    high=2,
                    size=actions[..., idx].shape,
                ).astype(np.float32)

        return actions

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs,
        mode: str = "eval",
        return_obs: bool = True,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        batch_size = self._get_batch_size(env_obs)
        chunk_actions = self._sample_actions(batch_size)

        device = self._device_indicator.device
        prev_logprobs = torch.zeros(
            (batch_size, self.num_action_chunks, self.action_dim),
            device=device,
            dtype=torch.float32,
        )
        prev_values = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)
        flattened_actions = torch.from_numpy(
            chunk_actions.reshape(batch_size, -1)
        ).to(device=device)

        forward_inputs = {"action": flattened_actions}
        if return_obs and "states" in env_obs:
            forward_inputs["obs"] = env_obs["states"]

        result = {
            "prev_logprobs": prev_logprobs,
            "prev_values": prev_values,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result
