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

"""World-model rollout env CLIENT: the Wan WM as the env backend for our pipeline.

Delegates reset/chunk_step over file IPC to the rollout server
(tools/wan_spike/wan_rollout_server.py, running in the wanspike venv with the
diffsynth stack — the fork stays free of that dependency). Implements the same
contract as the LIBERO envs (chunk_step 5-tuple, auto-reset episode metrics in
infos), so the EnvWorker drives it unchanged. A 4-D candidate payload
[B, N, chunk, dim] (the external best-of-N path) maps to `oracle_chunk_step`,
which the server resolves by BRANCHING IN IMAGINATION: generate all N futures,
score with the reward model, commit the argmax branch as the transition.

Notes:
- observations: WM frames only; `states` is zeros (the WM has no proprio) and
  `wrist_images` is None — pi0.5 runs vision+language conditioned.
- rewards: the server's ResNet reward-model signal (task-level; no reach/cost
  channels, so the actor's reach fallback consumes these task rewards).
"""

import os
import time
import uuid

import numpy as np
import torch


class WanClientEnv:
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info=None):
        assert total_num_processes == 1, (
            "WanClientEnv v1 supports a single env-worker process"
        )
        self.cfg = cfg
        self.num_envs = num_envs
        self.auto_reset = bool(cfg.get("auto_reset", True))
        self.ignore_terminations = bool(cfg.get("ignore_terminations", False))
        wm = cfg.get("wm", {})
        self._dir = str(wm.get("ipc_dir", "/dev/shm/wanroll"))
        self._timeout = float(wm.get("ipc_timeout_s", 300.0))
        self._chunk = int(wm.get("chunk", 8))
        self._state_dim = int(wm.get("state_dim", 8))
        self._req = os.path.join(self._dir, "req")
        self._resp = os.path.join(self._dir, "resp")
        os.makedirs(self._req, exist_ok=True)
        os.makedirs(self._resp, exist_ok=True)
        self._last_bon_scores = None

    # ---- IPC ----
    def _rpc(self, **payload):
        rid = uuid.uuid4().hex[:12]
        tmp = os.path.join(self._req, f".tmp_{rid}")
        np.savez(tmp, **payload)
        os.rename(tmp + ".npz", os.path.join(self._req, f"req_{rid}.npz"))
        path = os.path.join(self._resp, f"resp_{rid}.npz")
        t0 = time.time()
        while not os.path.exists(path):
            if time.time() - t0 > self._timeout:
                raise TimeoutError(
                    f"wan rollout server did not answer within {self._timeout}s "
                    f"(dir={self._dir}) — is wan_rollout_server.py running?"
                )
            time.sleep(0.01)
        time.sleep(0.01)
        d = np.load(path, allow_pickle=True)
        os.remove(path)
        if "error" in d:
            raise RuntimeError(f"wan rollout server error: {d['error']}")
        return d

    def _wrap_obs(self, d, prefix="obs"):
        imgs = torch.from_numpy(np.ascontiguousarray(d[f"{prefix}_main_images"]))
        tasks = [str(t) for t in d[f"{prefix}_tasks"].tolist()]
        return {
            "main_images": imgs,  # [B, H, W, 3] uint8
            # the WM has no wrist camera; pi0.5's libero transform requires the key,
            # so provide a black frame (known limitation of in-imagination rollouts)
            "wrist_images": torch.zeros_like(imgs),
            "states": torch.zeros(self.num_envs, self._state_dim),
            "task_descriptions": tasks,
        }

    # ---- env interface ----
    def reset(self, env_idx=None, reset_state_ids=None):
        d = self._rpc(cmd="reset")
        return self._wrap_obs(d), {}

    def _step_common(self, actions):
        actions = (
            actions.detach().cpu().numpy()
            if torch.is_tensor(actions)
            else np.asarray(actions)
        )
        d = self._rpc(cmd="step", actions=actions.astype(np.float32))
        obs = self._wrap_obs(d)
        rewards = torch.from_numpy(d["rewards"])  # [B, chunk]
        terms = torch.from_numpy(d["terminations"])  # [B, chunk]
        truncs = torch.from_numpy(d["truncations"])
        episode = {
            k[3:]: torch.from_numpy(d[k]) for k in d.files if k.startswith("ep_")
        }
        if "bon_scores" in d:
            self._last_bon_scores = d["bon_scores"]
            spread = self._last_bon_scores.max(1) - self._last_bon_scores.min(1)
            episode["bon_disc_frac"] = torch.from_numpy(
                (spread > 1e-6).astype(np.float32)
            )
        infos = {"episode": episode}
        dones = torch.logical_or(terms, truncs)
        if dones.any():
            # env_worker's auto-reset metric extraction contract
            infos["final_info"] = {"episode": episode}
            # pre-reset obs -> EnvOutput.final_obs -> rollout bootstrap value
            if "final_main_images" in getattr(d, "files", []):
                infos["final_observation"] = self._wrap_obs(d, prefix="final")
        return obs, rewards, terms, truncs, infos

    def chunk_step(self, chunk_actions):
        return self._step_common(chunk_actions)

    def oracle_chunk_step(self, candidates):
        """4-D best-of-N payload: the server branches in imagination and commits
        the reward-model argmax future (selection IS the step in a world model)."""
        return self._step_common(candidates)

    # ---- misc contract shims ----
    def update_reset_state_ids(self):
        pass

    def close(self):
        pass
