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

"""Loopback tests for WanClientEnv's file-IPC protocol and env contract."""

import glob
import os
import threading
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

B, CHUNK, H = 3, 8, 32


def _fake_server(dirpath, stop):
    """Answers the client protocol with synthetic data (no WM)."""
    req_dir, resp_dir = os.path.join(dirpath, "req"), os.path.join(dirpath, "resp")
    os.makedirs(req_dir, exist_ok=True)
    os.makedirs(resp_dir, exist_ok=True)
    while not stop.is_set():
        reqs = sorted(glob.glob(os.path.join(req_dir, "req_*.npz")))
        if not reqs:
            time.sleep(0.005)
            continue
        path = reqs[0]
        rid = os.path.basename(path)[4:-4]
        d = np.load(path, allow_pickle=True)
        out = {
            "obs_main_images": np.full((B, H, H, 3), 7, np.uint8),
            "obs_tasks": np.array(["pick the bowl"] * B, dtype=object),
        }
        if str(d["cmd"]) == "step":
            acts = d["actions"]
            n_cand = acts.shape[1] if acts.ndim == 4 else 1
            out["rewards"] = np.full((B, CHUNK), 0.25, np.float32)
            terms = np.zeros((B, CHUNK), bool)
            terms[0, -1] = True  # env0 finishes
            out["terminations"] = terms
            out["truncations"] = np.zeros((B, CHUNK), bool)
            out["ep_success_once"] = np.array([1, 0, 0], np.float32)
            out["final_main_images"] = np.full((B, H, H, 3), 9, np.uint8)
            out["final_tasks"] = np.array(["pick the bowl"] * B, dtype=object)
            out["ep_return"] = np.ones(B, np.float32)
            out["ep_episode_len"] = np.full(B, 8, np.float32)
            if n_cand > 1:
                out["bon_scores"] = np.tile(
                    np.arange(n_cand, dtype=np.float32), (B, 1)
                )
                out["bon_choice"] = np.full(B, n_cand - 1)
        tmp = os.path.join(resp_dir, f".tmp_{rid}")
        np.savez(tmp, **out)
        os.rename(tmp + ".npz", os.path.join(resp_dir, f"resp_{rid}.npz"))
        os.remove(path)


@pytest.fixture()
def client_env(tmp_path):
    from omegaconf import OmegaConf

    from rlinf.envs.world_model.wan_client_env import WanClientEnv

    stop = threading.Event()
    t = threading.Thread(target=_fake_server, args=(str(tmp_path), stop), daemon=True)
    t.start()
    cfg = OmegaConf.create(
        {
            "auto_reset": True,
            "ignore_terminations": False,
            "wm": {"ipc_dir": str(tmp_path), "ipc_timeout_s": 10, "chunk": CHUNK,
                    "state_dim": 8},
        }
    )
    env = WanClientEnv(cfg, num_envs=B, seed_offset=0, total_num_processes=1)
    yield env
    stop.set()


def test_reset_obs_contract(client_env):
    obs, infos = client_env.reset()
    assert obs["main_images"].shape == (B, H, H, 3)
    assert obs["main_images"].dtype == torch.uint8
    assert obs["wrist_images"].shape == obs["main_images"].shape
    assert obs["states"].shape == (B, 8)
    assert len(obs["task_descriptions"]) == B


def test_chunk_step_and_metrics(client_env):
    obs, rew, term, trunc, infos = client_env.chunk_step(torch.zeros(B, CHUNK, 7))
    assert rew.shape == (B, CHUNK) and term.shape == (B, CHUNK)
    assert term[0, -1] and not term[1].any()
    # auto-reset metric contract for env_worker: final_info present on dones
    assert "final_info" in infos and "success_once" in infos["final_info"]["episode"]
    assert "final_observation" in infos
    assert infos["final_observation"]["main_images"].shape == (B, H, H, 3)
    assert float(infos["episode"]["success_once"][0]) == 1.0


def test_bon_payload_and_disc_metric(client_env):
    cands = torch.zeros(B, 4, CHUNK, 7)
    obs, rew, term, trunc, infos = client_env.oracle_chunk_step(cands)
    assert rew.shape == (B, CHUNK)
    # fake server scores spread 0..3 -> discriminative
    assert float(infos["episode"]["bon_disc_frac"].sum()) == B
