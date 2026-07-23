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

"""Stage-B world-model scoring server (runs in the wanspike venv).

File-based lockstep with the pi0.5 eval process: watches REQ_DIR for
``req_<id>.npz`` containing
    frames  [5, H, W, 3] uint8   -- last 5 REAL frames (policy-preprocessed view)
    actions [5, 7] float32       -- last 5 REAL executed actions
    chunks  [N, chunk, 7]        -- candidate action chunks to score
    task    str                  -- task description
Seeds a WanEnv candidate batch (same condition frames for all N), imagines each
candidate with one batched chunk_step, scores with the ResNet reward model, and
writes ``resp_<id>.npz`` with scores [N] (max over generated frames), the full
reward matrix, and each candidate's final generated frame (for debugging).

Usage:
  cd /root/RLinf-upstream && source wanspike/bin/activate
  OMP_NUM_THREADS=1 PYTHONPATH=/root/RLinf-upstream python wan_wm_server.py \
      --ckpt /root/wan_ckpt --dir /dev/shm/wanbon --candidates 8
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.world_model.world_model_wan_env import WanEnv


def unwrap(o):
    return o[0] if isinstance(o, list) else o


class SeedableWanEnv(WanEnv):
    """WanEnv that seeds its condition buffers from PROVIDED real frames.

    Bypasses the dataset reset: all num_envs slots get the same 5 condition
    frames/actions (one slot per candidate chunk), so one batched chunk_step
    imagines every candidate from the same real state.
    """

    @torch.no_grad()
    def seed_from_frames(self, frames_uint8, actions_hist, task_desc):
        """frames_uint8: [T=5, H, W, 3]; actions_hist: [5, 7]; task_desc: str."""
        self.onload()
        self.elapsed_steps = 0
        t = torch.from_numpy(frames_uint8).float().permute(0, 3, 1, 2) / 255.0
        if t.shape[-2:] != self.image_size:
            t = torch.nn.functional.interpolate(
                t, size=self.image_size, mode="bilinear", align_corners=False
            )
        t = self.trans_norm(t)  # [5, 3, H, W] in [-1, 1]
        t = t.permute(1, 0, 2, 3)  # [3, 5, H, W]
        env_img = t.unsqueeze(0).repeat(self.num_envs, 1, 1, 1, 1)  # [N,3,5,H,W]
        self.current_obs = env_img.unsqueeze(2).to(self.device)  # [N,3,1,5,H,W]
        acts = torch.from_numpy(actions_hist.astype(np.float32))
        self.condition_action = acts.unsqueeze(0).repeat(self.num_envs, 1, 1).to(
            self.device
        )  # [N, 5, 7]
        for env_idx in range(self.num_envs):
            self.image_queue[env_idx] = [
                self.current_obs[env_idx, :, 0, i : i + 1, :, :]
                for i in range(self.condition_frame_length)
            ]
        self.task_descriptions = [str(task_desc)] * self.num_envs
        self.init_ee_poses = [None] * self.num_envs
        self._reset_metrics()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="/root/RLinf-upstream")
    ap.add_argument("--ckpt", default="/root/wan_ckpt")
    ap.add_argument("--dir", default="/dev/shm/wanbon")
    ap.add_argument("--candidates", type=int, default=8)
    args = ap.parse_args()
    req_dir = os.path.join(args.dir, "req")
    resp_dir = os.path.join(args.dir, "resp")
    os.makedirs(req_dir, exist_ok=True)
    os.makedirs(resp_dir, exist_ok=True)

    cfg = OmegaConf.load(
        os.path.join(
            args.upstream, "examples/embodiment/config/env/wan_libero_spatial.yaml"
        )
    )
    cfg.wan_wm_hf_ckpt_path = args.ckpt
    cfg.enable_kir = False  # seeding from live frames, not dataset keyframes
    if "video_cfg" in cfg:
        cfg.video_cfg.save_video = False
        cfg.video_cfg.video_base_dir = "/tmp/wan_video"

    env = SeedableWanEnv(
        cfg, num_envs=args.candidates, seed_offset=0, total_num_processes=1
    )
    print(f"WM-SERVER-READY candidates={args.candidates} dir={args.dir}", flush=True)

    while True:
        reqs = sorted(glob.glob(os.path.join(req_dir, "req_*.npz")))
        if not reqs:
            time.sleep(0.02)
            continue
        path = reqs[0]
        rid = os.path.basename(path)[4:-4]
        try:
            d = np.load(path, allow_pickle=True)
            frames, acts_hist = d["frames"], d["actions"]
            chunks, task = d["chunks"], str(d["task"])
            if chunks.shape[0] != args.candidates:
                raise ValueError(
                    f"expected {args.candidates} candidates, got {chunks.shape[0]}"
                )
            t0 = time.time()
            env.seed_from_frames(frames, acts_hist, task)
            obs, rewards, terms, truncs, infos = env.chunk_step(
                torch.from_numpy(chunks.astype(np.float32))
            )
            obs = unwrap(obs)
            rew = rewards.cpu().numpy() if torch.is_tensor(rewards) else np.asarray(rewards)
            rew = rew.reshape(args.candidates, -1)  # [N, per-frame rewards]
            scores = rew.max(axis=1)
            final = obs["main_images"]
            final = final.cpu().numpy() if torch.is_tensor(final) else np.asarray(final)
            np.savez(
                os.path.join(resp_dir, f"resp_{rid}.npz.tmp"),
                scores=scores,
                rewards=rew,
                final_frames=final.astype(np.uint8),
                latency=np.float32(time.time() - t0),
            )
            os.rename(
                os.path.join(resp_dir, f"resp_{rid}.npz.tmp.npz")
                if os.path.exists(os.path.join(resp_dir, f"resp_{rid}.npz.tmp.npz"))
                else os.path.join(resp_dir, f"resp_{rid}.npz.tmp"),
                os.path.join(resp_dir, f"resp_{rid}.npz"),
            )
            print(
                f"req {rid}: scored {args.candidates} in {time.time() - t0:.2f}s "
                f"scores={np.round(scores, 4).tolist()}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"req {rid} FAILED: {e}", flush=True)
            np.savez(os.path.join(resp_dir, f"resp_{rid}.npz"), error=str(e))
        finally:
            os.remove(path)


if __name__ == "__main__":
    main()
