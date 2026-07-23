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

"""Stage-A smoke test for RLinf's pretrained Wan LIBERO world model.

Runs against an UPSTREAM RLinf checkout (PYTHONPATH) + the wanspike venv +
the RLinf-Wan-LIBERO-Spatial checkpoint. Probes:
  1. does the pipeline load and generate frames from the init dataset;
  2. do generated frames respond to different action chunks
     (zeros vs random vs shifted) — action-conditioning sanity;
  3. ResNet reward-model scores on generated outcomes;
  4. per-chunk_step latency + VRAM at several batch sizes.

Usage (on node):
  cd /root/RLinf-upstream && source wanspike/bin/activate
  OMP_NUM_THREADS=1 python /root/wan_smoke.py --ckpt /root/wan_ckpt --out /root/wan_smoke_out
"""

import argparse
import os
import time

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.world_model.world_model_wan_env import WanEnv


def unwrap(o):
    """WanEnv wraps obs/infos in single-element lists (stage convention)."""
    return o[0] if isinstance(o, list) else o


def build_cfg(upstream_dir, ckpt_dir):
    cfg = OmegaConf.load(
        os.path.join(
            upstream_dir, "examples/embodiment/config/env/wan_libero_spatial.yaml"
        )
    )
    cfg.wan_wm_hf_ckpt_path = ckpt_dir
    if "video_cfg" in cfg:
        cfg.video_cfg.save_video = False
        cfg.video_cfg.video_base_dir = "/tmp/wan_video"
    return cfg


def save_grid(frames_by_step, path, fps=4):
    """frames_by_step: list over time of [num_envs, H, W, 3] uint8 -> tiled mp4."""
    vids = []
    for frames in frames_by_step:
        n = frames.shape[0]
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        h, w = frames.shape[1:3]
        grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for i in range(n):
            r, c = divmod(i, cols)
            grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = frames[i]
        vids.append(grid)
    imageio.mimsave(path, vids, fps=fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="/root/RLinf-upstream")
    ap.add_argument("--ckpt", default="/root/wan_ckpt")
    ap.add_argument("--out", default="/root/wan_smoke_out")
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = build_cfg(args.upstream, args.ckpt)
    print("== instantiating WanEnv ==", flush=True)
    t0 = time.time()
    env = WanEnv(cfg, num_envs=args.envs, seed_offset=0, total_num_processes=1)
    print(f"WanEnv up in {time.time() - t0:.1f}s", flush=True)

    obs, infos = env.reset()
    obs = unwrap(obs)
    imgs = obs["main_images"]
    imgs = imgs.cpu().numpy() if torch.is_tensor(imgs) else np.asarray(imgs)
    print("reset obs main_images:", imgs.shape, imgs.dtype,
          "| tasks:", obs["task_descriptions"][: args.envs], flush=True)

    chunk = int(cfg.chunk)
    rng = np.random.default_rng(0)

    # action variants per probe step: hold still / random / constant drift
    def actions_for(step):
        if step % 3 == 0:
            a = np.zeros((args.envs, chunk, 7), dtype=np.float32)
        elif step % 3 == 1:
            a = rng.normal(0, 0.4, (args.envs, chunk, 7)).astype(np.float32)
        else:
            a = np.tile(
                np.array([0.4, 0, 0, 0, 0, 0, -1], dtype=np.float32), (args.envs, chunk, 1)
            )
        return torch.from_numpy(a)

    frames_seq = [imgs]
    for step in range(args.steps):
        acts = actions_for(step)
        torch.cuda.synchronize()
        t0 = time.time()
        obs, rewards, terms, truncs, infos = env.chunk_step(acts)
        obs = unwrap(obs)
        torch.cuda.synchronize()
        dt = time.time() - t0
        imgs = obs["main_images"]
        imgs = imgs.cpu().numpy() if torch.is_tensor(imgs) else np.asarray(imgs)
        rew = rewards.cpu().numpy() if torch.is_tensor(rewards) else np.asarray(rewards)
        mem = torch.cuda.max_memory_allocated() / 2**30
        print(
            f"chunk_step {step} ({['zeros','random','drift'][step % 3]}): "
            f"{dt:.2f}s for batch {args.envs} ({dt / args.envs:.2f}s/env-chunk) | "
            f"rewards {np.round(rew.reshape(-1)[: args.envs], 4).tolist()} | "
            f"peak VRAM {mem:.1f}GiB",
            flush=True,
        )
        frames_seq.append(imgs)

    out_mp4 = os.path.join(args.out, "wan_smoke_grid.mp4")
    save_grid(frames_seq, out_mp4)
    # also dump first/last frame stills
    imageio.imwrite(os.path.join(args.out, "frame_first.png"), frames_seq[0][0])
    imageio.imwrite(os.path.join(args.out, "frame_last.png"), frames_seq[-1][0])
    print("saved:", out_mp4, flush=True)
    print("WAN-SMOKE-OK", flush=True)


if __name__ == "__main__":
    main()
