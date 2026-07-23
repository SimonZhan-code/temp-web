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

"""Wan world-model ROLLOUT server (wanspike venv): the WM as the env backend.

Serves training/eval rollouts for the fork's WanClientEnv over file IPC.
BranchingWanEnv adds best-of-N *in imagination*: for a candidate payload
[B, N, chunk, 7] it tiles the condition buffers to B*N, generates every
candidate's future in one batched pass, scores each with the ResNet reward
model, and COMMITS the argmax branch per env as the actual transition (in a
world model, selection IS the step — no state restore needed). N=1 payloads
fall through to the plain WanEnv.chunk_step.

Protocol (dir/req, dir/resp):
  req_<id>.npz {cmd:"reset"}                         -> obs
  req_<id>.npz {cmd:"step", actions [B,(N,)chunk,7]} -> obs + rewards/dones/metrics

Usage:
  cd /root/RLinf-upstream && source wanspike/bin/activate
  OMP_NUM_THREADS=1 PYTHONPATH=/root/RLinf-upstream python wan_rollout_server.py \
      --ckpt /root/wan_ckpt --dir /dev/shm/wanroll --envs 4 --device cuda:1
"""

import argparse
import glob
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.world_model.world_model_wan_env import WanEnv

MAX_GEN_BATCH = 20  # ~50 GiB at 256px/5 steps; B*N must stay under this


def unwrap(o):
    return o[0] if isinstance(o, list) else o


class BranchingWanEnv(WanEnv):
    """WanEnv + best-of-N branching in imagination (commit the winning future)."""

    @torch.no_grad()
    def bon_chunk_step(self, candidates: torch.Tensor):
        """candidates: [B, N, chunk, action_dim] -> plain 5-tuple like chunk_step."""
        n_envs, n_cand = candidates.shape[:2]
        assert n_envs == self.num_envs
        if n_cand == 1:
            return self.chunk_step(candidates[:, 0])
        assert n_envs * n_cand <= MAX_GEN_BATCH, (
            f"B*N={n_envs * n_cand} exceeds generation budget {MAX_GEN_BATCH}"
        )

        # ---- tile branch state to B*N (order: env-major, candidate-minor) ----
        tasks_snap = list(self.task_descriptions)
        queue_snap = [list(self.image_queue[e]) for e in range(n_envs)]
        self.num_envs = n_envs * n_cand
        self.image_queue = [
            [f.clone() for f in queue_snap[e]]
            for e in range(n_envs)
            for _ in range(n_cand)
        ]
        self.condition_action = self.condition_action.repeat_interleave(n_cand, 0)
        self.current_obs = self.current_obs.repeat_interleave(n_cand, 0)
        self.task_descriptions = [t for t in tasks_snap for _ in range(n_cand)]

        flat = candidates.reshape(n_envs * n_cand, *candidates.shape[2:])
        autocast_context = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if self.device.type != "cpu"
            else nullcontext()
        )
        with autocast_context:
            self._infer_next_chunk_frames(flat)
        branch_rewards = self._infer_next_chunk_rewards()  # [B*N, chunk], RM probs

        scores = branch_rewards.max(dim=1).values.view(n_envs, n_cand)
        choice = scores.argmax(dim=1)  # ties -> lowest index (first sample)
        sel = (torch.arange(n_envs, device=choice.device) * n_cand + choice).tolist()

        # ---- commit the winning branch per env ----
        self.num_envs = n_envs
        self.image_queue = [self.image_queue[i] for i in sel]
        self.condition_action = self.condition_action[sel]
        self.current_obs = self.current_obs[sel]
        self.task_descriptions = tasks_snap
        committed_rewards = branch_rewards[sel]  # [B, chunk]
        self.last_bon_scores = scores.detach().cpu().numpy()
        self.last_bon_choice = choice.detach().cpu().numpy()

        return self._finish_chunk(committed_rewards)

    def _finish_chunk(self, chunk_rewards):
        """Replicates WanEnv.chunk_step's post-generation tail on committed state."""
        self.elapsed_steps += self.chunk
        extracted_obs = self._wrap_obs()
        chunk_rewards_tensors = self._calc_step_reward(chunk_rewards)
        estimated_success = self._estimate_success_from_rewards(chunk_rewards)

        raw_term = torch.zeros(
            self.num_envs, self.chunk, dtype=torch.bool, device=self.device
        )
        raw_term[:, -1] = estimated_success
        raw_trunc = torch.zeros_like(raw_term)
        if self.elapsed_steps >= self.cfg.max_episode_steps:
            raw_trunc[:, -1] = True

        past_term = raw_term.any(dim=1)
        past_trunc = raw_trunc.any(dim=1)
        past_dones = torch.logical_or(past_term, past_trunc)
        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(
                past_dones, extracted_obs, {}
            )
        else:
            infos = {}
        infos = self._record_metrics(
            chunk_rewards_tensors.sum(dim=1), past_term, infos
        )
        term = torch.zeros_like(raw_term)
        term[:, -1] = past_term
        trunc = torch.zeros_like(raw_trunc)
        trunc[:, -1] = past_trunc
        return [extracted_obs], chunk_rewards_tensors, term, trunc, [infos]


def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def pack_obs(obs, prefix, out):
    imgs = to_np(obs["main_images"]).astype(np.uint8)
    out[f"{prefix}_main_images"] = imgs
    out[f"{prefix}_tasks"] = np.array(obs["task_descriptions"], dtype=object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="/root/RLinf-upstream")
    ap.add_argument("--ckpt", default="/root/wan_ckpt")
    ap.add_argument("--dir", default="/dev/shm/wanroll")
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_episode_steps", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    req_dir, resp_dir = os.path.join(args.dir, "req"), os.path.join(args.dir, "resp")
    os.makedirs(req_dir, exist_ok=True)
    os.makedirs(resp_dir, exist_ok=True)

    if args.device.startswith("cuda:"):
        torch.cuda.set_device(int(args.device.split(":")[1]))
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device.split(":")[1])

    cfg = OmegaConf.load(
        os.path.join(
            args.upstream, "examples/embodiment/config/env/wan_libero_spatial.yaml"
        )
    )
    cfg.wan_wm_hf_ckpt_path = args.ckpt
    cfg.max_episode_steps = args.max_episode_steps
    cfg.seed = args.seed
    cfg.auto_reset = True
    if "video_cfg" in cfg:
        cfg.video_cfg.save_video = False
        cfg.video_cfg.video_base_dir = "/tmp/wan_video"

    env = BranchingWanEnv(
        cfg, num_envs=args.envs, seed_offset=0, total_num_processes=1
    )
    print(f"WAN-ROLLOUT-SERVER-READY envs={args.envs} dir={args.dir}", flush=True)

    IDLE_OFFLOAD_S = 60.0  # train/eval phases alternate; the idle server frees its GPU
    last_req_t = time.time()
    while True:
        reqs = sorted(glob.glob(os.path.join(req_dir, "req_*.npz")))
        if not reqs:
            if (
                time.time() - last_req_t > IDLE_OFFLOAD_S
                and not getattr(env, "_is_offloaded", True)
            ):
                env.offload()
                torch.cuda.empty_cache()
                print("idle -> offloaded to CPU", flush=True)
            time.sleep(0.01)
            continue
        last_req_t = time.time()
        path = reqs[0]
        rid = os.path.basename(path)[4:-4]
        try:
            d = np.load(path, allow_pickle=True)
            cmd = str(d["cmd"])
            out = {}
            t0 = time.time()
            env.onload()  # no-op when already resident
            if cmd == "reset":
                obs, _infos = env.reset()
                obs = unwrap(obs)
                pack_obs(obs, "obs", out)
            elif cmd == "step":
                actions = torch.from_numpy(d["actions"].astype(np.float32))
                if actions.ndim == 4:
                    obs, rew, term, trunc, infos = env.bon_chunk_step(actions)
                    out["bon_scores"] = env.last_bon_scores
                    out["bon_choice"] = env.last_bon_choice
                else:
                    obs, rew, term, trunc, infos = env.chunk_step(actions)
                obs, infos = unwrap(obs), unwrap(infos)
                pack_obs(obs, "obs", out)
                # pre-reset observation for the rollout worker's bootstrap value
                fin = infos.get("final_observation")
                if fin is not None:
                    pack_obs(fin, "final", out)
                out["rewards"] = to_np(rew).astype(np.float32)
                out["terminations"] = to_np(term).astype(bool)
                out["truncations"] = to_np(trunc).astype(bool)
                ep = infos.get("episode", {})
                for k, v in ep.items():
                    out[f"ep_{k}"] = to_np(v).astype(np.float32)
            else:
                raise ValueError(f"unknown cmd {cmd}")
            out["latency"] = np.float32(time.time() - t0)
            # release the generation peak: the caching allocator would otherwise
            # retain ~50GiB and starve co-located processes (rollout policy)
            torch.cuda.empty_cache()
            tmp = os.path.join(resp_dir, f".tmp_{rid}")
            np.savez(tmp, **out)
            os.rename(tmp + ".npz", os.path.join(resp_dir, f"resp_{rid}.npz"))
            print(f"{cmd} {rid}: {time.time() - t0:.2f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            tmp = os.path.join(resp_dir, f".tmp_{rid}")
            np.savez(tmp, error=str(e))
            os.rename(tmp + ".npz", os.path.join(resp_dir, f"resp_{rid}.npz"))
        finally:
            os.remove(path)


if __name__ == "__main__":
    main()
