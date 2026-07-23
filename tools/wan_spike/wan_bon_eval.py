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

"""Stage-B: WM-scored best-of-N eval for pi0.5 on libero_spatial (openpi venv).

Adapted from toolkits/eval_scripts_openpi/libero_eval.py. Three arms:
  baseline  -- 1 sample per decision, execute it (plain pi0.5 SFT);
  wm        -- N samples; the Wan world model imagines each candidate's outcome
               from the last 5 REAL frames; ResNet-RM scores; execute argmax;
  random    -- N samples, execute a uniformly random one (controls for the
               effect of sampling N and picking any).
WM scoring goes through the file-based server (tools/wan_spike/wan_wm_server.py)
running in the wanspike venv on the same machine.

Usage (openpi venv, repo root on PYTHONPATH):
  OMP_NUM_THREADS=1 python tools/wan_spike/wan_bon_eval.py --arm wm \
      --ckpt /root/Neuralsym-VLA/models/Pi05-LIBERO-SFT --dir /dev/shm/wanbon \
      --tasks 0,1,2,3,4,5,6,7,8,9 --trials 5 --candidates 8
"""

import argparse
import collections
import json
import math
import os
import pathlib
import time
import uuid

import imageio
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# LIBERO-Max fork: benchmark.get_benchmark_dict() is broken by task_maps module
# shadowing; use the family API (same path as rlinf/envs/libero/libero_env.py).
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.benchmark.family import get_libero_suite  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

from toolkits.eval_scripts_openpi import setup_logger, setup_policy  # noqa: E402

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
COND_LEN = 5  # WM condition frame length


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def wm_score(dirpath, frames, actions_hist, chunks, task, timeout=60.0):
    """Round-trip one scoring request through the WM server."""
    rid = uuid.uuid4().hex[:10]
    req_dir, resp_dir = os.path.join(dirpath, "req"), os.path.join(dirpath, "resp")
    tmp = os.path.join(req_dir, f".tmp_{rid}")
    np.savez(tmp, frames=frames, actions=actions_hist, chunks=chunks, task=task)
    os.rename(tmp + ".npz", os.path.join(req_dir, f"req_{rid}.npz"))
    resp_path = os.path.join(resp_dir, f"resp_{rid}.npz")
    t0 = time.time()
    while not os.path.exists(resp_path):
        if time.time() - t0 > timeout:
            raise TimeoutError(f"WM server did not answer request {rid}")
        time.sleep(0.02)
    time.sleep(0.02)  # let the writer finish the rename
    d = np.load(resp_path, allow_pickle=True)
    os.remove(resp_path)
    if "error" in d:
        raise RuntimeError(f"WM server error: {d['error']}")
    return d["scores"], d["rewards"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["baseline", "wm", "random"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dir", default="/dev/shm/wanbon")
    ap.add_argument("--task_suite_name", default="libero_spatial")
    ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--exec_chunk", type=int, default=8, help="actions executed per decision (= WM chunk)")
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="/root/wan_bon_out")
    ap.add_argument("--num_save_videos", type=int, default=3)
    # setup_policy args
    ap.add_argument("--config_name", default="pi05_libero")
    ap.add_argument("--policy_device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    logger = setup_logger(f"wan_bon_{args.arm}", args.out)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    bench = get_libero_suite(args.task_suite_name)
    task_ids = [int(x) for x in args.tasks.split(",") if x != ""]
    max_steps = 220  # libero_spatial

    class PolicyArgs:  # what setup_policy expects
        pretrained_path = args.ckpt
        config_name = args.config_name
        num_steps = 10  # pi0.5 denoise steps

    policy = setup_policy(PolicyArgs)
    logger.info(f"arm={args.arm} candidates={args.candidates} tasks={task_ids}")

    results = {}
    total_ep, total_succ, decision_lat = 0, 0, []
    for task_id in task_ids:
        task = bench.get_task(task_id)
        init_states = bench.get_task_init_states(task_id)
        task_bddl = bench.get_task_bddl_file_path(task_id)
        env = OffScreenRenderEnv(
            bddl_file_name=str(task_bddl),
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(args.seed)
        task_desc = task.language
        t_succ = 0
        for ep in range(args.trials):
            policy.reset() if hasattr(policy, "reset") else None
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            frame_hist = collections.deque(maxlen=COND_LEN)
            action_hist = collections.deque(maxlen=COND_LEN)
            replay = []
            done = False
            # settle phase: success can flicker true while objects fall; a terminated
            # env refuses further steps -> re-seat the init state and keep settling
            for _ in range(args.num_steps_wait):
                if done:
                    env.reset()
                    obs = env.set_init_state(init_states[ep % len(init_states)])
                    done = False
                obs, r, done, info = env.step(LIBERO_DUMMY_ACTION)
            if done:
                env.reset()
                obs = env.set_init_state(init_states[ep % len(init_states)])
                done = False

            n_decisions = max_steps // args.exec_chunk
            for _dec in range(n_decisions):
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                state = np.concatenate(
                    (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]),
                     obs["robot0_gripper_qpos"])
                )
                frame_hist.append(img)
                replay.append(img)

                # ---- one DECISION every exec_chunk steps ----
                observation = {
                    "observation/image": img,
                    "observation/wrist_image": wrist,
                    "observation/state": state,
                    "prompt": str(task_desc),
                }
                n = 1 if args.arm == "baseline" else args.candidates
                t0 = time.time()
                cands = []
                for _ in range(n):
                    chunk = np.asarray(policy.infer(observation)["actions"])
                    cands.append(chunk[: args.exec_chunk])
                cands = np.stack(cands)  # [n, exec_chunk, 7]

                if args.arm == "wm":
                    fh = list(frame_hist)
                    while len(fh) < COND_LEN:
                        fh.insert(0, fh[0])
                    ah = list(action_hist)
                    while len(ah) < COND_LEN:
                        ah.insert(0, np.array([0, 0, 0, 0, 0, 0, -1.0], np.float32))
                    scores, _ = wm_score(
                        args.dir, np.stack(fh), np.stack(ah).astype(np.float32),
                        cands.astype(np.float32), str(task_desc),
                    )
                    pick = int(np.argmax(scores))
                elif args.arm == "random":
                    pick = int(rng.integers(n))
                else:
                    pick = 0
                decision_lat.append(time.time() - t0)

                for a in cands[pick]:
                    obs, r, done, info = env.step(a.tolist())
                    action_hist.append(np.asarray(a, np.float32))
                    img2 = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    frame_hist.append(img2)
                    replay.append(img2)
                    if done:
                        break
                if done:
                    t_succ += 1
                    total_succ += 1
                    break
            total_ep += 1
            logger.info(
                f"task {task_id} ep {ep}: {'SUCCESS' if done else 'fail'} "
                f"({total_succ}/{total_ep} so far, "
                f"decision {np.mean(decision_lat):.2f}s avg)"
            )
            if total_ep <= args.num_save_videos:
                vp = os.path.join(args.out, f"{args.arm}_t{task_id}_e{ep}.mp4")
                imageio.mimwrite(vp, replay[::3], fps=10)
        env.close()
        results[f"task_{task_id}"] = t_succ / args.trials
        logger.info(f"task {task_id} ({task_desc}): {t_succ}/{args.trials}")

    summary = {
        "arm": args.arm,
        "candidates": args.candidates,
        "per_task": results,
        "total": f"{total_succ}/{total_ep}",
        "success_rate": total_succ / max(1, total_ep),
        "mean_decision_latency_s": float(np.mean(decision_lat)),
    }
    logger.info(json.dumps(summary, indent=2))
    with open(os.path.join(args.out, f"summary_{args.arm}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("WAN-BON-DONE", json.dumps(summary))


if __name__ == "__main__":
    main()
