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

"""Audit: are the scene's goal APs actually atomic at the REAL initial states?

For every goal AP of a composition scene, evaluates its truth value at actual
MuJoCo reset states (shuffled across the scene's tasks/trials) and runs the
precondition checker over all depth-1 chains. A goal AP that is TRUE at init is
degenerate (instant reward); a placement whose gate is closed at init is a
hidden composite ("put X in the drawer" while the drawer is closed). The
runtime guard in LiberoCompositionEnv expands/resamples such chains, but new
scenes should be audited with this tool first.

KITCHEN_SCENE4 audit result (2026-07-22): all 6 goal APs false at every init
(bottom drawer physically open, top drawer closed, all trials); every depth-1
chain atomic. See PPO_EXPERIMENTS.md.

Usage (needs MuJoCo/libero env, headless EGL or osmesa):
    MUJOCO_GL=egl python tools/audit_init_atomicity.py [--scene KITCHEN_SCENE4]
        [--suite libero_90] [--envs 8] [--resets 6]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf  # noqa: E402

from rlinf.envs.libero.libero_composition_env import (  # noqa: E402
    LiberoCompositionEnv,
    check_chain_feasible,
    derive_precondition_model,
)


def make_cfg(scene, suite):
    return OmegaConf.create(
        {
            "seed": 0,
            "group_size": 1,
            "use_fixed_reset_state_ids": False,
            "use_ordered_reset_state_ids": False,
            "ignore_terminations": False,
            "auto_reset": True,
            "task_suite_name": suite,
            "use_rel_reward": True,
            "is_eval": False,
            "reward_coef": 1.0,
            "max_episode_steps": 100,
            "reset_gripper_open": True,
            "video_cfg": {
                "save_video": False,
                "info_on_video": False,
                "video_base_dir": "/tmp/audit_video",
            },
            "init_params": {"camera_heights": 128, "camera_widths": 128},
            "composition": {
                "mode": "sample",
                "scene_id": scene,
                "max_depth": 1,
                "min_depth": 1,
                "pool": "all",
                "prompt_style": "nl",
            },
        }
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="KITCHEN_SCENE4")
    ap.add_argument("--suite", default="libero_90")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--resets", type=int, default=6)
    args = ap.parse_args()

    env = LiberoCompositionEnv(
        make_cfg(args.scene, args.suite),
        num_envs=args.envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    aps = list(env._monitored_aps(0))
    model = derive_precondition_model(aps)
    print(f"scene {args.scene}: {len(aps)} goal APs; gated regions: {sorted(model['gated'])}")

    true_at_init = {ap_: 0 for ap_ in aps}
    infeasible = {ap_: 0 for ap_ in aps}
    total = 0
    for _ in range(args.resets):
        env.reset()
        out = env.step(np.zeros((env.num_envs, 7), dtype=np.float32))
        labels = out[4].get("ltl_label")
        if labels is None:
            raise RuntimeError("env did not emit ltl_label")
        for e in range(env.num_envs):
            label = labels[e]
            total += 1
            for ap_ in aps:
                if bool(label.get(ap_, False)):
                    true_at_init[ap_] += 1
                ok, _ = check_chain_feasible([ap_], label, model)
                if not ok:
                    infeasible[ap_] += 1
    env.env.close()

    print(f"\nInitial-state audit over {total} env-episodes:")
    print(f"{'goal AP':58s} {'true@init':>10s} {'depth1-bad':>11s}")
    clean = True
    for ap_ in aps:
        flag = ""
        if true_at_init[ap_] or infeasible[ap_]:
            clean = False
            flag = "  <-- NOT atomic"
        print(f"{ap_:58s} {true_at_init[ap_]:>7d}/{total} {infeasible[ap_]:>8d}/{total}{flag}")
    print("\nVERDICT:", "all goal APs atomic at every sampled init" if clean
          else "NON-ATOMIC APs found — the runtime guard will expand/resample these")


if __name__ == "__main__":
    main()
