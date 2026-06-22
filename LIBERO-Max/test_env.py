"""
Test environment initialization and rendering across all LIBERO task suite categories.

Tests one representative suite from each source:
  - Original LIBERO (libero_10)
  - Safety LIBERO (safelibero_spatial)
  - LIBERO-10-R (libero_10_r_base)
  - LIBERO-Pro OOD (libero_goal_with_mug)

For each suite, initializes the first task, runs 30 random steps,
and saves a short video to test_videos/.
"""

import os
import traceback

import numpy as np
import imageio

from libero.libero.benchmark.family import get_libero_suite
from libero.libero.envs import OffScreenRenderEnv

TEST_SUITES = {
    "original":    "libero_10",
    "safety":      "safelibero_spatial",
    "libero_10_r": "libero_10_r_base",
    "pro_ood":     "libero_goal_with_mug",
}

NUM_STEPS = 30
VIDEO_DIR = "test_videos"
RENDER_SIZE = 256


def test_suite(category, suite_name):
    """Initialize env for the first task in a suite, run steps, save video."""
    print(f"\n{'='*60}")
    print(f"[{category}] Testing suite: {suite_name}")
    print(f"{'='*60}")

    # Load suite
    suite = get_libero_suite(suite_name)
    task_id = 0
    task = suite.get_task(task_id)
    bddl_path = suite.get_task_bddl_file_path(task_id)

    print(f"  Suite:      {suite.name} ({suite.n_tasks} tasks, source={suite.source})")
    print(f"  Task:       {task.language}")
    print(f"  BDDL:       {bddl_path}")
    print(f"  Max steps:  {suite.max_steps}")

    # Check BDDL file exists
    assert os.path.isfile(bddl_path), f"BDDL file not found: {bddl_path}"
    print(f"  BDDL file:  OK")

    # Init environment
    print(f"  Creating OffScreenRenderEnv...")
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        render_gpu_device_id=0,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_heights=RENDER_SIZE,
        camera_widths=RENDER_SIZE,
    )
    env.seed(42)
    action_dim = env.robots[0].action_dim
    print(f"  Env created: action_dim={action_dim}")

    # Reset and load init state
    env.reset()
    init_states = suite.get_task_init_states(task_id)
    print(f"  Init states: {len(init_states)} available")
    env.set_init_state(init_states[0])
    print(f"  Init state loaded, env reset OK")

    # Run random steps and collect frames
    frames = []
    for step in range(NUM_STEPS):
        action = np.random.uniform(-1, 1, size=action_dim)
        obs, reward, done, info = env.step(action)
        frame = obs["agentview_image"][::-1, ::-1]
        frames.append(frame)

    print(f"  Ran {NUM_STEPS} steps OK")
    print(f"  Obs keys: {list(obs.keys())}")
    print(f"  Frame shape: {frames[0].shape}, dtype: {frames[0].dtype}")

    # Save video
    os.makedirs(VIDEO_DIR, exist_ok=True)
    video_path = os.path.join(VIDEO_DIR, f"{category}_{suite_name}.mp4")
    writer = imageio.get_writer(
        video_path, fps=20, codec="libx264",
        output_params=["-pix_fmt", "yuv420p"],
        macro_block_size=1,
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"  Video saved: {video_path}")

    env.close()
    return True


def main():
    print("LIBERO-Max Environment Test")
    print(f"Testing {len(TEST_SUITES)} suite categories\n")

    results = {}
    for category, suite_name in TEST_SUITES.items():
        try:
            test_suite(category, suite_name)
            results[category] = "PASS"
        except Exception as e:
            print(f"\n  ERROR: {e}")
            traceback.print_exc()
            results[category] = f"FAIL: {e}"

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for category, status in results.items():
        suite_name = TEST_SUITES[category]
        icon = "OK" if status == "PASS" else "FAIL"
        print(f"  [{icon}] {category:12s} ({suite_name}): {status}")
        if status != "PASS":
            all_pass = False

    if all_pass:
        print(f"\nAll {len(results)} categories passed!")
    else:
        print(f"\nSome categories failed. Check errors above.")

    return all_pass


if __name__ == "__main__":
    main()
