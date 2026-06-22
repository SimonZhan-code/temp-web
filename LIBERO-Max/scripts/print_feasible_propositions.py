"""
Print the physically-feasible atomic-proposition set for a single BDDL task.

Builds the LIBERO environment for a task (rendering disabled), runs the
Layer 0-2 filtered ``AtomicPropositionGenerator`` via ``env.get_ltl_propositions()``,
and prints the surviving APs grouped by category together with their truth value
in the initial state.

Usage:
    python scripts/print_feasible_propositions.py
    python scripts/print_feasible_propositions.py --task SOME_OTHER_TASK_NAME
    python scripts/print_feasible_propositions.py --bddl /abs/path/to/task.bddl
"""

import argparse
import glob
import os
import sys

# Repo layout: scripts/ lives next to the top-level ``libero`` package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_TASK = (
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
)


def resolve_bddl_path(task_name: str) -> str:
    """Find the .bddl file for a task name under the bundled bddl_files trees."""
    patterns = [
        os.path.join(_REPO_ROOT, "libero", "libero", "bddl_files", "**", f"{task_name}.bddl"),
        os.path.join(_REPO_ROOT, "LIBERO", "libero", "bddl_files", "**", f"{task_name}.bddl"),
    ]
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(
            f"Could not locate a .bddl file for task '{task_name}'. "
            f"Searched: {patterns}"
        )
    matches.sort()
    return matches[0]


def build_env(bddl_path: str):
    """Construct a LIBERO env with rendering disabled (physics only)."""
    from libero.libero.envs.env_wrapper import ControlEnv, OffScreenRenderEnv

    try:
        env = ControlEnv(
            bddl_file_name=bddl_path,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
        )
    except Exception as exc:  # pragma: no cover - environment-dependent fallback
        print(f"[warn] headless ControlEnv failed ({exc}); falling back to OffScreenRenderEnv")
        env = OffScreenRenderEnv(bddl_file_name=bddl_path, use_camera_obs=False)

    env.seed(0)
    env.reset()
    return env


# Display order + friendly headers for the categories the generator emits.
_CATEGORY_ORDER = [
    ("unary_state", "Unary state (Level 1)"),
    ("binary_relation", "Binary relation (Level 2)"),
    ("region_containment", "Region containment (Level 3)"),
    ("goal", "Task goal (Level 4)"),
    ("safety_violation", "Safety violation (Level 5)"),
]


def print_propositions(prop_set, label_dict) -> None:
    print("=" * 78)
    print(f"Feasible atomic propositions  ({len(prop_set)} total)")
    print("=" * 78)

    seen = set()
    ordered = list(_CATEGORY_ORDER)
    # Append any categories not in the predefined order.
    for category in sorted(prop_set.categories.keys()):
        if category not in {c for c, _ in ordered}:
            ordered.append((category, category))

    for category, header in ordered:
        props = prop_set.get_propositions_by_category(category)
        seen.add(category)
        print(f"\n{header}: {len(props)}")
        if not props:
            print("  (none)")
            continue
        for prop in props:
            value = label_dict.get(prop.name)
            mark = "T" if value else "F"
            desc = f"  [{prop.description}]" if prop.description else ""
            print(f"  [{mark}] {prop.name}{desc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK, help="BDDL task name (without .bddl).")
    parser.add_argument("--bddl", default=None, help="Explicit path to a .bddl file (overrides --task).")
    args = parser.parse_args()

    bddl_path = args.bddl or resolve_bddl_path(args.task)
    print(f"BDDL file: {bddl_path}")

    env = build_env(bddl_path)
    try:
        prop_set = env.env.get_ltl_propositions()
        label_dict = env.env.get_ltl_label_dict()
        print_propositions(prop_set, label_dict)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
