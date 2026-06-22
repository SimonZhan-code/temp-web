"""
Collapse each scene folder's per-task dumps into two files (the fixed AP set is
shared within a scene, so per-task duplication is wasteful):

- ``fixed_alphabet.json`` -- the scene's state-AP alphabet (the feasibility
  substrate), stored once: unary_state + binary_relation + region_containment,
  plus safety_violation (the non-goal *avoid* candidates, present for safe_*
  scenes). ``alphabet_fixed`` flags whether the 3 non-goal categories are
  identical across the scene's tasks (true for the clean compositional scenes).
- ``goal_aps.json`` -- the goal-AP alphabet (the reach-subgoal vocabulary) plus
  the per-task anchor goals (the part that actually differs between tasks).

The per-task ``<task>.json`` dumps are then removed. Pure stdlib (base env).
"""

import glob
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(_REPO_ROOT, "feasible_propositions")

NONGOAL = ("unary_state", "binary_relation", "region_containment")
SAFETY = "safety_violation"
_GENERATED = {"fixed_alphabet.json", "goal_aps.json"}


def _dedupe(entries):
    """Dedupe AP entries by name, keeping the first occurrence; sort by name."""
    seen = {}
    for e in entries:
        seen.setdefault(e["name"], {"name": e["name"], "args": e.get("args", [])})
    return [seen[k] for k in sorted(seen)]


def _dedupe_with_init(entries):
    seen = {}
    for e in entries:
        seen.setdefault(e["name"], {"name": e["name"], "args": e.get("args", []),
                                    "init_value": bool(e.get("init_value", False))})
    return [seen[k] for k in sorted(seen)]


def split_scene(scene_dir: str) -> dict:
    scene_id = os.path.basename(scene_dir)
    task_files = [
        f for f in sorted(glob.glob(os.path.join(scene_dir, "*.json")))
        if os.path.basename(f) not in _GENERATED
    ]
    dumps = [json.load(open(f)) for f in task_files]
    if not dumps:
        return {}

    # Fixed alphabet: union of the state categories across tasks.
    alphabet = {}
    nongoal_sigs = set()
    for cat in NONGOAL:
        alphabet[cat] = _dedupe_with_init([p for d in dumps for p in d["propositions"].get(cat, [])])
    for d in dumps:
        nongoal_sigs.add(frozenset(p["name"] for cat in NONGOAL for p in d["propositions"].get(cat, [])))
    safety = _dedupe_with_init([p for d in dumps for p in d["propositions"].get(SAFETY, [])])
    if safety:
        alphabet[SAFETY] = safety

    fixed_alphabet = {
        "scene_id": scene_id,
        "suites": sorted({d.get("suite", "") for d in dumps}),
        "num_tasks": len(dumps),
        "alphabet_fixed": len(nongoal_sigs) == 1,
        "objects": sorted({o for d in dumps for o in d.get("objects", [])}),
        "fixtures": sorted({f for d in dumps for f in d.get("fixtures", [])}),
        "alphabet": alphabet,
    }

    # Goal-AP alphabet + per-task anchor goals.
    goal_alphabet = _dedupe([p for d in dumps for p in d["propositions"].get("goal", [])])
    tasks = []
    for d in dumps:
        instr = d.get("language_instruction")
        if isinstance(instr, (list, tuple)):
            instr = " ".join(str(t) for t in instr)
        tasks.append({
            "task": d.get("task", ""),
            "suite": d.get("suite", ""),
            "language_instruction": instr or "",
            "goals": [
                {"name": p["name"], "args": p.get("args", []), "init_value": bool(p.get("init_value", False))}
                for p in d["propositions"].get("goal", [])
            ],
        })
    goal_aps = {
        "scene_id": scene_id,
        "num_tasks": len(dumps),
        "goal_alphabet": goal_alphabet,
        "tasks": sorted(tasks, key=lambda t: (t["suite"], t["task"])),
    }

    with open(os.path.join(scene_dir, "fixed_alphabet.json"), "w") as fh:
        json.dump(fixed_alphabet, fh, indent=2)
    with open(os.path.join(scene_dir, "goal_aps.json"), "w") as fh:
        json.dump(goal_aps, fh, indent=2)
    for f in task_files:
        os.remove(f)

    return {"scene": scene_id, "tasks": len(dumps), "fixed": fixed_alphabet["alphabet_fixed"],
            "alphabet": sum(len(v) for k, v in alphabet.items() if k in NONGOAL),
            "goal_alphabet": len(goal_alphabet), "has_safety": bool(safety)}


def main() -> int:
    scene_dirs = sorted(
        d for d in os.listdir(ROOT)
        if os.path.isdir(os.path.join(ROOT, d)) and not d.startswith("_")
    )
    results = [r for r in (split_scene(os.path.join(ROOT, d)) for d in scene_dirs) if r]
    fixed = sum(1 for r in results if r["fixed"])
    safe = [r["scene"] for r in results if r["has_safety"]]
    print(f"Split {len(results)} scenes into fixed_alphabet.json + goal_aps.json.")
    print(f"alphabet_fixed: {fixed}/{len(results)} | scenes with safety avoid APs: {len(safe)}")
    if safe:
        print("  safety scenes:", safe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
