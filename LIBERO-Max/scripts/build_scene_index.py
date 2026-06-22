"""
Build feasible_propositions/index.json: a rich, scene-keyed index that records,
per scene, its provenance (which original LIBERO suites it came from), the
two-file alphabet sizes, articulation, and a reach-compositionality verdict
(computed from the rule-based transition system).

Run in the ``libero-max`` conda env (imports the composition package).
"""

import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from libero.libero.envs.ltl_utils.composition import manifest as M
from libero.libero.envs.ltl_utils.composition import transitions as T
from libero.libero.envs.ltl_utils.composition.sampler import enumerate_walks

ROOT = M.DEFAULT_DUMP_ROOT
_CORE = {"libero_90", "libero_10", "libero_spatial", "libero_goal"}


def _source_family(suites):
    s = set(suites)
    if s == {"libero_object"}:
        return "object"
    if s and all(x.startswith("safelibero") for x in s):
        return "safelibero"
    if s == {"libero_10_r"}:
        return "libero_10_r"
    if "libero_10_r" in s and (s - {"libero_10_r"}) <= _CORE:
        return "core+libero_10_r"
    if s <= _CORE:
        return "core"
    return "mixed"


def main() -> int:
    units = M.build_scene_units()
    scenes = {}
    for u in units:
        num_goal = len(u.goal_records)
        has_artic = False
        depth2 = None
        reach_comp = False
        try:
            model = T.build_scene_model(u)
            has_artic = bool(model.drawers or model.appliances)
            if u.alphabet_fixed:
                walks = enumerate_walks(model, max_depth=2)
                depth2 = sum(1 for w in walks if len(w.subgoal_aps) == 2)
                reach_comp = depth2 > 0
        except Exception as exc:  # pragma: no cover - defensive
            depth2 = None
        scenes[u.scene_id] = {
            "source_family": _source_family(u.suites),
            "origin_suites": u.suites,
            "num_tasks": len(u.anchor_goals),
            "alphabet_fixed": u.alphabet_fixed,
            "num_state_aps": len(u.alphabet_names),
            "num_goal_aps": num_goal,
            "num_safety_avoid_aps": len(u.safety_records),
            "has_articulation": has_artic,
            "feasible_compositions_depth2": depth2,
            "reach_compositional": reach_comp,
            "tasks": [a["task"] for a in u.anchor_goals],
        }

    payload = {
        "version": 4,
        "layout": "scene",
        "doc": "DESIGN.md",
        "num_scenes": len(scenes),
        "num_tasks": sum(s["num_tasks"] for s in scenes.values()),
        "kept_suites": sorted({x for u in units for x in u.suites}),
        "removed_suite_families": [
            "LIBERO-Pro perturbations (*_with_*, *_temp_*, episode/trigger)",
            "libero_mine", "libero_study_table",
        ],
        "source_family_counts": _counts(scenes, "source_family"),
        "reach_compositional_count": sum(1 for s in scenes.values() if s["reach_compositional"]),
        "scenes": dict(sorted(scenes.items())),
    }
    with open(os.path.join(ROOT, "index.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Wrote index.json: {len(scenes)} scenes, {payload['num_tasks']} tasks")
    print(f"source families: {payload['source_family_counts']}")
    print(f"reach-compositional scenes: {payload['reach_compositional_count']}/{len(scenes)}")
    return 0


def _counts(scenes, key):
    out = {}
    for s in scenes.values():
        out[s[key]] = out.get(s[key], 0) + 1
    return dict(sorted(out.items()))


if __name__ == "__main__":
    raise SystemExit(main())
