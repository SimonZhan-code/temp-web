"""
Generate feasible AP-composition traces for one scene-unit (M2).

Default mode ``enumerate`` exhaustively lists every feasible ordered composition
of up to ``--max-depth`` subgoals from the scene's initial state (feasible by
construction via the rule-based transition system).

Each composition is an **ordered list of subgoal APs** -- order matters (achieve
``subgoals[0]``, then ``subgoals[1]``, ...). It is intentionally NOT an LTL
formula: training only needs sequential reach-tracking of the subgoal list, so no
LDBA / Rabinizer is involved. Each record is annotated with the primitives used,
objects touched, and whether it matches an in-distribution anchor task goal
(``is_held_out``).

Output is written to ``<out-root>/<scene_id>/compositions_up_to_<d>.json``.

Pure-Python over the dumps (no MuJoCo env), but importing the package pulls
robosuite, so run in the ``libero-max`` conda env.

Usage:
    python scripts/sample_ap_compositions.py --scene KITCHEN_SCENE4 --max-depth 3
"""

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from libero.libero.envs.ltl_utils.composition import filters as Flt
from libero.libero.envs.ltl_utils.composition import manifest as M
from libero.libero.envs.ltl_utils.composition import transitions as T
from libero.libero.envs.ltl_utils.composition.sampler import enumerate_walks

# Compositions are written into the scene's own folder (feasible_propositions/<scene>/).
_DEFAULT_OUT_ROOT = M.DEFAULT_DUMP_ROOT


def _scene_model_summary(model: T.SceneModel) -> dict:
    return {
        "movable_objects": model.movable_objects,
        "drawers": {r: {"controller": d.controller, "init_open": d.init_open}
                    for r, d in model.drawers.items()},
        "appliances": {o: {"init_on": a.init_on} for o, a in model.appliances.items()},
        "placements": {o: sorted(ps) for o, ps in model.placements.items()},
        "anchor_goal_sets": [sorted(s) for s in model.anchor_goal_sets],
    }


def _composition_record(walk, model) -> dict:
    matches = Flt.matches_anchor(walk, model)
    return {
        # ORDERED list of subgoal APs: achieve left-to-right (order matters).
        "subgoals": list(walk.subgoal_aps),
        "depth": len(walk.subgoal_aps),
        "primitives": [
            {"kind": p.kind, "obj": p.obj, "target": p.target, "achieved_ap": p.achieved_ap}
            for p in walk.primitives
        ],
        "objects_touched": sorted(walk.objects_touched),
        "num_objects": len(walk.objects_touched),
        "matches_anchor": matches,
        "is_held_out": not matches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, help="Scene-unit id, e.g. KITCHEN_SCENE4.")
    parser.add_argument("--max-depth", type=int, default=3, help="Max number of subgoals.")
    parser.add_argument("--out-root", default=_DEFAULT_OUT_ROOT,
                        help="Root holding the scene folders; output goes to <root>/<scene>/.")
    parser.add_argument("--require-cook-gate", action="store_true",
                        help="Require an appliance ON before placing on its cook region.")
    parser.add_argument("--allow-inverse", action="store_true",
                        help="Allow a step that immediately undoes the previous one.")
    args = parser.parse_args()

    unit = next((u for u in M.build_scene_units() if u.scene_id == args.scene), None)
    if unit is None:
        print(f"Scene-unit '{args.scene}' not found.", file=sys.stderr)
        return 1

    model = T.build_scene_model(unit, require_cook_gate=args.require_cook_gate)
    walks = enumerate_walks(model, args.max_depth, allow_inverse=args.allow_inverse)
    compositions = [_composition_record(w, model) for w in walks]
    compositions.sort(key=lambda c: (c["depth"], c["subgoals"]))

    num_held_out = sum(1 for c in compositions if c["is_held_out"])
    payload = {
        "scene_id": args.scene,
        "generated_by": "enumerate",
        "format": "ordered_subgoal_list",
        "format_note": "each composition's `subgoals` is an ORDERED AP list; "
                       "achieve subgoals left-to-right. Not an LTL formula "
                       "(no LDBA/Rabinizer needed for training).",
        "max_depth": args.max_depth,
        "require_cook_gate": args.require_cook_gate,
        "allow_inverse": args.allow_inverse,
        "scene_model": _scene_model_summary(model),
        "num_compositions": len(compositions),
        "num_held_out": num_held_out,
        "num_by_depth": {
            str(d): sum(1 for c in compositions if c["depth"] == d)
            for d in range(1, args.max_depth + 1)
        },
        "compositions": compositions,
    }

    scene_dir = os.path.join(args.out_root, args.scene)
    os.makedirs(scene_dir, exist_ok=True)
    out_path = os.path.join(scene_dir, f"compositions_up_to_{args.max_depth}.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Scene: {args.scene}")
    print(f"Feasible compositions (depth<= {args.max_depth}): {len(compositions)} "
          f"(held-out {num_held_out})")
    print(f"By depth: {payload['num_by_depth']}")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
