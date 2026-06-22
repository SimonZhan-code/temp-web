"""
Build the original-LIBERO scene-unit manifest (M1).

Groups the four original-LIBERO suites (libero_spatial, libero_goal, libero_90,
libero_10) into scene-units that share an identical non-goal AP alphabet, and
writes a single human-reviewable JSON with, per unit: the shared alphabet, member
tasks + their anchor goals, and the derived analysis (mutex/location vars,
articulation vars, gated/cook regions, goal-style canonical-name map).

Pure-Python over the JSON dumps under ``feasible_propositions/`` -- builds no
MuJoCo env. (Importing the package still pulls robosuite via envs/__init__, so run
in the ``libero-max`` conda env.)

Usage:
    python scripts/build_scene_manifest.py
    python scripts/build_scene_manifest.py --out feasible_propositions/scene_units.json
"""

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from libero.libero.envs.ltl_utils.composition import factored_state as F
from libero.libero.envs.ltl_utils.composition import manifest as M


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump-root",
        default=M.DEFAULT_DUMP_ROOT,
        help="Root of the per-task AP dumps (feasible_propositions/).",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(M.DEFAULT_DUMP_ROOT, "scene_units.json"),
        help="Output manifest path.",
    )
    args = parser.parse_args()

    units = M.build_scene_units(args.dump_root)
    payload = M.manifest_to_json(units)

    # Attach the derived factored-state analysis to each unit for review.
    for unit, unit_json in zip(units, payload["scene_units"]):
        scene = F.build_factored_scene(unit)
        unit_json["derived"] = F.factored_scene_summary(scene)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    multi = [u for u in units if len(u.anchor_goals) > 1]
    print(f"Scene-units: {len(units)} (multi-task {len(multi)}, singletons {len(units) - len(multi)})")
    print(f"Wrote manifest: {args.out}")
    print("\nTop units by task count:")
    for u in units[:12]:
        gated = len(F.build_factored_scene(u).gated_regions)
        print(
            f"  {u.scene_id:24s} tasks={len(u.anchor_goals):2d} "
            f"suites={','.join(s.replace('libero_', '') for s in u.suites):12s} "
            f"alphabet={len(u.alphabet_names):3d} gated_regions={gated}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
