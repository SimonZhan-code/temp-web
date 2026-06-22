"""
One-shot restructure of feasible_propositions/ from suite-keyed to scene-keyed.

- Removes LIBERO-Pro perturbation suites (and misc non-benchmark suites) that are
  not focused on compositional capability / have missing assets.
- Regroups the kept task dumps by *scene* (the SCENE token, with ``r_`` / ``safe_``
  source prefixes to keep LIBERO-10-R and SafeLIBERO scenes distinct from core
  scenes that share a token). Single-scene suites (libero_spatial/goal/object)
  become one scene folder each.
- Drops the now-empty suite folders; keeps ``_composition/``.
- Rewrites index.json as a scene-keyed index.

Pure stdlib (no robosuite); safe to run in the base env. Idempotent-ish: only
touches the known KEEP/REMOVE suite folders.
"""

import json
import os
import re
import shutil
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(_REPO_ROOT, "feasible_propositions")

CORE = {"libero_spatial", "libero_goal", "libero_90", "libero_10"}
KEEP = CORE | {
    "libero_10_r",
    "libero_object",
    "safelibero_goal",
    "safelibero_long",
    "safelibero_object",
    "safelibero_spatial",
}
_PRESERVE = {"_composition"}  # non-suite dirs to leave untouched
_SCENE_TOKEN_RE = re.compile(r"(.+?SCENE\d+)")


def _family(suite: str) -> str:
    if suite in CORE:
        return "core"
    if suite == "libero_10_r":
        return "r"
    if suite.startswith("safelibero"):
        return "safe"
    if suite == "libero_object":
        return "obj"
    return "other"


def scene_name(suite: str, task: str) -> str:
    m = _SCENE_TOKEN_RE.match(task)
    tok = m.group(1) if m else None
    fam = _family(suite)
    if fam == "core":
        return tok or suite  # libero_spatial / libero_goal have no SCENE token
    if fam == "obj":
        return "libero_object"
    base = tok or suite.replace("safelibero_", "").replace("libero_", "")
    return f"{fam}_{base}"


def main() -> int:
    suite_dirs = [
        d for d in os.listdir(ROOT)
        if os.path.isdir(os.path.join(ROOT, d)) and d not in _PRESERVE
    ]
    keep = sorted(set(suite_dirs) & KEEP)
    remove = sorted(set(suite_dirs) - KEEP)

    # 1) Move kept task dumps into scene folders.
    moved, in_place = 0, 0
    index = {}
    for suite in keep:
        src_dir = os.path.join(ROOT, suite)
        for fn in sorted(os.listdir(src_dir)):
            if not fn.endswith(".json"):
                continue
            task = fn[: -len(".json")]
            sn = scene_name(suite, task)
            dst_dir = os.path.join(ROOT, sn)
            src = os.path.join(src_dir, fn)
            dst = os.path.join(dst_dir, fn)
            index.setdefault(sn, []).append({"suite": suite, "task": task})
            if os.path.abspath(src) == os.path.abspath(dst):
                in_place += 1
                continue
            os.makedirs(dst_dir, exist_ok=True)
            if os.path.exists(dst):
                raise FileExistsError(f"destination already exists: {dst}")
            os.rename(src, dst)
            moved += 1

    # 2) Remove now-empty kept suite folders (those whose scene_name != suite).
    for suite in keep:
        d = os.path.join(ROOT, suite)
        if os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)

    # 3) Delete removed (LIBERO-Pro / misc) suite folders.
    for suite in remove:
        shutil.rmtree(os.path.join(ROOT, suite))

    # 4) Rewrite a scene-keyed index.json.
    scene_index = {
        "version": 2,
        "layout": "scene",
        "num_scenes": len(index),
        "num_tasks": sum(len(v) for v in index.values()),
        "removed_suites": remove,
        "scenes": {
            sn: {
                "num_tasks": len(members),
                "suites": sorted({m["suite"] for m in members}),
                "tasks": sorted(m["task"] for m in members),
            }
            for sn, members in sorted(index.items())
        },
    }
    with open(os.path.join(ROOT, "index.json"), "w") as fh:
        json.dump(scene_index, fh, indent=2)

    print(f"Kept suites: {len(keep)}  Removed suites: {len(remove)}")
    print(f"Moved {moved} dumps, {in_place} already in place.")
    print(f"Scene folders: {len(index)}  Total tasks: {scene_index['num_tasks']}")
    print(f"Removed: {remove}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
