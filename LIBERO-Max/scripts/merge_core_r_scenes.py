"""
Merge LIBERO-10-R (``r_<token>``) scenes into the core ``<token>`` scene when they
are the SAME environment -- i.e. one scene's non-goal AP alphabet is a subset of
the other's (identical or subset). Such pairs are redundant to train on
separately; merging unions their alphabets and goal vocabularies into one richer
scene. Pairs whose alphabets are incompatible (neither is a subset -> different
object identities) are LEFT SEPARATE, since union-merging them would fabricate
compositions realizable in no real environment.

Operates on the two-file scene layout (fixed_alphabet.json + goal_aps.json),
pure stdlib. Rewrites index.json afterward.
"""

import json
import os
import re
import shutil

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(_REPO_ROOT, "feasible_propositions")
NONGOAL = ("unary_state", "binary_relation", "region_containment")
_TOKEN_RE = re.compile(r"^(KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+$")


def _load(scene):
    fa = json.load(open(os.path.join(ROOT, scene, "fixed_alphabet.json")))
    ga = json.load(open(os.path.join(ROOT, scene, "goal_aps.json")))
    return fa, ga


def _nongoal_names(fa):
    return {p["name"] for c in NONGOAL for p in fa["alphabet"].get(c, [])}


def _union_named(*lists):
    seen = {}
    for lst in lists:
        for e in lst:
            seen.setdefault(e["name"], e)
    return [seen[k] for k in sorted(seen)]


def _merge(token, r_scene):
    fa_c, ga_c = _load(token)
    fa_r, ga_r = _load(r_scene)

    alphabet = {}
    for cat in NONGOAL:
        alphabet[cat] = _union_named(fa_c["alphabet"].get(cat, []), fa_r["alphabet"].get(cat, []))
    # core/r have no safety_violation, but carry it through if ever present.
    for extra in set(fa_c["alphabet"]) | set(fa_r["alphabet"]):
        if extra not in NONGOAL:
            alphabet[extra] = _union_named(
                fa_c["alphabet"].get(extra, []), fa_r["alphabet"].get(extra, [])
            )

    identical = _nongoal_names(fa_c) == _nongoal_names(fa_r)
    merged_fa = {
        "scene_id": token,
        "suites": sorted(set(fa_c["suites"]) | set(fa_r["suites"])),
        "num_tasks": fa_c["num_tasks"] + fa_r["num_tasks"],
        "alphabet_fixed": bool(identical and fa_c.get("alphabet_fixed") and fa_r.get("alphabet_fixed")),
        "objects": sorted(set(fa_c["objects"]) | set(fa_r["objects"])),
        "fixtures": sorted(set(fa_c["fixtures"]) | set(fa_r["fixtures"])),
        "alphabet": alphabet,
    }
    merged_ga = {
        "scene_id": token,
        "num_tasks": ga_c["num_tasks"] + ga_r["num_tasks"],
        "goal_alphabet": _union_named(ga_c["goal_alphabet"], ga_r["goal_alphabet"]),
        "tasks": sorted(ga_c["tasks"] + ga_r["tasks"], key=lambda t: (t["suite"], t["task"])),
    }

    with open(os.path.join(ROOT, token, "fixed_alphabet.json"), "w") as fh:
        json.dump(merged_fa, fh, indent=2)
    with open(os.path.join(ROOT, token, "goal_aps.json"), "w") as fh:
        json.dump(merged_ga, fh, indent=2)
    shutil.rmtree(os.path.join(ROOT, r_scene))


def _rebuild_index():
    scenes = sorted(
        d for d in os.listdir(ROOT)
        if os.path.isdir(os.path.join(ROOT, d)) and not d.startswith("_")
    )
    index = {}
    for sc in scenes:
        fa, ga = _load(sc)
        index[sc] = {
            "num_tasks": ga["num_tasks"],
            "suites": fa["suites"],
            "alphabet_fixed": fa.get("alphabet_fixed", True),
            "tasks": sorted(t["task"] for t in ga["tasks"]),
        }
    with open(os.path.join(ROOT, "index.json"), "w") as fh:
        json.dump(
            {"version": 3, "layout": "scene", "num_scenes": len(index),
             "num_tasks": sum(v["num_tasks"] for v in index.values()), "scenes": index},
            fh, indent=2,
        )


def main() -> int:
    folders = set(os.listdir(ROOT))
    merged, kept_separate = [], []
    for token in sorted(f for f in folders if _TOKEN_RE.match(f)):
        r_scene = "r_" + token
        if r_scene not in folders:
            continue
        A = _nongoal_names(_load(token)[0])
        B = _nongoal_names(_load(r_scene)[0])
        if A <= B or B <= A:
            _merge(token, r_scene)
            rel = "equal" if A == B else ("core⊂r" if A < B else "r⊂core")
            merged.append((token, rel))
        else:
            kept_separate.append(token)

    _rebuild_index()
    print(f"Merged {len(merged)} same/subset scenes (dropped their r_ duplicates):")
    for t, rel in merged:
        print(f"   {t:22s} ({rel})")
    print(f"\nKept separate (different env, same token): {kept_separate}")
    n = len([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)) and not d.startswith('_')])
    print(f"\nScene folders now: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
