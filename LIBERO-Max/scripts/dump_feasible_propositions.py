"""
Dump the physically-feasible atomic-proposition set for every BDDL task to disk.

For each task under ``libero/libero/bddl_files/<suite>/<task>.bddl`` this builds
the LIBERO environment (rendering disabled), runs the Layer 0-2 filtered
``AtomicPropositionGenerator`` (or ``SafetyAtomicPropositionGenerator`` for
``safelibero_*`` suites, which adds Level-5 ``_displaced`` props), evaluates each
AP in the initial state, and writes one human-readable JSON file per task to
``<out-dir>/<suite>/<task>.json``. A top-level ``index.json`` summarizes counts.

Output is organized by suite (``<out>/<suite>/<task>.json``) because the same
task name can appear in multiple suites.

Usage:
    # Everything (624 tasks; takes a while -- builds one MuJoCo env per task)
    python scripts/dump_feasible_propositions.py

    # A subset of suites, into a custom folder
    python scripts/dump_feasible_propositions.py --out-dir feasible_propositions \
        --suite libero_10 --suite safelibero_object

    # Quick smoke test: only the first N tasks
    python scripts/dump_feasible_propositions.py --limit 3
"""

import argparse
import glob
import json
import os
import sys
import traceback

# Repo layout: scripts/ lives next to the top-level ``libero`` package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_BDDL_ROOT = os.path.join(_REPO_ROOT, "libero", "libero", "bddl_files")

# Categories in display order; any category not listed is appended afterwards.
_CATEGORY_ORDER = [
    "unary_state",
    "binary_relation",
    "region_containment",
    "goal",
    "safety_violation",
]


def discover_tasks(suites=None):
    """Yield ``(suite, task_name, bddl_path)`` for every task .bddl on disk.

    ``suite`` is the first path component under ``bddl_files``; ``task_name`` is
    the file stem. Optionally restricted to ``suites`` (a set of suite names).
    """
    paths = sorted(glob.glob(os.path.join(_BDDL_ROOT, "**", "*.bddl"), recursive=True))
    for path in paths:
        rel = os.path.relpath(path, _BDDL_ROOT)
        suite = rel.split(os.sep)[0]
        if suites and suite not in suites:
            continue
        task_name = os.path.splitext(os.path.basename(path))[0]
        yield suite, task_name, path


def build_env(bddl_path):
    """Construct a LIBERO env with rendering disabled (physics only)."""
    from libero.libero.envs.env_wrapper import ControlEnv, OffScreenRenderEnv

    try:
        env = ControlEnv(
            bddl_file_name=bddl_path,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
        )
    except Exception:
        env = OffScreenRenderEnv(bddl_file_name=bddl_path, use_camera_obs=False)
    env.seed(0)
    env.reset()
    return env


def generate_prop_set(env, suite):
    """Return the feasible ``PropositionSet`` for ``env``.

    ``safelibero_*`` suites use ``SafetyAtomicPropositionGenerator`` so the
    Level-5 ``_displaced`` safety props are included; everything else uses the
    canonical ``env.get_ltl_propositions()`` path (base Levels 1-4).
    """
    if suite.startswith("safelibero"):
        from libero.libero.envs.ltl_utils.safety_proposition_generator import (
            SafetyAtomicPropositionGenerator,
        )

        return SafetyAtomicPropositionGenerator(env, verbose=False).generate_all(
            include_goals=True
        )
    return env.get_ltl_propositions()


def _to_jsonable(value):
    """Coerce numpy bools / tuples to plain JSON-serializable values."""
    try:
        return bool(value)
    except Exception:
        return value


def serialize_task(suite, task_name, bddl_path, env, prop_set, label_dict):
    """Build the readable JSON-ready dict for one task."""
    parsed = getattr(env, "parsed_problem", {}) or {}

    # parsed_problem stores the instruction as a token list; join for readability.
    instruction = parsed.get("language_instruction")
    if isinstance(instruction, (list, tuple)):
        instruction = " ".join(str(t) for t in instruction)

    propositions = {}
    counts = {}
    categories = list(_CATEGORY_ORDER)
    for cat in sorted(prop_set.categories.keys()):
        if cat not in categories:
            categories.append(cat)

    for cat in categories:
        props = prop_set.get_propositions_by_category(cat)
        # Stable, readable ordering (goals keep BDDL order; others sorted by name).
        if cat != "goal":
            props = sorted(props, key=lambda p: p.name)
        entries = []
        for p in props:
            entries.append(
                {
                    "name": p.name,
                    "description": p.description,
                    "args": list(p.args) if isinstance(p.args, (list, tuple)) else p.args,
                    "init_value": _to_jsonable(label_dict.get(p.name)),
                }
            )
        propositions[cat] = entries
        counts[cat] = len(entries)
    counts["total"] = len(prop_set)

    return {
        "task": task_name,
        "suite": suite,
        "problem_name": parsed.get("problem_name"),
        "bddl_file": os.path.relpath(bddl_path, _REPO_ROOT),
        "language_instruction": instruction,
        "objects": sorted(getattr(env, "objects_dict", {}).keys()),
        "fixtures": sorted(getattr(env, "fixtures_dict", {}).keys()),
        "counts": counts,
        "propositions": propositions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=os.path.join(_REPO_ROOT, "feasible_propositions"),
        help="Output folder (one <suite>/<task>.json per task).",
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=None,
        help="Restrict to this suite (repeatable). Default: all suites.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N tasks."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-dump tasks whose JSON already exists (default: skip them).",
    )
    args = parser.parse_args()

    suites = set(args.suite) if args.suite else None
    tasks = list(discover_tasks(suites))
    if args.limit is not None:
        tasks = tasks[: args.limit]

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output dir : {args.out_dir}")
    print(f"Tasks found: {len(tasks)}")

    index = []
    failures = []
    for i, (suite, task_name, bddl_path) in enumerate(tasks, 1):
        suite_dir = os.path.join(args.out_dir, suite)
        out_path = os.path.join(suite_dir, f"{task_name}.json")
        prefix = f"[{i}/{len(tasks)}] {suite}/{task_name}"

        if os.path.exists(out_path) and not args.overwrite:
            print(f"{prefix}: skip (exists)")
            continue

        env = None
        try:
            env = build_env(bddl_path)
            domain = env.env  # underlying BDDLBaseDomain (wrapper delegates)
            prop_set = generate_prop_set(domain, suite)
            label_dict = prop_set.get_label_dict(domain)
            record = serialize_task(
                suite, task_name, bddl_path, domain, prop_set, label_dict
            )
        except Exception as exc:  # noqa: BLE001 - record & continue
            failures.append({"suite": suite, "task": task_name, "error": str(exc)})
            print(f"{prefix}: ERROR {exc}")
            traceback.print_exc()
            continue
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

        os.makedirs(suite_dir, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(record, fh, indent=2, sort_keys=False)
        index.append(record["counts"])  # count this run's writes for the summary line
        print(f"{prefix}: {record['counts']['total']} APs -> {out_path}")

    # Build a COMPLETE index from every JSON on disk so incremental / --suite runs
    # never truncate it. "missing" lists every discovered task lacking a dump
    # (across ALL suites), annotated with this run's error message when available.
    disk_index = []
    present = set()
    for jf in sorted(glob.glob(os.path.join(args.out_dir, "*", "*.json"))):
        try:
            rec = json.load(open(jf))
        except Exception:
            continue
        disk_index.append(
            {
                "suite": rec.get("suite"),
                "task": rec.get("task"),
                "file": os.path.relpath(jf, args.out_dir),
                "counts": rec.get("counts"),
            }
        )
        present.add((rec.get("suite"), rec.get("task")))

    run_errors = {(f["suite"], f["task"]): f["error"] for f in failures}
    missing = [
        {"suite": s, "task": t, "error": run_errors.get((s, t), "")}
        for s, t, _ in discover_tasks(None)
        if (s, t) not in present
    ]

    with open(os.path.join(args.out_dir, "index.json"), "w") as fh:
        json.dump(
            {"tasks": disk_index, "missing": missing, "num_missing": len(missing)},
            fh,
            indent=2,
        )

    print(
        f"\nDone. {len(index)} written this run; "
        f"{len(disk_index)} total on disk; {len(missing)} tasks missing."
    )
    if failures:
        print("This run's failures:")
        for f in failures:
            print(f"  {f['suite']}/{f['task']}: {f['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
