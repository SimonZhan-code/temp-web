"""
Generate canonical per-task LTL labels for evaluation (authoritative, env-based).

For every task in the original-LIBERO, SafeLIBERO, and LIBERO-10-R suites, builds
the env headless with ``LIBERO_LTL_SPEC_MODE=auto`` and calls the SAME runtime
resolver (``env.env.get_ltl_task_spec`` -> ``task_specs.get_task_ltl_spec``) so the
recorded ``resolved_formula`` is byte-identical to what eval produces (it is the
automaton cache key, so parity is essential).

Writes ``libero/libero/ltl_monitor/canonical_task_ltl_labels.json``:
{task_id: {suite, bddl_path, mode, formula, resolved_formula, proposition_aliases,
           goal_atoms, safety_atoms, source}} + an index with per-suite counts and
flagged tasks (empty goals / trivially true).

Run in the libero-max conda env (MuJoCo). Usage:
    MUJOCO_GL=egl python scripts/build_ltl_labels.py
    MUJOCO_GL=egl python scripts/build_ltl_labels.py --suite libero_10 --suite safelibero_goal
"""

import argparse
import glob
import json
import os
import sys
import traceback

os.environ.setdefault("LIBERO_LTL_SPEC_MODE", "auto")  # ordering-aware + safety

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_BDDL_ROOT = os.path.join(_REPO_ROOT, "libero", "libero", "bddl_files")
_OUT = os.path.join(_REPO_ROOT, "libero", "libero", "ltl_monitor", "canonical_task_ltl_labels.json")

IN_SCOPE_SUITES = (
    "libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_10",
    "safelibero_goal", "safelibero_long", "safelibero_object", "safelibero_spatial",
    "libero_10_r",
)


def discover_tasks(suites):
    for suite in suites:
        for path in sorted(glob.glob(os.path.join(_BDDL_ROOT, suite, "*.bddl"))):
            task = os.path.splitext(os.path.basename(path))[0]
            yield suite, task, path


def build_env(bddl_path):
    from libero.libero.envs.env_wrapper import ControlEnv, OffScreenRenderEnv

    try:
        env = ControlEnv(
            bddl_file_name=bddl_path,
            has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        )
    except Exception:
        env = OffScreenRenderEnv(bddl_file_name=bddl_path, use_camera_obs=False)
    env.seed(0)
    env.reset()
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", default=None,
                        help="Restrict to suite(s) (repeatable). Default: all in-scope.")
    parser.add_argument("--out", default=_OUT)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-resolve tasks already present in the output file.")
    args = parser.parse_args()

    suites = args.suite or list(IN_SCOPE_SUITES)
    tasks = list(discover_tasks(suites))

    labels = {}
    if os.path.exists(args.out):  # always preserve existing entries; --overwrite re-does processed ones
        try:
            labels = json.load(open(args.out)).get("labels", {})
        except Exception:
            labels = {}

    failures, flagged = [], []
    for i, (suite, task, bddl) in enumerate(tasks, 1):
        key = f"{suite}/{task}"  # composite: task stems collide across libero_10 / libero_10_r
        prefix = f"[{i}/{len(tasks)}] {key}"
        if key in labels and not args.overwrite:
            print(f"{prefix}: skip (exists)")
            continue
        env = None
        try:
            env = build_env(bddl)
            domain = env.env
            spec = domain.get_ltl_task_spec()  # uses LIBERO_LTL_SPEC_MODE=auto, task_id=bddl stem
            prop_set = domain.get_ltl_propositions()
            goal_atoms = sorted(p.name for p in prop_set.get_propositions_by_category("goal"))
            safety_atoms = sorted(
                p.name
                for cat in ("safety_violation", "safety_aggregate")
                for p in prop_set.get_propositions_by_category(cat)
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"suite": suite, "task": task, "error": str(exc)})
            print(f"{prefix}: ERROR {exc}")
            traceback.print_exc()
            continue
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

        resolved = spec.get("resolved_formula", "")
        rec = {
            "suite": suite,
            "task": task,
            "bddl_path": os.path.relpath(bddl, _REPO_ROOT),
            "mode": os.environ["LIBERO_LTL_SPEC_MODE"],
            "formula": spec.get("formula", ""),
            "resolved_formula": resolved,
            "proposition_aliases": spec.get("proposition_aliases", {}),
            "source": spec.get("source", ""),
            "goal_atoms": goal_atoms,
            "safety_atoms": safety_atoms,
        }
        labels[key] = rec
        if not goal_atoms:
            flagged.append({"task": key, "reason": "no_goal_atoms"})
        elif resolved.strip() in ("", "true"):
            flagged.append({"task": key, "reason": "trivial_formula"})
        print(f"{prefix}: {resolved}")

    by_suite = {}
    for r in labels.values():
        by_suite[r["suite"]] = by_suite.get(r["suite"], 0) + 1
    payload = {
        "version": 1,
        "spec_mode": os.environ["LIBERO_LTL_SPEC_MODE"],
        "num_labels": len(labels),
        "num_unique_resolved_formulas": len({r["resolved_formula"] for r in labels.values()}),
        "by_suite": dict(sorted(by_suite.items())),
        "flagged": flagged,
        "failures": failures,
        "labels": labels,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nLabels: {len(labels)} | unique formulas: {payload['num_unique_resolved_formulas']}"
          f" | flagged: {len(flagged)} | failures: {len(failures)}")
    print(f"by suite: {payload['by_suite']}")
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
