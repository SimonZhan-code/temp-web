"""
Prebuild & cache the LDBA (HOA) for every canonical task LTL formula, so online
eval/training never calls Rabinizer.

Reads ``canonical_task_ltl_labels.json``, collects the UNIQUE ``resolved_formula``
strings (many tasks share one), runs Rabinizer once per unique formula, sanity-
parses each HOA, and writes the formula-keyed cache
``libero/libero/ltl_monitor/hoa_store.json`` (``resolved_formula -> hoa_text``).
``builder._load_prebuilt_hoa`` consults this (via ``PREBUILT_HOA``) before ever
invoking Rabinizer.

Requires Rabinizer (Java 11+). Point RABINIZER_PATH at the ltl2ldba launcher and
ensure a Java 11 runtime is on PATH, e.g.:

    PATH=/path/to/jdk11/bin:$PATH \
    RABINIZER_PATH=./rabinizer4/bin/ltl2ldba \
    python scripts/prebuild_ltl_automata.py
"""

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from libero.libero.ltl_monitor.hoa_parser import HOAParser
from libero.libero.ltl_monitor.rabinizer import run_rabinizer
from libero.libero.ltl_monitor.temporal_monitor import is_atomic_token, tokenize

_LABELS = os.path.join(_REPO_ROOT, "libero", "libero", "ltl_monitor", "canonical_task_ltl_labels.json")
_STORE = os.path.join(_REPO_ROOT, "libero", "libero", "ltl_monitor", "hoa_store.json")


def formula_atoms(formula: str):
    return {t for t in tokenize(formula) if is_atomic_token(t)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=_LABELS)
    parser.add_argument("--out", default=_STORE)
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild formulas already present in the store.")
    args = parser.parse_args()

    labels = json.load(open(args.labels))["labels"]
    formulas = sorted({r["resolved_formula"] for r in labels.values() if r["resolved_formula"].strip()})

    store = {}
    if os.path.exists(args.out):
        try:
            store = json.load(open(args.out))
        except Exception:
            store = {}

    built, skipped, failures = 0, 0, []
    for i, formula in enumerate(formulas, 1):
        if formula in store and not args.overwrite:
            skipped += 1
            continue
        try:
            hoa = run_rabinizer(formula)
            atoms = formula_atoms(formula)
            ldba = HOAParser(formula, hoa, atoms).parse_hoa()  # sanity-parse
            ldba.compute_sccs()
            store[formula] = hoa
            built += 1
            print(f"[{i}/{len(formulas)}] OK  {formula}")
        except Exception as exc:  # noqa: BLE001
            failures.append({"formula": formula, "error": str(exc)})
            print(f"[{i}/{len(formulas)}] FAIL {formula} :: {exc}")

    with open(args.out, "w") as fh:
        json.dump(store, fh, indent=2)

    print(f"\nUnique formulas: {len(formulas)} | built: {built} | skipped: {skipped} "
          f"| failures: {len(failures)} | store size: {len(store)}")
    if failures:
        for f in failures[:10]:
            print(f"  FAIL {f['formula']}: {f['error']}")
    print(f"Wrote: {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
