"""
Verify that every canonical task LTL label builds an LDBA FROM THE CACHE with
Rabinizer disabled -- i.e. online eval/training needs no Rabinizer.

For each task in ``canonical_task_ltl_labels.json``:
  - point RABINIZER_PATH at a non-existent binary (so a cache miss can't silently
    fall through to a live build),
  - ``build_ldba(resolved_formula, atoms)`` -> must succeed (cache hit),
  - assert the automaton is NON-EMPTY (a reach-avoid sequence exists).

Exits non-zero if any task lacks a cached automaton or yields an empty language.
Run in any env with the ltl_monitor package importable (no MuJoCo needed):
    python scripts/verify_ltl_cache.py
"""

import json
import os
import sys

# Disable live Rabinizer BEFORE importing the monitor stack.
os.environ["RABINIZER_PATH"] = "/nonexistent/ltl2ldba"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from libero.libero.ltl_monitor.builder import LDBABuildError, build_ldba
from libero.libero.ltl_monitor.search import ExhaustiveSearchSimple, NoPathsException
from libero.libero.ltl_monitor.temporal_monitor import is_atomic_token, tokenize

_LABELS = os.path.join(_REPO_ROOT, "libero", "libero", "ltl_monitor", "canonical_task_ltl_labels.json")


def main() -> int:
    labels = json.load(open(_LABELS))["labels"]
    cache_miss, empty, ok = [], [], 0

    for key, rec in sorted(labels.items()):
        formula = rec["resolved_formula"]
        # Use ONLY the atoms in the formula (as the runtime does); declared
        # per-object safety atoms aren't in the formula and would needlessly
        # enlarge the assignment space.
        atoms = {t for t in tokenize(formula) if is_atomic_token(t)}
        try:
            ldba = build_ldba(formula, atoms)
        except LDBABuildError as exc:
            cache_miss.append((key, str(exc)))
            continue
        try:
            ExhaustiveSearchSimple(atoms)(ldba, [ldba.initial_state])
            ok += 1
        except NoPathsException:
            empty.append(key)

    print(f"Tasks: {len(labels)} | cache-built+non-empty: {ok} "
          f"| cache MISS: {len(cache_miss)} | EMPTY language: {len(empty)}")
    for k, e in cache_miss[:10]:
        print(f"  MISS  {k}: {e}")
    for k in empty[:10]:
        print(f"  EMPTY {k}")
    return 0 if (not cache_miss and not empty) else 1


if __name__ == "__main__":
    raise SystemExit(main())
