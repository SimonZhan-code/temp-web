"""Single entry point for building an ``LDBA`` from an LTL formula.

Resolution order (cheapest first):

1. Sidecar ``.hoa`` file next to the BDDL file (e.g.
   ``bddl_files/libero_10/KITCHEN_SCENE3_*.hoa`` next to
   ``bddl_files/libero_10/KITCHEN_SCENE3_*.bddl``). The user generates these
   ahead of time with Rabinizer and commits them.
2. ``PREBUILT_HOA`` dict (defined in ``task_specs.py``), keyed by the resolved
   formula string. Used in unit tests where no BDDL path is in scope.
3. Live Rabinizer subprocess via :func:`run_rabinizer`. Only invoked when the
   ``RABINIZER_PATH`` env var points to a working ``ltl2ldba`` binary.

If all three fail, raises ``LDBABuildError`` with a clear message.
"""

from __future__ import annotations

import os
from typing import Optional

from .automata import LDBA
from .hoa_parser import HOAParser
from .logic import Assignment
from .rabinizer import run_rabinizer


class LDBABuildError(RuntimeError):
    pass


def _load_sidecar_hoa(bddl_file_name: Optional[str]) -> Optional[str]:
    if not bddl_file_name:
        return None
    base, _ = os.path.splitext(bddl_file_name)
    candidate = base + ".hoa"
    if not os.path.exists(candidate):
        return None
    try:
        with open(candidate, "r") as fh:
            return fh.read()
    except OSError:
        return None


def _load_prebuilt_hoa(formula: str) -> Optional[str]:
    # Imported lazily to avoid a circular import; PREBUILT_HOA lives in
    # task_specs.py alongside the registry.
    try:
        from .task_specs import PREBUILT_HOA
    except ImportError:
        return None
    return PREBUILT_HOA.get(formula)


def _try_rabinizer(formula: str) -> Optional[str]:
    try:
        return run_rabinizer(formula)
    except (FileNotFoundError, RuntimeError):
        return None


def resolve_hoa(formula: str, bddl_file_name: Optional[str] = None) -> Optional[str]:
    """Run the cache resolution chain without raising.

    Useful for callers that want to know whether a formula has a cached HOA
    before paying the cost of construction.
    """

    return (
        _load_sidecar_hoa(bddl_file_name)
        or _load_prebuilt_hoa(formula)
        or _try_rabinizer(formula)
    )


def build_ldba(
    formula: str,
    propositions: set[str],
    *,
    bddl_file_name: Optional[str] = None,
    possible_assignments: Optional[list[Assignment]] = None,
) -> LDBA:
    """Resolve ``formula`` to HOA, parse, prune, complete, and compute SCCs.

    Args:
        formula: LTL formula in Rabinizer syntax (atoms, ``! & |``, ``F G X U``).
        propositions: Atomic propositions referenced by the formula. May be a
            superset of the HOA's APs.
        bddl_file_name: If provided, the directory and stem are used to look
            up a sidecar ``.hoa`` file. Optional but strongly recommended for
            production use.
        possible_assignments: If provided, restricts transition validity to
            only these assignments (Neuralsym ``prune`` step).

    Raises:
        LDBABuildError: when no cache hit and Rabinizer is unavailable.
    """

    hoa = resolve_hoa(formula, bddl_file_name=bddl_file_name)
    if hoa is None:
        raise LDBABuildError(
            f"Could not build LDBA for formula '{formula}'. Tried sidecar HOA"
            f"{f' (bddl: {bddl_file_name})' if bddl_file_name else ''}, "
            "PREBUILT_HOA cache, and live Rabinizer subprocess. Either set "
            "RABINIZER_PATH to a working ltl2ldba binary, generate a sidecar "
            ".hoa file alongside the BDDL, or add an entry to PREBUILT_HOA."
        )
    ldba = HOAParser(formula, hoa, propositions).parse_hoa()
    if possible_assignments:
        ldba.prune(possible_assignments)
    ldba.complete_sink_state()
    ldba.compute_sccs()
    return ldba
