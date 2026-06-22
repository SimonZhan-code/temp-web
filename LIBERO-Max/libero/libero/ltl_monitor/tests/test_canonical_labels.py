"""
Correctness checks for the canonical per-task LTL labels (no Rabinizer / MuJoCo).

Validates ``libero/libero/ltl_monitor/canonical_task_ltl_labels.json``: every
formula tokenizes, every atomic token is a declared goal/safety atom, every goal
atom is referenced, and no non-empty-goal task collapses to a trivial formula.
"""

import json
import os

import pytest

try:
    from libero.libero.ltl_monitor.temporal_monitor import is_atomic_token, tokenize
except ModuleNotFoundError:  # pragma: no cover - import path fallback
    from libero.ltl_monitor.temporal_monitor import is_atomic_token, tokenize

_LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "canonical_task_ltl_labels.json")


def _labels():
    if not os.path.exists(_LABELS_PATH):
        pytest.skip("canonical_task_ltl_labels.json not generated yet")
    return json.load(open(_LABELS_PATH))["labels"]


def test_every_formula_tokenizes_and_atoms_declared():
    bad = []
    for key, rec in _labels().items():
        declared = set(rec["goal_atoms"]) | set(rec["safety_atoms"])
        atoms = {t for t in tokenize(rec["resolved_formula"]) if is_atomic_token(t)}
        unknown = atoms - declared
        if unknown:
            bad.append((key, sorted(unknown)))
    assert not bad, f"formulas reference undeclared atoms: {bad[:5]}"


def test_all_goal_atoms_referenced():
    bad = []
    for key, rec in _labels().items():
        atoms = {t for t in tokenize(rec["resolved_formula"]) if is_atomic_token(t)}
        missing = set(rec["goal_atoms"]) - atoms
        if missing:
            bad.append((key, sorted(missing)))
    assert not bad, f"goal atoms missing from formula: {bad[:5]}"


def test_no_nonempty_goal_collapses_to_trivial():
    bad = [
        key for key, rec in _labels().items()
        if rec["goal_atoms"] and rec["resolved_formula"].strip() in ("", "true")
    ]
    assert not bad, f"tasks with goals but trivial formula: {bad[:5]}"


def test_coverage_three_families():
    labels = _labels()
    suites = {rec["suite"] for rec in labels.values()}
    # original LIBERO + LIBERO-10-R present (SafeLIBERO handled separately).
    for s in ("libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_10", "libero_10_r"):
        assert s in suites, f"missing suite {s}"
    assert len(labels) >= 160
