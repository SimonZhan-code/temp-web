"""
Regression: every canonical task LTL label builds its LDBA from the prebuilt
cache with Rabinizer DISABLED, and the language is non-empty. Proves online
eval/training needs no Rabinizer. No MuJoCo required.
"""

import json
import os

import pytest

os.environ["RABINIZER_PATH"] = "/nonexistent/ltl2ldba"  # forbid live builds

try:
    from libero.libero.ltl_monitor.builder import LDBABuildError, build_ldba
    from libero.libero.ltl_monitor.search import ExhaustiveSearchSimple, NoPathsException
    from libero.libero.ltl_monitor.temporal_monitor import is_atomic_token, tokenize
except ModuleNotFoundError:  # pragma: no cover
    from libero.ltl_monitor.builder import LDBABuildError, build_ldba
    from libero.ltl_monitor.search import ExhaustiveSearchSimple, NoPathsException
    from libero.ltl_monitor.temporal_monitor import is_atomic_token, tokenize

_LABELS = os.path.join(os.path.dirname(__file__), "..", "canonical_task_ltl_labels.json")


def _labels():
    if not os.path.exists(_LABELS):
        pytest.skip("canonical_task_ltl_labels.json not generated yet")
    return json.load(open(_LABELS))["labels"]


def _atoms(rec):
    # Only the atoms appearing in the formula (matches the runtime monitor).
    return {t for t in tokenize(rec["resolved_formula"]) if is_atomic_token(t)}


def test_every_label_builds_from_cache_without_rabinizer():
    miss = []
    for key, rec in _labels().items():
        try:
            build_ldba(rec["resolved_formula"], _atoms(rec))
        except LDBABuildError as exc:
            miss.append((key, str(exc)))
    assert not miss, f"{len(miss)} tasks not covered by prebuilt cache: {miss[:5]}"


def test_every_label_language_non_empty():
    empty = []
    for key, rec in _labels().items():
        atoms = _atoms(rec)
        ldba = build_ldba(rec["resolved_formula"], atoms)
        try:
            ExhaustiveSearchSimple(atoms)(ldba, [ldba.initial_state])
        except NoPathsException:
            empty.append(key)
    assert not empty, f"{len(empty)} tasks have empty language: {empty[:5]}"
