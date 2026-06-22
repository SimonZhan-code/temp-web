"""
Dump-driven tests for the scene-keyed manifest builder.

These read the real scene-keyed ``feasible_propositions/<scene>/<task>.json``
layout (core + libero_10_r + safelibero_* + libero_object). No MuJoCo env is
built, but importing the package pulls robosuite, so run in the ``libero-max``
conda env.
"""

import pytest

from libero.libero.envs.ltl_utils.composition import manifest as M

_UNITS = None


def units():
    global _UNITS
    if _UNITS is None:
        _UNITS = M.build_scene_units()
    if not _UNITS:
        pytest.skip("no feasible_propositions scene folders found")
    return _UNITS


def test_unit_count_in_expected_range():
    # Scene-keyed layout after merging same/subset core+r_ scenes: 47 scene
    # folders. Range guards against gross regressions.
    u = units()
    assert 42 <= len(u) <= 52, f"unexpected scene-unit count: {len(u)}"


def test_kitchen_scene3_merges_90_and_10():
    by_id = {x.scene_id: x for x in units()}
    assert "KITCHEN_SCENE3" in by_id
    k3 = by_id["KITCHEN_SCENE3"]
    # The 90/10 boundary dissolves: short- and long-horizon tasks share one alphabet.
    assert {"libero_90", "libero_10"} <= set(k3.suites)


def test_some_unit_spans_both_horizon_suites():
    spanning = [x for x in units() if {"libero_90", "libero_10"} <= set(x.suites)]
    assert spanning, "expected at least one scene-unit merging libero_90 and libero_10"


def test_spatial_and_goal_are_single_scene_units():
    by_id = {x.scene_id: x for x in units()}
    assert "libero_spatial" in by_id and len(by_id["libero_spatial"].anchor_goals) == 10
    assert "libero_goal" in by_id and len(by_id["libero_goal"].anchor_goals) == 10


def test_alphabet_is_non_goal_only():
    for x in units():
        # Only the three non-goal categories form the alphabet.
        assert set(x.alphabet_by_cat.keys()) == set(M.ALPHABET_CATEGORIES)
        assert x.alphabet_names, f"{x.scene_id} has empty alphabet"


def test_members_share_identical_alphabet():
    # By construction every member of a unit must have the same non-goal AP set.
    for x in units():
        assert len(x.alphabet_names) == len(set(x.alphabet_names))
