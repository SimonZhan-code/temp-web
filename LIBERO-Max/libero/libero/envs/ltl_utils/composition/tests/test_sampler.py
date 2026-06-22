"""
Tests for the random-walk sampler + meaningfulness filters (M2), on KITCHEN_SCENE4.

Run in the ``libero-max`` conda env.
"""

import random

import pytest

from libero.libero.envs.ltl_utils.composition import filters as Flt
from libero.libero.envs.ltl_utils.composition import manifest as M
from libero.libero.envs.ltl_utils.composition import transitions as T
from libero.libero.envs.ltl_utils.composition.sampler import (
    SampleConfig,
    Walk,
    enumerate_walks,
    lift_to_ltl,
    random_walk,
)

IN_BOWL_BOTTOM = "in_akita_black_bowl_1_white_cabinet_1_bottom_region"
CLOSE_BOTTOM = "close_white_cabinet_1_bottom_region"
BOTTOM = "white_cabinet_1_bottom_region"


def model():
    for u in M.build_scene_units():
        if u.scene_id == "KITCHEN_SCENE4":
            return T.build_scene_model(u)
    pytest.skip("KITCHEN_SCENE4 unit not present")


def test_walk_is_deterministic_for_seed():
    m = model()
    cfg = SampleConfig(depth=3, seed=7)
    w1 = random_walk(m, cfg, random.Random(cfg.seed))
    w2 = random_walk(m, cfg, random.Random(cfg.seed))
    assert w1.subgoal_aps == w2.subgoal_aps


def test_walk_is_feasible_by_construction():
    m = model()
    for seed in range(20):
        walk = random_walk(m, SampleConfig(depth=4, seed=seed), random.Random(seed))
        state = m.init_state
        for prim in walk.primitives:
            assert prim in T.enabled_primitives(m, state), "walk replayed an infeasible step"
            state = T.apply(state, prim)


def test_lift_ordered_nesting():
    alias_f, aliases, resolved = lift_to_ltl(["a", "b", "c"], ordered=True)
    assert alias_f == "F(g1 & F(g2 & F(g3)))"
    assert resolved == "F(a & F(b & F(c)))"
    assert aliases == {"g1": "a", "g2": "b", "g3": "c"}


def test_lift_order_free():
    alias_f, _, resolved = lift_to_ltl(["a", "b"], ordered=False)
    assert alias_f == "F(g1 & g2)"
    assert resolved == "F(a & b)"


def _build_close_it_walk(m):
    """Manually build the 'put bowl in bottom drawer and close it' anchor walk."""
    place = next(
        p for p in T.enabled_primitives(m, m.init_state)
        if p.kind == "move" and p.achieved_ap == IN_BOWL_BOTTOM
    )
    s2 = T.apply(m.init_state, place)
    close = next(
        p for p in T.enabled_primitives(m, s2)
        if p.kind == "close" and p.target == BOTTOM
    )
    s3 = T.apply(s2, close)
    return Walk(
        primitives=[place, close],
        subgoal_aps=[place.achieved_ap, close.achieved_ap],
        objects_touched={place.obj, close.obj},
        end_state=s3,
    )


def test_anchor_match_is_excluded_as_held_out():
    m = model()
    walk = _build_close_it_walk(m)
    # This composition equals the libero_10 'and close it' task goal set.
    assert Flt.matches_anchor(walk, m)
    res = Flt.evaluate_walk(walk, m, SampleConfig(ordered=True), exclude_anchor_matches=True)
    assert not res.accepted
    assert not res.is_held_out
    assert "anchor" in res.reason


def test_enumerate_walks_are_all_feasible():
    m = model()
    walks = enumerate_walks(m, max_depth=2)
    assert walks
    assert {len(w.subgoal_aps) for w in walks} == {1, 2}
    for w in walks:
        state = m.init_state
        for prim in w.primitives:
            assert prim in T.enabled_primitives(m, state)
            state = T.apply(state, prim)
        # no duplicate subgoal within a walk
        assert len(w.subgoal_aps) == len(set(w.subgoal_aps))


def test_order_free_consistency_rejects_double_placement():
    m = model()
    # Two placements of the same object are mutex-inconsistent for a conjunction.
    place_aps = list(m.placements["akita_black_bowl_1"].keys())
    assert len(place_aps) >= 2
    fake = Walk(
        primitives=[
            T.Primitive("move", "akita_black_bowl_1", "t1", place_aps[0]),
            T.Primitive("move", "akita_black_bowl_1", "t2", place_aps[1]),
        ],
        subgoal_aps=place_aps[:2],
        objects_touched={"akita_black_bowl_1"},
        end_state=m.init_state,
    )
    assert not Flt.is_order_free_consistent(fake)
