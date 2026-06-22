"""
Tests for the rule-based transition system (M2), driven by KITCHEN_SCENE4.

KITCHEN_SCENE4 (white cabinet + bowl + wine bottle): the bottom drawer starts
OPEN, the top drawer starts CLOSED, and ``<cabinet>_top_side`` is a surface (not a
gated drawer). Run in the ``libero-max`` conda env.
"""

import pytest

from libero.libero.envs.ltl_utils.composition import manifest as M
from libero.libero.envs.ltl_utils.composition import transitions as T

BOTTOM = "white_cabinet_1_bottom_region"
TOP = "white_cabinet_1_top_region"
IN_BOWL_BOTTOM = "in_akita_black_bowl_1_white_cabinet_1_bottom_region"
ON_BOWL_TOPSIDE = "on_akita_black_bowl_1_white_cabinet_1_top_side"
OPEN_TOP = "open_white_cabinet_1_top_region"
CLOSE_BOTTOM = "close_white_cabinet_1_bottom_region"

_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        unit = None
        for u in M.build_scene_units():
            if u.scene_id == "KITCHEN_SCENE4":
                unit = u
                break
        if unit is None:
            pytest.skip("KITCHEN_SCENE4 unit not present")
        _MODEL = T.build_scene_model(unit)
    return _MODEL


def test_per_region_init_open_state():
    m = model()
    assert m.drawers[BOTTOM].init_open is True   # bottom drawer starts open
    assert m.drawers[TOP].init_open is False     # top drawer starts closed


def test_top_side_is_surface_not_drawer():
    m = model()
    # top_side prefix-matches the cabinet but has no open/close achiever -> not a drawer.
    assert "white_cabinet_1_top_side" not in m.drawers


def test_place_into_open_bottom_enabled_at_init():
    m = model()
    achieved = {p.achieved_ap for p in T.enabled_primitives(m, m.init_state)}
    assert IN_BOWL_BOTTOM in achieved          # bottom is open -> placement enabled
    assert ON_BOWL_TOPSIDE in achieved         # surface placement always enabled
    assert OPEN_TOP in achieved                # top closed -> can open it


def test_closing_bottom_blocks_placement_into_it():
    m = model()
    # Close the bottom drawer, then placing into it must no longer be enabled.
    close = next(
        p for p in T.enabled_primitives(m, m.init_state)
        if p.kind == "close" and p.target == BOTTOM
    )
    state2 = T.apply(m.init_state, close)
    achieved = {p.achieved_ap for p in T.enabled_primitives(m, state2)}
    assert IN_BOWL_BOTTOM not in achieved      # open-before-place gate now blocks it
    assert ON_BOWL_TOPSIDE in achieved         # surface still fine


def test_single_variable_change_per_primitive():
    m = model()
    s = m.init_state
    for p in T.enabled_primitives(m, s):
        s2 = T.apply(s, p)
        diffs = 0
        diffs += sum(1 for a, b in zip(s.locations, s2.locations) if a != b)
        diffs += sum(1 for a, b in zip(s.region_open, s2.region_open) if a != b)
        diffs += sum(1 for a, b in zip(s.appliance_on, s2.appliance_on) if a != b)
        assert diffs == 1, f"{p} changed {diffs} variables"


def test_no_self_loop_moves():
    m = model()
    for p in T.enabled_primitives(m, m.init_state):
        if p.kind == "move":
            assert p.achieved_ap != m.init_state.location_of(p.obj)
