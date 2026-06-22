"""
Dump-driven tests for the factored-state / mutex / canonical-map extractor (M1).

Run in the ``libero-max`` conda env (importing the package pulls robosuite).
"""

import pytest

from libero.libero.envs.ltl_utils.composition import factored_state as F
from libero.libero.envs.ltl_utils.composition import manifest as M

_UNITS = None


def units():
    global _UNITS
    if _UNITS is None:
        _UNITS = M.build_scene_units()
    if not _UNITS:
        pytest.skip("no feasible_propositions dumps found for original-LIBERO suites")
    return _UNITS


def unit_by_id(scene_id):
    for u in units():
        if u.scene_id == scene_id:
            return u
    return None


def test_gated_region_controller_is_openable():
    # Find any scene with an openable container and at least one gated region.
    target = None
    for u in units():
        sc = F.build_factored_scene(u)
        if sc.gated_regions and any(a.openable for a in sc.articulation_vars.values()):
            target = sc
            break
    assert target is not None, "expected at least one scene with a gated container region"
    for region, obj in target.gated_regions.items():
        assert target.articulation_vars[obj].openable
        assert region.startswith(obj + "_")


def test_wooden_cabinet_region_gated():
    # KITCHEN_SCENE1 has an openable wooden_cabinet_1 with drawer regions.
    u = unit_by_id("KITCHEN_SCENE1")
    if u is None:
        pytest.skip("KITCHEN_SCENE1 unit not present")
    sc = F.build_factored_scene(u)
    cab = sc.articulation_vars.get("wooden_cabinet_1")
    assert cab is not None and cab.openable
    assert any(o == "wooden_cabinet_1" for o in sc.gated_regions.values())


def test_canonical_map_basket_in_goal():
    u = unit_by_id("LIVING_ROOM_SCENE2")
    if u is None:
        pytest.skip("LIVING_ROOM_SCENE2 unit not present")
    sc = F.build_factored_scene(u)
    assert sc.canonical, "expected a non-empty canonical map for a basket scene"
    assert any(v.startswith(("in_", "on_")) for v in sc.canonical.values())
    key = "alphabet_soup_1_in_basket_1_contain_region"
    if key in sc.canonical:
        assert sc.canonical[key] == "in_alphabet_soup_1_basket_1_contain_region"


def test_stove_is_appliance_not_openable():
    found = False
    for u in units():
        sc = F.build_factored_scene(u)
        a = sc.articulation_vars.get("flat_stove_1")
        if a is not None:
            found = True
            assert a.appliance and not a.openable, "flat_stove must be toggle-only, not openable"
    if not found:
        pytest.skip("no scene with flat_stove_1 found")


def test_own_init_region_excluded_from_destinations():
    u = unit_by_id("LIVING_ROOM_SCENE2")
    if u is None:
        pytest.skip("LIVING_ROOM_SCENE2 unit not present")
    sc = F.build_factored_scene(u)
    lv = sc.location_vars.get("alphabet_soup_1")
    assert lv is not None
    if lv.origin_ap is not None:
        assert lv.origin_ap not in lv.destination_aps
        assert lv.init_location == lv.origin_ap


def test_init_state_covers_all_factored_vars():
    sc = F.build_factored_scene(units()[0])
    st = sc.init_assignment
    assert len(st.locations) == len(sc.location_vars)
    openable = {o for o, a in sc.articulation_vars.items() if a.openable}
    appliance = {o for o, a in sc.articulation_vars.items() if a.appliance}
    assert {o for o, _ in st.open_state} == openable
    assert {o for o, _ in st.on_state} == appliance
