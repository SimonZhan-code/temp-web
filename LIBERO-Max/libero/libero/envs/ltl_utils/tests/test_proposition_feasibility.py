"""
Unit tests for Layer 0-2 physical-feasibility filtering of atomic propositions.

The logic tests use a lightweight fake environment (no MuJoCo) modeled on the
``LIVING_ROOM_SCENE2 ... in the basket`` task, mirroring the fake-env style of
``libero/libero/ltl_monitor/tests/test_monitor.py``. A final integration test
builds the real environment for that scene and is skipped if the simulator
cannot be constructed in this environment.
"""

from __future__ import annotations

import os
import sys

import pytest

try:
    from libero.libero.envs.ltl_utils import feasibility
    from libero.libero.envs.ltl_utils.proposition_generator import (
        AtomicPropositionGenerator,
    )
except ModuleNotFoundError:  # pragma: no cover - import path fallback
    from libero.envs.ltl_utils import feasibility
    from libero.envs.ltl_utils.proposition_generator import (
        AtomicPropositionGenerator,
    )


# --------------------------------------------------------------------------- #
# Fake environment building blocks (no MuJoCo)
# --------------------------------------------------------------------------- #


class _FakeObject:
    """Underlying MuJoCo-object stand-in with optional affordances."""

    def __init__(self, category_name, openable=False, toggleable=False, has_in_box=False):
        self.category_name = category_name
        if openable:
            self.is_open = lambda qpos=0.0: False
            self.is_close = lambda qpos=0.0: True
        if toggleable:
            self.turn_on = lambda qpos=0.0: False
            self.turn_off = lambda qpos=0.0: True
        if has_in_box:
            self.in_box = lambda *a, **k: False


class _FakeState:
    """ObjectState/SiteObjectState stand-in exposing the eval-path methods."""

    def __init__(self, *, ontop=False, contain=False):
        if ontop:
            self.check_ontop = lambda other: False
        if contain:
            self.check_contain = lambda other: False
            self.check_contact = lambda other: True


class _FakeEnv:
    def __init__(self):
        # Underlying objects
        foods = [
            "alphabet_soup_1",
            "cream_cheese_1",
            "tomato_sauce_1",
            "ketchup_1",
            "orange_juice_1",
            "milk_1",
            "butter_1",
        ]
        self.objects_dict = {f: _FakeObject(f.rsplit("_", 1)[0]) for f in foods}
        self.objects_dict["basket_1"] = _FakeObject("basket")  # container, but no in_box
        # The table is the arena anchor (a region *target*), NOT an instantiated
        # object: it has no entry in fixtures_dict / object_states_dict and
        # get_object("living_room_table") returns None. This mirrors the real
        # scene, where env.fixtures_dict is empty even though parsed_problem
        # still lists living_room_table under "fixtures".
        self.fixtures_dict = {}

        # Region sites
        self.object_sites_dict = {
            "living_room_table_alphabet_soup_init_region": object(),
            "living_room_table_basket_init_region": object(),
            "basket_1_contain_region": object(),
        }

        # State wrappers (objects support on/in eval paths; sites support contain)
        self.object_states_dict = {}
        for name in list(self.objects_dict) + list(self.fixtures_dict):
            self.object_states_dict[name] = _FakeState(ontop=True, contain=True)
        for region in self.object_sites_dict:
            self.object_states_dict[region] = _FakeState(contain=True)

        self.parsed_problem = {
            "objects": {
                **{f.rsplit("_", 1)[0]: [f] for f in foods},
                "basket": ["basket_1"],
            },
            "fixtures": {"living_room_table": ["living_room_table"]},
            "regions": {
                "living_room_table_alphabet_soup_init_region": {
                    "target": "living_room_table",
                    "ranges": [[-0.125, -0.175, -0.075, -0.125]],
                },
                "living_room_table_basket_init_region": {
                    "target": "living_room_table",
                    "ranges": [[-0.01, 0.25, 0.01, 0.27]],
                },
                "basket_1_contain_region": {"target": "basket_1", "ranges": []},
            },
            "goal_state": [
                ("In", "alphabet_soup_1", "basket_1_contain_region"),
                ("In", "tomato_sauce_1", "basket_1_contain_region"),
            ],
        }

    def get_object(self, name):
        for d in (self.fixtures_dict, self.objects_dict, self.object_sites_dict):
            if name in d:
                return d[name]
        return None

    def _eval_predicate(self, predicate):  # used by goal eval closures only
        return False


@pytest.fixture
def fake_env():
    return _FakeEnv()


# --------------------------------------------------------------------------- #
# Layer 0 — evaluability gate
# --------------------------------------------------------------------------- #


def test_unary_predicate_key_map():
    assert feasibility.map_unary_predicate("is_open") == "open"
    assert feasibility.map_unary_predicate("is_close") == "close"
    assert feasibility.map_unary_predicate("turn_on") == "turnon"
    assert feasibility.map_unary_predicate("turn_off") == "turnoff"
    assert feasibility.map_unary_predicate("nonsense") is None


def test_capability_detection_on_underlying_object():
    assert feasibility.is_openable(_FakeObject("wooden_cabinet", openable=True))
    assert not feasibility.is_openable(_FakeObject("alphabet_soup"))
    assert feasibility.is_toggleable(_FakeObject("flat_stove", toggleable=True))
    assert not feasibility.is_toggleable(_FakeObject("basket"))
    assert not feasibility.is_openable(None)


def test_toggle_only_object_not_openable():
    """A FlatStove-like object inherits the ArticulatedObject is_open stub.

    The base class declares is_open/is_close as NotImplementedError stubs; a
    toggle-only subclass (FlatStove) overrides turn_on but inherits the stubs.
    The gate must treat it as toggleable but NOT openable, otherwise it emits
    permanently-false flat_stove_*_is_open APs (the real bug this guards).
    """

    class ArticulatedObject:  # mirrors the real base: callable but unimplemented
        def is_open(self, qpos):
            raise NotImplementedError

        def is_close(self, qpos):
            raise NotImplementedError

    class FlatStove(ArticulatedObject):  # toggleable, inherits is_open stub
        def turn_on(self, qpos):
            return True

        def turn_off(self, qpos):
            return False

    class Microwave(ArticulatedObject):  # genuinely openable: overrides is_open
        def is_open(self, qpos):
            return False

        def is_close(self, qpos):
            return True

    stove = FlatStove()
    assert feasibility.is_toggleable(stove)  # turn_on is a real override
    assert not feasibility.is_openable(stove)  # is_open is only the inherited stub

    microwave = Microwave()
    assert feasibility.is_openable(microwave)  # real is_open override
    assert not feasibility.is_toggleable(microwave)  # no turn_on at all


# --------------------------------------------------------------------------- #
# Layer 1 — type compatibility
# --------------------------------------------------------------------------- #


def test_support_surface_classification(fake_env):
    # The table is the arena anchor, not an instantiated fixture object, so it is
    # NOT a usable support surface (matches the real scene: fixtures_dict empty,
    # get_object("living_room_table") is None, no check_ontop state).
    assert not feasibility.is_support_surface(fake_env, "living_room_table")
    assert not feasibility.is_support_surface(fake_env, "alphabet_soup_1")
    assert not feasibility.is_support_surface(fake_env, "basket_1")
    # A plate is a known movable support-surface category.
    fake_env.objects_dict["plate_1"] = _FakeObject("plate")
    fake_env.parsed_problem["objects"]["plate"] = ["plate_1"]
    assert feasibility.is_support_surface(fake_env, "plate_1")


def test_container_classification(fake_env):
    # basket is a container category AND target of a *_contain_region
    assert feasibility.is_container(fake_env, "basket_1")
    assert not feasibility.is_container(fake_env, "alphabet_soup_1")
    assert not feasibility.is_container(fake_env, "living_room_table")


# --------------------------------------------------------------------------- #
# Layer 2 — region geometry / validity
# --------------------------------------------------------------------------- #


def test_region_validity_and_self_region(fake_env):
    table_region = "living_room_table_alphabet_soup_init_region"
    assert feasibility.region_is_valid(fake_env, table_region)
    assert feasibility.region_feasible_for_object(fake_env, "milk_1", table_region)

    # self-region: basket_1 cannot be "in" its own contain region
    assert not feasibility.region_feasible_for_object(
        fake_env, "basket_1", "basket_1_contain_region"
    )
    # but a food object in the basket's contain region is feasible
    assert feasibility.region_feasible_for_object(
        fake_env, "alphabet_soup_1", "basket_1_contain_region"
    )


def test_region_validity_rejects_missing_and_degenerate(fake_env):
    assert not feasibility.region_is_valid(fake_env, "does_not_exist")
    fake_env.parsed_problem["regions"]["living_room_table_alphabet_soup_init_region"][
        "ranges"
    ] = [[0.1, 0.1, 0.05, 0.05]]  # xmax < xmin -> degenerate
    assert not feasibility.region_is_valid(
        fake_env, "living_room_table_alphabet_soup_init_region"
    )


# --------------------------------------------------------------------------- #
# Generator integration (fake env)
# --------------------------------------------------------------------------- #


def _generate(env):
    gen = AtomicPropositionGenerator(env, verbose=False)
    return gen.generate_all(include_goals=True)


def test_no_unary_props_for_non_articulated_scene(fake_env):
    props = _generate(fake_env)
    assert props.get_propositions_by_category("unary_state") == []


def test_unary_props_emitted_for_articulated_object(fake_env):
    # Add an articulated cabinet + a toggleable stove and regenerate.
    fake_env.objects_dict["wooden_cabinet_1"] = _FakeObject(
        "wooden_cabinet", openable=True
    )
    fake_env.objects_dict["flat_stove_1"] = _FakeObject("flat_stove", toggleable=True)
    fake_env.parsed_problem["objects"]["wooden_cabinet"] = ["wooden_cabinet_1"]
    fake_env.parsed_problem["objects"]["flat_stove"] = ["flat_stove_1"]
    fake_env.object_states_dict["wooden_cabinet_1"] = _FakeState(ontop=True, contain=True)
    fake_env.object_states_dict["flat_stove_1"] = _FakeState(ontop=True, contain=True)

    names = {p.name for p in _generate(fake_env).get_propositions_by_category("unary_state")}
    assert "wooden_cabinet_1_is_open" in names
    assert "wooden_cabinet_1_is_close" in names
    assert "flat_stove_1_turn_on" in names
    assert "flat_stove_1_turn_off" in names
    # No spurious open/on props on food items
    assert not any(n.startswith("alphabet_soup_1_") for n in names)


def test_no_binary_props_without_support_surface(fake_env):
    names = {
        p.name for p in _generate(fake_env).get_propositions_by_category("binary_relation")
    }
    # This scene has no instantiated object-level support surface (the table is an
    # arena anchor, not an object) and basket_1 has no in_box, so there is nothing
    # to put an object on/in -> zero binary APs.
    assert names == set()


def test_binary_props_on_support_surface(fake_env):
    # Introduce a real, instantiated support object (a plate) with a check_ontop
    # state -- the only legitimate way an "on" AP is generated.
    fake_env.objects_dict["plate_1"] = _FakeObject("plate")
    fake_env.parsed_problem["objects"]["plate"] = ["plate_1"]
    fake_env.object_states_dict["plate_1"] = _FakeState(ontop=True, contain=True)

    names = {
        p.name for p in _generate(fake_env).get_propositions_by_category("binary_relation")
    }
    # Every other movable object can rest on the plate...
    assert "alphabet_soup_1_on_plate_1" in names
    assert "basket_1_on_plate_1" in names
    # ...nonsense pairs (food on food / food in basket-without-in_box) are pruned...
    assert "alphabet_soup_1_on_milk_1" not in names
    assert "alphabet_soup_1_on_basket_1" not in names
    # ...the plate is not "on" itself...
    assert "plate_1_on_plate_1" not in names
    # ...and the plate is the only support surface, so every AP is "on plate_1".
    assert all(n.endswith("_on_plate_1") for n in names)
    assert len(names) == len(fake_env.objects_dict) - 1  # all movables except the plate


def test_object_level_in_pruned_without_in_box(fake_env):
    names = {
        p.name for p in _generate(fake_env).get_propositions_by_category("binary_relation")
    }
    # basket_1 has no in_box -> object-level containment is not emitted
    assert not any("_in_basket_1" in n for n in names)


def test_in_kept_when_container_has_in_box(fake_env):
    fake_env.objects_dict["real_bowl_1"] = _FakeObject("bowl", has_in_box=True)
    fake_env.parsed_problem["objects"]["bowl"] = ["real_bowl_1"]
    fake_env.object_states_dict["real_bowl_1"] = _FakeState(ontop=True, contain=True)

    names = {
        p.name for p in _generate(fake_env).get_propositions_by_category("binary_relation")
    }
    assert "alphabet_soup_1_in_real_bowl_1" in names


def test_region_props_prune_self_region(fake_env):
    names = {
        p.name
        for p in _generate(fake_env).get_propositions_by_category("region_containment")
    }
    assert "alphabet_soup_1_in_basket_1_contain_region" in names
    assert "basket_1_in_basket_1_contain_region" not in names  # self-region pruned
    # food objects get region props for the table init regions too
    assert "milk_1_in_living_room_table_basket_init_region" in names


def test_goal_props_match_bddl_goal(fake_env):
    names = {p.name for p in _generate(fake_env).get_propositions_by_category("goal")}
    assert len(names) == 2
    assert any("alphabet_soup_1" in n and "basket_1_contain_region" in n for n in names)
    assert any("tomato_sauce_1" in n and "basket_1_contain_region" in n for n in names)


# --------------------------------------------------------------------------- #
# Integration with the real environment (skipped if MuJoCo unavailable)
# --------------------------------------------------------------------------- #


_TASK = "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"


def _real_env():
    import glob

    repo_root = os.path.dirname(
        os.path.abspath(os.path.join(__file__, "..", "..", "..", "..", ".."))
    )
    matches = glob.glob(
        os.path.join(repo_root, "libero", "libero", "bddl_files", "**", f"{_TASK}.bddl"),
        recursive=True,
    )
    if not matches:
        pytest.skip(f"BDDL for {_TASK} not found")
    from libero.libero.envs.env_wrapper import ControlEnv

    env = ControlEnv(
        bddl_file_name=sorted(matches)[0],
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    env.seed(0)
    env.reset()
    return env


def test_real_scene_feasible_counts():
    try:
        env = _real_env()
    except pytest.skip.Exception:
        raise
    except Exception as exc:  # MuJoCo / GL / asset issues -> skip, don't fail
        pytest.skip(f"could not build real env: {exc}")

    try:
        prop_set = env.env.get_ltl_propositions()
        label_dict = env.env.get_ltl_label_dict()
    finally:
        env.close()

    unary = prop_set.get_propositions_by_category("unary_state")
    binary = prop_set.get_propositions_by_category("binary_relation")
    region = prop_set.get_propositions_by_category("region_containment")
    goal = prop_set.get_propositions_by_category("goal")

    assert len(unary) == 0
    # No instantiated object-level support surface in this scene (the table is the
    # arena anchor, not an object; there is no plate/stove/rack), so no on/in
    # binary APs are emitted. "On the table" is expressed via region sites and is
    # captured by the Level-3 region-containment APs instead.
    assert len(binary) == 0
    assert len(goal) == 2
    assert len(region) > 0
    # The goal objects' basket-containment region APs exist and are False at init
    # (the objects start on the table, not yet inside the basket). Labels may be
    # numpy bools, so test truthiness rather than identity.
    assert not label_dict["alphabet_soup_1_in_basket_1_contain_region"]
    assert not label_dict["tomato_sauce_1_in_basket_1_contain_region"]
