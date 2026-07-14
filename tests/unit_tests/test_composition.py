# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the KITCHEN-style AP-composition training path.

Pure logic only (no MuJoCo / GPU): the ordered-subgoal tracker reward and the
composition sampler. Skipped gracefully if the LIBERO composition data is absent.
"""

import numpy as np
import pytest

from rlinf.envs.libero.libero_composition_env import (
    advance_ordered_subgoals,
    parse_goal_subgoals,
    render_subgoal_ap,
    render_subgoal_nl,
)


# --------------------------------------------------------------------------- #
# AP-format prompt rendering (no natural language)
# --------------------------------------------------------------------------- #
def test_render_predicate_move_on():
    prim = {"kind": "move", "obj": "bowl_1", "target": "cabinet_1_top_side"}
    assert (
        render_subgoal_ap("on_bowl_1_cabinet_1_top_side", prim)
        == "on(bowl_1, cabinet_1_top_side)"
    )


def test_render_predicate_move_in():
    prim = {"kind": "move", "obj": "bowl_1", "target": "cabinet_1_bottom_region"}
    assert (
        render_subgoal_ap("in_bowl_1_cabinet_1_bottom_region", prim)
        == "in(bowl_1, cabinet_1_bottom_region)"
    )


def test_render_predicate_open_close():
    assert (
        render_subgoal_ap("open_cabinet_1_top_region", {"kind": "open", "target": "cabinet_1_top_region"})
        == "open(cabinet_1_top_region)"
    )


def test_render_raw_format():
    prim = {"kind": "move", "obj": "bowl_1", "target": "cabinet_1_top_side"}
    name = "on_bowl_1_cabinet_1_top_side"
    assert render_subgoal_ap(name, prim, fmt="raw") == name


def test_render_no_natural_language():
    # AP rendering must never contain articles/verbs like "put"/"the".
    prim = {"kind": "move", "obj": "bowl_1", "target": "cabinet_1_top_side"}
    out = render_subgoal_ap("on_bowl_1_cabinet_1_top_side", prim).lower()
    assert "put" not in out and " the " not in out


# --------------------------------------------------------------------------- #
# NL-format prompt rendering (prompt_style="nl") — in-distribution for the SFT VLA
# --------------------------------------------------------------------------- #
def test_render_nl_move_in():
    prim = {"kind": "move", "obj": "akita_black_bowl_1", "target": "white_cabinet_1_bottom_region"}
    assert (
        render_subgoal_nl("in_akita_black_bowl_1_white_cabinet_1_bottom_region", prim)
        == "put the akita black bowl in the white cabinet bottom region"
    )


def test_render_nl_move_on():
    prim = {"kind": "move", "obj": "akita_black_bowl_1", "target": "white_cabinet_1_top_side"}
    assert (
        render_subgoal_nl("on_akita_black_bowl_1_white_cabinet_1_top_side", prim)
        == "put the akita black bowl on the white cabinet top side"
    )


def test_render_nl_open_close():
    assert (
        render_subgoal_nl("open_white_cabinet_1_top_region", {"kind": "open", "target": "white_cabinet_1_top_region"})
        == "open the white cabinet top region"
    )
    assert (
        render_subgoal_nl("close_white_cabinet_1_bottom_region", {"kind": "close", "target": "white_cabinet_1_bottom_region"})
        == "close the white cabinet bottom region"
    )


def test_render_nl_is_natural_language():
    # NL rendering must NOT contain predicate() syntax, underscores, or instance ids.
    prim = {"kind": "move", "obj": "akita_black_bowl_1", "target": "white_cabinet_1_bottom_region"}
    out = render_subgoal_nl("in_akita_black_bowl_1_white_cabinet_1_bottom_region", prim)
    assert "(" not in out and "_" not in out and " 1 " not in out
    assert out.startswith("put the ")


# --------------------------------------------------------------------------- #
# BDDL :goal parsing (task_goals eval mode)
# --------------------------------------------------------------------------- #
def test_parse_goal_subgoals_and_block(tmp_path):
    bddl = tmp_path / "t.bddl"
    bddl.write_text(
        "(define (problem x)\n"
        "  (:goal\n"
        "    (And (Close white_cabinet_1_bottom_region) "
        "(Open white_cabinet_1_top_region))\n"
        "  )\n)\n"
    )
    sub, prims = parse_goal_subgoals(str(bddl))
    assert sub == [
        "close_white_cabinet_1_bottom_region",
        "open_white_cabinet_1_top_region",
    ]
    assert prims[0]["kind"] == "close" and prims[1]["kind"] == "open"
    # rendered AP form matches the prompt format
    assert render_subgoal_ap(sub[0], prims[0]) == "close(white_cabinet_1_bottom_region)"


def test_parse_goal_subgoals_single(tmp_path):
    bddl = tmp_path / "t.bddl"
    bddl.write_text("(:goal\n    (And (On bowl_1 wine_rack_1_top_region))\n  )\n")
    sub, prims = parse_goal_subgoals(str(bddl))
    assert sub == ["on_bowl_1_wine_rack_1_top_region"]
    assert render_subgoal_ap(sub[0], prims[0]) == "on(bowl_1, wine_rack_1_top_region)"


# --------------------------------------------------------------------------- #
# Ordered-subgoal tracker
# --------------------------------------------------------------------------- #
def test_tracker_no_progress_when_unsatisfied():
    sub = ["a", "b", "c"]
    ptr, reach, acc = advance_ordered_subgoals(sub, 0, {"a": False, "b": False})
    assert ptr == 0 and reach == 0.0 and acc is False


def test_tracker_advances_in_order():
    # +1 per subgoal achieved this step
    sub = ["a", "b", "c"]
    ptr, reach, acc = advance_ordered_subgoals(sub, 0, {"a": True})
    assert ptr == 1 and reach == pytest.approx(1.0) and acc is False


def test_tracker_does_not_skip_out_of_order():
    # later subgoal true but current one false -> pointer must not jump
    sub = ["a", "b", "c"]
    ptr, reach, acc = advance_ordered_subgoals(sub, 0, {"a": False, "c": True})
    assert ptr == 0 and reach == 0.0 and acc is False


def test_tracker_multi_advance_single_step():
    # two subgoals completed in one step -> +2
    sub = ["a", "b", "c"]
    ptr, reach, acc = advance_ordered_subgoals(sub, 0, {"a": True, "b": True})
    assert ptr == 2 and reach == pytest.approx(2.0) and acc is False


def test_tracker_final_subgoal_reward():
    # completing the last (only newly-done) subgoal this step -> +1, accepted
    sub = ["a", "b"]
    ptr, reach, acc = advance_ordered_subgoals(sub, 1, {"a": True, "b": True})
    assert ptr == 2 and reach == pytest.approx(1.0) and acc is True


def test_tracker_holds_on_missing_label():
    # no label -> no new completion this step -> 0 reward
    sub = ["a", "b"]
    ptr, reach, acc = advance_ordered_subgoals(sub, 1, None)
    assert ptr == 1 and reach == pytest.approx(0.0) and acc is False


def test_tracker_empty_subgoals():
    assert advance_ordered_subgoals([], 0, {"x": True}) == (0, 0.0, False)


# --------------------------------------------------------------------------- #
# CompositionSampler (requires LIBERO composition data; skip if absent)
# --------------------------------------------------------------------------- #
def _sampler_or_skip():
    try:
        from rlinf.envs.libero.composition_sampler import CompositionSampler

        return CompositionSampler("KITCHEN_SCENE4", max_depth=3, pool="all")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"KITCHEN_SCENE4 composition data unavailable: {e}")


def test_sampler_loads_ordered_subgoal_list():
    s = _sampler_or_skip()
    assert len(s) > 0
    c = s.sample(np.random.default_rng(0))
    assert isinstance(c.subgoals, list) and len(c.subgoals) >= 1


def test_sampler_subgoals_within_goal_alphabet():
    # identity precondition: every subgoal must be a goal_alphabet AP (so the
    # all-goals BDDL emits a matching ltl_label key).
    s = _sampler_or_skip()
    ga = set(s.goal_alphabet_names())
    for comp in s.compositions:
        for sg in comp.subgoals:
            assert sg in ga, f"subgoal {sg} not in goal_alphabet"


def test_sampler_covers_pool():
    s = _sampler_or_skip()
    rng = np.random.default_rng(0)
    seen = {tuple(s.sample(rng).subgoals) for _ in range(3000)}
    assert len(seen) == len(s)


def test_count_goal_predicates_for_max_goals_filter(tmp_path):
    # The depth-1 eval's max_goals filter keys off LiberoEnv._count_goal_predicates:
    # single-goal tasks -> 1, compositional (And ...) tasks -> N.
    from rlinf.envs.libero.libero_env import LiberoEnv

    single = tmp_path / "one.bddl"
    single.write_text("(:goal\n    (Open white_cabinet_1_top_region)\n  )\n")
    assert LiberoEnv._count_goal_predicates(str(single)) == 1

    two = tmp_path / "two.bddl"
    two.write_text(
        "(:goal\n    (And (Close white_cabinet_1_bottom_region) "
        "(Open white_cabinet_1_top_region))\n  )\n"
    )
    assert LiberoEnv._count_goal_predicates(str(two)) == 2
    # max_goals=1 keeps the single, drops the two-goal:
    assert LiberoEnv._count_goal_predicates(str(single)) <= 1
    assert LiberoEnv._count_goal_predicates(str(two)) > 1


def test_sampler_depth1_curriculum():
    # max_depth=1 must yield ONLY single-subgoal compositions, even though the shipped
    # file is compositions_up_to_3.json (file fallback + per-composition upper filter).
    try:
        from rlinf.envs.libero.composition_sampler import CompositionSampler

        s = CompositionSampler("KITCHEN_SCENE4", max_depth=1, pool="all")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"KITCHEN_SCENE4 composition data unavailable: {e}")
    assert len(s) > 0
    assert all(c.depth == 1 for c in s.compositions)
    rng = np.random.default_rng(0)
    assert all(len(s.sample(rng).subgoals) == 1 for _ in range(200))
