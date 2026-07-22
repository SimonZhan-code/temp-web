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


def test_validate_identity_is_per_env():
    # Regression: task_goals eval runs different real tasks per env, so each env's
    # ltl_label only has its own task's goal keys. Validation must be PER-ENV, not
    # every-env's-subgoals vs one env's label (which crashed the depth-1 eval).
    import types
    from rlinf.envs.libero.libero_composition_env import LiberoCompositionEnv

    def stub():
        return types.SimpleNamespace(
            _identity_checked=False, scene_id="KITCHEN_SCENE4", _mode="task_goals",
            _subgoals=[["on_bowl_top"], ["in_wine_bottom"]], task_ids=[25, 26],
        )

    # Each env's subgoal is in its OWN label -> passes (this used to falsely fail).
    s = stub()
    LiberoCompositionEnv._validate_identity_once(
        s, [{"on_bowl_top": False}, {"in_wine_bottom": False}]
    )
    assert s._identity_checked is True

    # env1's subgoal genuinely absent from env1's label -> must raise.
    s = stub()
    with pytest.raises(RuntimeError):
        LiberoCompositionEnv._validate_identity_once(
            s, [{"on_bowl_top": False}, {"on_bowl_top": False}]
        )


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


# --------------------------------------------------------------------------- #
# Avoid-violation counting (reach-avoid shaping)
# --------------------------------------------------------------------------- #
def test_avoid_no_prev_label_no_violation():
    from rlinf.envs.libero.libero_composition_env import count_avoid_violations

    # right after reset there is no previous label -> never a violation
    assert count_avoid_violations(None, {"open_top": True}, ["open_top"], []) == 0
    assert count_avoid_violations({"open_top": True}, None, ["open_top"], []) == 0


def test_avoid_detects_uncommanded_toggle():
    from rlinf.envs.libero.libero_composition_env import count_avoid_violations

    # the memorized-tail case: drawer closes while no subgoal asked for it
    prev = {"close_bottom": False, "in_bowl_bottom": True}
    curr = {"close_bottom": True, "in_bowl_bottom": True}
    aps = ["close_bottom", "in_bowl_bottom"]
    assert count_avoid_violations(prev, curr, aps, exempt_aps=[]) == 1


def test_avoid_exempts_achieved_subgoal():
    from rlinf.envs.libero.libero_composition_env import count_avoid_violations

    # the commanded subgoal flipping True is the reward event, not a violation
    prev = {"in_bowl_bottom": False, "close_bottom": False}
    curr = {"in_bowl_bottom": True, "close_bottom": False}
    aps = ["in_bowl_bottom", "close_bottom"]
    assert (
        count_avoid_violations(prev, curr, aps, exempt_aps=["in_bowl_bottom"]) == 0
    )


def test_avoid_undoing_achieved_subgoal_counts():
    from rlinf.envs.libero.libero_composition_env import count_avoid_violations

    # undoing a PREVIOUSLY achieved subgoal (re-closing what we opened) is a violation:
    # it is not in this step's exempt set (only subgoals achieved THIS step are).
    prev = {"open_top": True}
    curr = {"open_top": False}
    assert count_avoid_violations(prev, curr, ["open_top"], exempt_aps=[]) == 1


def test_avoid_skips_aps_missing_from_either_label():
    from rlinf.envs.libero.libero_composition_env import count_avoid_violations

    # task_goals mode: per-task labels only contain that task's goal predicates
    prev = {"open_top": True}
    curr = {"open_top": True, "close_bottom": True}
    assert (
        count_avoid_violations(prev, curr, ["open_top", "close_bottom"], []) == 0
    )


def test_avoid_counts_multiple_toggles():
    from rlinf.envs.libero.libero_composition_env import count_avoid_violations

    prev = {"a": False, "b": True, "c": False}
    curr = {"a": True, "b": False, "c": False}
    assert count_avoid_violations(prev, curr, ["a", "b", "c"], []) == 2


# --------------------------------------------------------------------------- #
# chunk_step reach/cost channel collection (first-class reward channels)
# --------------------------------------------------------------------------- #
class _ChunkStubEnv:
    """Scripted env exercising the REAL LiberoEnv.chunk_step / _handle_auto_reset.

    Mimics LiberoCompositionEnv.step()'s contract: emits obs["ltl_reach_rewards"]
    per sim step; reset obs LACKS the ltl keys (like the real composition reset).
    """

    def __init__(self, complete_at_step, done_on_complete, num_envs=2):
        from rlinf.envs.libero.libero_env import LiberoEnv

        self._chunk_step = LiberoEnv.chunk_step
        self._auto_reset_impl = LiberoEnv._handle_auto_reset
        self.num_envs = num_envs
        self.auto_reset = True
        self.ignore_terminations = False
        self.use_fixed_reset_state_ids = False
        self.reset_state_ids = np.zeros(num_envs, dtype=int)

        class _Cfg(dict):
            __getattr__ = dict.get

        self.cfg = _Cfg(is_eval=False)
        self.t = 0
        self.complete_at_step = complete_at_step
        self.done_on_complete = done_on_complete

    def chunk_step(self, chunk_actions):
        return self._chunk_step(self, chunk_actions)

    def _handle_auto_reset(self, dones, obs, infos):
        return self._auto_reset_impl(self, dones, obs, infos)

    def update_reset_state_ids(self):
        pass

    def reset(self, env_idx=None, reset_state_ids=None):
        # like LiberoCompositionEnv.reset(): NO ltl_* keys in the reset obs
        return {"task_descriptions": ["sg"] * self.num_envs}, {}

    def step(self, actions=None, auto_reset=True):
        import torch

        self.t += 1
        reach = np.zeros(self.num_envs, dtype=np.float32)
        term = np.zeros(self.num_envs, dtype=bool)
        if self.t == self.complete_at_step:
            reach[0] = 1.0  # env0 achieves its subgoal at this sim step
            if self.done_on_complete:
                term[0] = True
        obs = {
            "task_descriptions": ["sg"] * self.num_envs,
            "ltl_reach_rewards": torch.from_numpy(reach),
            "ltl_cost_rewards": torch.full((self.num_envs,), -1.0),
        }
        return (
            obs,
            torch.from_numpy(term.astype(np.float32)),
            torch.from_numpy(term),
            torch.zeros(self.num_envs, dtype=torch.bool),
            {},
        )


def test_chunk_step_collects_midchunk_reach():
    import torch

    # a subgoal completed at sim step 3 of 10 must land in the [B, chunk] channel
    env = _ChunkStubEnv(complete_at_step=3, done_on_complete=False)
    _, _, _, _, infos = env.chunk_step(torch.zeros(2, 10, 7))
    reach = infos["chunk_reach_rewards"]
    assert reach.shape == (2, 10)
    assert reach[0].tolist() == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    assert reach[1].sum().item() == 0
    assert infos["chunk_cost_rewards"].shape == (2, 10)


def test_chunk_reach_survives_auto_reset():
    import torch

    # depth-1: completion terminates -> auto-reset replaces obs/infos mid-pipeline;
    # the collected channel must still carry the +1 (the old obs-piggyback lost it)
    env = _ChunkStubEnv(complete_at_step=3, done_on_complete=True)
    obs, _, _, _, infos = env.chunk_step(torch.zeros(2, 10, 7))
    assert "ltl_reach_rewards" not in obs  # reset obs (key gone) - the old dead path
    reach = infos["chunk_reach_rewards"]  # the new first-class channel survives
    assert reach.shape == (2, 10)
    assert reach[0, 2].item() == 1.0
    assert reach.sum().item() == 1.0


def test_envoutput_to_dict_carries_reach_channels():
    import torch

    try:
        from rlinf.data.io_struct import EnvOutput
    except ModuleNotFoundError as e:  # io_struct pulls in ray via the scheduler
        pytest.skip(f"io_struct deps unavailable in this env: {e}")

    out = EnvOutput(
        obs={"task_descriptions": ["sg"]},
        rewards=torch.zeros(2, 10),
        reach_rewards=torch.ones(2, 10),
        cost_rewards=torch.full((2, 10), -1.0),
    )
    d = out.to_dict()
    assert d["reach_rewards"].shape == (2, 10)
    assert d["cost_rewards"].shape == (2, 10)
    # envs that don't emit the channels -> None (actor falls back to task reward)
    d2 = EnvOutput(obs={"task_descriptions": ["sg"]}).to_dict()
    assert d2["reach_rewards"] is None and d2["cost_rewards"] is None


def test_preprocess_accepts_3d_reach_rewards():
    import torch

    from rlinf.algorithms.utils import preprocess_embodied_advantages_inputs

    # reach stacked as [n_chunk, B, chunk] (same 3-D shape as task rewards) must pass
    # the chunk_level preprocess; the per-chunk sum must preserve mid-chunk events.
    n_chunk, bsz, chunk = 4, 2, 10
    reach = torch.zeros(n_chunk, bsz, chunk)
    reach[1, 0, 2] = 1.0  # mid-chunk subgoal event
    out = preprocess_embodied_advantages_inputs(
        rewards=reach,
        dones=torch.zeros(n_chunk + 1, bsz, chunk),
        values=torch.zeros(n_chunk + 1, bsz, 1),
        loss_mask=None,
        loss_mask_sum=None,
        reward_type="chunk_level",
        adv_type="gae",
        task_type="embodied",
        gamma=0.99,
        gae_lambda=0.95,
    )
    flat = out["rewards"]  # [n_steps, bsz] with chunk_size collapsed to 1
    assert flat.shape == (n_chunk, bsz)
    assert flat[1, 0].item() == 1.0  # the mid-chunk +1 survives the chunk sum
    assert flat.sum().item() == 1.0


# --------------------------------------------------------------------------- #
# Canonical NL prompts (original LIBERO task language)
# --------------------------------------------------------------------------- #
def test_split_task_language_single_and_compound():
    from rlinf.envs.libero.libero_composition_env import split_task_language

    assert split_task_language("put the black bowl on top of the cabinet", 1) == [
        "put the black bowl on top of the cabinet"
    ]
    # compound close+open task splits by clause, aligned with goal order
    assert split_task_language(
        "close the bottom drawer of the cabinet and open the top drawer", 2
    ) == ["close the bottom drawer of the cabinet", "open the top drawer"]
    # count mismatch -> None (caller falls back to mechanical rendering)
    assert split_task_language("do the thing", 2) is None
    assert split_task_language("", 1) is None


def test_build_canonical_nl_map_kitchen4():
    from rlinf.envs.libero.libero_composition_env import build_canonical_nl_map

    entries = [
        (["close_white_cabinet_1_bottom_region", "open_white_cabinet_1_top_region"],
         "close the bottom drawer of the cabinet and open the top drawer"),
        (["in_akita_black_bowl_1_white_cabinet_1_bottom_region"],
         "put the black bowl in the bottom drawer of the cabinet"),
        (["on_akita_black_bowl_1_white_cabinet_1_top_side"],
         "put the black bowl on top of the cabinet"),
        (["bad_ap"], ""),  # empty language skipped
    ]
    m = build_canonical_nl_map(entries)
    assert m["open_white_cabinet_1_top_region"] == "open the top drawer"
    assert (
        m["in_akita_black_bowl_1_white_cabinet_1_bottom_region"]
        == "put the black bowl in the bottom drawer of the cabinet"
    )
    assert m["on_akita_black_bowl_1_white_cabinet_1_top_side"] == (
        "put the black bowl on top of the cabinet"
    )
    assert "bad_ap" not in m


def test_parse_task_language(tmp_path):
    from rlinf.envs.libero.libero_composition_env import parse_task_language

    b = tmp_path / "t.bddl"
    b.write_text("(define (problem X)\n  (:language put the black bowl on top of the cabinet)\n)")
    assert parse_task_language(str(b)) == "put the black bowl on top of the cabinet"
    assert parse_task_language(str(tmp_path / "missing.bddl")) == ""


# --------------------------------------------------------------------------- #
# Precondition model + chain feasibility + expansion
# --------------------------------------------------------------------------- #
_K4_ALPHABET = [
    "close_white_cabinet_1_bottom_region",
    "in_akita_black_bowl_1_white_cabinet_1_bottom_region",
    "in_wine_bottle_1_white_cabinet_1_bottom_region",
    "on_akita_black_bowl_1_white_cabinet_1_top_side",
    "on_wine_bottle_1_wine_rack_1_top_region",
    "open_white_cabinet_1_top_region",
]
# the audited real init: bottom open (close_bottom false), top closed (open_top false)
_K4_INIT = {ap: False for ap in _K4_ALPHABET}


def test_precondition_model_derivation():
    from rlinf.envs.libero.libero_composition_env import derive_precondition_model

    m = derive_precondition_model(_K4_ALPHABET)
    assert m["gated"] == {
        "white_cabinet_1_bottom_region",
        "white_cabinet_1_top_region",
    }
    # top_side (surface) and wine_rack are NOT gated
    assert "white_cabinet_1_top_side" not in m["gated"]
    assert m["open_ap"]["white_cabinet_1_top_region"] == "open_white_cabinet_1_top_region"
    assert m["close_ap"]["white_cabinet_1_bottom_region"] == "close_white_cabinet_1_bottom_region"


def test_chain_feasible_all_kitchen4_depth1():
    from rlinf.envs.libero.libero_composition_env import (
        check_chain_feasible,
        derive_precondition_model,
    )

    m = derive_precondition_model(_K4_ALPHABET)
    for ap in _K4_ALPHABET:  # audit result: every depth-1 chain atomic at real init
        ok, reason = check_chain_feasible([ap], _K4_INIT, m)
        assert ok, f"{ap}: {reason}"


def test_chain_infeasible_place_into_closed_gate():
    from rlinf.envs.libero.libero_composition_env import (
        check_chain_feasible,
        derive_precondition_model,
    )

    m = derive_precondition_model(_K4_ALPHABET)
    # hypothetical init with the bottom drawer CLOSED -> placement is a hidden composite
    init = dict(_K4_INIT, close_white_cabinet_1_bottom_region=True)
    ok, reason = check_chain_feasible(
        ["in_akita_black_bowl_1_white_cabinet_1_bottom_region"], init, m
    )
    assert not ok and "hidden composite" in reason
    # close-then-place is infeasible even from the real (open) init
    ok, reason = check_chain_feasible(
        ["close_white_cabinet_1_bottom_region",
         "in_akita_black_bowl_1_white_cabinet_1_bottom_region"], _K4_INIT, m)
    assert not ok


def test_chain_feasible_open_then_place_via_effect():
    from rlinf.envs.libero.libero_composition_env import (
        check_chain_feasible,
        derive_precondition_model,
    )

    alphabet = _K4_ALPHABET + ["in_akita_black_bowl_1_white_cabinet_1_top_region"]
    m = derive_precondition_model(alphabet)
    init = {ap: False for ap in alphabet}  # top closed
    # explicit composite: open the top drawer, then place into it -> feasible
    ok, reason = check_chain_feasible(
        ["open_white_cabinet_1_top_region",
         "in_akita_black_bowl_1_white_cabinet_1_top_region"], init, m)
    assert ok, reason
    # placement alone is the hidden composite
    ok, _ = check_chain_feasible(
        ["in_akita_black_bowl_1_white_cabinet_1_top_region"], init, m)
    assert not ok


def test_chain_degenerate_already_true():
    from rlinf.envs.libero.libero_composition_env import (
        check_chain_feasible,
        derive_precondition_model,
    )

    m = derive_precondition_model(_K4_ALPHABET)
    init = dict(_K4_INIT, open_white_cabinet_1_top_region=True)
    ok, reason = check_chain_feasible(["open_white_cabinet_1_top_region"], init, m)
    assert not ok and "degenerate" in reason


def test_expand_inserts_open_before_placement():
    from rlinf.envs.libero.libero_composition_env import (
        check_chain_feasible,
        derive_precondition_model,
        expand_chain_with_preconditions,
    )

    alphabet = _K4_ALPHABET + ["in_akita_black_bowl_1_white_cabinet_1_top_region"]
    m = derive_precondition_model(alphabet)
    init = {ap: False for ap in alphabet}  # top closed
    out = expand_chain_with_preconditions(
        ["in_akita_black_bowl_1_white_cabinet_1_top_region"],
        [{"kind": "move", "obj": "akita_black_bowl_1",
          "target": "white_cabinet_1_top_region"}],
        init, m)
    assert out is not None
    sg, pr, n = out
    assert n == 1
    assert sg == ["open_white_cabinet_1_top_region",
                  "in_akita_black_bowl_1_white_cabinet_1_top_region"]
    assert pr[0]["kind"] == "open"  # renderable by render_subgoal_nl/ap unchanged
    ok, reason = check_chain_feasible(sg, init, m)  # expanded chain is feasible
    assert ok, reason


def test_expand_fails_without_open_ap_in_alphabet():
    from rlinf.envs.libero.libero_composition_env import (
        derive_precondition_model,
        expand_chain_with_preconditions,
    )

    # KITCHEN_SCENE4's bottom drawer has close_ only -> the open precondition is
    # not expressible -> None (caller falls back to resampling)
    m = derive_precondition_model(_K4_ALPHABET)
    init = dict(_K4_INIT, close_white_cabinet_1_bottom_region=True)  # bottom closed
    out = expand_chain_with_preconditions(
        ["in_akita_black_bowl_1_white_cabinet_1_bottom_region"], [None], init, m)
    assert out is None
