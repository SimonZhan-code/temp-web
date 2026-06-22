"""Tests for the Rabinizer-based LTL monitor.

Adapted from Neuralsym-VLA/ltl_benchmark/tests/test_automaton.py and extended
with a LIBERO-Max integration smoke test that exercises the cached-HOA path
end-to-end without requiring Rabinizer or MuJoCo.

Run with:

    PYTHONPATH=. pytest libero/libero/ltl_monitor/tests/test_monitor.py -v
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import pytest

# Allow `python -m pytest <file>` from any cwd.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:  # noqa: E402
    from libero.libero.ltl_monitor.automata import LDBA
    from libero.libero.ltl_monitor.builder import build_ldba
    from libero.libero.ltl_monitor.hoa_parser import HOAParser
    from libero.libero.ltl_monitor.ldba_sequence import LDBASequence
    from libero.libero.ltl_monitor.logic import Assignment
    from libero.libero.ltl_monitor.logic.boolean_parser import parse
    from libero.libero.ltl_monitor.monitor import LTLMonitor
    from libero.libero.ltl_monitor.search import (
        ExhaustiveSearchSimple,
        NoPathsException,
    )
    from libero.libero.ltl_monitor.task_specs import (
        PREBUILT_HOA,
        TaskLTLSpec,
        build_monitor_from_spec,
        get_task_ltl_spec,
    )
except ImportError:  # noqa: E402
    from libero.ltl_monitor.automata import LDBA
    from libero.ltl_monitor.builder import build_ldba
    from libero.ltl_monitor.hoa_parser import HOAParser
    from libero.ltl_monitor.ldba_sequence import LDBASequence
    from libero.ltl_monitor.logic import Assignment
    from libero.ltl_monitor.logic.boolean_parser import parse
    from libero.ltl_monitor.monitor import LTLMonitor
    from libero.ltl_monitor.search import ExhaustiveSearchSimple, NoPathsException
    from libero.ltl_monitor.task_specs import (
        PREBUILT_HOA,
        TaskLTLSpec,
        build_monitor_from_spec,
        get_task_ltl_spec,
    )


SCENE3_FORMULA = (
    "F ( turnon_flat_stove_1 & F ( on_moka_pot_1_flat_stove_1_cook_region ) )"
)
SCENE3_PROPS = {
    "turnon_flat_stove_1",
    "on_moka_pot_1_flat_stove_1_cook_region",
}


# ---- Boolean parser ----


class TestBooleanParser:
    def test_simple_and(self):
        ast = parse("a & b")
        assert ast.eval({"a": True, "b": True}) is True
        assert ast.eval({"a": True, "b": False}) is False

    def test_simple_or(self):
        ast = parse("a | b")
        assert ast.eval({"a": False, "b": True}) is True
        assert ast.eval({"a": False, "b": False}) is False

    def test_negation(self):
        ast = parse("!a")
        assert ast.eval({"a": False}) is True
        assert ast.eval({"a": True}) is False

    def test_complex_expression(self):
        ast = parse("a & !b | c")
        assert ast.eval({"a": True, "b": False, "c": False}) is True
        assert ast.eval({"a": False, "b": False, "c": True}) is True
        assert ast.eval({"a": False, "b": True, "c": False}) is False


# ---- Assignment ----


class TestAssignment:
    def test_satisfies(self):
        a = Assignment({"x": True, "y": False})
        assert a.satisfies("x") is True
        assert a.satisfies("y") is False
        assert a.satisfies("x & !y") is True
        assert a.satisfies("t") is True

    def test_all_possible_assignments(self):
        all_a = Assignment.all_possible_assignments(("a", "b"))
        assert len(all_a) == 4
        true_counts = sorted(sum(1 for v in a.values() if v) for a in all_a)
        assert true_counts == [0, 1, 1, 2]

    def test_single_proposition(self):
        props = {"a", "b", "c"}
        a = Assignment.single_proposition("b", props)
        assert a["b"] is True
        assert a["a"] is False
        assert a["c"] is False

    def test_frozen_assignment(self):
        a = Assignment({"x": True, "y": False})
        fa = a.to_frozen()
        assert fa.get_true_propositions() == frozenset({"x"})
        assert hash(fa) == hash(fa)


# ---- HOA parsing + LDBA construction ----


def _build_scene3_ldba() -> LDBA:
    hoa = PREBUILT_HOA[SCENE3_FORMULA]
    ldba = HOAParser(SCENE3_FORMULA, hoa, SCENE3_PROPS).parse_hoa()
    ldba.complete_sink_state()
    ldba.compute_sccs()
    return ldba


class TestHOAParser:
    def test_parse_ap_line(self):
        props = HOAParser.parse_ap_line('2 "a" "b"', 0)
        assert props == ["a", "b"]

    def test_parse_full_hoa(self):
        ldba = _build_scene3_ldba()
        # 3 HOA states + 1 sink completion
        assert ldba.num_states == 4
        assert ldba.initial_state == 0


class TestLDBA:
    def test_epsilon_transition(self):
        ldba = _build_scene3_ldba()
        eps = [t for t in ldba.state_to_transitions[0] if t.is_epsilon()]
        assert len(eps) == 1
        assert eps[0].target == 1

    def test_get_next_states_epsilon(self):
        ldba = _build_scene3_ldba()
        next_states = ldba.get_next_states(0, set(), take_epsilon=True)
        assert next_states == [(1, False)]

    def test_get_next_states_normal(self):
        ldba = _build_scene3_ldba()
        next_states = ldba.get_next_states(1, {"turnon_flat_stove_1"})
        targets = {t for t, _ in next_states}
        assert 2 in targets

    def test_finite_specification(self):
        ldba = _build_scene3_ldba()
        assert ldba.is_finite_specification() is True


# ---- Reach-avoid search ----


class TestExhaustiveSearch:
    def test_search_finds_sequence(self):
        ldba = _build_scene3_ldba()
        search = ExhaustiveSearchSimple(SCENE3_PROPS, num_loops=1)
        seq = search(ldba, [1])
        assert len(seq) >= 1

    def test_search_reach_contains_target(self):
        ldba = _build_scene3_ldba()
        search = ExhaustiveSearchSimple(SCENE3_PROPS, num_loops=1)
        seq = search(ldba, [1])
        reach, _avoid = seq[0]
        reach_props: set[str] = set()
        for fa in reach:
            reach_props.update(fa.get_true_propositions())
        assert "turnon_flat_stove_1" in reach_props


# ---- LTLMonitor (LIBERO-Max integration) ----


@dataclass
class _FakeProp:
    name: str
    description: str
    args: tuple
    category: str


class _FakePropSet:
    """Minimal stand-in for ``PropositionSet`` used by ``get_task_ltl_spec``."""

    def __init__(self, goals: list[_FakeProp], safety: list[_FakeProp] = ()):
        self._goals = list(goals)
        self._safety = list(safety)
        self.prop_dict = {p.name: p for p in self._goals + list(safety)}

    def get_propositions_by_category(self, category: str):
        if category == "goal":
            return self._goals
        if category == "safety_violation":
            return self._safety
        return []


def _make_scene3_spec() -> TaskLTLSpec:
    prop_set = _FakePropSet(
        goals=[
            _FakeProp(
                "turnon_flat_stove_1",
                "turnon(flat_stove_1)",
                ("Turnon", "flat_stove_1"),
                "goal",
            ),
            _FakeProp(
                "on_moka_pot_1_flat_stove_1_cook_region",
                "on(moka_pot_1, flat_stove_1_cook_region)",
                ("On", "moka_pot_1", "flat_stove_1_cook_region"),
                "goal",
            ),
        ]
    )
    return get_task_ltl_spec(
        prop_set,
        task_id="KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    )


class TestLiberoMaxIntegration:
    def test_spec_resolution_uses_registry(self, monkeypatch):
        monkeypatch.setenv("LIBERO_LTL_SPEC_MODE", "auto")
        spec = _make_scene3_spec()
        assert spec.source == "registry"
        assert spec.resolved_formula == SCENE3_FORMULA

    def test_goal_only_mode_uses_simple_conjunction(self, monkeypatch):
        monkeypatch.delenv("LIBERO_LTL_SPEC_MODE", raising=False)
        spec = _make_scene3_spec()
        assert spec.source == "goal_only"
        assert spec.formula == "F(goal_1 & goal_2)"
        assert (
            spec.resolved_formula
            == "F ( turnon_flat_stove_1 & on_moka_pot_1_flat_stove_1_cook_region )"
        )

    def test_build_monitor_from_spec_returns_monitor(self, monkeypatch):
        monkeypatch.setenv("LIBERO_LTL_SPEC_MODE", "auto")
        spec = _make_scene3_spec()
        monitor = build_monitor_from_spec(spec)
        assert isinstance(monitor, LTLMonitor)
        assert monitor.task_spec is spec

    def test_step_emits_expected_info_keys(self, monkeypatch):
        monkeypatch.setenv("LIBERO_LTL_SPEC_MODE", "auto")
        spec = _make_scene3_spec()
        monitor = build_monitor_from_spec(spec)
        info = monitor.step({})
        for key in [
            "ltl_task_spec",
            "ltl_formula",
            "ltl_monitor_state",
            "ltl_state_changed",
            "ltl_accepted",
            "ltl_violated",
            "ltl_reach_props",
            "ltl_avoid_props",
            "ltl_reach_reward",
            "ltl_safety_cost",
            "ltl_reach_avoid_text",
            "ltl_accepting_visits",
        ]:
            assert key in info, f"missing info key: {key}"

    def test_acceptance_trace(self, monkeypatch):
        monkeypatch.setenv("LIBERO_LTL_SPEC_MODE", "auto")
        spec = _make_scene3_spec()
        monitor = build_monitor_from_spec(spec)

        # Step 1: nothing true -> not accepted
        info = monitor.step({})
        assert info["ltl_accepted"] is False
        assert info["ltl_violated"] is False
        # Reach should hint at turning the stove on
        assert "turnon_flat_stove_1" in info["ltl_reach_props"]

        # Step 2: stove turns on -> state changes, still not accepted
        info = monitor.step({"turnon_flat_stove_1": True})
        assert info["ltl_accepted"] is False
        assert info["ltl_state_changed"] is True
        # Reach should now hint at placing the moka pot
        assert "on_moka_pot_1_flat_stove_1_cook_region" in info["ltl_reach_props"]

        # Step 3: stove on AND moka pot placed -> accepting
        info = monitor.step(
            {
                "turnon_flat_stove_1": True,
                "on_moka_pot_1_flat_stove_1_cook_region": True,
            }
        )
        assert info["ltl_accepted"] is True
        assert info["ltl_violated"] is False
        assert info["ltl_reach_reward"] == pytest.approx(10.0)

    def test_goal_only_acceptance_trace(self, monkeypatch):
        monkeypatch.delenv("LIBERO_LTL_SPEC_MODE", raising=False)
        spec = _make_scene3_spec()
        monitor = build_monitor_from_spec(spec)

        info = monitor.step({})
        assert info["ltl_accepted"] is False
        assert info["ltl_violated"] is False
        assert "turnon_flat_stove_1" in info["ltl_reach_props"]
        assert "on_moka_pot_1_flat_stove_1_cook_region" in info["ltl_reach_props"]

        info = monitor.step({"turnon_flat_stove_1": True})
        assert info["ltl_accepted"] is False
        assert info["ltl_violated"] is False

        info = monitor.step(
            {
                "turnon_flat_stove_1": True,
                "on_moka_pot_1_flat_stove_1_cook_region": True,
            }
        )
        assert info["ltl_accepted"] is True
        assert info["ltl_violated"] is False


# ---- LDBA state trace (lower-level, no monitor wrapper) ----


class TestLDBAStateTrace:
    def test_kitchen_scene3_trace(self):
        ldba = _build_scene3_ldba()

        # Epsilon from 0 -> 1
        next_states = ldba.get_next_states(0, set(), take_epsilon=True)
        assert next_states == [(1, False)]

        # Stove not on -> stay at 1
        next_states = ldba.get_next_states(1, set())
        assert any(s == 1 for s, _ in next_states)

        # Stove on -> advance to 2
        next_states = ldba.get_next_states(1, {"turnon_flat_stove_1"})
        assert any(s == 2 for s, _ in next_states)

        # At 2, both true -> accepting transition
        next_states = ldba.get_next_states(
            2,
            {
                "turnon_flat_stove_1",
                "on_moka_pot_1_flat_stove_1_cook_region",
            },
        )
        accepting = [acc for s, acc in next_states if s == 2]
        assert any(accepting)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
