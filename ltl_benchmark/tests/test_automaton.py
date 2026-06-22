"""Unit tests for LDBA construction, reach-avoid extraction, and env wrapper logic.
No MuJoCo/LIBERO dependency required.
"""
import pytest

from ltl_benchmark.logic import Assignment, FrozenAssignment
from ltl_benchmark.logic.boolean_parser import parse
from ltl_benchmark.automata.ldba import LDBA, LDBATransition, SCC
from ltl_benchmark.automata.ldba_sequence import LDBASequence
from ltl_benchmark.automata.hoa_parser import HOAParser
from ltl_benchmark.search import ExhaustiveSearchSimple, NoPathsException
from ltl_benchmark.env_wrapper import LDBAEnvWrapper, CurrentState, format_reach_text, format_avoid_text
from ltl_benchmark.task_specs import TASK_SPECS, PREBUILT_HOA


# ---- Boolean Parser Tests ----

class TestBooleanParser:
    def test_simple_and(self):
        ast = parse('a & b')
        assert ast.eval({'a': True, 'b': True}) is True
        assert ast.eval({'a': True, 'b': False}) is False

    def test_simple_or(self):
        ast = parse('a | b')
        assert ast.eval({'a': False, 'b': True}) is True
        assert ast.eval({'a': False, 'b': False}) is False

    def test_negation(self):
        ast = parse('!a')
        assert ast.eval({'a': False}) is True
        assert ast.eval({'a': True}) is False

    def test_complex_expression(self):
        ast = parse('a & !b | c')
        assert ast.eval({'a': True, 'b': False, 'c': False}) is True
        assert ast.eval({'a': False, 'b': False, 'c': True}) is True
        assert ast.eval({'a': False, 'b': True, 'c': False}) is False

    def test_implication(self):
        ast = parse('a => b')
        assert ast.eval({'a': True, 'b': True}) is True
        assert ast.eval({'a': True, 'b': False}) is False
        assert ast.eval({'a': False, 'b': False}) is True


# ---- Assignment Tests ----

class TestAssignment:
    def test_satisfies(self):
        a = Assignment({'x': True, 'y': False})
        assert a.satisfies('x') is True
        assert a.satisfies('y') is False
        assert a.satisfies('x & !y') is True
        assert a.satisfies('t') is True  # tautology

    def test_all_possible_assignments(self):
        all_a = Assignment.all_possible_assignments(('a', 'b'))
        assert len(all_a) == 4
        true_counts = [sum(1 for v in a.values() if v) for a in all_a]
        assert sorted(true_counts) == [0, 1, 1, 2]

    def test_single_proposition(self):
        props = {'a', 'b', 'c'}
        a = Assignment.single_proposition('b', props)
        assert a['b'] is True
        assert a['a'] is False
        assert a['c'] is False

    def test_frozen_assignment(self):
        a = Assignment({'x': True, 'y': False})
        fa = a.to_frozen()
        assert fa.get_true_propositions() == frozenset({'x'})
        assert hash(fa) == hash(fa)

    def test_where(self):
        props = {'a', 'b', 'c'}
        a = Assignment.where('a', 'c', propositions=props)
        assert a['a'] is True
        assert a['b'] is False
        assert a['c'] is True


# ---- LDBA Tests ----

class TestLDBA:
    def _build_simple_ldba(self) -> LDBA:
        """Build LDBA for F(a & F(b)) from HOA."""
        hoa = PREBUILT_HOA["F(flat_stove_1_turn_on & F(moka_pot_1_on_flat_stove_1))"]
        props = {'flat_stove_1_turn_on', 'moka_pot_1_on_flat_stove_1'}
        ldba = HOAParser('F(a & F(b))', hoa, props).parse_hoa()
        ldba.complete_sink_state()
        ldba.compute_sccs()
        return ldba

    def test_state_count(self):
        ldba = self._build_simple_ldba()
        # 3 HOA states + 1 sink = 4
        assert ldba.num_states == 4

    def test_initial_state(self):
        ldba = self._build_simple_ldba()
        assert ldba.initial_state == 0

    def test_epsilon_transition(self):
        ldba = self._build_simple_ldba()
        transitions = ldba.state_to_transitions[0]
        eps = [t for t in transitions if t.is_epsilon()]
        assert len(eps) == 1
        assert eps[0].target == 1

    def test_get_next_states_epsilon(self):
        ldba = self._build_simple_ldba()
        next_states = ldba.get_next_states(0, set(), take_epsilon=True)
        assert next_states == [(1, False)]

    def test_get_next_states_normal(self):
        ldba = self._build_simple_ldba()
        # From state 1, with stove turned on
        next_states = ldba.get_next_states(1, {'flat_stove_1_turn_on'})
        targets = {t for t, _ in next_states}
        assert 2 in targets

    def test_scc_structure(self):
        ldba = self._build_simple_ldba()
        # State 2 should be in an accepting bottom SCC (finite spec)
        scc2 = ldba.state_to_scc[2]
        assert scc2.accepting is True
        assert scc2.bottom is True
        # Sink state should be violating
        sink_scc = ldba.state_to_scc[ldba.sink_state]
        assert sink_scc.accepting is False
        assert sink_scc.bottom is True
        assert ldba.is_state_violating(ldba.sink_state) is True

    def test_finite_specification(self):
        ldba = self._build_simple_ldba()
        assert ldba.is_finite_specification() is True


class TestHOAParser:
    def test_parse_ap_line(self):
        props = HOAParser.parse_ap_line('2 "a" "b"', 0)
        assert props == ['a', 'b']

    def test_parse_full_hoa(self):
        hoa = PREBUILT_HOA["F(flat_stove_1_turn_on & F(moka_pot_1_on_flat_stove_1))"]
        props = {'flat_stove_1_turn_on', 'moka_pot_1_on_flat_stove_1'}
        ldba = HOAParser('test', hoa, props).parse_hoa()
        assert ldba.num_states == 3
        assert ldba.initial_state == 0
        assert ldba.num_transitions == 5


# ---- Search Tests ----

class TestExhaustiveSearch:
    def _build_ldba_and_search(self):
        hoa = PREBUILT_HOA["F(flat_stove_1_turn_on & F(moka_pot_1_on_flat_stove_1))"]
        props = {'flat_stove_1_turn_on', 'moka_pot_1_on_flat_stove_1'}
        ldba = HOAParser('test', hoa, props).parse_hoa()
        ldba.complete_sink_state()
        ldba.compute_sccs()
        search = ExhaustiveSearchSimple(props, num_loops=1)
        return ldba, search, props

    def test_search_finds_sequence(self):
        ldba, search, props = self._build_ldba_and_search()
        # Search from state 1 (after epsilon from 0)
        seq = search(ldba, [1])
        assert len(seq) >= 1

    def test_search_reach_contains_target(self):
        ldba, search, props = self._build_ldba_and_search()
        seq = search(ldba, [1])
        reach, avoid = seq[0]
        # The reach set should contain assignments where stove is on
        reach_true_props = set()
        for fa in reach:
            reach_true_props.update(fa.get_true_propositions())
        assert 'flat_stove_1_turn_on' in reach_true_props

    def test_search_with_safety(self):
        """Test with F(a) & G(!c) formula."""
        hoa = """HOA: v1
States: 4
Start: 0
AP: 2 "a" "c"
acc-name: Buchi
Acceptance: 1 Inf(0)
--BODY--
State: 0
1
State: 1
[!a & !c] 1
[a & !c] 2
[c] 3
State: 2
[!c] 2 {0}
[c] 3
State: 3
[!a & !c | !a & c | a & !c | a & c] 3
--END--"""
        props = {'a', 'c'}
        ldba = HOAParser('F(a) & G(!c)', hoa, props).parse_hoa()
        ldba.complete_sink_state()
        ldba.compute_sccs()
        search = ExhaustiveSearchSimple(props, num_loops=1)
        seq = search(ldba, [1])

        # Should have reach and loop parts
        assert len(seq) >= 1
        reach, avoid = seq[0]
        # Reach should include 'a', avoid should include 'c'
        reach_props = set()
        for fa in reach:
            reach_props.update(fa.get_true_propositions())
        avoid_props = set()
        for fa in avoid:
            avoid_props.update(fa.get_true_propositions())
        assert 'a' in reach_props
        assert 'c' in avoid_props


# ---- State Tracking Tests ----

class TestCurrentState:
    def test_successor(self):
        cs = CurrentState(state=0, accepting=False)
        s1 = cs.get_successor(1, True)
        assert s1.state == 1
        assert s1.accepting is True
        assert s1.num_accepting_visits == 1

    def test_multiple_accepting(self):
        cs = CurrentState(state=0, accepting=False)
        s1 = cs.get_successor(1, True)
        s2 = s1.get_successor(2, True)
        assert s2.num_accepting_visits == 2


# ---- Text Formatting Tests ----

class TestFormatting:
    def test_format_reach_text(self):
        props = {'a', 'b'}
        fa = Assignment.single_proposition('a', props).to_frozen()
        text = format_reach_text(frozenset({fa}), props)
        assert 'a' in text

    def test_format_avoid_text_empty(self):
        text = format_avoid_text(frozenset(), {'a', 'b'})
        assert text == "none"

    def test_format_reach_text_epsilon(self):
        text = format_reach_text(LDBASequence.EPSILON, {'a'})
        assert text == "none"


# ---- LDBA State Trace Test ----

class TestLDBAStateTrace:
    def test_kitchen_scene3_trace(self):
        """Simulate the KITCHEN_SCENE3 task through LDBA state transitions."""
        hoa = PREBUILT_HOA["F(flat_stove_1_turn_on & F(moka_pot_1_on_flat_stove_1))"]
        props = {'flat_stove_1_turn_on', 'moka_pot_1_on_flat_stove_1'}
        ldba = HOAParser('test', hoa, props).parse_hoa()
        ldba.complete_sink_state()
        ldba.compute_sccs()

        # Start at state 0, take epsilon to state 1
        state = 0
        next_states = ldba.get_next_states(state, set(), take_epsilon=True)
        state = next_states[0][0]
        assert state == 1

        # Nothing true yet -> stay at 1
        next_states = ldba.get_next_states(state, set())
        assert any(s == 1 for s, _ in next_states)

        # Turn on stove -> go to 2
        next_states = ldba.get_next_states(state, {'flat_stove_1_turn_on'})
        state = [s for s, _ in next_states if s == 2][0]
        assert state == 2

        # Moka pot not on stove yet -> stay at 2
        next_states = ldba.get_next_states(state, {'flat_stove_1_turn_on'})
        assert any(s == 2 for s, _ in next_states)

        # Place moka pot on stove -> accepting transition at 2
        next_states = ldba.get_next_states(state, {'flat_stove_1_turn_on', 'moka_pot_1_on_flat_stove_1'})
        accepting = [acc for s, acc in next_states if s == 2]
        assert any(accepting), "Should have accepting transition when both conditions met"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
