"""Runtime LTL monitor over an LDBA built by :mod:`builder`.

Replaces the earlier single-state template monitor with a multi-state runtime
ported from Neuralsym-VLA/ltl_benchmark/env_wrapper.py. Tracks every reachable
LDBA state simultaneously (Rabinizer LDBAs use epsilon transitions to
nondeterministically branch into acceptance modes), drains all available
epsilon transitions after each label step, and exposes the current
reach-avoid subgoal via :class:`LDBASequence` extracted by
:class:`ExhaustiveSearchSimple`.

The public surface is unchanged:

- ``LTLMonitor(ldba, *, goal_props, safety_props, task_name, task_spec)``
- ``monitor.reset()``
- ``monitor.step(label_dict_or_set) -> info_dict``

The returned info dict preserves every key the existing
``LiberoEnv._update_ltl_monitor_infos`` reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .automata import LDBA
from .ldba_sequence import LDBASequence
from .search import ExhaustiveSearchSimple, NoPathsException


@dataclass
class MonitorState:
    state: int
    accepting: bool = False
    num_accepting_visits: int = 0

    def get_successor(self, state: int, accepting: bool) -> "MonitorState":
        return MonitorState(
            state=state,
            accepting=accepting,
            num_accepting_visits=self.num_accepting_visits + int(accepting),
        )


def _format_assignments(assignments) -> str:
    if assignments is None or assignments == LDBASequence.EPSILON:
        return "none"
    parts: list[str] = []
    for fa in assignments:
        true_props = fa.get_true_propositions()
        if true_props:
            parts.append(" & ".join(sorted(true_props)))
    return " | ".join(parts) if parts else "none"


def _flatten_assignment_props(assignments) -> list[str]:
    if assignments is None or assignments == LDBASequence.EPSILON:
        return []
    out: set[str] = set()
    for fa in assignments:
        out.update(fa.get_true_propositions())
    return sorted(out)


class LTLMonitor:
    """Tracks LDBA progress from per-step proposition labels."""

    def __init__(
        self,
        ldba: LDBA,
        *,
        goal_props: Optional[set[str]] = None,
        safety_props: Optional[set[str]] = None,
        task_name: str = "",
        task_spec=None,
        num_loops: int = 1,
    ):
        self.ldba = ldba
        self.goal_props = set(goal_props or ())
        self.safety_props = set(safety_props or ())
        self.task_name = task_name
        self.task_spec = task_spec
        self.num_loops = num_loops
        self.terminate_on_acceptance = ldba.is_finite_specification()
        self.search = ExhaustiveSearchSimple(
            set(ldba.propositions), num_loops=num_loops
        )
        self.reset()

    def reset(self) -> None:
        assert self.ldba.initial_state is not None
        self.states: List[MonitorState] = [
            MonitorState(state=self.ldba.initial_state, accepting=False)
        ]
        self.violating_states: List[MonitorState] = []
        self._take_epsilon_transitions()
        self.accepted = any(s.accepting for s in self.states)
        self.violated = len(self.states) == 0
        self.steps = 0
        self.last_true_props: set[str] = set()
        self.sequence: Optional[LDBASequence] = self._extract_sequence()

    def step(self, label) -> dict:
        true_props = self._extract_true_props(label)
        # Restrict to APs the LDBA actually knows about.
        ldba_props = set(self.ldba.propositions)
        relevant_true_props = true_props & ldba_props

        self.last_true_props = true_props
        self.steps += 1

        if self.violated:
            return self.info(state_changed=False)

        prev_state_indices = [s.state for s in self.states]
        new_states: dict[tuple[int, bool], MonitorState] = {}
        for state in self.states:
            try:
                successors = self.ldba.get_next_states(
                    state.state, relevant_true_props
                )
            except ValueError:
                continue
            for target, accepting in successors:
                successor = state.get_successor(target, accepting)
                if self.ldba.is_state_violating(successor.state):
                    self.violating_states.append(successor)
                    continue
                key = (target, accepting)
                existing = new_states.get(key)
                if (
                    existing is None
                    or existing.num_accepting_visits < successor.num_accepting_visits
                ):
                    new_states[key] = successor
        self.states = [new_states[k] for k in sorted(new_states.keys())]

        self._take_epsilon_transitions()

        state_changed = prev_state_indices != [s.state for s in self.states]
        self.violated = len(self.states) == 0
        self.accepted = self.accepted or any(s.accepting for s in self.states)

        if state_changed or self.sequence is None:
            self.sequence = self._extract_sequence()

        return self.info(state_changed=state_changed)

    def _extract_true_props(self, label) -> set[str]:
        if isinstance(label, dict):
            return {name for name, value in label.items() if value}
        return set(label or ())

    def _take_epsilon_transitions(self) -> None:
        """Drain all available epsilon transitions from current states."""

        changed = True
        while changed:
            changed = False
            new_states: dict[tuple[int, bool], MonitorState] = {}
            for state in self.states:
                eps_transitions = [
                    t
                    for t in self.ldba.state_to_transitions.get(state.state, [])
                    if t.is_epsilon()
                ]
                if eps_transitions:
                    for t in eps_transitions:
                        successor = state.get_successor(t.target, t.accepting)
                        if self.ldba.is_state_violating(successor.state):
                            self.violating_states.append(successor)
                            continue
                        key = (t.target, t.accepting)
                        existing = new_states.get(key)
                        if (
                            existing is None
                            or existing.num_accepting_visits
                            < successor.num_accepting_visits
                        ):
                            new_states[key] = successor
                            changed = True
                else:
                    key = (state.state, state.accepting)
                    existing = new_states.get(key)
                    if (
                        existing is None
                        or existing.num_accepting_visits
                        < state.num_accepting_visits
                    ):
                        new_states[key] = state
            self.states = [new_states[k] for k in sorted(new_states.keys())]

    def _extract_sequence(self) -> Optional[LDBASequence]:
        if not self.states:
            return None
        # Once accepted on a finite spec, no further reach goal to extract.
        if any(s.accepting for s in self.states) and self.terminate_on_acceptance:
            return None
        try:
            return self.search(self.ldba, [s.state for s in self.states])
        except (NoPathsException, IndexError):
            # IndexError is the Neuralsym search's symptom when the current
            # state is already in the (singleton) accepting bottom SCC of a
            # finite spec — the loop part collapses to length 0. Fall back to
            # surfacing the accepting transitions' guards directly so the
            # agent still has a reach goal to chase.
            return self._fallback_accepting_sequence()

    def _fallback_accepting_sequence(self) -> Optional[LDBASequence]:
        accepting_assignments: set = set()
        for s in self.states:
            for t in self.ldba.state_to_transitions.get(s.state, []):
                if t.accepting and t.is_feasible():
                    accepting_assignments |= t.feasible_assignments
        if not accepting_assignments:
            return None
        return LDBASequence([(frozenset(accepting_assignments), frozenset())])

    def _current_reach_avoid(self):
        if self.sequence is None or len(self.sequence) == 0:
            return None, None
        reach, avoid = self.sequence[0]
        return reach, avoid

    def current_reach_props(self) -> list[str]:
        """Props required to make any forward progress from the current
        state(s). Computed structurally from the LDBA's outgoing non-self,
        non-sink transitions: intersection of true props across each
        transition's valid assignments (so a guard like ``[a]`` yields
        ``{a}`` even though both ``a&b`` and ``a&!b`` satisfy it), union
        across transitions.
        """

        if self.violated or not self.states:
            return []
        if any(s.accepting for s in self.states) and self.terminate_on_acceptance:
            return []
        reach: set[str] = set()
        for s in self.states:
            for t in self.ldba.state_to_transitions.get(s.state, []):
                if t.is_epsilon():
                    continue
                if t.target == s.state:  # self-loop (wait/partial)
                    continue
                if (
                    self.ldba.sink_state is not None
                    and t.target == self.ldba.sink_state
                ):
                    continue
                if not t.feasible_assignments:
                    continue
                required = None
                for fa in t.feasible_assignments:
                    props = fa.get_true_propositions()
                    required = props if required is None else required & props
                if required:
                    reach |= required
        # If we're at the (singleton) accepting bottom SCC of a finite spec
        # but haven't yet fired the accepting transition, the only outgoing
        # forward edge is the accepting self-loop — handle it as a reach goal.
        if not reach and self.terminate_on_acceptance:
            for s in self.states:
                for t in self.ldba.state_to_transitions.get(s.state, []):
                    if not t.accepting or not t.feasible_assignments:
                        continue
                    required = None
                    for fa in t.feasible_assignments:
                        props = fa.get_true_propositions()
                        required = props if required is None else required & props
                    if required:
                        reach |= required
        return sorted(reach)

    def current_avoid_props(self) -> list[str]:
        return sorted(self.safety_props)

    def reach_reward(self, state_changed: bool) -> float:
        if self.accepted:
            return 10.0
        if state_changed and not self.violated:
            return 1.0
        return 0.0

    def safety_cost(self) -> float:
        if self.violated:
            return 1.0
        if self.safety_props and (self.safety_props & self.last_true_props):
            return 1.0
        return 0.0

    def reach_avoid_text(self) -> str:
        reach_props = self.current_reach_props()
        avoid_props = self.current_avoid_props()
        reach_text = " & ".join(reach_props) if reach_props else "none"
        avoid_text = " | ".join(avoid_props) if avoid_props else "none"
        return f"Reach: {reach_text} | Avoid: {avoid_text}"

    def info(self, *, state_changed: bool) -> dict:
        accepting_visits = (
            max(s.num_accepting_visits for s in self.states)
            if self.states
            else (
                max(
                    s.num_accepting_visits for s in self.violating_states
                )
                if self.violating_states
                else 0
            )
        )
        return {
            "ltl_task_spec": self.task_spec.as_dict() if self.task_spec else None,
            "ltl_formula": self.ldba.formula,
            "ltl_monitor_state": [s.state for s in self.states],
            "ltl_state_changed": state_changed,
            "ltl_accepted": self.accepted,
            "ltl_violated": self.violated,
            "ltl_reach_props": self.current_reach_props(),
            "ltl_avoid_props": self.current_avoid_props(),
            "ltl_reach_reward": float(self.reach_reward(state_changed)),
            "ltl_safety_cost": float(self.safety_cost()),
            "ltl_reach_avoid_text": self.reach_avoid_text(),
            "ltl_accepting_visits": accepting_visits,
        }
