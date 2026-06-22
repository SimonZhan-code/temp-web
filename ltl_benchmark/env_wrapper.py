from dataclasses import dataclass
from typing import Optional, List

from ltl_benchmark.automata.ldba import LDBA
from ltl_benchmark.automata.ldba_sequence import LDBASequence
from ltl_benchmark.logic import Assignment, FrozenAssignment
from ltl_benchmark.search import ExhaustiveSearchSimple, NoPathsException


@dataclass
class CurrentState:
    state: int
    accepting: bool
    num_accepting_visits: int = 0

    def get_successor(self, state, accepting) -> 'CurrentState':
        return CurrentState(
            state,
            accepting,
            num_accepting_visits=self.num_accepting_visits + int(accepting),
        )


def format_reach_text(reach: frozenset, propositions: set[str]) -> str:
    """Format reach assignments as human-readable text for VLA conditioning."""
    if not reach or reach == LDBASequence.EPSILON:
        return "none"
    parts = []
    for fa in reach:
        true_props = fa.get_true_propositions()
        if true_props:
            parts.append(' & '.join(sorted(true_props)))
    return ' | '.join(parts) if parts else "none"


def format_avoid_text(avoid: frozenset, propositions: set[str]) -> str:
    """Format avoid assignments as human-readable text for VLA conditioning."""
    if not avoid:
        return "none"
    parts = []
    for fa in avoid:
        true_props = fa.get_true_propositions()
        if true_props:
            parts.append(' & '.join(sorted(true_props)))
    return ' | '.join(parts) if parts else "none"


class LDBAEnvWrapper:
    """Wraps a LIBERO environment with LDBA tracking for LTL reach-avoid subgoals."""

    def __init__(self, env, ldba: LDBA, num_loops: int = 1):
        """
        Args:
            env: A LIBERO BDDLBaseDomain or SafeLiberoBaseDomain environment.
            ldba: Pre-built LDBA (already pruned, completed, with SCCs computed).
            num_loops: Number of times to repeat the accepting loop in sequences.
        """
        self.env = env
        self.ldba = ldba
        self.propositions = set(ldba.propositions)
        self.terminate_on_acceptance = ldba.is_finite_specification()
        self.search = ExhaustiveSearchSimple(self.propositions, num_loops=num_loops)

        # Runtime state
        self.states: List[CurrentState] = []
        self.violating_states: List[CurrentState] = []
        self.sequence: Optional[LDBASequence] = None

    @property
    def ldba_state(self) -> Optional[int]:
        return self.states[0].state if self.states else None

    def reset(self):
        obs = self.env.reset()
        self.states = [CurrentState(state=self.ldba.initial_state, accepting=False)]
        self.violating_states = []

        # Take epsilon transitions from initial state if present
        self._take_epsilon_transitions()

        # Extract initial reach-avoid sequence
        try:
            self.sequence = self.search(self.ldba, [s.state for s in self.states])
        except NoPathsException:
            self.sequence = None

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Get true propositions from labeling function
        label_dict = info.get('ltl_label', {})
        true_props = {name for name, val in label_dict.items() if val}

        # Advance LDBA states
        prev_state_indices = [s.state for s in self.states]
        new_states = {}
        for state in self.states:
            for key in self.ldba.get_next_states(state.state, true_props):
                successor = state.get_successor(*key)
                if self.ldba.is_state_violating(successor.state):
                    self.violating_states.append(successor)
                elif key not in new_states or new_states[key].num_accepting_visits < successor.num_accepting_visits:
                    new_states[key] = successor
        self.states = [new_states[k] for k in sorted(new_states.keys())]

        # Take any epsilon transitions
        self._take_epsilon_transitions()

        # Detect state changes
        state_changed = prev_state_indices != [s.state for s in self.states]
        violated = len(self.states) == 0
        accepted = any(s.accepting for s in self.states) if self.states else False

        # Re-extract reach-avoid if state changed
        if state_changed or self.sequence is None:
            if self.states:
                try:
                    self.sequence = self.search(self.ldba, [s.state for s in self.states])
                except NoPathsException:
                    self.sequence = None

        # Extract current subgoal
        if self.sequence is not None and len(self.sequence) > 0:
            reach, avoid = self.sequence[0]
        else:
            reach, avoid = frozenset(), frozenset()

        # Format text for VLA conditioning
        reach_text = format_reach_text(reach, self.propositions)
        avoid_text = format_avoid_text(avoid, self.propositions)

        # Compute rewards
        reach_reward = 0.0
        if accepted:
            reach_reward = 10.0
        elif state_changed and not violated:
            reach_reward = 1.0
        safety_cost = 1.0 if violated else 0.0

        # Augment info
        info['ldba_state'] = [s.state for s in self.states]
        info['ldba_state_changed'] = state_changed
        info['ldba_violated'] = violated
        info['ldba_accepted'] = accepted
        info['reach_avoid_reach'] = reach
        info['reach_avoid_avoid'] = avoid
        info['reach_avoid_text'] = f"Reach: {reach_text}\nAvoid: {avoid_text}"
        info['reach_reward'] = reach_reward
        info['safety_cost'] = safety_cost
        if self.states or self.violating_states:
            info['num_accepting_visits'] = max(
                s.num_accepting_visits for s in (self.states if self.states else self.violating_states)
            )

        # Terminate on violation or acceptance (for finite specs)
        spec_done = violated or (accepted and self.terminate_on_acceptance)

        return obs, reward, done or spec_done, info

    def _take_epsilon_transitions(self):
        """Advance through any epsilon transitions from current states."""
        changed = True
        while changed:
            changed = False
            new_states = {}
            for state in self.states:
                eps_transitions = [t for t in self.ldba.state_to_transitions[state.state] if t.is_epsilon()]
                if eps_transitions:
                    for t in eps_transitions:
                        key = (t.target, t.accepting)
                        successor = state.get_successor(t.target, t.accepting)
                        if self.ldba.is_state_violating(successor.state):
                            self.violating_states.append(successor)
                        elif key not in new_states or new_states[key].num_accepting_visits < successor.num_accepting_visits:
                            new_states[key] = successor
                            changed = True
                else:
                    key = (state.state, state.accepting)
                    if key not in new_states or new_states[key].num_accepting_visits < state.num_accepting_visits:
                        new_states[key] = state
            self.states = [new_states[k] for k in sorted(new_states.keys())]
