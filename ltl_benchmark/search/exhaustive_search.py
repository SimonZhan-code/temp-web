from dataclasses import dataclass
from typing import Optional, List

from ltl_benchmark.automata.ldba import LDBA, LDBATransition
from ltl_benchmark.automata.ldba_sequence import LDBASequence
from ltl_benchmark.logic import Assignment, FrozenAssignment


class NoPathsException(Exception):
    pass


@dataclass
class Path:
    reach_avoid: list[tuple[LDBATransition, set[LDBATransition]]]
    loop_index: int

    def __len__(self):
        return len(self.reach_avoid)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return Path(self.reach_avoid[item], self.loop_index)
        return self.reach_avoid[item]

    def prepend(self, reach: LDBATransition, avoid: set[LDBATransition]) -> 'Path':
        return Path([(reach, avoid)] + self.reach_avoid, self.loop_index)

    def to_sequence(self, num_loops: int) -> LDBASequence:
        seq = [self.reach_avoid_to_assignments(r, a) for r, a in self.reach_avoid[:self.loop_index] if not r.is_epsilon()]
        loop = [self.reach_avoid_to_assignments(r, a) for r, a in self.reach_avoid[self.loop_index:] if not r.is_epsilon()]
        seq = seq + loop * num_loops
        return LDBASequence(seq)

    @staticmethod
    def reach_avoid_to_assignments(reach: LDBATransition, avoid: set[LDBATransition]) -> tuple[frozenset, frozenset]:
        avoid_sets = [a.valid_assignments for a in avoid]
        avoid_all = set() if not avoid_sets else set.union(*avoid_sets)
        if reach.is_epsilon():
            new_reach = LDBASequence.EPSILON
        else:
            new_reach = frozenset(reach.feasible_assignments)
            assert new_reach
        return new_reach, frozenset(avoid_all)


class ExhaustiveSearchSimple:
    """Simplified exhaustive search that selects the shortest valid sequence (no neural network)."""

    def __init__(self, propositions: set[str], num_loops: int = 1):
        self.propositions = propositions
        self.num_loops = num_loops

    def __call__(self, ldba: LDBA, ldba_states: List[int]) -> LDBASequence:
        seqs = self.all_sequences(ldba, ldba_states, self.num_loops)

        processed_seqs = []
        for seq in seqs:
            # An empty sequence means this path is already at acceptance (no reach
            # requirement left); skip it rather than indexing seq[0] (IndexError). If
            # every path is empty, processed_seqs stays empty -> NoPathsException below,
            # which callers treat as "spec satisfied / no current subgoal".
            if len(seq) == 0:
                continue
            reach_list, avoid = seq[0]
            suffix = seq[1:]

            # Eliminate avoid assignments that are subsets of another
            avoid_assignments = []
            avoid_sets = []
            for a, s in sorted([(a, a.get_true_propositions()) for a in avoid], key=lambda x: len(x[1])):
                if not any(other <= s for other in avoid_sets):
                    avoid_sets.append(s)
                    avoid_assignments.append(a)
            new_avoid = frozenset(avoid_assignments)

            for reach in reach_list:
                true_props = reach.get_true_propositions()
                if not any(avoid_set <= true_props for avoid_set in avoid_sets):
                    new_reach = frozenset(
                        [Assignment.single_proposition(p[0], self.propositions).to_frozen()
                         for p in reach if p[1]]
                    )
                    new_seq = [(new_reach, new_avoid)] + list(suffix)
                    processed_seqs.append(LDBASequence(new_seq))

        if not processed_seqs:
            raise NoPathsException()

        # Select shortest sequence (simple heuristic without neural value function)
        return min(processed_seqs, key=lambda s: len(s))

    def all_sequences(self, ldba: LDBA, ldba_states: List[int], num_loops: int = 1) -> List[LDBASequence]:
        num_loops = 0 if ldba.is_finite_specification() else num_loops
        return [
            path.to_sequence(num_loops)
            for ldba_state in ldba_states
            for path in self.dfs(ldba, ldba_state, [], {}, None, num_loops)
        ]

    def dfs(self, ldba: LDBA, state: int, current_path: list[LDBATransition],
            state_to_path_index: dict[int, int],
            accepting_transition: Optional[LDBATransition], num_loops: int = 1) -> list[Path]:
        state_to_path_index[state] = len(current_path)
        neg_transitions = set()
        paths = []
        for transition in ldba.state_to_transitions[state]:
            if not transition.is_feasible():
                continue
            scc = ldba.state_to_scc[transition.target]
            if scc.bottom and not scc.accepting:
                neg_transitions.add(transition)
            else:
                current_path.append(transition)
                stays_in_scc = scc == ldba.state_to_scc[transition.source]
                updated_accepting_transition = accepting_transition
                if transition.accepting and stays_in_scc:
                    updated_accepting_transition = transition
                if transition.target in state_to_path_index:  # found cycle
                    if updated_accepting_transition in current_path[state_to_path_index[transition.target]:]:
                        path = Path(reach_avoid=[], loop_index=state_to_path_index[transition.target])
                        future_paths = [path]
                    else:
                        current_path.pop()
                        if transition.source != transition.target:
                            neg_transitions.add(transition)
                        continue
                else:
                    future_paths = self.dfs(ldba, transition.target, current_path, state_to_path_index,
                                            updated_accepting_transition, num_loops)
                    if len(future_paths) == 0:
                        neg_transitions.add(transition)
                for fp in future_paths:
                    paths.append(fp.prepend(transition, set()))
                current_path.pop()

        del state_to_path_index[state]
        paths = self.prune_paths(paths)
        for path in paths:
            path[0][1].update(neg_transitions)
        return paths

    @staticmethod
    def prune_paths(paths: list[Path]) -> list[Path]:
        to_remove = set()
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if i in to_remove or j in to_remove:
                    continue
                if len(paths[i]) < len(paths[j]):
                    if ExhaustiveSearchSimple.check_path_contained(paths[j], paths[i]):
                        to_remove.add(j)
                elif len(paths[i]) > len(paths[j]):
                    if ExhaustiveSearchSimple.check_path_contained(paths[i], paths[j]):
                        to_remove.add(i)
                if i in to_remove:
                    break
        return [paths[i] for i in range(len(paths)) if i not in to_remove]

    @staticmethod
    def check_path_contained(path1: Path, path2: Path) -> bool:
        assert len(path2) < len(path1)
        p1 = [t[0].valid_assignments for t in path1]
        p2 = [t[0].valid_assignments for t in path2]
        acc_pos = 0
        for p in p1:
            if p.issubset(p2[acc_pos]):
                acc_pos += 1
                if acc_pos == len(p2):
                    return True
        return False
