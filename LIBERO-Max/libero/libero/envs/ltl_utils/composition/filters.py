"""
Meaningfulness filters for sampled walks (M2).

A walk is feasible by construction (it came from the transition system). These
filters keep only the *meaningful* ones for training supervision:

- ``min_distinct_objects``: require genuine composition (>= k distinct entities).
- mutex-consistency for order-free compositions: an order-free ``F(a & b & ...)``
  is only achievable if the subgoals are simultaneously satisfiable (no object in
  two places; no open & close of the same drawer; no on & off of one appliance).
  Ordered walks are exempt (the sequence resolves the conflict).
- held-out: optionally drop walks whose subgoal *set* equals an existing task's
  goal set, so the sampled set is a true compositional-generalization signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from libero.libero.envs.ltl_utils.composition.sampler import SampleConfig, Walk
from libero.libero.envs.ltl_utils.composition.transitions import SceneModel


@dataclass
class FilterResult:
    accepted: bool
    is_held_out: bool
    reason: str = ""


def num_distinct_objects(walk: Walk) -> int:
    return len(walk.objects_touched)


def matches_anchor(walk: Walk, model: SceneModel) -> bool:
    """True if the subgoal set equals a member task's goal set (in-distribution)."""
    return frozenset(walk.subgoal_aps) in model.anchor_goal_sets


def is_order_free_consistent(walk: Walk) -> bool:
    """Mutex-consistency check for treating the subgoals as a simultaneous conjunction."""
    move_objs = [p.obj for p in walk.primitives if p.kind == "move"]
    if len(move_objs) != len(set(move_objs)):
        return False  # same object placed in two locations
    opened = {p.target for p in walk.primitives if p.kind == "open"}
    closed = {p.target for p in walk.primitives if p.kind == "close"}
    if opened & closed:
        return False  # open & close the same drawer
    turned_on = {p.obj for p in walk.primitives if p.kind == "turn_on"}
    turned_off = {p.obj for p in walk.primitives if p.kind == "turn_off"}
    if turned_on & turned_off:
        return False
    return True


def evaluate_walk(
    walk: Walk,
    model: SceneModel,
    cfg: SampleConfig,
    *,
    exclude_anchor_matches: bool = True,
    min_subgoals: int = 2,
) -> FilterResult:
    """Apply all meaningfulness filters; return acceptance + held-out flag."""
    held_out = not matches_anchor(walk, model)

    if len(walk.subgoal_aps) < min_subgoals:
        return FilterResult(False, held_out, "too few subgoals")
    if num_distinct_objects(walk) < cfg.min_distinct_objects:
        return FilterResult(False, held_out, "not enough distinct objects")
    if not cfg.ordered and not is_order_free_consistent(walk):
        return FilterResult(False, held_out, "order-free conjunction is mutex-inconsistent")
    if exclude_anchor_matches and not held_out:
        return FilterResult(False, held_out, "matches an in-distribution anchor goal")

    return FilterResult(True, held_out, "ok")
