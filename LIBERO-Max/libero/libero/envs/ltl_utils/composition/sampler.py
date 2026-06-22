"""
Random-walk sampler over the rule-based transition system (M2).

A *walk* is a feasible-by-construction sequence of single primitives from the
scene's initial state; its ordered ``achieved_ap`` list is a composition of
goal-style subgoals. ``lift_to_ltl`` turns that list into an LTL formula:
ordered -> nested ``F(a1 & F(a2 & ... F(ak)))`` (the walk order encodes
precedence: open before place, place before close); order-free ->
``F(a1 & a2 & ... & ak)``.

Determinism: pass an explicit ``seed``; identical (model, config, seed) yields an
identical walk. (``random`` is used, not the disallowed ``Math.random``.)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from libero.libero.envs.ltl_utils.composition.transitions import (
    Primitive,
    SceneModel,
    WalkState,
    apply,
    enabled_primitives,
)


@dataclass
class SampleConfig:
    depth: int = 3                 # target number of subgoals (curriculum knob)
    ordered: bool = True           # nested F(...) vs order-free conjunction
    min_distinct_objects: int = 2  # genuine composition
    seed: int = 0
    require_cook_gate: bool = False


@dataclass
class Walk:
    primitives: List[Primitive]
    subgoal_aps: List[str]
    objects_touched: Set[str]
    end_state: WalkState


def _is_inverse(p: Primitive, last: Primitive) -> bool:
    """True if p immediately undoes last (open<->close region, on<->off appliance)."""
    if {p.kind, last.kind} == {"open", "close"} and p.target == last.target:
        return True
    if {p.kind, last.kind} == {"turn_on", "turn_off"} and p.obj == last.obj:
        return True
    return False


def random_walk(model: SceneModel, cfg: SampleConfig, rng: Optional[random.Random] = None) -> Walk:
    """Sample one feasible K-step walk from the scene's initial state."""
    rng = rng or random.Random(cfg.seed)
    state = model.init_state
    prims: List[Primitive] = []
    subgoals: List[str] = []
    touched: Set[str] = set()

    for _ in range(cfg.depth):
        enabled = enabled_primitives(model, state)
        # Avoid undoing the previous step and re-achieving an existing subgoal.
        if prims:
            enabled = [p for p in enabled if not _is_inverse(p, prims[-1])]
        enabled = [p for p in enabled if p.achieved_ap not in subgoals]
        if not enabled:
            break
        # Bias toward genuine composition: while we still need new objects, prefer
        # primitives that touch an object we haven't used yet.
        novel = [p for p in enabled if p.obj not in touched]
        pool = novel if (novel and len(touched) < cfg.min_distinct_objects) else enabled
        prim = pool[rng.randrange(len(pool))]
        state = apply(state, prim)
        prims.append(prim)
        subgoals.append(prim.achieved_ap)
        touched.add(prim.obj)

    return Walk(primitives=prims, subgoal_aps=subgoals, objects_touched=touched, end_state=state)


def enumerate_walks(
    model: SceneModel, max_depth: int, *, allow_inverse: bool = False
) -> List[Walk]:
    """Exhaustively enumerate every feasible ordered walk of length 1..max_depth.

    Feasible by construction (each step is an enabled primitive). Within a single
    walk a subgoal AP is never re-achieved, and (unless ``allow_inverse``) a step
    never immediately undoes the previous one. Different orderings of the same
    subgoal set are distinct walks (ordering matters for the LTL).
    """
    results: List[Walk] = []

    def dfs(state: WalkState, prims: List[Primitive], subgoals: List[str], touched: Set[str]):
        if prims:
            results.append(Walk(list(prims), list(subgoals), set(touched), state))
        if len(prims) >= max_depth:
            return
        for p in enabled_primitives(model, state):
            if p.achieved_ap in subgoals:
                continue
            if prims and not allow_inverse and _is_inverse(p, prims[-1]):
                continue
            dfs(apply(state, p), prims + [p], subgoals + [p.achieved_ap], touched | {p.obj})

    dfs(model.init_state, [], [], set())
    return results


def _nest(names: List[str], ordered: bool) -> str:
    if not names:
        return "true"
    if not ordered:
        return "F(" + " & ".join(names) + ")"
    expr = f"F({names[-1]})"
    for name in reversed(names[:-1]):
        expr = f"F({name} & {expr})"
    return expr


def lift_to_ltl(subgoal_aps: List[str], ordered: bool = True) -> Tuple[str, Dict[str, str], str]:
    """Return (alias_formula, aliases, resolved_formula) for a subgoal sequence."""
    aliases = {f"g{i + 1}": ap for i, ap in enumerate(subgoal_aps)}
    alias_names = list(aliases.keys())
    alias_formula = _nest(alias_names, ordered)
    resolved_formula = _nest(list(subgoal_aps), ordered)
    return alias_formula, aliases, resolved_formula
