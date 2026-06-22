"""Task LTL specification selection and auto-generation for LIBERO-Max.

The auto-generator turns LIBERO proposition sets into task formulas for three
evaluation categories:

* success: eventually satisfy all BDDL goal propositions
* safety: globally avoid every ``safety_violation`` proposition
* ordering: use registry specs or conservative open/turnon -> goal -> close
  heuristics where the BDDL goals expose an order signal

Compilation of the resolved LTL formula to a runtime LDBA goes through
:func:`build_ldba` (Rabinizer + cached HOA fallback).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence

from .builder import LDBABuildError, build_ldba
from .monitor import LTLMonitor
from .temporal_monitor import is_atomic_token, tokenize


@dataclass
class TaskLTLSpec:
    task_id: str
    description: str
    formula: str
    resolved_formula: str
    proposition_aliases: Dict[str, str] = field(default_factory=dict)
    source: str = "registry"

    def as_dict(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "formula": self.formula,
            "resolved_formula": self.resolved_formula,
            "proposition_aliases": dict(self.proposition_aliases),
            "source": self.source,
        }


TASK_LTL_SPECS: Dict[str, Dict[str, object]] = {
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it": {
        "description": "Turn on the stove before eventually placing the moka pot on it.",
        "formula": "F(stove_on & F(moka_pot_on_stove))",
        "proposition_aliases": {
            "stove_on": "turnon_flat_stove_1",
            "moka_pot_on_stove": "on_moka_pot_1_flat_stove_1_cook_region",
        },
    },
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it": {
        "description": "Put the bowl in the bottom drawer before eventually closing the drawer.",
        "formula": "F(bowl_in_drawer & F(drawer_closed))",
        "proposition_aliases": {
            "bowl_in_drawer": "in_akita_black_bowl_1_white_cabinet_1_bottom_region",
            "drawer_closed": "close_white_cabinet_1_bottom_region",
        },
    },
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it": {
        "description": "Place the mug in the microwave before eventually closing the microwave door.",
        "formula": "F(mug_in_microwave & F(microwave_closed))",
        "proposition_aliases": {
            "mug_in_microwave": "in_white_yellow_mug_1_microwave_1_heating_region",
            "microwave_closed": "close_microwave_1",
        },
    },
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": {
        # BDDL goal is (On pot1)(On pot2)(Turnon stove): the stove must be turned
        # on too. Turn on first, then eventually both pots on it.
        "description": "Turn on the stove, then eventually both moka pots on it.",
        "formula": "F(stove_on & F(left_moka_pot_on_stove & right_moka_pot_on_stove))",
        "proposition_aliases": {
            "stove_on": "turnon_flat_stove_1",
            "left_moka_pot_on_stove": "on_moka_pot_1_flat_stove_1_cook_region",
            "right_moka_pot_on_stove": "on_moka_pot_2_flat_stove_1_cook_region",
        },
    },
}

# Backward-compatible name from the previous lightweight monitor module.
TASK_SPECS = TASK_LTL_SPECS

LTL_SPEC_MODE_ENV = "LIBERO_LTL_SPEC_MODE"
DEFAULT_LTL_SPEC_MODE = "goal_only"


# Cached HOA strings keyed by the *resolved* LTL formula (atoms = real BDDL
# proposition names, no aliases). Populated per task once Rabinizer settles.
# Sidecar ``.hoa`` files next to BDDL files take precedence; this dict is the
# in-memory fallback for environments that don't have BDDL paths in scope
# (unit tests, scripted analysis).
PREBUILT_HOA: Dict[str, str] = {
    # Goal-only Scene3 variant: F(turnon_flat_stove_1 & on_moka_pot_1_flat_stove_1_cook_region)
    # Accept once both goal predicates hold simultaneously.
    "F ( turnon_flat_stove_1 & on_moka_pot_1_flat_stove_1_cook_region )": (
        "HOA: v1\n"
        "States: 2\n"
        "Start: 0\n"
        'AP: 2 "turnon_flat_stove_1" "on_moka_pot_1_flat_stove_1_cook_region"\n'
        "acc-name: Buchi\n"
        "Acceptance: 1 Inf(0)\n"
        "--BODY--\n"
        "State: 0\n"
        "[0&1] 1 {0}\n"
        "[!0 | !1] 0\n"
        "State: 1\n"
        "[t] 1 {0}\n"
        "--END--"
    ),
    # KITCHEN_SCENE3: F(turnon_flat_stove_1 & F(on_moka_pot_1_flat_stove_1_cook_region))
    # Pre-generated reference output mirroring Rabinizer4 -p -d -e behavior:
    # - State 0: epsilon to state 1 (acceptance branch)
    # - State 1: wait for stove on, then transition to state 2
    # - State 2: wait for moka pot on stove, accepting self-loop on satisfaction
    "F ( turnon_flat_stove_1 & F ( on_moka_pot_1_flat_stove_1_cook_region ) )": (
        "HOA: v1\n"
        "States: 3\n"
        "Start: 0\n"
        'AP: 2 "turnon_flat_stove_1" "on_moka_pot_1_flat_stove_1_cook_region"\n'
        "acc-name: Buchi\n"
        "Acceptance: 1 Inf(0)\n"
        "--BODY--\n"
        "State: 0\n"
        "1\n"
        "State: 1\n"
        "[!0] 1\n"
        "[0] 2\n"
        "State: 2\n"
        "[1] 2 {0}\n"
        "[!1] 2\n"
        "--END--"
    ),
}


def _load_hoa_store() -> Dict[str, str]:
    """Load the prebuilt formula -> HOA cache (``hoa_store.json``), if present.

    Generated offline by ``scripts/prebuild_ltl_automata.py`` (Rabinizer). Merged
    over the hand-written fallback entries so ``build_ldba`` resolves every
    canonical task formula without invoking Rabinizer at runtime.
    """
    path = os.path.join(os.path.dirname(__file__), "hoa_store.json")
    try:
        with open(path) as fh:
            return json.load(fh)
    except (FileNotFoundError, ValueError, OSError):
        return {}


# Merge the prebuilt store as a FALLBACK: hand-written reference entries above
# stay authoritative (tests assert against them); the store fills in every other
# canonical formula so build_ldba never needs Rabinizer.
for _formula, _hoa in _load_hoa_store().items():
    PREBUILT_HOA.setdefault(_formula, _hoa)


def _task_key(task_name: str) -> str:
    return task_name.split("_demo")[0] if "_demo" in task_name else task_name


def _get_ltl_spec_mode() -> str:
    mode = os.environ.get(LTL_SPEC_MODE_ENV, DEFAULT_LTL_SPEC_MODE).strip().lower()
    if mode not in {"goal_only", "auto"}:
        return DEFAULT_LTL_SPEC_MODE
    return mode


def _task_id_candidates(
    task_id: str, bddl_file_name: Optional[str]
) -> Iterable[str]:
    if task_id:
        yield _task_key(task_id)
    if bddl_file_name:
        yield os.path.splitext(os.path.basename(bddl_file_name))[0]


def _replace_aliases(formula: str, aliases: Dict[str, str]) -> str:
    resolved_tokens = []
    for token in tokenize(formula):
        resolved_tokens.append(aliases.get(token, token))
    return " ".join(resolved_tokens)


def _formula_atoms(formula: str) -> set[str]:
    return {token for token in tokenize(formula) if is_atomic_token(token)}


def _validate_propositions(
    proposition_set: Any,
    names: Iterable[str],
    task_id: str,
) -> None:
    prop_dict = getattr(proposition_set, "prop_dict", {})
    missing = sorted({name for name in names if name not in prop_dict})
    if missing:
        raise KeyError(
            f"Task '{task_id}' references unknown propositions: {', '.join(missing)}"
        )


def _validate_formula_atoms(
    proposition_set: Any,
    formula: str,
    task_id: str,
) -> None:
    _validate_propositions(proposition_set, _formula_atoms(formula), task_id)


def _goal_mentions_target(goal_prop: Any, target: str) -> bool:
    """Return whether a goal predicate references a target object or fixture."""

    args = getattr(goal_prop, "args", None)
    if not isinstance(args, (tuple, list)):
        return False
    if len(args) == 2:
        return str(args[1]) == target
    if len(args) == 3:
        return target in str(args[2]) or target in str(args[1])
    return False


def _build_ordered_goal_formula(goal_props: Sequence[Any]) -> Optional[str]:
    """Build a conservative ordering-aware goal formula from BDDL goals."""

    if not goal_props:
        return None

    unary_predicates = {"open", "close", "turnon", "turnoff"}
    unary_goals = []
    for idx, prop in enumerate(goal_props):
        args = getattr(prop, "args", ())
        if not isinstance(args, (tuple, list)) or len(args) != 2:
            continue
        pred = str(args[0]).lower()
        if pred in unary_predicates:
            unary_goals.append((idx, pred, str(args[1])))

    if not unary_goals:
        return None

    gating_indices = set()
    for idx, _pred, target in unary_goals:
        for other_idx, other_prop in enumerate(goal_props):
            if other_idx == idx:
                continue
            if _goal_mentions_target(other_prop, target):
                gating_indices.add(idx)
                break

    if not gating_indices:
        return None

    start_stage = []
    end_stage = []
    for idx, pred, _target in unary_goals:
        if idx not in gating_indices:
            continue
        if pred in {"open", "turnon"}:
            start_stage.append(idx)
        elif pred in {"close", "turnoff"}:
            end_stage.append(idx)

    middle_stage = [
        idx
        for idx in range(len(goal_props))
        if idx not in set(start_stage) | set(end_stage)
    ]
    if not middle_stage:
        return None

    def conj(indices: Sequence[int]) -> str:
        return " & ".join(f"goal_{i + 1}" for i in indices)

    if start_stage and end_stage:
        return (
            f"F({conj(start_stage)} & "
            f"F({conj(middle_stage)} & F({conj(end_stage)})))"
        )
    if start_stage:
        return f"F({conj(start_stage)} & F({conj(middle_stage)}))"
    if end_stage:
        return f"F({conj(middle_stage)} & F({conj(end_stage)}))"
    return None


def _build_default_spec(task_id: str, proposition_set: Any) -> TaskLTLSpec:
    goal_props = proposition_set.get_propositions_by_category("goal")
    # Prefer a single aggregate safety atom ("any obstacle displaced") over the
    # per-object disjunction: G(!(s1|...|sN)) over ~13 atoms makes the LDBA
    # build blow up (2^N assignment enumeration). The aggregate keeps the
    # automaton small while still baking safety in; per-object displacement is
    # still reported by SafeLiberoBaseDomain.step().
    agg_props = proposition_set.get_propositions_by_category("safety_aggregate")
    safety_props = agg_props or proposition_set.get_propositions_by_category(
        "safety_violation"
    )
    aliases = {f"goal_{idx + 1}": prop.name for idx, prop in enumerate(goal_props)}
    aliases.update(
        {f"safety_{idx + 1}": prop.name for idx, prop in enumerate(safety_props)}
    )

    ordered_goal = _build_ordered_goal_formula(goal_props)
    if ordered_goal is not None:
        goal_formula = ordered_goal
        goal_desc = (
            "Auto-generated ordered spec from goal predicates "
            "(open/turnon -> goals -> close/turnoff)."
        )
        goal_source = "auto_order_goal"
    elif goal_props:
        body = " & ".join(f"goal_{idx + 1}" for idx in range(len(goal_props)))
        goal_formula = f"F({body})"
        goal_desc = "Auto-generated eventual conjunction of all BDDL goal predicates."
        goal_source = "auto_goal"
    else:
        goal_formula = "true"
        goal_desc = (
            "Auto-generated trivial spec because no goal propositions were found."
        )
        goal_source = "auto_goal"

    if goal_props and safety_props:
        safety_body = " | ".join(
            f"safety_{idx + 1}" for idx in range(len(safety_props))
        )
        raw_formula = f"G(!({safety_body})) & {goal_formula}"
        description = (
            "Auto-generated SafeLIBERO spec: avoid every safety violation globally "
            "while satisfying the task goal specification."
        )
        source = "auto_safe_goal"
    else:
        raw_formula = goal_formula
        description = goal_desc
        source = goal_source

    return TaskLTLSpec(
        task_id=task_id,
        description=description,
        formula=raw_formula,
        resolved_formula=_replace_aliases(raw_formula, aliases),
        proposition_aliases=aliases,
        source=source,
    )


def _build_goal_only_spec(task_id: str, proposition_set: Any) -> TaskLTLSpec:
    goal_props = proposition_set.get_propositions_by_category("goal")
    aliases = {f"goal_{idx + 1}": prop.name for idx, prop in enumerate(goal_props)}

    if goal_props:
        body = " & ".join(f"goal_{idx + 1}" for idx in range(len(goal_props)))
        raw_formula = f"F({body})"
        description = "Goal-only test spec: eventually satisfy all BDDL goal predicates."
    else:
        raw_formula = "true"
        description = "Goal-only test spec: no BDDL goal predicates were found."

    return TaskLTLSpec(
        task_id=task_id,
        description=description,
        formula=raw_formula,
        resolved_formula=_replace_aliases(raw_formula, aliases),
        proposition_aliases=aliases,
        source="goal_only",
    )


def get_task_ltl_spec(
    proposition_set: Any,
    task_id: str,
    bddl_file_name: Optional[str] = None,
) -> TaskLTLSpec:
    """Return a registry or auto-generated LTL spec for a LIBERO task."""

    if _get_ltl_spec_mode() == "goal_only":
        spec = _build_goal_only_spec(task_id=task_id, proposition_set=proposition_set)
        _validate_formula_atoms(proposition_set, spec.resolved_formula, task_id)
        return spec

    for candidate in _task_id_candidates(task_id, bddl_file_name):
        entry = TASK_LTL_SPECS.get(candidate)
        if entry is None:
            continue
        aliases = dict(entry.get("proposition_aliases", {}))
        _validate_propositions(proposition_set, aliases.values(), candidate)
        resolved_formula = _replace_aliases(str(entry["formula"]), aliases)
        _validate_formula_atoms(proposition_set, resolved_formula, candidate)
        return TaskLTLSpec(
            task_id=candidate,
            description=str(entry.get("description", "")),
            formula=str(entry["formula"]),
            resolved_formula=resolved_formula,
            proposition_aliases=aliases,
            source="registry",
        )

    spec = _build_default_spec(task_id=task_id, proposition_set=proposition_set)
    _validate_formula_atoms(proposition_set, spec.resolved_formula, task_id)
    return spec


def infer_goal_props_from_info(goal_desc: dict | None) -> set[str]:
    """Return goal proposition names from ``info['ltl_goal_desc']``."""

    if not isinstance(goal_desc, dict):
        return set()
    return {str(name) for name in goal_desc.keys()}


def infer_safety_props_from_label(label: dict | None) -> set[str]:
    """Infer safety APs from a label dictionary."""

    if not isinstance(label, dict):
        return set()
    return {str(name) for name in label.keys() if str(name).endswith("_displaced")}


def _strip_safety_formula(formula: str) -> tuple[str, set[str]]:
    """Peel a leading ``G(!(s1 | s2 | ...))`` and collect the safety atoms."""

    tokens = tokenize(formula)
    safety_props: set[str] = set()
    if not tokens or tokens[0] != "G":
        return formula, safety_props

    depth = 0
    end_idx: Optional[int] = None
    for idx, token in enumerate(tokens[1:], start=1):
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
            if depth == 0:
                end_idx = idx
                break
    if end_idx is None:
        return formula, safety_props

    safety_props = {
        token for token in tokens[1 : end_idx + 1] if is_atomic_token(token)
    }
    rest = tokens[end_idx + 1 :]
    if rest and rest[0] == "&":
        rest = rest[1:]
    return " ".join(rest) if rest else "true", safety_props


def _split_goal_safety(resolved_formula: str) -> tuple[set[str], set[str]]:
    """Return ``(goal_atoms, safety_atoms)`` for the resolved LTL formula."""

    _, safety_atoms = _strip_safety_formula(resolved_formula)
    all_atoms = _formula_atoms(resolved_formula)
    return all_atoms - safety_atoms, safety_atoms


def _monitor_from_spec(
    spec: TaskLTLSpec,
    *,
    bddl_file_name: Optional[str] = None,
) -> Optional[LTLMonitor]:
    formula = spec.resolved_formula or "true"
    if formula.strip() == "true":
        return None
    goal_props, safety_props = _split_goal_safety(formula)
    propositions = goal_props | safety_props
    try:
        ldba = build_ldba(
            formula,
            propositions,
            bddl_file_name=bddl_file_name,
        )
    except LDBABuildError:
        raise
    return LTLMonitor(
        ldba,
        goal_props=goal_props,
        safety_props=safety_props,
        task_name=spec.task_id,
        task_spec=spec,
    )


def build_monitor_from_spec(
    spec: TaskLTLSpec | dict,
    *,
    bddl_file_name: Optional[str] = None,
) -> Optional[LTLMonitor]:
    """Build a runtime monitor from a task LTL specification."""

    if isinstance(spec, dict):
        spec = TaskLTLSpec(
            task_id=str(spec.get("task_id", "")),
            description=str(spec.get("description", "")),
            formula=str(spec.get("formula", "")),
            resolved_formula=str(spec.get("resolved_formula", "")),
            proposition_aliases=dict(spec.get("proposition_aliases", {})),
            source=str(spec.get("source", "")),
        )
    return _monitor_from_spec(spec, bddl_file_name=bddl_file_name)


def build_monitor_for_task(
    task_name: str,
    *,
    proposition_set: Any | None = None,
    bddl_file_name: str | None = None,
    goal_props: set[str] | None = None,
    safety_props: set[str] | None = None,
) -> Optional[LTLMonitor]:
    """Build a monitor for a task.

    ``proposition_set`` enables full registry/default spec generation. The
    ``goal_props`` / ``safety_props`` path is kept for lightweight callers
    that only have ``info`` dictionaries (e.g. ``LiberoEnv``).
    """

    key = _task_key(task_name)
    if proposition_set is not None:
        spec = get_task_ltl_spec(
            proposition_set,
            task_id=key,
            bddl_file_name=bddl_file_name,
        )
        return _monitor_from_spec(spec, bddl_file_name=bddl_file_name)

    if _get_ltl_spec_mode() == "goal_only":
        goals = set(goal_props or ())
        if not goals:
            return None
        formula = f"F({' & '.join(sorted(goals))})"
        spec = TaskLTLSpec(
            task_id=key,
            description="Runtime-generated goal-only test spec from info dictionaries.",
            formula=formula,
            resolved_formula=formula,
            source="runtime_goal_only",
        )
        return _monitor_from_spec(spec, bddl_file_name=bddl_file_name)

    entry = TASK_LTL_SPECS.get(key)
    if entry is not None:
        aliases = dict(entry.get("proposition_aliases", {}))
        spec = TaskLTLSpec(
            task_id=key,
            description=str(entry.get("description", "")),
            formula=str(entry["formula"]),
            resolved_formula=_replace_aliases(str(entry["formula"]), aliases),
            proposition_aliases=aliases,
            source="registry",
        )
        return _monitor_from_spec(spec, bddl_file_name=bddl_file_name)

    goals = set(goal_props or ())
    if not goals:
        return None
    safety = set(safety_props or ())
    body = " & ".join(sorted(goals))
    formula = f"F({body})"
    if safety:
        formula = f"G(!({' | '.join(sorted(safety))})) & {formula}"
    spec = TaskLTLSpec(
        task_id=key,
        description="Runtime-generated spec from info dictionaries.",
        formula=formula,
        resolved_formula=formula,
        source="runtime_info",
    )
    return _monitor_from_spec(spec, bddl_file_name=bddl_file_name)
