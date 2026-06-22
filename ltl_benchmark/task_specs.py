"""
Task specifications mapping LIBERO tasks to LTL formulas.

Each entry maps a task name to:
  - formula: LTL reachability specification (using Rabinizer syntax)
  - safety_formula: LTL safety specification (optional, G(!...) form)
  - description: Human-readable description
  - propositions: Set of atomic propositions used in the formula

Proposition naming follows LIBERO convention:
  Level 1 (unary state):    {obj_name}_{pred_name}         e.g., flat_stove_1_turnon
  Level 2 (binary relation): {obj1}_{pred}_{obj2}          e.g., moka_pot_1_on_flat_stove_1
  Level 3 (region):         {obj}_in_{region}              e.g., akita_black_bowl_1_in_wooden_cabinet_1_top_region
  Level 4 (goal predicate): {pred}_{obj1}_{obj2}           e.g., turnon_flat_stove_1
  Level 5 (safety):         {obj_name}_displaced            e.g., chefmate_8_frypan_1_displaced
"""

TASK_SPECS = {
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it": {
        "formula": "F(flat_stove_1_turn_on & F(moka_pot_1_on_flat_stove_1))",
        "safety_formula": "G(!chefmate_8_frypan_1_displaced)",
        "description": "Sequential: turn on stove, then place moka pot on it",
        "propositions": {
            "flat_stove_1_turn_on",
            "moka_pot_1_on_flat_stove_1",
            "chefmate_8_frypan_1_displaced",
        },
    },
}


# Pre-built HOA strings for tasks when Rabinizer is not available (Java 11+ required).
# These are the expected Rabinizer outputs for the formulas above.
PREBUILT_HOA = {
    # F(a & F(b)) where a=flat_stove_1_turn_on, b=moka_pot_1_on_flat_stove_1
    # 3 states: 0 (init) -> epsilon -> 1 (wait for stove) -> 2 (wait for moka, accepting loop)
    # Uses numeric AP indices as Rabinizer outputs: 0=flat_stove_1_turn_on, 1=moka_pot_1_on_flat_stove_1
    "F(flat_stove_1_turn_on & F(moka_pot_1_on_flat_stove_1))": """HOA: v1
States: 3
Start: 0
AP: 2 "flat_stove_1_turn_on" "moka_pot_1_on_flat_stove_1"
acc-name: Buchi
Acceptance: 1 Inf(0)
--BODY--
State: 0
1
State: 1
[!0] 1
[0] 2
State: 2
[1] 2 {0}
[!1] 2
--END--""",
}


def get_combined_formula(task_name: str) -> str:
    """Get the combined (reach + safety) LTL formula for a task."""
    spec = TASK_SPECS[task_name]
    formula = spec["formula"]
    safety = spec.get("safety_formula")
    if safety:
        return f"({formula}) & ({safety})"
    return formula


def get_task_propositions(task_name: str) -> set[str]:
    """Get the set of atomic propositions for a task."""
    return TASK_SPECS[task_name]["propositions"]
