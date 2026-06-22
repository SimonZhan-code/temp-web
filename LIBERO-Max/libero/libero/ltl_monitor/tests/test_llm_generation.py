from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:  # noqa: E402
    from libero.libero.benchmark.family import get_libero_suite_spec
    from libero.libero.ltl_monitor import llm_generation
    from libero.libero.ltl_monitor.llm_generation import (
        build_generation_messages,
        build_task_generation_context,
        compose_generation_record,
        iter_suite_generation_contexts,
        validate_generated_constraints,
    )
except ImportError:  # noqa: E402
    from libero.benchmark.family import get_libero_suite_spec
    from libero.ltl_monitor import llm_generation
    from libero.ltl_monitor.llm_generation import (
        build_generation_messages,
        build_task_generation_context,
        compose_generation_record,
        iter_suite_generation_contexts,
        validate_generated_constraints,
    )


def _mock_parse_problem_scene3(_path: str) -> dict:
    return {
        "goal_state": [
            ("Turnon", "flat_stove_1"),
            ("On", "moka_pot_1", "flat_stove_1_cook_region"),
        ],
        "language_instruction": [
            "turn",
            "on",
            "the",
            "stove",
            "and",
            "put",
            "the",
            "moka",
            "pot",
            "on",
            "it",
        ],
        "obj_of_interest": ["moka_pot_1", "flat_stove_1"],
        "objects": {"moka_pot": ["moka_pot_1"]},
    }


def _mock_parse_problem_safe_object(_path: str) -> dict:
    return {
        "goal_state": [
            ("In", "bbq_sauce_1", "basket_1_inside_region"),
        ],
        "language_instruction": [
            "pick",
            "up",
            "the",
            "bbq",
            "sauce",
            "and",
            "place",
            "it",
            "in",
            "the",
            "basket",
        ],
        "obj_of_interest": ["bbq_sauce_1", "basket_1"],
        "objects": {
            "bbq_sauce": ["bbq_sauce_1"],
            "basket": ["basket_1"],
            "milk": ["milk_1"],
        },
    }


def test_build_task_generation_context_for_libero_10_scene3(monkeypatch):
    monkeypatch.setattr(llm_generation, "robosuite_parse_problem", _mock_parse_problem_scene3)
    task_name = "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
    context = build_task_generation_context(
        suite_name="libero_10",
        task_id=task_name,
        bddl_file_path="/tmp/mock_scene3.bddl",
    )

    assert context.task_id == task_name
    assert context.taxonomy_level_1 == "baseline_id"
    assert context.goal_formula_local == (
        "F(turnon_flat_stove_1 & on_moka_pot_1_flat_stove_1_cook_region)"
    )
    assert [prop.name for prop in context.goal_atomic_propositions] == [
        "turnon_flat_stove_1",
        "on_moka_pot_1_flat_stove_1_cook_region",
    ]
    assert context.safety_atomic_propositions == ()


def test_build_task_generation_context_for_safelibero_adds_safety_props(monkeypatch):
    monkeypatch.setattr(
        llm_generation,
        "robosuite_parse_problem",
        _mock_parse_problem_safe_object,
    )
    task_name = "pick_up_the_bbq_sauce_and_place_it_in_the_basket"
    context = build_task_generation_context(
        suite_name="safelibero_object",
        task_id=task_name,
        bddl_file_path="/tmp/mock_safe_object.bddl",
    )

    assert context.taxonomy_level_1 == "constraint_safety"
    assert context.safety_atomic_propositions
    assert all(prop.name.endswith("_displaced") for prop in context.safety_atomic_propositions)


def test_build_task_generation_context_strips_bddl_comments_for_safelibero():
    repo_root = Path(_REPO_ROOT)
    bddl_path = (
        repo_root
        / "LIBERO"
        / "libero"
        / "bddl_files"
        / "safelibero_spatial"
        / "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl"
    )
    context = build_task_generation_context(
        suite_name="safelibero_spatial",
        task_id="pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        bddl_file_path=str(bddl_path),
    )

    safety_names = {prop.name for prop in context.safety_atomic_propositions}
    assert "akita_black_bowl_1_displaced" in safety_names
    assert "moka_pot_obstacle_1_displaced" in safety_names
    assert "Obstacles_displaced" not in safety_names
    assert "box_base_displaced" not in safety_names
    assert all(";" not in name for name in safety_names)


def test_build_generation_messages_embeds_task_payload(monkeypatch):
    monkeypatch.setattr(llm_generation, "robosuite_parse_problem", _mock_parse_problem_scene3)
    task_name = "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
    context = build_task_generation_context(
        suite_name="libero_10",
        task_id=task_name,
        bddl_file_path="/tmp/mock_scene3.bddl",
    )
    messages = build_generation_messages(context)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert task_name in messages[1]["content"]
    assert "turnon_flat_stove_1" in messages[1]["content"]


def test_compose_generation_record_passes_full_formula_through(monkeypatch):
    monkeypatch.setattr(
        llm_generation,
        "robosuite_parse_problem",
        _mock_parse_problem_safe_object,
    )
    task_name = "pick_up_the_bbq_sauce_and_place_it_in_the_basket"
    context = build_task_generation_context(
        suite_name="safelibero_object",
        task_id=task_name,
        bddl_file_path="/tmp/mock_safe_object.bddl",
    )
    record = compose_generation_record(
        context,
        {
            "ltl_formula": (
                "G(!(milk_1_displaced)) & F(in_bbq_sauce_1_basket_1_inside_region)"
            ),
            "description": "Reach the goal while never displacing the milk.",
            "notes": ["example"],
        },
    )

    assert record["ltl_formula"] == (
        "G(!(milk_1_displaced)) & F(in_bbq_sauce_1_basket_1_inside_region)"
    )
    assert record["description"] == "Reach the goal while never displacing the milk."
    assert record["notes"] == ["example"]
    assert "all_atomic_propositions" in record


def test_generated_constraints_reject_unknown_atomic_proposition(monkeypatch):
    monkeypatch.setattr(llm_generation, "robosuite_parse_problem", _mock_parse_problem_scene3)
    task_name = "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
    context = build_task_generation_context(
        suite_name="libero_10",
        task_id=task_name,
        bddl_file_path="/tmp/mock_scene3.bddl",
    )

    with pytest.raises(ValueError, match="Unknown atomic proposition"):
        validate_generated_constraints(
            context,
            ltl_formula=(
                "G(!(chefmate_8_frypan_1_displaced)) & "
                "F(turnon_flat_stove_1 & F(on_moka_pot_1_flat_stove_1_cook_region))"
            ),
        )


def test_generated_constraints_reject_malformed_formula(monkeypatch):
    monkeypatch.setattr(llm_generation, "robosuite_parse_problem", _mock_parse_problem_scene3)
    task_name = "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
    context = build_task_generation_context(
        suite_name="libero_10",
        task_id=task_name,
        bddl_file_path="/tmp/mock_scene3.bddl",
    )

    with pytest.raises(ValueError, match="Unexpected end of formula|Expected token"):
        validate_generated_constraints(
            context,
            ltl_formula="F(turnon_flat_stove_1 & F(on_moka_pot_1_flat_stove_1_cook_region)",
        )


def _build_raises_unavailable(*_args, **_kwargs):
    raise llm_generation.LDBABuildError("Rabinizer unavailable in this test.")


def _build_returns_non_finite_ldba(*_args, **_kwargs):
    class _StubLDBA:
        def is_finite_specification(self) -> bool:
            return False

    return _StubLDBA()


def test_generated_constraints_warn_when_ldba_pipeline_unavailable(monkeypatch):
    """Without Rabinizer / cached HOA the semantic check skips with a warning."""

    monkeypatch.setattr(llm_generation, "robosuite_parse_problem", _mock_parse_problem_scene3)
    task_name = "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
    context = build_task_generation_context(
        suite_name="libero_10",
        task_id=task_name,
        bddl_file_path="/tmp/mock_scene3.bddl",
    )

    monkeypatch.setattr(llm_generation, "build_ldba", _build_raises_unavailable)

    with pytest.warns(RuntimeWarning, match="Skipping semantic LTL check"):
        validate_generated_constraints(
            context,
            ltl_formula=(
                "F(turnon_flat_stove_1 & F(on_moka_pot_1_flat_stove_1_cook_region))"
            ),
        )


def test_generated_constraints_reject_non_finite_specification(monkeypatch):
    """When build_ldba succeeds but the spec isn't finite, validation fails."""

    monkeypatch.setattr(llm_generation, "robosuite_parse_problem", _mock_parse_problem_scene3)
    task_name = "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
    context = build_task_generation_context(
        suite_name="libero_10",
        task_id=task_name,
        bddl_file_path="/tmp/mock_scene3.bddl",
    )

    monkeypatch.setattr(llm_generation, "build_ldba", _build_returns_non_finite_ldba)

    with pytest.raises(ValueError, match="finite specification"):
        validate_generated_constraints(
            context,
            ltl_formula=(
                "F(turnon_flat_stove_1 & F(on_moka_pot_1_flat_stove_1_cook_region))"
            ),
        )


def test_iter_suite_generation_contexts_returns_all_suite_tasks(monkeypatch):
    monkeypatch.setattr(
        llm_generation,
        "robosuite_parse_problem",
        _mock_parse_problem_safe_object,
    )
    contexts = iter_suite_generation_contexts("safelibero_goal")
    suite_spec = get_libero_suite_spec("safelibero_goal")
    assert len(contexts) == len(suite_spec.tasks)
