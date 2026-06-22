import sys
import types

sys.modules.setdefault("torch", types.SimpleNamespace(load=lambda *args, **kwargs: None))

from libero.benchmark.family import (
    LIBERO_TAXONOMY_LEVEL_1,
    get_libero_suite_names_by_taxonomy_level_1,
    get_libero_suite_spec,
    get_libero_suites_grouped_by_taxonomy_level_1,
)


def test_suite_specs_expose_level_one_taxonomy():
    assert get_libero_suite_spec("libero_spatial").taxonomy_level_1 == "baseline_id"
    assert get_libero_suite_spec("safelibero_spatial").taxonomy_level_1 == "constraint_safety"
    assert get_libero_suite_spec("libero_10_r_base").taxonomy_level_1 == "baseline_id"
    assert (
        get_libero_suite_spec("libero_10_r_ood_visual").taxonomy_level_1
        == "distribution_shift_ood"
    )


def test_taxonomy_level_one_is_available_as_tag():
    assert "baseline_id" in get_libero_suite_spec("libero_spatial").tags
    assert "constraint_safety" in get_libero_suite_spec("safelibero_goal").tags


def test_get_suite_names_by_taxonomy_level_one_with_source_filter():
    names = get_libero_suite_names_by_taxonomy_level_1(
        "baseline_id",
        sources=("original", "safety", "libero_10_r"),
    )
    assert "libero_spatial" in names
    assert "libero_10_r_base" in names
    assert "safelibero_spatial" not in names
    assert "libero_10_r_ood" not in names


def test_group_suites_by_taxonomy_level_one_for_core_sources():
    grouped = get_libero_suites_grouped_by_taxonomy_level_1(
        sources=("original", "safety", "libero_10_r"),
    )
    assert set(grouped) == set(LIBERO_TAXONOMY_LEVEL_1)
    assert "libero_goal" in grouped["baseline_id"]
    assert "safelibero_long" in grouped["constraint_safety"]
    assert "libero_10_r_ood_visual_scene" in grouped["distribution_shift_ood"]
