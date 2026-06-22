"""Unified LIBERO-family suite registry."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

try:
    from libero.libero import get_libero_path
    from libero.libero.benchmark import Task, task_orders
    from libero.libero.benchmark.libero_suite_task_map import libero_task_map
    from libero.libero.benchmark.task_maps import (
        LIBERO_10_R_ALL,
        LIBERO_10_R_BASE,
        LIBERO_10_R_OOD,
        LIBERO_10_R_OOD_COMPOSITION,
        LIBERO_10_R_OOD_VISUAL,
        LIBERO_10_R_OOD_VISUAL_DISTRACTOR,
        LIBERO_10_R_OOD_VISUAL_SCENE,
        LIBERO_SUITE_MAX_STEPS,
        ORIGINAL_LIBERO_SUITES,
        ORIGINAL_LIBERO_TASK_MAP,
        SAFELIBERO_SUITES,
        SAFELIBERO_TASK_MAP,
    )
except ImportError:
    from libero import get_libero_path
    from libero.benchmark import Task, task_orders
    from libero.benchmark.libero_suite_task_map import libero_task_map
    from libero.benchmark.task_maps import (
        LIBERO_10_R_ALL,
        LIBERO_10_R_BASE,
        LIBERO_10_R_OOD,
        LIBERO_10_R_OOD_COMPOSITION,
        LIBERO_10_R_OOD_VISUAL,
        LIBERO_10_R_OOD_VISUAL_DISTRACTOR,
        LIBERO_10_R_OOD_VISUAL_SCENE,
        LIBERO_SUITE_MAX_STEPS,
        ORIGINAL_LIBERO_SUITES,
        ORIGINAL_LIBERO_TASK_MAP,
        SAFELIBERO_SUITES,
        SAFELIBERO_TASK_MAP,
    )

# Register safety object aliases on import
try:
    from libero.libero.envs.safety import _register_safelibero_object_aliases
    _register_safelibero_object_aliases()
except Exception:
    try:
        from libero.envs.safety import _register_safelibero_object_aliases
        _register_safelibero_object_aliases()
    except Exception:
        pass

# LIBERO-PRO task map (included in the unified package)
LIBERO_PRO_TASK_MAP = libero_task_map


def _grab_language_from_filename(filename: str) -> str:
    if filename[0].isupper():
        if "SCENE10" in filename:
            language = " ".join(filename[filename.find("SCENE") + 8 :].split("_"))
        else:
            language = " ".join(filename[filename.find("SCENE") + 7 :].split("_"))
    else:
        language = " ".join(filename.split("_"))
    end = language.find(".bddl")
    return language[:end] if end != -1 else language


def _libero_root(key: str) -> str:
    return get_libero_path(key)


def _default_init_file(task_name: str) -> str:
    return f"{task_name}.pruned_init"


def _safety_default_init_file(task_name: str) -> str:
    return f"{task_name}_LevelI.pruned_init"


def _safety_init_candidates(task_name: str, level: str = "I") -> tuple[str, ...]:
    normalized_level = str(level).upper()
    return (
        f"{task_name}_Level{normalized_level}.pruned_init",
        f"{task_name}_level_{normalized_level}.pruned_init",
    )


def _make_tasks(
    task_names: tuple[str, ...] | list[str],
    *,
    problem: str,
    problem_folder: str,
    init_file_fn: Callable[[str], str] = _default_init_file,
) -> tuple[Task, ...]:
    return tuple(
        Task(
            name=task_name,
            language=_grab_language_from_filename(f"{task_name}.bddl"),
            problem=problem,
            problem_folder=problem_folder,
            bddl_file=f"{task_name}.bddl",
            init_states_file=init_file_fn(task_name),
        )
        for task_name in task_names
    )


@dataclass(frozen=True)
class LiberoSuiteSpec:
    name: str
    family: str
    source: str
    taxonomy_level_1: str
    evaluation_tags: tuple[str, ...]
    tasks: tuple[Task, ...]
    bddl_root_fn: Callable[[str], str]
    init_root_fn: Callable[[str], str]
    max_steps: int
    uses_task_orders: bool = False
    supports_level_init: bool = False

    @property
    def variant(self) -> str:
        """Backward-compatible alias for older call sites."""
        return self.source

    @property
    def tags(self) -> tuple[str, ...]:
        """Evaluation-first taxonomy tags."""
        combined_tags = list(self.evaluation_tags)
        if self.taxonomy_level_1 not in combined_tags:
            combined_tags.append(self.taxonomy_level_1)
        if self.source not in combined_tags:
            combined_tags.append(self.source)
        return tuple(combined_tags)

    @property
    def bddl_root(self) -> str:
        return self.bddl_root_fn("bddl_files")

    @property
    def init_root(self) -> str:
        return self.init_root_fn("init_states")

    @property
    def eval_axis(self) -> str:
        """Top-level evaluation taxonomy label."""
        return self.taxonomy_level_1


class LiberoFamilySuite:
    """Benchmark-compatible suite wrapper."""

    def __init__(self, spec: LiberoSuiteSpec, task_order_index: int = 0):
        self.spec = spec
        self.name = spec.name
        self.family = spec.family
        self.source = spec.source
        self.taxonomy_level_1 = spec.taxonomy_level_1
        self.variant = spec.variant
        self.tags = spec.tags
        self.max_steps = spec.max_steps
        self.task_order_index = task_order_index
        self.task_embs = None
        self.tasks = self._ordered_tasks()
        self.n_tasks = len(self.tasks)

    def _ordered_tasks(self) -> list[Task]:
        tasks = list(self.spec.tasks)
        if self.spec.uses_task_orders:
            order = task_orders[self.task_order_index]
            tasks = [tasks[i] for i in order]
        return tasks

    def get_num_tasks(self) -> int:
        return self.n_tasks

    def get_task_names(self) -> list[str]:
        return [task.name for task in self.tasks]

    def get_task_problems(self) -> list[str]:
        return [task.problem for task in self.tasks]

    def get_task_bddl_files(self) -> list[str]:
        return [task.bddl_file for task in self.tasks]

    def get_task_bddl_file_path(self, i: int) -> str:
        task = self.tasks[i]
        return os.path.join(self.spec.bddl_root, task.problem_folder, task.bddl_file)

    def get_task_demonstration(self, i: int) -> str:
        task = self.tasks[i]
        return f"{task.problem_folder}/{task.name}_demo.hdf5"

    def get_task(self, i: int) -> Task:
        return self.tasks[i]

    def get_task_emb(self, i: int):
        return self.task_embs[i]

    def _get_task_init_state_path(self, i: int, level: str | None = None) -> str:
        task = self.tasks[i]
        if self.spec.supports_level_init:
            requested_level = str(level or "I").upper()
            if requested_level not in {"I", "II"}:
                raise ValueError(f"Unsupported safety level: {requested_level}")
            candidate_files = _safety_init_candidates(task.name, level=requested_level)
            for candidate_file in candidate_files:
                candidate_path = os.path.join(
                    self.spec.init_root,
                    task.problem_folder,
                    candidate_file,
                )
                if os.path.exists(candidate_path):
                    return candidate_path
            init_file = candidate_files[0]
        else:
            init_file = task.init_states_file
        return os.path.join(self.spec.init_root, task.problem_folder, init_file)

    def get_task_init_states(self, i: int):
        import torch

        return torch.load(self._get_task_init_state_path(i), weights_only=False)

    def get_task_init_states_by_level(self, i: int, level: str = "I"):
        import torch

        return torch.load(
            self._get_task_init_state_path(i, level=level),
            weights_only=False,
        )

    def set_task_embs(self, task_embs) -> None:
        self.task_embs = task_embs


def _infer_libero_pro_eval_tags(suite_name: str) -> tuple[str, ...]:
    if suite_name.endswith("_temp"):
        return ("ood", "combined")
    if suite_name.endswith("_env"):
        return ("ood", "environment")
    if suite_name.endswith("_lan") or suite_name.endswith("_semantic_ood"):
        return ("ood", "semantic")
    if suite_name.endswith("_swap") or "_with_diffpos_" in suite_name or "_with_rotated_" in suite_name:
        return ("ood", "position")
    if suite_name.endswith("_task") or suite_name.endswith("_relation_ood"):
        return ("ood", "task")
    if (
        suite_name.endswith("_object")
        or suite_name.endswith("_object_ood")
        or "_with_" in suite_name
    ):
        return ("ood", "visual", "visual_object")
    if (
        "trigger" in suite_name
        or "episode" in suite_name
        or suite_name in {"libero_mine", "libero_study_table"}
    ):
        return ("experimental",)
    return ("pro",)


LIBERO_TAXONOMY_LEVEL_1 = (
    "baseline_id",
    "constraint_safety",
    "distribution_shift_ood",
)


def _infer_taxonomy_level_1(
    *,
    source: str,
    suite_name: str,
    evaluation_tags: tuple[str, ...],
) -> str:
    normalized_source = str(source).lower()
    normalized_tags = {str(tag).lower() for tag in evaluation_tags}

    if normalized_source == "safety":
        return "constraint_safety"
    if normalized_source == "original":
        return "baseline_id"
    if normalized_source in {"pro", "libero_10_r"}:
        if suite_name == "libero_10_r_base" or {"id", "base"}.issubset(normalized_tags):
            return "baseline_id"
        return "distribution_shift_ood"
    return "distribution_shift_ood"


def _get_available_pro_suites() -> list[str]:
    """Return PRO suite names whose bddl_files and init_files dirs exist."""
    try:
        bddl_root = get_libero_path("bddl_files")
        init_root = get_libero_path("init_states")
    except Exception:
        return []

    available = []
    for suite_name in sorted(LIBERO_PRO_TASK_MAP):
        if suite_name in ORIGINAL_LIBERO_SUITES:
            continue
        bddl_dir = os.path.join(bddl_root, suite_name)
        init_dir = os.path.join(init_root, suite_name)
        if os.path.isdir(bddl_dir) and os.path.isdir(init_dir):
            available.append(suite_name)
    return available


def _build_registry() -> dict[str, LiberoSuiteSpec]:
    registry: dict[str, LiberoSuiteSpec] = {}

    # Original LIBERO suites
    for suite_name in ORIGINAL_LIBERO_SUITES:
        registry[suite_name] = LiberoSuiteSpec(
            name=suite_name,
            family="libero",
            source="original",
            taxonomy_level_1="baseline_id",
            evaluation_tags=("id", "base"),
            tasks=_make_tasks(
                ORIGINAL_LIBERO_TASK_MAP[suite_name],
                problem="Libero",
                problem_folder=suite_name,
            ),
            bddl_root_fn=_libero_root,
            init_root_fn=_libero_root,
            max_steps=LIBERO_SUITE_MAX_STEPS[suite_name],
            uses_task_orders=suite_name in {
                "libero_spatial",
                "libero_object",
                "libero_goal",
                "libero_10",
            },
        )

    # Aggregated libero_130
    aggregated_tasks = []
    seen_task_names: set[str] = set()
    for suite_name in ORIGINAL_LIBERO_SUITES:
        for task in registry[suite_name].tasks:
            if task.name in seen_task_names:
                continue
            seen_task_names.add(task.name)
            aggregated_tasks.append(task)
    registry["libero_130"] = LiberoSuiteSpec(
        name="libero_130",
        family="libero",
        source="original",
        taxonomy_level_1="baseline_id",
        evaluation_tags=("id", "base", "aggregated"),
        tasks=tuple(aggregated_tasks),
        bddl_root_fn=_libero_root,
        init_root_fn=_libero_root,
        max_steps=LIBERO_SUITE_MAX_STEPS["libero_130"],
    )

    # SafeLIBERO suites
    for suite_name in SAFELIBERO_SUITES:
        registry[suite_name] = LiberoSuiteSpec(
            name=suite_name,
            family="libero",
            source="safety",
            taxonomy_level_1="constraint_safety",
            evaluation_tags=("safety",),
            tasks=_make_tasks(
                SAFELIBERO_TASK_MAP[suite_name],
                problem="SafeLibero",
                problem_folder=suite_name,
                init_file_fn=_safety_default_init_file,
            ),
            bddl_root_fn=_libero_root,
            init_root_fn=_libero_root,
            max_steps=LIBERO_SUITE_MAX_STEPS[suite_name],
            supports_level_init=True,
        )

    # LIBERO-PRO suites (all data is now in the unified package)
    for suite_name in _get_available_pro_suites():
        if suite_name in registry:
            continue

        base_suite = next(
            (
                candidate
                for candidate in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
                if suite_name.startswith(f"{candidate}_")
            ),
            None,
        )

        task_names = LIBERO_PRO_TASK_MAP.get(suite_name)
        if not task_names:
            continue

        registry[suite_name] = LiberoSuiteSpec(
            name=suite_name,
            family="libero",
            source="pro",
            taxonomy_level_1=_infer_taxonomy_level_1(
                source="pro",
                suite_name=suite_name,
                evaluation_tags=_infer_libero_pro_eval_tags(suite_name),
            ),
            evaluation_tags=_infer_libero_pro_eval_tags(suite_name),
            tasks=_make_tasks(
                task_names,
                problem="Libero-Pro",
                problem_folder=suite_name,
            ),
            bddl_root_fn=_libero_root,
            init_root_fn=_libero_root,
            max_steps=LIBERO_SUITE_MAX_STEPS.get(
                base_suite or suite_name,
                LIBERO_SUITE_MAX_STEPS["libero_10"],
            ),
        )

    # LIBERO-10-R suites
    libero_10_r_suites = {
        "libero_10_r": (
            LIBERO_10_R_ALL,
            ("full", "id", "ood", "composition", "visual"),
            "libero_10_r",
        ),
        "libero_10_r_base": (
            LIBERO_10_R_BASE,
            ("id", "base"),
            "libero_10_r",
        ),
        "libero_10_r_ood": (
            LIBERO_10_R_OOD,
            ("ood",),
            "libero_10_r",
        ),
        "libero_10_r_ood_composition": (
            LIBERO_10_R_OOD_COMPOSITION,
            ("ood", "composition"),
            "libero_10_r",
        ),
        "libero_10_r_ood_visual": (
            LIBERO_10_R_OOD_VISUAL,
            ("ood", "visual"),
            "libero_10_r",
        ),
        "libero_10_r_ood_visual_scene": (
            LIBERO_10_R_OOD_VISUAL_SCENE,
            ("ood", "visual", "scene_shift"),
            "libero_10_r",
        ),
        "libero_10_r_ood_visual_distractor": (
            LIBERO_10_R_OOD_VISUAL_DISTRACTOR,
            ("ood", "visual", "distractor_shift"),
            "libero_10_r",
        ),
    }
    for suite_name, (task_names, evaluation_tags, problem_folder) in libero_10_r_suites.items():
        registry[suite_name] = LiberoSuiteSpec(
            name=suite_name,
            family="libero",
            source="libero_10_r",
            taxonomy_level_1=_infer_taxonomy_level_1(
                source="libero_10_r",
                suite_name=suite_name,
                evaluation_tags=evaluation_tags,
            ),
            evaluation_tags=evaluation_tags,
            tasks=_make_tasks(
                task_names,
                problem="Libero-10-R",
                problem_folder=problem_folder,
            ),
            bddl_root_fn=_libero_root,
            init_root_fn=_libero_root,
            max_steps=LIBERO_SUITE_MAX_STEPS[suite_name],
        )

    return registry


LIBERO_FAMILY_REGISTRY = _build_registry()

LIBERO_FAMILY_ALIASES = {
    "libero_10_r_full": "libero_10_r",
    "libero_10_r_all": "libero_10_r",
    "libero_10_r_comp": "libero_10_r_ood_composition",
    "libero_10_r_composition": "libero_10_r_ood_composition",
    "libero_10_r_visual_scene": "libero_10_r_ood_visual_scene",
    "libero_10_r_visual_distractor": "libero_10_r_ood_visual_distractor",
}


def resolve_libero_suite_name(name: str) -> str:
    canonical_name = str(name).lower()
    return LIBERO_FAMILY_ALIASES.get(canonical_name, canonical_name)


def get_libero_suite_spec(name: str) -> LiberoSuiteSpec:
    canonical_name = resolve_libero_suite_name(name)
    if canonical_name not in LIBERO_FAMILY_REGISTRY:
        available = ", ".join(sorted(LIBERO_FAMILY_REGISTRY))
        raise KeyError(f"Unknown LIBERO-family suite '{name}'. Available: {available}")
    return LIBERO_FAMILY_REGISTRY[canonical_name]


def get_libero_suite(name: str, task_order_index: int = 0) -> LiberoFamilySuite:
    return LiberoFamilySuite(get_libero_suite_spec(name), task_order_index=task_order_index)


def get_libero_suite_names(include_aliases: bool = False) -> list[str]:
    names = sorted(LIBERO_FAMILY_REGISTRY)
    if include_aliases:
        names.extend(sorted(LIBERO_FAMILY_ALIASES))
    return names


def _extend_suite_aliases(names: list[str]) -> list[str]:
    names.extend(
        alias
        for alias, canonical_name in sorted(LIBERO_FAMILY_ALIASES.items())
        if canonical_name in names
    )
    return names


def get_libero_suite_names_by_eval_tags(
    *tags: str, include_aliases: bool = False
) -> list[str]:
    wanted = {str(tag).lower() for tag in tags}
    names = [
        suite_name
        for suite_name, spec in sorted(LIBERO_FAMILY_REGISTRY.items())
        if wanted.issubset({tag.lower() for tag in spec.tags})
    ]
    if include_aliases:
        return _extend_suite_aliases(names)
    return names


def get_libero_suite_names_by_tags(*tags: str, include_aliases: bool = False) -> list[str]:
    return get_libero_suite_names_by_eval_tags(
        *tags,
        include_aliases=include_aliases,
    )


def get_libero_suite_names_by_source(
    *sources: str, include_aliases: bool = False
) -> list[str]:
    wanted = {str(source).lower() for source in sources}
    names = list(sorted(LIBERO_FAMILY_REGISTRY))
    if wanted:
        names = [
            suite_name
            for suite_name, spec in sorted(LIBERO_FAMILY_REGISTRY.items())
            if spec.source.lower() in wanted
        ]
    if include_aliases:
        return _extend_suite_aliases(names)
    return names


def get_libero_suite_names_by_taxonomy_level_1(
    *labels: str,
    include_aliases: bool = False,
    sources: tuple[str, ...] | None = None,
) -> list[str]:
    wanted = {str(label).lower() for label in labels}
    wanted_sources = {str(source).lower() for source in (sources or ())}
    names = [
        suite_name
        for suite_name, spec in sorted(LIBERO_FAMILY_REGISTRY.items())
        if (not wanted or spec.taxonomy_level_1.lower() in wanted)
        and (not wanted_sources or spec.source.lower() in wanted_sources)
    ]
    if include_aliases:
        return _extend_suite_aliases(names)
    return names


def get_libero_suites_grouped_by_taxonomy_level_1(
    *,
    include_aliases: bool = False,
    sources: tuple[str, ...] | None = None,
) -> dict[str, list[str]]:
    wanted_sources = {str(source).lower() for source in (sources or ())}
    grouped: dict[str, list[str]] = {label: [] for label in LIBERO_TAXONOMY_LEVEL_1}
    for suite_name, spec in sorted(LIBERO_FAMILY_REGISTRY.items()):
        if wanted_sources and spec.source.lower() not in wanted_sources:
            continue
        grouped.setdefault(spec.taxonomy_level_1, []).append(suite_name)
    if include_aliases:
        return {
            label: _extend_suite_aliases(list(names))
            for label, names in grouped.items()
            if names
        }
    return {label: names for label, names in grouped.items() if names}


def get_libero_suite_max_steps(name: str) -> int:
    return get_libero_suite_spec(name).max_steps
