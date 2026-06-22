"""
Scene-unit manifest builder for the original-LIBERO suites (M1).

Reads the per-task atomic-proposition dumps under
``feasible_propositions/<suite>/<task>.json`` for the four original-LIBERO suites
and groups tasks into *scene-units*: maximal sets of tasks that share an
identical **non-goal** AP alphabet (unary_state + binary_relation +
region_containment). That alphabet is a property of the scene's object/region
inventory, so a scene-unit is the natural training unit for sampling feasible AP
compositions; each member task contributes one in-distribution *goal* composition
("anchor goal").

``libero_object`` is intentionally excluded: its object inventory rotates per task
so it has no fixed alphabet. ``libero_90`` and ``libero_10`` share scene tokens
with identical alphabets, so grouping merges short- and long-horizon tasks.

This module is pure-Python over the JSON dumps -- it builds no MuJoCo env.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

# Repo root: .../LIBERO-Max/libero/libero/envs/ltl_utils/composition/manifest.py
_REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_DUMP_ROOT = str(_REPO_ROOT / "feasible_propositions")

# The four original-LIBERO suites in scope (libero_object excluded by design).
ORIGINAL_SUITES: Tuple[str, ...] = (
    "libero_spatial",
    "libero_goal",
    "libero_90",
    "libero_10",
)

# Non-goal categories whose union forms the scene "alphabet".
ALPHABET_CATEGORIES: Tuple[str, ...] = (
    "unary_state",
    "binary_relation",
    "region_containment",
)

_SCENE_TOKEN_RE = re.compile(r"(.+?SCENE\d+)")


@dataclass(frozen=True)
class APRecord:
    """One atomic proposition as stored in a dump."""

    name: str
    category: str
    args: tuple
    description: str
    init_value: bool

    @staticmethod
    def from_json(category: str, entry: dict) -> "APRecord":
        return APRecord(
            name=entry["name"],
            category=category,
            args=tuple(entry.get("args") or ()),
            description=entry.get("description", ""),
            init_value=bool(entry.get("init_value", False)),
        )


@dataclass
class TaskDump:
    """A single task's dumped AP set."""

    suite: str
    task: str
    objects: List[str]
    fixtures: List[str]
    language_instruction: str
    props_by_cat: Dict[str, List[APRecord]]

    def records(self, category: str) -> List[APRecord]:
        return self.props_by_cat.get(category, [])

    @property
    def goal_records(self) -> List[APRecord]:
        return self.records("goal")

    def scene_token(self) -> Optional[str]:
        m = _SCENE_TOKEN_RE.match(self.task)
        return m.group(1) if m else None


@dataclass
class SceneUnit:
    """A group of tasks sharing one non-goal AP alphabet."""

    scene_id: str
    suites: List[str]
    objects: List[str]
    fixtures: List[str]
    alphabet_by_cat: Dict[str, List[APRecord]]
    # Per-member in-distribution goal composition.
    anchor_goals: List[dict]
    # Deduped union of all members' goal APs (records, for canonical-map building).
    goal_records: List[APRecord]
    # Non-goal *avoid* candidates (safety_violation APs), for SafeLIBERO scenes.
    safety_records: List[APRecord] = field(default_factory=list)
    alphabet_fixed: bool = True

    @property
    def alphabet_names(self) -> List[str]:
        return [r.name for cat in ALPHABET_CATEGORIES for r in self.alphabet_by_cat.get(cat, [])]


def load_task_dump(path: str) -> TaskDump:
    with open(path) as fh:
        d = json.load(fh)
    props = d.get("propositions", {}) or {}
    by_cat: Dict[str, List[APRecord]] = {}
    for category, entries in props.items():
        by_cat[category] = [APRecord.from_json(category, e) for e in (entries or [])]
    instruction = d.get("language_instruction")
    if isinstance(instruction, (list, tuple)):
        instruction = " ".join(str(t) for t in instruction)
    return TaskDump(
        suite=d.get("suite", ""),
        task=d.get("task", ""),
        objects=list(d.get("objects", []) or []),
        fixtures=list(d.get("fixtures", []) or []),
        language_instruction=instruction or "",
        props_by_cat=by_cat,
    )


def alphabet_key(dump: TaskDump) -> Tuple[str, ...]:
    """Frozen fingerprint of the non-goal AP names (the scene alphabet)."""
    names = {r.name for cat in ALPHABET_CATEGORIES for r in dump.records(cat)}
    return tuple(sorted(names))


def _dedupe_records(records: List[APRecord]) -> List[APRecord]:
    seen: Dict[str, APRecord] = {}
    for r in records:
        seen.setdefault(r.name, r)
    return sorted(seen.values(), key=lambda r: r.name)


def scene_dirs(root: str = DEFAULT_DUMP_ROOT) -> List[str]:
    """Scene-folder names under ``root`` (excludes ``_`` dirs like ``_composition``)."""
    return sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith("_")
    )


SAFETY_CATEGORY = "safety_violation"


def _records_from_entries(category: str, entries) -> List[APRecord]:
    return [
        APRecord(
            name=e["name"],
            category=category,
            args=tuple(e.get("args") or ()),
            description=e.get("description", ""),
            init_value=bool(e.get("init_value", False)),
        )
        for e in (entries or [])
    ]


def load_scene_unit(scene_dir: str) -> Optional[SceneUnit]:
    """Load one SceneUnit from a scene folder's ``fixed_alphabet.json`` +
    ``goal_aps.json`` (the two-file scene layout)."""
    fa_path = os.path.join(scene_dir, "fixed_alphabet.json")
    ga_path = os.path.join(scene_dir, "goal_aps.json")
    if not (os.path.exists(fa_path) and os.path.exists(ga_path)):
        return None
    with open(fa_path) as fh:
        fa = json.load(fh)
    with open(ga_path) as fh:
        ga = json.load(fh)

    alpha = fa.get("alphabet", {}) or {}
    alphabet_by_cat = {cat: _records_from_entries(cat, alpha.get(cat, [])) for cat in ALPHABET_CATEGORIES}
    safety_records = _records_from_entries(SAFETY_CATEGORY, alpha.get(SAFETY_CATEGORY, []))

    goal_seen: Dict[str, APRecord] = {}
    anchor_goals: List[dict] = []
    for t in ga.get("tasks", []):
        for g in t.get("goals", []):
            goal_seen.setdefault(g["name"], _records_from_entries("goal", [g])[0])
        anchor_goals.append(
            {
                "suite": t.get("suite", ""),
                "task": t.get("task", ""),
                "description": t.get("language_instruction", ""),
                "goal_ap_names": sorted(g["name"] for g in t.get("goals", [])),
            }
        )

    return SceneUnit(
        scene_id=fa.get("scene_id", os.path.basename(scene_dir)),
        suites=list(fa.get("suites", [])),
        objects=list(fa.get("objects", [])),
        fixtures=list(fa.get("fixtures", [])),
        alphabet_by_cat=alphabet_by_cat,
        anchor_goals=sorted(anchor_goals, key=lambda a: (a["suite"], a["task"])),
        goal_records=sorted(goal_seen.values(), key=lambda r: r.name),
        safety_records=safety_records,
        alphabet_fixed=bool(fa.get("alphabet_fixed", True)),
    )


def build_scene_units(root: str = DEFAULT_DUMP_ROOT) -> List[SceneUnit]:
    """Build one SceneUnit per scene folder from the two-file scene layout."""
    units = [load_scene_unit(os.path.join(root, d)) for d in scene_dirs(root)]
    units = [u for u in units if u is not None]
    units.sort(key=lambda u: (-len(u.anchor_goals), u.scene_id))
    return units


def _records_to_json(records: List[APRecord]) -> List[dict]:
    return [
        {"name": r.name, "args": list(r.args), "init_value": r.init_value}
        for r in records
    ]


def scene_unit_to_json(unit: SceneUnit) -> dict:
    out = {
        "scene_id": unit.scene_id,
        "suites": unit.suites,
        "num_tasks": len(unit.anchor_goals),
        "alphabet_fixed": unit.alphabet_fixed,
        "objects": unit.objects,
        "fixtures": unit.fixtures,
        "alphabet": {
            cat: _records_to_json(unit.alphabet_by_cat.get(cat, []))
            for cat in ALPHABET_CATEGORIES
        },
        "anchor_goals": unit.anchor_goals,
    }
    if unit.safety_records:
        out["safety_avoid_aps"] = _records_to_json(unit.safety_records)
    return out


def manifest_to_json(units: List[SceneUnit]) -> dict:
    return {
        "version": 2,
        "source_suites": sorted({s for u in units for s in u.suites}),
        "num_scene_units": len(units),
        "scene_units": [scene_unit_to_json(u) for u in units],
    }
