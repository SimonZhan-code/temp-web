"""
Factored-state model for a scene-unit (M1).

From a ``SceneUnit`` (manifest.py) this derives the structures the composition
sampler needs, all from the dumped APs (no MuJoCo):

- **Mutex groups**: per object, all its location APs (region_containment +
  binary on/in) are mutually exclusive ("an object is in one place"); per
  articulated object, ``is_open``/``is_close`` and ``turn_on``/``turn_off`` are
  XOR pairs.
- **Goal-style canonical map**: ``region_containment``/``binary_relation`` AP name
  -> the goal-shaped AP name for the same (object, region/target), when one exists
  among the scene's goal APs. Sampled formulas prefer the goal-style name.
- **Gated-region detection**: which region requires an openable container to be
  open before an object can be placed inside it (open-before-place), and which
  regions are appliance "cook" regions.
- **Pruning**: an object's own ``*_init_region`` is its origin/home, not a valid
  destination subgoal.
- A factored ``State`` and the best-effort initial assignment.

After the FlatStove ``is_open`` fix, the dumps already exclude silently-false
articulation APs, so articulation capability is read directly from AP presence:
an object is openable iff it has an ``is_open`` AP, an appliance iff ``turn_on``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from libero.libero.envs.ltl_utils.composition.manifest import (
    ALPHABET_CATEGORIES,
    APRecord,
    SceneUnit,
)

INIT = "INIT"  # sentinel location: object at its origin / resting state
_TRAILING_INDEX_RE = re.compile(r"_\d+$")
_LOCATION_PREDS = ("on", "in")


def object_category(obj: str) -> str:
    """``alphabet_soup_1`` -> ``alphabet_soup``; ``chefmate_8_frypan_1`` -> ``chefmate_8_frypan``."""
    return _TRAILING_INDEX_RE.sub("", obj)


# --------------------------------------------------------------------------- #
# Factored variables
# --------------------------------------------------------------------------- #


@dataclass
class LocationVar:
    """An object's location: one categorical variable over destination APs."""

    obj: str
    origin_ap: Optional[str]          # the object's own *_init_region (home), or None
    destination_aps: List[str]        # candidate destination location APs (no origin)
    table_region_aps: List[str]       # subset of destinations that are table/init regions
    init_location: str                # origin_ap if known else INIT


@dataclass
class ArticulationVar:
    """An articulated object's open/close and on/off state."""

    obj: str
    is_open_ap: Optional[str]
    is_close_ap: Optional[str]
    turn_on_ap: Optional[str]
    turn_off_ap: Optional[str]
    open_goal_aps: List[str]          # goal-style open_<region> achievers for this object
    turnon_goal_ap: Optional[str]     # goal-style turnon_<obj>
    init_open: bool
    init_on: bool

    @property
    def openable(self) -> bool:
        return self.is_open_ap is not None

    @property
    def appliance(self) -> bool:
        return self.turn_on_ap is not None


@dataclass
class FactoredScene:
    scene_id: str
    location_vars: Dict[str, LocationVar]
    articulation_vars: Dict[str, ArticulationVar]
    canonical: Dict[str, str]              # region/binary AP name -> goal-style name
    gated_regions: Dict[str, str]          # region name -> controlling openable object
    cook_regions: Dict[str, str]           # region name -> appliance object
    usable_location_aps: set               # all destination APs across objects
    init_assignment: "State"
    dest_target: Dict[str, str] = field(default_factory=dict)  # location AP -> target region/surface

    def canonical_name(self, ap_name: str) -> str:
        return self.canonical.get(ap_name, ap_name)


@dataclass(frozen=True)
class State:
    """A mutex-consistent factored truth assignment."""

    locations: Tuple[Tuple[str, str], ...]      # sorted ((obj, location_ap|INIT), ...)
    open_state: Tuple[Tuple[str, bool], ...]    # ((obj, is_open?), ...)
    on_state: Tuple[Tuple[str, bool], ...]      # ((obj, is_on?), ...)

    def location_of(self, obj: str) -> Optional[str]:
        for o, loc in self.locations:
            if o == obj:
                return loc
        return None


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _region_of(record: APRecord) -> Optional[str]:
    """Destination/target name for a location AP (region or binary obj2)."""
    if record.category == "region_containment" and len(record.args) >= 2:
        return record.args[1]
    if record.category == "binary_relation" and len(record.args) >= 3:
        return record.args[2]
    return None


def build_canonical_map(unit: SceneUnit) -> Dict[str, str]:
    """Map region_containment / binary_relation AP names to goal-style names.

    A goal AP ``[pred, obj, target]`` (pred in on/in) is the goal-shaped twin of a
    location AP for the same (obj, target). Keying on (obj, target) means the same
    physical state is reachable under either name.
    """
    goal_by_key: Dict[Tuple[str, str], str] = {}
    for g in unit.goal_records:
        if len(g.args) == 3 and str(g.args[0]).lower() in _LOCATION_PREDS:
            goal_by_key[(g.args[1], g.args[2])] = g.name

    canonical: Dict[str, str] = {}
    for cat in ("region_containment", "binary_relation"):
        for r in unit.alphabet_by_cat.get(cat, []):
            obj = r.args[0] if r.args else None
            target = _region_of(r)
            if obj is None or target is None:
                continue
            goal_name = goal_by_key.get((obj, target))
            if goal_name:
                canonical[r.name] = goal_name
    return canonical


def _build_articulation_vars(unit: SceneUnit) -> Dict[str, ArticulationVar]:
    # Per-object unary predicate -> record.
    by_obj: Dict[str, Dict[str, APRecord]] = {}
    for r in unit.alphabet_by_cat.get("unary_state", []):
        if len(r.args) >= 2:
            by_obj.setdefault(r.args[0], {})[r.args[1]] = r

    # Goal-style achievers: open_<region> and turnon_<obj>.
    open_goals_by_obj: Dict[str, List[str]] = {}
    turnon_goal_by_obj: Dict[str, str] = {}
    for g in unit.goal_records:
        pred = str(g.args[0]).lower() if g.args else ""
        if pred == "open" and len(g.args) >= 2:
            region = g.args[1]
            for obj in by_obj:
                if region == obj or region.startswith(obj + "_"):
                    open_goals_by_obj.setdefault(obj, []).append(g.name)
        elif pred == "turnon" and len(g.args) >= 2:
            turnon_goal_by_obj[g.args[1]] = g.name

    arts: Dict[str, ArticulationVar] = {}
    for obj, preds in by_obj.items():
        arts[obj] = ArticulationVar(
            obj=obj,
            is_open_ap=preds["is_open"].name if "is_open" in preds else None,
            is_close_ap=preds["is_close"].name if "is_close" in preds else None,
            turn_on_ap=preds["turn_on"].name if "turn_on" in preds else None,
            turn_off_ap=preds["turn_off"].name if "turn_off" in preds else None,
            open_goal_aps=sorted(open_goals_by_obj.get(obj, [])),
            turnon_goal_ap=turnon_goal_by_obj.get(obj),
            init_open=bool("is_open" in preds and preds["is_open"].init_value),
            init_on=bool("turn_on" in preds and preds["turn_on"].init_value),
        )
    return arts


def _detect_gated_and_cook(
    unit: SceneUnit, arts: Dict[str, ArticulationVar]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    openables = sorted((o for o, a in arts.items() if a.openable), key=len, reverse=True)
    appliances = sorted((o for o, a in arts.items() if a.appliance), key=len, reverse=True)

    regions = set()
    for r in unit.alphabet_by_cat.get("region_containment", []):
        target = _region_of(r)
        if target:
            regions.add(target)

    gated: Dict[str, str] = {}
    cook: Dict[str, str] = {}
    for region in sorted(regions):
        for o in openables:
            if region.startswith(o + "_"):
                gated[region] = o
                break
        for o in appliances:
            if region.startswith(o + "_"):
                cook[region] = o
                break
    return gated, cook


def _build_location_vars(unit: SceneUnit) -> Dict[str, LocationVar]:
    # Gather location APs grouped by subject object.
    by_obj: Dict[str, List[APRecord]] = {}
    for cat in ("region_containment", "binary_relation"):
        for r in unit.alphabet_by_cat.get(cat, []):
            if r.args:
                by_obj.setdefault(r.args[0], []).append(r)

    loc_vars: Dict[str, LocationVar] = {}
    for obj, records in by_obj.items():
        category = object_category(obj)
        own_init_suffix = f"_{category}_init_region"
        origin_ap: Optional[str] = None
        destinations: List[str] = []
        table_regions: List[str] = []
        for r in records:
            target = _region_of(r) or ""
            is_init_region = target.endswith("_init_region")
            is_own_origin = target.endswith(own_init_suffix)
            if is_own_origin:
                origin_ap = r.name
                continue  # origin is not a destination
            destinations.append(r.name)
            if is_init_region:
                table_regions.append(r.name)
        loc_vars[obj] = LocationVar(
            obj=obj,
            origin_ap=origin_ap,
            destination_aps=sorted(destinations),
            table_region_aps=sorted(table_regions),
            init_location=origin_ap if origin_ap else INIT,
        )
    return loc_vars


def _build_init_state(
    loc_vars: Dict[str, LocationVar], arts: Dict[str, ArticulationVar]
) -> State:
    locations = tuple(sorted((o, v.init_location) for o, v in loc_vars.items()))
    open_state = tuple(
        sorted((o, a.init_open) for o, a in arts.items() if a.openable)
    )
    on_state = tuple(sorted((o, a.init_on) for o, a in arts.items() if a.appliance))
    return State(locations=locations, open_state=open_state, on_state=on_state)


def build_factored_scene(unit: SceneUnit) -> FactoredScene:
    canonical = build_canonical_map(unit)
    arts = _build_articulation_vars(unit)
    gated, cook = _detect_gated_and_cook(unit, arts)
    loc_vars = _build_location_vars(unit)
    usable = {ap for v in loc_vars.values() for ap in v.destination_aps}
    init_state = _build_init_state(loc_vars, arts)
    dest_target: Dict[str, str] = {}
    for cat in ("region_containment", "binary_relation"):
        for r in unit.alphabet_by_cat.get(cat, []):
            target = _region_of(r)
            if target is not None:
                dest_target[r.name] = target
    return FactoredScene(
        scene_id=unit.scene_id,
        location_vars=loc_vars,
        articulation_vars=arts,
        canonical=canonical,
        gated_regions=gated,
        cook_regions=cook,
        usable_location_aps=usable,
        init_assignment=init_state,
        dest_target=dest_target,
    )


def factored_scene_summary(scene: FactoredScene) -> dict:
    """JSON-able view of the derived structures, for manifest embedding / review."""
    return {
        "location_vars": {
            obj: {
                "origin_ap": v.origin_ap,
                "init_location": v.init_location,
                "num_destinations": len(v.destination_aps),
                "destination_aps": v.destination_aps,
                "table_region_aps": v.table_region_aps,
            }
            for obj, v in sorted(scene.location_vars.items())
        },
        "articulation_vars": {
            obj: {
                "openable": a.openable,
                "appliance": a.appliance,
                "is_open_ap": a.is_open_ap,
                "is_close_ap": a.is_close_ap,
                "turn_on_ap": a.turn_on_ap,
                "turn_off_ap": a.turn_off_ap,
                "open_goal_aps": a.open_goal_aps,
                "turnon_goal_ap": a.turnon_goal_ap,
                "init_open": a.init_open,
                "init_on": a.init_on,
            }
            for obj, a in sorted(scene.articulation_vars.items())
        },
        "gated_regions": scene.gated_regions,
        "cook_regions": scene.cook_regions,
        "canonical_map": scene.canonical,
        "num_usable_location_aps": len(scene.usable_location_aps),
    }
