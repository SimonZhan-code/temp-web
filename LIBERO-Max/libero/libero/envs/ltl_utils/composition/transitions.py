"""
Rule-based abstract transition system for feasible AP-composition sampling (M2).

The subgoal vocabulary of a scene is its set of **goal-style** APs (the union of
the member tasks' goal predicates): placements (``in_*`` / ``on_*``), drawer
articulation (``open_*`` / ``close_*``), and appliance toggling
(``turnon_*`` / ``turnoff_*``). A *composition* is a feasible ordering of such
subgoals; this module makes "feasible" precise and constructive.

State is factored:
- one categorical **location** per movable object (sentinel ``INIT`` = origin),
- a boolean **open** per *true* openable drawer region (one that has an
  ``open_*`` or ``close_*`` achiever -- this excludes surface false-positives like
  ``<cabinet>_top_side``),
- a boolean **on** per appliance.

Transitions are single feasible primitives (single-gripper: one variable changes
per step) with preconditions:
- placing an object *into a gated drawer* requires that drawer open
  (open-before-place);
- placing onto a surface / cook region has no precondition by default
  (``require_cook_gate`` can require the appliance on first);
- opening requires closed, closing requires open, toggling flips on/off.

Every walk through this system is therefore feasible by construction (no LDBA /
Rabinizer needed). Per-region initial open-state is read from the open/close
goal-AP ``init_value`` (e.g. a drawer with ``close_<r>`` false at init starts
open).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

from libero.libero.envs.ltl_utils.composition import factored_state as FS
from libero.libero.envs.ltl_utils.composition.manifest import APRecord, SceneUnit

INIT = FS.INIT


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Primitive:
    """A single feasible action; ``achieved_ap`` is the goal-style AP it makes true."""

    kind: str  # "move" | "open" | "close" | "turn_on" | "turn_off"
    obj: str  # the manipulated / toggled object (or drawer controller for open/close)
    target: Optional[str]  # region/surface for move; drawer region for open/close
    achieved_ap: str

    def touches(self) -> str:
        """The entity whose 'distinctness' this primitive contributes to a composition."""
        return self.obj


@dataclass
class Drawer:
    region: str
    controller: str
    open_ap: Optional[str]
    close_ap: Optional[str]
    init_open: bool


@dataclass
class Appliance:
    obj: str
    on_ap: Optional[str]
    off_ap: Optional[str]
    init_on: bool


@dataclass
class SceneModel:
    scene_id: str
    movable_objects: List[str]
    # obj -> {achieved goal-style placement AP -> target region/surface}
    placements: Dict[str, Dict[str, str]]
    drawers: Dict[str, Drawer]  # region -> Drawer
    appliances: Dict[str, Appliance]  # obj -> Appliance
    cook_targets: Dict[str, str]  # cook-region -> appliance obj
    init_state: "WalkState"
    anchor_goal_sets: List[FrozenSet[str]]
    require_cook_gate: bool = False


@dataclass(frozen=True)
class WalkState:
    locations: Tuple[Tuple[str, str], ...]      # ((obj, placement_ap|INIT), ...)
    region_open: Tuple[Tuple[str, bool], ...]   # ((drawer_region, open?), ...)
    appliance_on: Tuple[Tuple[str, bool], ...]  # ((appliance_obj, on?), ...)

    def location_of(self, obj: str) -> str:
        for o, loc in self.locations:
            if o == obj:
                return loc
        return INIT

    def is_open(self, region: str) -> bool:
        for r, v in self.region_open:
            if r == region:
                return v
        return False

    def is_on(self, obj: str) -> bool:
        for o, v in self.appliance_on:
            if o == obj:
                return v
        return False


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #


def _achievers(unit: SceneUnit, pred: str) -> Dict[str, APRecord]:
    """Map target (args[1]) -> goal record for a unary goal predicate."""
    out: Dict[str, APRecord] = {}
    for g in unit.goal_records:
        if g.args and str(g.args[0]).lower() == pred and len(g.args) >= 2:
            out.setdefault(g.args[1], g)
    return out


def _region_open_init(
    region: str, open_ach: Dict[str, APRecord], close_ach: Dict[str, APRecord]
) -> bool:
    """Initial open-state of a drawer from open/close goal-AP init values."""
    if region in open_ach and open_ach[region].init_value:
        return True
    if region in close_ach and close_ach[region].init_value:
        return False
    if region in close_ach:  # close_<r> exists but false at init => not closed => open
        return True
    if region in open_ach:  # open_<r> exists but false at init => not open => closed
        return False
    return False


def build_scene_model(unit: SceneUnit, require_cook_gate: bool = False) -> SceneModel:
    fs = FS.build_factored_scene(unit)

    open_ach = _achievers(unit, "open")
    close_ach = _achievers(unit, "close")
    turnon_ach = _achievers(unit, "turnon")
    turnoff_ach = _achievers(unit, "turnoff")

    # Meaningful placements = the scene's goal-style placement predicates, taken
    # straight from the goal records (args = [pred, obj, target]). Keyed by the
    # goal-style achieved AP name; value is the placement target (region/surface).
    placements: Dict[str, Dict[str, str]] = {}
    for g in unit.goal_records:
        if g.args and str(g.args[0]).lower() in ("in", "on") and len(g.args) >= 3:
            obj, target = g.args[1], g.args[2]
            placements.setdefault(obj, {})[g.name] = target

    # True openable drawers: gated regions that have an open or close achiever.
    drawers: Dict[str, Drawer] = {}
    for region, controller in fs.gated_regions.items():
        if region in open_ach or region in close_ach:
            drawers[region] = Drawer(
                region=region,
                controller=controller,
                open_ap=open_ach[region].name if region in open_ach else None,
                close_ap=close_ach[region].name if region in close_ach else None,
                init_open=_region_open_init(region, open_ach, close_ach),
            )

    appliances: Dict[str, Appliance] = {}
    for obj in set(turnon_ach) | set(turnoff_ach):
        appliances[obj] = Appliance(
            obj=obj,
            on_ap=turnon_ach[obj].name if obj in turnon_ach else None,
            off_ap=turnoff_ach[obj].name if obj in turnoff_ach else None,
            init_on=bool(obj in turnon_ach and turnon_ach[obj].init_value),
        )

    movable_objects = sorted(placements.keys())
    init_state = WalkState(
        locations=tuple(sorted((o, INIT) for o in movable_objects)),
        region_open=tuple(sorted((r, d.init_open) for r, d in drawers.items())),
        appliance_on=tuple(sorted((o, a.init_on) for o, a in appliances.items())),
    )

    anchor_goal_sets = [
        frozenset(ag["goal_ap_names"]) for ag in unit.anchor_goals if ag["goal_ap_names"]
    ]

    return SceneModel(
        scene_id=unit.scene_id,
        movable_objects=movable_objects,
        placements=placements,
        drawers=drawers,
        appliances=appliances,
        cook_targets=dict(fs.cook_regions),
        init_state=init_state,
        anchor_goal_sets=anchor_goal_sets,
        require_cook_gate=require_cook_gate,
    )


# --------------------------------------------------------------------------- #
# Transition relation
# --------------------------------------------------------------------------- #


def _place_precondition_met(model: SceneModel, state: WalkState, target: str) -> bool:
    if target in model.drawers:  # gated drawer: must be open
        return state.is_open(target)
    if model.require_cook_gate and target in model.cook_targets:
        return state.is_on(model.cook_targets[target])
    return True


def enabled_primitives(model: SceneModel, state: WalkState) -> List[Primitive]:
    prims: List[Primitive] = []

    # Placements (move one object to a new goal-style location).
    for obj in model.movable_objects:
        cur = state.location_of(obj)
        for achieved, target in sorted(model.placements[obj].items()):
            if achieved == cur:
                continue  # no self-loop
            if not _place_precondition_met(model, state, target):
                continue
            prims.append(Primitive("move", obj, target, achieved))

    # Drawer open / close.
    for region, drawer in sorted(model.drawers.items()):
        opened = state.is_open(region)
        if not opened and drawer.open_ap:
            prims.append(Primitive("open", drawer.controller, region, drawer.open_ap))
        if opened and drawer.close_ap:
            prims.append(Primitive("close", drawer.controller, region, drawer.close_ap))

    # Appliance toggle.
    for obj, appliance in sorted(model.appliances.items()):
        on = state.is_on(obj)
        if not on and appliance.on_ap:
            prims.append(Primitive("turn_on", obj, None, appliance.on_ap))
        if on and appliance.off_ap:
            prims.append(Primitive("turn_off", obj, None, appliance.off_ap))

    return prims


def apply(state: WalkState, prim: Primitive) -> WalkState:
    if prim.kind == "move":
        locations = tuple(
            sorted(
                (o, prim.achieved_ap if o == prim.obj else loc)
                for o, loc in state.locations
            )
        )
        return WalkState(locations, state.region_open, state.appliance_on)
    if prim.kind in ("open", "close"):
        new_open = prim.kind == "open"
        region_open = tuple(
            sorted((r, new_open if r == prim.target else v) for r, v in state.region_open)
        )
        return WalkState(state.locations, region_open, state.appliance_on)
    if prim.kind in ("turn_on", "turn_off"):
        new_on = prim.kind == "turn_on"
        appliance_on = tuple(
            sorted((o, new_on if o == prim.obj else v) for o, v in state.appliance_on)
        )
        return WalkState(state.locations, state.region_open, appliance_on)
    raise ValueError(f"unknown primitive kind: {prim.kind}")
