"""
Physical-feasibility filtering for atomic-proposition generation.

This module hosts the static (no action-rollout) heuristics used by
``AtomicPropositionGenerator`` to keep only atomic propositions (APs) that are
both *evaluable* (the eval path actually queries the simulator instead of
silently hitting ``except -> False``) and *physically feasible* (there exists a
reachable simulator state in which the AP can be True).

Three layers are implemented here as small, testable helpers:

- Layer 0 (evaluability gate): predicate-name mapping to the registry keys, and
  capability/method introspection on the *underlying* object.
- Layer 1 (type/structural compatibility): support-surface / container typing.
- Layer 2 (region geometry): region validity check (degenerate / self regions).

The generator wires these in; the actual region containment *evaluation* reuses
the existing ``eval_predicate_fn("in", obj_state, region_state)`` machinery so it
is geometry-correct and consistent with how ``:init`` / ``:goal`` ``In``
predicates are evaluated.
"""

from __future__ import annotations

from typing import Optional


# ===================== Layer 0: evaluability gate =====================

# The proposition generator describes unary predicates with verb-ish tokens, but
# the predicate registry (``libero/libero/envs/predicates/__init__.py``) is keyed
# differently. Without this mapping every unary AP raises inside
# ``eval_predicate_fn`` and is silently swallowed to a constant ``False``.
UNARY_PREDICATE_KEY_MAP = {
    "is_open": "open",
    "is_close": "close",
    "turn_on": "turnon",
    "turn_off": "turnoff",
}


def map_unary_predicate(pred_name: str) -> Optional[str]:
    """Map a generator unary token to its predicate-registry key.

    Returns the registry key, or ``None`` if there is no valid mapping (the AP
    must then be skipped rather than emitted as a silently-false proposition).
    """
    return UNARY_PREDICATE_KEY_MAP.get(pred_name)


# The ``ArticulatedObject`` base class declares ``is_open``/``is_close`` as
# ``raise NotImplementedError`` stubs (see ``envs/objects/articulated_objects.py``).
# A toggleable-but-not-openable subclass like ``FlatStove`` overrides ``turn_on``
# but inherits those stubs, so a bare ``callable(...)`` check is a false positive:
# the method exists and is callable, but raises when evaluated -- producing a
# permanently-false ("silently-false") AP, exactly what this gate must prevent.
# We detect a genuine override via ``__qualname__`` (which records the defining
# class) instead of importing ``ArticulatedObject`` -- keeping this module
# importable without robosuite, as the fake-env tests rely on.
_ARTICULATED_STUB_CLASS = "ArticulatedObject"


def _has_real_method(obj, method_name: str) -> bool:
    """True if ``obj`` exposes ``method_name`` as a genuinely-implemented method.

    Rejects the inherited ``ArticulatedObject`` ``NotImplementedError`` stubs
    (callable, but not actually implemented).
    """
    fn = getattr(obj, method_name, None)
    if not callable(fn):
        return False
    # __qualname__ is "<...scope...>.DefiningClass.method"; the class that defines
    # the (possibly inherited) method is the component just before the method name.
    parts = getattr(fn, "__qualname__", "").split(".")
    defining_cls = parts[-2] if len(parts) >= 2 else ""
    return defining_cls != _ARTICULATED_STUB_CLASS


def is_openable(obj) -> bool:
    """True if the *underlying* object genuinely implements articulated open/close.

    Real openables (cabinets, microwaves, windows, ...) override ``is_open``;
    rigid objects (food, bowls, baskets) lack it entirely, and toggle-only
    appliances (``FlatStove``/``YellowStove``) only *inherit* the base stub --
    all three are correctly rejected. Probe the underlying MuJoCo object
    (``env.get_object(name)``), NOT the ``ObjectState`` wrapper, which always
    defines ``is_open``.
    """
    return obj is not None and _has_real_method(obj, "is_open")


def is_toggleable(obj) -> bool:
    """True if the underlying object genuinely implements on/off toggling (e.g. a stove)."""
    return obj is not None and _has_real_method(obj, "turn_on")


# ===================== Layer 1: type compatibility =====================

# Movable object categories that can act as a support surface for ``on(a, b)``.
# Fixtures (tables, counters, ...) are always treated as support surfaces; these
# are the additional non-fixture categories that objects can rest on.
SUPPORT_SURFACE_CATEGORIES = {
    "plate",
    "rack",
    "tray",
    "wooden_tray",
    "wood_tray",
    "chefmate_8_frypan",
    "moka_pot",
}

# Object categories that can contain another object for ``in(a, b)``.
CONTAINER_CATEGORIES = {
    "basket",
    "white_bowl",
    "akita_black_bowl",
    "wooden_bowl",
    "yellow_bowl",
    "bowl",
    "wooden_cabinet",
    "short_cabinet",
    "wooden_two_layer_shelf",
}


def get_object_category(env, name: str) -> Optional[str]:
    """Resolve an instance name to its BDDL category.

    Primary source is the parsed problem (``objects`` / ``fixtures`` map
    ``category -> [instance names]``); falls back to the object's
    ``category_name`` attribute.
    """
    parsed = getattr(env, "parsed_problem", {}) or {}
    for group in ("objects", "fixtures"):
        for category, instances in (parsed.get(group, {}) or {}).items():
            if name in instances:
                return category
    obj = env.get_object(name)
    return getattr(obj, "category_name", None) if obj is not None else None


def is_fixture(env, name: str) -> bool:
    return name in getattr(env, "fixtures_dict", {})


def is_support_surface(env, name: str) -> bool:
    """True if ``name`` can support another object placed ``on`` it."""
    if is_fixture(env, name):
        return True
    return get_object_category(env, name) in SUPPORT_SURFACE_CATEGORIES


def is_container(env, name: str) -> bool:
    """True if ``name`` can contain another object placed ``in`` it.

    Either the category is a known container, or the object is the ``target`` of
    a ``*_contain_region`` declared in the BDDL.
    """
    if get_object_category(env, name) in CONTAINER_CATEGORIES:
        return True
    regions = (getattr(env, "parsed_problem", {}) or {}).get("regions", {}) or {}
    for region_name, region_dict in regions.items():
        if region_dict.get("target") == name and "contain" in region_name:
            return True
    return False


def object_supports_method(env, name: str, method_name: str) -> bool:
    """True if the object's *state wrapper* exposes a callable ``method_name``.

    Used as the Layer-0 method-existence gate before emitting a binary AP whose
    eval path will call e.g. ``arg2.check_ontop`` / ``arg2.check_contain``.
    """
    state = (getattr(env, "object_states_dict", {}) or {}).get(name)
    return state is not None and callable(getattr(state, method_name, None))


# ===================== Layer 2: region geometry =====================


def region_target(env, region_name: str) -> Optional[str]:
    regions = (getattr(env, "parsed_problem", {}) or {}).get("regions", {}) or {}
    region_dict = regions.get(region_name, {}) or {}
    return region_dict.get("target")


def region_is_valid(env, region_name: str) -> bool:
    """True if a region AP can be evaluated and is non-degenerate.

    Requires the region to have a registered site (so ``get_site_xpos`` and the
    ``SiteObjectState`` containment path work) and a non-empty, well-formed
    ``ranges`` box. Regions targeting an object (e.g. ``*_contain_region``) may
    have no ``ranges`` of their own; those are accepted as long as a site exists.
    """
    sites = getattr(env, "object_sites_dict", {}) or {}
    states = getattr(env, "object_states_dict", {}) or {}
    if region_name not in sites or region_name not in states:
        return False

    regions = (getattr(env, "parsed_problem", {}) or {}).get("regions", {}) or {}
    region_dict = regions.get(region_name, {}) or {}
    ranges = region_dict.get("ranges") or []
    if ranges:
        rect = ranges[0]
        if len(rect) < 4 or rect[2] <= rect[0] or rect[3] <= rect[1]:
            return False
    return True


def region_feasible_for_object(env, obj_name: str, region_name: str) -> bool:
    """True if ``obj in region`` is a feasible, non-redundant region AP.

    Drops self-regions (a region whose target is the object itself, e.g.
    ``basket_1`` in ``basket_1_contain_region``) and invalid/degenerate regions.
    Reachable table-placement regions are kept for all movable objects.
    """
    if not region_is_valid(env, region_name):
        return False
    if region_target(env, region_name) == obj_name:
        return False
    return True
