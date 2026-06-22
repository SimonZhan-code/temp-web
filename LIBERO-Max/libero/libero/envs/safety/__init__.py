"""
SafeLIBERO environment extensions.

Provides SafeLiberoBaseDomain for safety-aware LIBERO tasks with obstacle
displacement tracking and Level 5 LTL propositions.
"""

from libero.libero.envs.safety.safelibero_base_domain import (
    SafeLiberoBaseDomain,
    SafetyMixin,
)


def _register_safelibero_object_aliases():
    """Map SafeLIBERO obstacle object names onto existing LIBERO object classes."""

    try:
        from libero.libero.envs.base_object import OBJECTS_DICT
    except Exception:
        return

    alias_map = {
        "moka_pot_obstacle": "moka_pot",
        "moka_pot_small_obstacle": "moka_pot",
        "white_storage_box_obstacle": "white_storage_box",
        "box_base": "white_storage_box",
        "box_small_base": "white_storage_box",
        "milk_obstacle": "milk",
        "milk_small_obstacle": "milk",
        "wine_bottle_obstacle": "wine_bottle",
        "wine_bottle_small_obstacle": "wine_bottle",
        "red_coffee_mug_obstacle": "red_coffee_mug",
        "yellow_book_obstacle": "yellow_book",
    }
    for alias, base_name in alias_map.items():
        if alias in OBJECTS_DICT:
            continue
        if base_name in OBJECTS_DICT:
            OBJECTS_DICT[alias] = OBJECTS_DICT[base_name]


_register_safelibero_object_aliases()

# Imported after alias registration; lazy per-arena safety domain resolution.
from libero.libero.envs.safety.safelibero_problems import (  # noqa: E402
    get_safety_domain_class,
    is_safelibero_bddl,
)

__all__ = [
    "SafeLiberoBaseDomain",
    "SafetyMixin",
    "get_safety_domain_class",
    "is_safelibero_bddl",
    "_register_safelibero_object_aliases",
]
