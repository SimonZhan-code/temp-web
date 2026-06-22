"""
Feasible AP-composition tooling for the original-LIBERO scene-based training paradigm.

This package groups the four original-LIBERO suites into *scene-units* (tasks that
share an identical atomic-proposition alphabet) and -- in later milestones -- samples
feasible, meaningful compositions of those APs as training-time LTL specifications.

M1 (this milestone) is pure-Python over the JSON dumps under ``feasible_propositions/``
and does not build any MuJoCo environment.

Public API:
- ``manifest``: scene-unit grouping (``build_scene_units``, ``SceneUnit``, ...).
- ``factored_state``: mutex groups, goal-style canonical-name map, gated-region
  detection, and the factored ``State`` model (``build_factored_scene``, ...).
"""

from libero.libero.envs.ltl_utils.composition import manifest, factored_state

__all__ = ["manifest", "factored_state"]
