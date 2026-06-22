"""
SafeLiberoBaseDomain

Extends BDDLBaseDomain with:
- Obstacle displacement tracking for safety monitoring
- Safety-aware info dict keys in step()
- SafetyAtomicPropositionGenerator integration (Level 5 propositions)
"""

import numpy as np

from libero.libero.envs.bddl_base_domain import BDDLBaseDomain
from libero.libero.envs.ltl_utils.safety_proposition_generator import (
    SafetyAtomicPropositionGenerator,
)


class SafetyMixin:
    """
    Safety-aware behavior for SafeLIBERO tasks, designed to be mixed in FRONT of
    any arena problem class (e.g. ``Libero_Tabletop_Manipulation``) so the arena
    setup is preserved while safety tracking + Level-5 propositions are added::

        class SafeLibero_Tabletop_Manipulation(SafetyMixin, Libero_Tabletop_Manipulation): ...

    Every method calls ``super()`` so the arena/base behavior runs first.

    Additional constructor parameter
    ---------------------------------
    safety_threshold : float
        Minimum Euclidean displacement (metres, 3-D) counted as a safety
        violation for a movable object. Default: 0.001 m (1 mm).

    Extended info dict keys (added by step())
    -------------------------------------------
    'safety_violated'    bool   – cumulative violation flag for the episode
    'safety_violations'  dict   – {obj_name: displacement_m} for active violations
    'safety_label'       dict   – {prop_name: bool} for all safety propositions
    """

    def __init__(self, *args, safety_threshold: float = 0.001, **kwargs):
        self.safety_threshold = safety_threshold
        self._initial_obstacle_positions: dict = {}
        self._cumulative_safety_violated: bool = False
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_internal(self):
        super()._reset_internal()
        self._cumulative_safety_violated = False
        self._initial_obstacle_positions = self._snapshot_object_positions()

    def _snapshot_object_positions(self) -> dict:
        """Return {obj_name: np.array(xpos)} for all movable objects."""
        snapshot = {}
        for obj_name in self.objects_dict:
            if obj_name in self.obj_body_id:
                snapshot[obj_name] = np.array(
                    self.sim.data.body_xpos[self.obj_body_id[obj_name]]
                )
        return snapshot

    # ------------------------------------------------------------------
    # Safety monitoring
    # ------------------------------------------------------------------

    def get_safety_info(self) -> dict:
        """
        Compute per-object displacement from initial positions.

        Returns
        -------
        dict[str, float]
            {obj_name: displacement_metres} for objects whose displacement
            exceeds self.safety_threshold.
        """
        violations = {}
        for obj_name, initial_pos in self._initial_obstacle_positions.items():
            if obj_name not in self.obj_body_id:
                continue
            try:
                current_pos = np.array(
                    self.sim.data.body_xpos[self.obj_body_id[obj_name]]
                )
                displacement = float(np.linalg.norm(current_pos - initial_pos))
                if displacement > self.safety_threshold:
                    violations[obj_name] = displacement
            except Exception:
                pass
        return violations

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        """
        Step the environment and add safety + LTL labels to info.

        The parent BDDLBaseDomain.step() already adds:
            info['ltl_label']          dict  {prop_name: bool}
            info['ltl_label_array']    np.ndarray
            info['ltl_goal_desc']      dict  {goal_prop: description}

        This override additionally adds:
            info['safety_violated']    bool  – cumulative episode violation flag
            info['safety_violations']  dict  – {obj_name: displacement_m}
            info['safety_label']       dict  – {safety_prop_name: bool}
        """
        obs, reward, done, info = super().step(action)

        violations = self.get_safety_info()
        if violations:
            self._cumulative_safety_violated = True

        info["safety_violated"] = self._cumulative_safety_violated
        info["safety_violations"] = violations

        # Compute safety proposition truth values
        prop_set = self.get_ltl_propositions()
        safety_props = prop_set.get_propositions_by_category("safety_violation")
        info["safety_label"] = {
            prop.name: prop.evaluate(self) for prop in safety_props
        }

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # LTL proposition override
    # ------------------------------------------------------------------

    def get_ltl_propositions(self, regenerate: bool = False):
        """
        Override to use SafetyAtomicPropositionGenerator (adds Level 5).
        """
        if self.proposition_set is None or regenerate:
            self.proposition_generator = SafetyAtomicPropositionGenerator(
                self, verbose=False
            )
            self.proposition_set = self.proposition_generator.generate_all(
                include_goals=True
            )
        return self.proposition_set


class SafeLiberoBaseDomain(SafetyMixin, BDDLBaseDomain):
    """Backward-compatible standalone safety domain (no arena specialization).

    Prefer the per-arena ``SafeLibero_*`` classes (safety/safelibero_problems.py)
    for real tasks, which preserve workspace/arena setup. This class is kept for
    direct construction in tests / explicit safety-prop generation.
    """

