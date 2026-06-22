"""
Safety Atomic Proposition Generator (Level 5)

Extends AtomicPropositionGenerator with safety-specific propositions:
- Object displacement violations (obstacle knocked over / moved)
"""

import numpy as np

from libero.libero.envs.ltl_utils.proposition_generator import AtomicPropositionGenerator
from libero.libero.envs.ltl_utils import AtomicProposition, SafetyPropositionType


class SafetyAtomicPropositionGenerator(AtomicPropositionGenerator):
    """
    Generates atomic propositions including Level 5 safety violations.

    Level 5 propositions track whether movable objects (obstacles) have been
    displaced beyond the environment's safety_threshold from their initial
    positions at reset time.
    """

    def generate_all(self, include_goals: bool = True):
        """
        Generate all propositions (Levels 1–4 via super, then Level 5 safety).
        """
        super().generate_all(include_goals=include_goals)
        self._generate_safety_violation_propositions()
        return self.propositions

    def _generate_safety_violation_propositions(self) -> None:
        """
        Level 5: create a displacement-violation proposition for each movable object.

        Each proposition `{obj_name}_displaced` is True when the object has moved
        more than `env.safety_threshold` meters (Euclidean, 3-D) from its initial
        position captured at the last reset.

        Requires the environment to be a SafeLiberoBaseDomain (or subclass) so that
        `env._initial_obstacle_positions` and `env.safety_threshold` are available.
        """
        if self.verbose:
            print("\n[Level 5] Generating safety violation propositions...")

        movable_objects = list(self.env.objects_dict.keys())
        count = 0

        for obj_name in movable_objects:
            prop_name = f"{obj_name}_displaced"

            def eval_fn(env, obj=obj_name):
                initial_positions = getattr(env, "_initial_obstacle_positions", {})
                if obj not in initial_positions:
                    return False
                try:
                    current_pos = np.array(
                        env.sim.data.body_xpos[env.obj_body_id[obj]]
                    )
                    initial_pos = initial_positions[obj]
                    displacement = float(np.linalg.norm(current_pos - initial_pos))
                    threshold = getattr(env, "safety_threshold", 0.001)
                    return displacement > threshold
                except Exception:
                    return False

            prop = AtomicProposition(
                name=prop_name,
                type=SafetyPropositionType.SAFETY_VIOLATION,
                args=(obj_name,),
                eval_fn=eval_fn,
                description=f"displaced({obj_name}) > safety_threshold",
                category="safety_violation",
            )
            self.propositions.add_proposition(prop)
            count += 1

        # Aggregate safety atom: True iff ANY movable object is displaced beyond
        # the threshold. Used by the LTL spec (G(!obstacle_displaced)) so the
        # automaton stays small instead of a per-object disjunction over ~13 atoms.
        if movable_objects:
            def aggregate_eval(env, objs=tuple(movable_objects)):
                initial_positions = getattr(env, "_initial_obstacle_positions", {})
                threshold = getattr(env, "safety_threshold", 0.001)
                for obj in objs:
                    if obj not in initial_positions:
                        continue
                    try:
                        current_pos = np.array(
                            env.sim.data.body_xpos[env.obj_body_id[obj]]
                        )
                        displacement = float(
                            np.linalg.norm(current_pos - initial_positions[obj])
                        )
                        if displacement > threshold:
                            return True
                    except Exception:
                        continue
                return False

            self.propositions.add_proposition(
                AtomicProposition(
                    name="obstacle_displaced",
                    type=SafetyPropositionType.SAFETY_VIOLATION,
                    args=(),
                    eval_fn=aggregate_eval,
                    description="any obstacle displaced > safety_threshold",
                    category="safety_aggregate",
                )
            )

        if self.verbose:
            print(f"  Created {count} safety violation propositions")
