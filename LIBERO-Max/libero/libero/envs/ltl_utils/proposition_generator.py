"""
Automatic Atomic Proposition Generator

Extracts all possible atomic propositions from a BDDL environment,
organized in levels:
- Level 1: Unary object state predicates
- Level 2: Binary object-object relations
- Level 3: Object-region spatial relationships
- Level 4: Task goal predicates
"""

import numpy as np
from typing import Set, Tuple
from libero.libero.envs.ltl_utils import (
    AtomicProposition, PropositionSet, PropositionType
)
from libero.libero.envs.ltl_utils import feasibility
from libero.libero.envs.predicates import eval_predicate_fn


class AtomicPropositionGenerator:
    """
    Automatically generates atomic propositions from BDDL environment.
    
    Key insight: Propositions are generated from three sources:
    1. Object intrinsic states (discovered via introspection)
    2. Object-object spatial relations (from binary predicates)
    3. Object-region containment (from BDDL regions)
    4. Task goals (from :goal in BDDL)
    """
    
    def __init__(self, env, verbose: bool = False):
        """
        Args:
            env: BDDLBaseDomain instance
            verbose: Print generation progress
        """
        self.env = env
        self.verbose = verbose
        self.propositions = PropositionSet(name=env.__class__.__name__)
        self._discovered_predicates: Set[str] = set()
        self._object_predicate_support: dict = {}  # obj → set of supported preds
    
    def generate_all(self, include_goals: bool = True) -> PropositionSet:
        """
        Main entry point: generate all atomic propositions.
        
        Args:
            include_goals: Whether to include task-specific goal propositions
            
        Returns:
            PropositionSet with all generated propositions
        """
        if self.verbose:
            print("\n=== Generating Atomic Propositions ===")
        
        self._generate_unary_propositions()
        self._generate_binary_propositions()
        self._generate_region_propositions()
        
        if include_goals:
            self._generate_goal_propositions()
        
        if self.verbose:
            self.propositions.print_summary()
        
        return self.propositions
    
    # ========== Level 1: Unary State Propositions ==========
    
    def _generate_unary_propositions(self) -> None:
        """
        Generate propositions for intrinsic object states.
        
        Examples:
            - moka_pot_1_is_open
            - flat_stove_1_is_on
            - cabinet_door_is_closed
        """
        if self.verbose:
            print("\n[Level 1] Generating unary state propositions...")
        
        # Predicates to check for each object
        unary_predicates = [
            "is_open",
            "is_close",
            "turn_on",
            "turn_off",
        ]
        
        all_objects = list(self.env.objects_dict.keys()) + \
                     list(self.env.fixtures_dict.keys())
        
        count = 0
        for obj_name in all_objects:
            if obj_name not in self.env.object_states_dict:
                continue

            for pred_name in unary_predicates:
                # Layer 0: skip predicates with no valid registry key.
                if feasibility.map_unary_predicate(pred_name) is None:
                    continue
                # Layer 1: only emit for objects whose underlying class actually
                # implements the affordance (articulated -> open/close,
                # toggleable -> on/off).
                if not self._object_supports_predicate(obj_name, pred_name):
                    continue

                prop_name = f"{obj_name}_{pred_name}"
                
                # Create proposition with closure over current values
                prop = self._create_unary_proposition(
                    prop_name, obj_name, pred_name
                )
                self.propositions.add_proposition(prop)
                self._discovered_predicates.add(pred_name)
                count += 1
        
        if self.verbose:
            print(f"Created {count} unary propositions")
    
    def _create_unary_proposition(
        self, prop_name: str, obj_name: str, pred_name: str
    ) -> AtomicProposition:
        """Create a unary state proposition with proper closure"""

        # Layer 0: resolve the generator token to its predicate-registry key.
        registry_key = feasibility.map_unary_predicate(pred_name)

        def eval_fn(env, obj=obj_name, key=registry_key):
            if obj not in env.object_states_dict:
                return False
            try:
                return eval_predicate_fn(key, env.object_states_dict[obj])
            except:
                return False

        return AtomicProposition(
            name=prop_name,
            type=PropositionType.UNARY_STATE,
            args=(obj_name, pred_name),
            eval_fn=eval_fn,
            description=f"{pred_name}({obj_name})",
            category="unary_state"
        )

    def _object_supports_predicate(self, obj_name: str, pred_name: str) -> bool:
        """
        Check whether the *underlying* object supports a unary predicate.

        Probes the real MuJoCo object (not the ``ObjectState`` wrapper, which
        always defines ``is_open``/``turn_on``), so only articulated objects get
        open/close props and only toggleable objects get on/off props.
        """
        obj = self.env.get_object(obj_name)
        if pred_name in ("is_open", "is_close"):
            return feasibility.is_openable(obj)
        if pred_name in ("turn_on", "turn_off"):
            return feasibility.is_toggleable(obj)
        return False
    
    # ========== Level 2: Binary Relation Propositions ==========
    
    def _generate_binary_propositions(self) -> None:
        """
        Generate propositions for object-object spatial relations.
        
        Examples:
            - moka_pot_1_on_flat_stove_1
            - frypan_1_in_cabinet
            - cup_1_contact_plate_1
        """
        if self.verbose:
            print("\n[Level 2] Generating binary relation propositions...")
        
        binary_predicates = [
            "on",
            "in",
        ]
        
        all_objects = list(self.env.objects_dict.keys()) + \
                     list(self.env.fixtures_dict.keys())
        
        count = 0
        # Consider both orderings of each pair so an asymmetric relation
        # (e.g. on(a, support)) is generated regardless of declaration order;
        # the type/evaluability filter keeps only the physically valid direction.
        for obj_1_name in all_objects:
            for obj_2_name in all_objects:
                if obj_1_name == obj_2_name:
                    continue

                for pred_name in binary_predicates:
                    if not self._is_valid_binary_relation(
                        obj_1_name, obj_2_name, pred_name
                    ):
                        continue

                    prop_name = f"{obj_1_name}_{pred_name}_{obj_2_name}"
                    prop = self._create_binary_proposition(
                        prop_name, obj_1_name, pred_name, obj_2_name
                    )
                    self.propositions.add_proposition(prop)
                    count += 1

        if self.verbose:
            print(f"  Created {count} binary propositions")
    
    def _create_binary_proposition(
        self, prop_name: str, obj_1_name: str, pred_name: str, obj_2_name: str
    ) -> AtomicProposition:
        """Create a binary relation proposition with proper closure"""
        
        def eval_fn(env, o1=obj_1_name, pred=pred_name, o2=obj_2_name):
            if o1 not in env.object_states_dict or \
               o2 not in env.object_states_dict:
                return False
            try:
                return eval_predicate_fn(
                    pred,
                    env.object_states_dict[o1],
                    env.object_states_dict[o2]
                )
            except:
                return False
        
        return AtomicProposition(
            name=prop_name,
            type=PropositionType.BINARY_RELATION,
            args=(obj_1_name, pred_name, obj_2_name),
            eval_fn=eval_fn,
            description=f"{pred_name}({obj_1_name}, {obj_2_name})",
            category="binary_relation"
        )
    
    def _is_valid_binary_relation(
        self, obj_1_name: str, obj_2_name: str, pred_name: str
    ) -> bool:
        """
        Layer 0/1 filter for ``pred(obj_1, obj_2)``.

        - ``on(a, b)``: ``b`` must be a support surface, ``a`` a non-fixture
          movable, and ``b``'s state must expose ``check_ontop``.
        - ``in(a, b)``: ``b`` must be a container whose underlying object exposes
          ``in_box`` (regular objects don't; their containment is a region AP),
          and ``a`` must be a non-fixture movable.
        """
        # Fixtures stay put; they can't be the placed object.
        if feasibility.is_fixture(self.env, obj_1_name):
            return False

        if pred_name == "on":
            return (
                feasibility.is_support_surface(self.env, obj_2_name)
                and feasibility.object_supports_method(
                    self.env, obj_2_name, "check_ontop"
                )
            )

        if pred_name == "in":
            obj_2 = self.env.get_object(obj_2_name)
            return (
                feasibility.is_container(self.env, obj_2_name)
                and feasibility.object_supports_method(
                    self.env, obj_2_name, "check_contain"
                )
                and obj_2 is not None
                and callable(getattr(obj_2, "in_box", None))
            )

        return False
    
    # ========== Level 3: Region Containment Propositions ==========
    
    def _generate_region_propositions(self) -> None:
        """
        Generate propositions for object-region spatial relationships.
        
        Examples:
            - moka_pot_1_in_cook_region
            - frypan_1_in_drawer_region
        """
        if self.verbose:
            print("\n[Level 3] Generating region containment propositions...")
        
        regions = self.env.parsed_problem.get("regions", {})
        movable_objects = list(self.env.objects_dict.keys())

        count = 0
        for obj_name in movable_objects:
            for region_name in regions.keys():
                # Layer 2: drop invalid/degenerate regions and self-regions
                # (a region whose target is the object itself).
                if not feasibility.region_feasible_for_object(
                    self.env, obj_name, region_name
                ):
                    continue

                prop_name = f"{obj_name}_in_{region_name}"

                prop = self._create_region_proposition(
                    prop_name, obj_name, region_name
                )
                self.propositions.add_proposition(prop)
                count += 1

        if self.verbose:
            print(f"  Created {count} region propositions")
    
    def _create_region_proposition(
        self, prop_name: str, obj_name: str, region_name: str
    ) -> AtomicProposition:
        """Create a region containment proposition"""
        
        def eval_fn(env, obj=obj_name, reg=region_name):
            try:
                return self._check_object_in_region(env, obj, reg)
            except:
                return False
        
        return AtomicProposition(
            name=prop_name,
            type=PropositionType.REGION_CONTAINMENT,
            args=(obj_name, region_name),
            eval_fn=eval_fn,
            description=f"in_region({obj_name}, {region_name})",
            category="region_containment"
        )
    
    def _check_object_in_region(
        self, env, obj_name: str, region_name: str
    ) -> bool:
        """
        Check whether an object is contained in a region.

        Reuses the same containment machinery as ``:init`` / ``:goal`` ``In``
        predicates: ``eval_predicate_fn("in", obj_state, region_state)`` where
        ``region_state`` is a ``SiteObjectState`` whose ``check_contain`` uses the
        region's true box geometry (``SiteObject.in_box``). This replaces the old
        ad-hoc 20 cm xy-distance threshold with geometry-correct, consistent eval.
        """
        if obj_name not in env.object_states_dict or \
           region_name not in env.object_states_dict:
            return False

        try:
            return eval_predicate_fn(
                "in",
                env.object_states_dict[obj_name],
                env.object_states_dict[region_name],
            )
        except:
            return False
    
    # ========== Level 4: Task Goal Propositions ==========
    
    def _generate_goal_propositions(self) -> None:
        """
        Generate propositions for task-specific goal states.
        
        These map directly to the :goal in BDDL file.
        """
        if self.verbose:
            print("\n[Level 4] Generating goal propositions...")
        
        goal_state = self.env.parsed_problem.get("goal_state", [])
        
        count = 0
        for i, goal_predicate in enumerate(goal_state):
            prop_name = self._format_goal_name(goal_predicate, i)

            # Create readable description
            desc = self._format_goal_description(goal_predicate)
            
            prop = self._create_goal_proposition(
                prop_name, goal_predicate, desc
            )
            self.propositions.add_proposition(prop)
            count += 1
        
        if self.verbose:
            print(f"  Created {count} goal propositions")

    def _format_goal_name(self, goal_predicate: Tuple, idx: int) -> str:
        tokens = [str(x) for x in goal_predicate]
        if not tokens:
            base = f"goal_{idx}"
        else:
            pred = tokens[0].lower()
            pred = self._normalize_goal_predicate(pred)
            if len(tokens) == 2:
                base = f"{pred}_{tokens[1]}"
            elif len(tokens) >= 3:
                base = f"{pred}_{tokens[1]}_{tokens[2]}"
            else:
                base = "_".join(tokens)
        base = (
            base.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")
            .lower()
        )
        name = base
        if name in self.propositions.prop_dict:
            name = f"{base}_goal_{idx}"
        return name

    def _normalize_goal_predicate(self, pred: str) -> str:
        return pred

    def _format_goal_description(self, goal_predicate: Tuple) -> str:
        tokens = [str(x) for x in goal_predicate]
        if not tokens:
            return ""
        pred = tokens[0]
        args = ", ".join(tokens[1:])
        return f"{pred}({args})"
    
    def _create_goal_proposition(
        self, prop_name: str, goal_predicate: Tuple, desc: str
    ) -> AtomicProposition:
        """Create a goal predicate proposition"""
        
        def eval_fn(env, gp=goal_predicate):
            try:
                return env._eval_predicate(gp)
            except:
                return False
        
        return AtomicProposition(
            name=prop_name,
            type=PropositionType.GOAL_PREDICATE,
            args=goal_predicate,
            eval_fn=eval_fn,
            description=desc,
            category="goal"
        )
    
    # ========== Summary & Stats ==========
    
    def get_statistics(self) -> dict:
        """Get generation statistics"""
        stats = {}
        for category in self.propositions.categories:
            count = len(self.propositions.categories[category])
            stats[category] = count
        stats["total"] = len(self.propositions)
        return stats
    
    def print_statistics(self):
        """Print generation statistics"""
        stats = self.get_statistics()
        print("\n=== Proposition Generation Statistics ===")
        for category, count in sorted(stats.items()):
            print(f"  {category}: {count}")
