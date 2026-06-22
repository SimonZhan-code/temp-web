"""
Atomic Proposition Framework for LTL Construction

This module provides automatic extraction and evaluation of atomic propositions
from BDDL environments, enabling LTL specification and verification.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Callable, Dict, Tuple
import numpy as np


class PropositionType(Enum):
    """Types of atomic propositions"""
    UNARY_STATE = "unary_state"              # obj.is_open()
    BINARY_RELATION = "binary_relation"      # on(obj_i, obj_j)
    REGION_CONTAINMENT = "region"            # in(obj_i, region)
    GOAL_PREDICATE = "goal"                  # goal state from BDDL
    GRIPPER_STATE = "gripper"                # gripper.holding(obj)
    COMPOSITE = "composite"                  # AND/OR of sub-propositions


@dataclass
class AtomicProposition:
    """Represents a single atomic proposition for LTL"""
    
    name: str                           # "moka_pot_1_on_flat_stove_1"
    type: PropositionType              # PropositionType.BINARY_RELATION
    
    # Arguments: stored for serialization/debugging
    # Format: (obj_name,) or (obj_1, pred_name, obj_2) etc.
    args: Tuple                        
    
    # Labeling function: takes env, returns bool
    eval_fn: Callable                  
    
    # Metadata
    description: str = ""
    category: str = ""
    
    def evaluate(self, env) -> bool:
        """
        Evaluate this proposition in the given environment state.
        
        Args:
            env: BDDLBaseDomain environment instance
            
        Returns:
            bool: Truth value of this proposition
        """
        try:
            return self.eval_fn(env)
        except Exception as e:
            print(f"Error evaluating {self.name}: {e}")
            return False
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return isinstance(other, AtomicProposition) and self.name == other.name
    
    def __repr__(self):
        return f"Prop({self.name})"


class PropositionSet:
    """Manages and evaluates a set of atomic propositions"""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.propositions: List[AtomicProposition] = []
        self.prop_dict: Dict[str, AtomicProposition] = {}
        self.categories: Dict[str, List[str]] = {}  # category → [prop_names]
    
    def add_proposition(self, prop: AtomicProposition) -> None:
        """Add a proposition to the set"""
        if prop.name not in self.prop_dict:
            self.propositions.append(prop)
            self.prop_dict[prop.name] = prop
            
            # Index by category
            if prop.category:
                if prop.category not in self.categories:
                    self.categories[prop.category] = []
                self.categories[prop.category].append(prop.name)
    
    def get_proposition(self, name: str) -> Optional[AtomicProposition]:
        """Retrieve a proposition by name"""
        return self.prop_dict.get(name)
    
    def get_propositions_by_category(self, category: str) \
            -> List[AtomicProposition]:
        """Get all propositions in a category"""
        names = self.categories.get(category, [])
        return [self.prop_dict[n] for n in names]
    
    def get_label(self, env) -> np.ndarray:
        """
        Get truth vector for all propositions.
        
        Args:
            env: BDDLBaseDomain environment instance
            
        Returns:
            np.ndarray: Boolean array of shape (num_propositions,)
        """
        return np.array([prop.evaluate(env) for prop in self.propositions],
                       dtype=bool)
    
    def get_label_dict(self, env) -> Dict[str, bool]:
        """
        Get truth values as dictionary.
        
        Args:
            env: BDDLBaseDomain environment instance
            
        Returns:
            dict: {prop_name: bool}
        """
        return {prop.name: prop.evaluate(env) for prop in self.propositions}
    
    def get_label_by_category(self, env) -> Dict[str, np.ndarray]:
        """
        Get truth values grouped by category.
        
        Args:
            env: BDDLBaseDomain environment instance
            
        Returns:
            dict: {category: boolean array}
        """
        result = {}
        for category in self.categories:
            props = self.get_propositions_by_category(category)
            result[category] = np.array([p.evaluate(env) for p in props],
                                      dtype=bool)
        return result
    
    def __len__(self) -> int:
        return len(self.propositions)
    
    def __iter__(self):
        return iter(self.propositions)
    
    def __getitem__(self, idx: int) -> AtomicProposition:
        return self.propositions[idx]
    
    def __repr__(self):
        return f"PropositionSet({self.name}, {len(self)} propositions)"
    
    def print_summary(self):
        """Print summary of propositions by category"""
        print(f"\n=== PropositionSet: {self.name} ===")
        print(f"Total propositions: {len(self)}\n")
        
        for category in sorted(self.categories.keys()):
            props = self.get_propositions_by_category(category)
            print(f"{category} ({len(props)} propositions):")
            for prop in props:
                print(f"  - {prop.name}")
                if prop.description:
                    print(f"    {prop.description}")
            print()


class SafetyPropositionType(Enum):
    """Additional proposition types for safety tracking."""
    SAFETY_VIOLATION = "safety_violation"   # object displaced beyond threshold
    DANGER_ZONE = "danger_zone"             # EEF inside a designated hazard region


class LTLLabelingFunction:
    """
    Wrapper for evaluating LTL formulas over atomic propositions.
    
    Usage:
        ltl_labeler = LTLLabelingFunction(proposition_set)
        label = ltl_labeler.evaluate(env)  # np.array of shape (num_props,)
    """
    
    def __init__(self, proposition_set: PropositionSet):
        self.prop_set = proposition_set
        self.num_propositions = len(proposition_set)
        self.prop_index = {prop.name: i for i, prop 
                          in enumerate(proposition_set)}
    
    def evaluate(self, env) -> Tuple[np.ndarray, Dict[str, bool]]:
        """
        Evaluate all propositions.
        
        Returns:
            Tuple of (label_array, label_dict)
        """
        label_array = self.prop_set.get_label(env)
        label_dict = self.prop_set.get_label_dict(env)
        return label_array, label_dict
    
    def evaluate_formula(self, env, formula: str) -> bool:
        """
        Evaluate a simple LTL-like formula.
        Supports: prop_name, ~prop_name, (prop_a & prop_b), (prop_a | prop_b)
        
        Args:
            env: Environment instance
            formula: Formula string, e.g., "moka_pot_on_stove & ~gripper_holding"
            
        Returns:
            bool: Truth value of formula
        """
        label_dict = self.prop_set.get_label_dict(env)
        
        # Simple formula evaluation (for basic cases)
        formula = formula.strip()
        
        # Replace proposition names with their values
        for prop_name in label_dict:
            value = "True" if label_dict[prop_name] else "False"
            formula = formula.replace(prop_name, value)
        
        # Replace logical operators
        formula = formula.replace("~", "not ")
        formula = formula.replace("&", "and")
        formula = formula.replace("|", "or")
        
        try:
            return eval(formula)
        except Exception as e:
            print(f"Error evaluating formula: {e}")
            return False
    
    def __repr__(self):
        return f"LTLLabeler({self.num_propositions} propositions)"
