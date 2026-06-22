"""Boolean-formula primitives backing HOA transition guards.

Ported from Neuralsym-VLA/ltl_benchmark/logic/.
"""

from .assignment import Assignment, FrozenAssignment

__all__ = ["Assignment", "FrozenAssignment"]
