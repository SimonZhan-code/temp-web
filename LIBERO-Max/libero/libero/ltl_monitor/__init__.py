"""LTL monitoring utilities for LIBERO-Max.

The monitor consumes per-step atomic proposition labels emitted by
``BDDLBaseDomain.step()`` and tracks task-level LTL progress via a real
LDBA built by Rabinizer (with cached HOA fallback). The default task monitor
checks eventual satisfaction of all BDDL goal propositions; task-specific
specs can encode ordered subgoals and safety constraints.

Compilation pipeline:

    LTL formula  ──►  HOA (sidecar | PREBUILT_HOA | Rabinizer subprocess)
                 ──►  LDBA  (HOAParser → prune → complete_sink_state → SCCs)
                 ──►  LTLMonitor  (multi-state, epsilon, reach-avoid via search)
"""

from .automata import LDBA, LDBATransition, SCC
from .builder import LDBABuildError, build_ldba, resolve_hoa
from .hoa_parser import HOAParser
from .ldba_sequence import LDBASequence
from .llm_generation import (
    OPENROUTER_API_KEY_ENV,
    OpenRouterClient,
    build_generation_messages,
    build_task_generation_context,
    compose_generation_record,
    iter_suite_generation_contexts,
    validate_generated_constraints,
    validate_ltl_formula_syntax,
)
from .monitor import LTLMonitor, MonitorState
from .rabinizer import RABINIZER_PATH, run_rabinizer
from .search import ExhaustiveSearchSimple, NoPathsException
from .task_specs import (
    PREBUILT_HOA,
    TASK_LTL_SPECS,
    TASK_SPECS,
    TaskLTLSpec,
    build_monitor_for_task,
    build_monitor_from_spec,
    get_task_ltl_spec,
    infer_goal_props_from_info,
    infer_safety_props_from_label,
)

__all__ = [
    # Automata core
    "LDBA",
    "LDBATransition",
    "SCC",
    "LDBASequence",
    "HOAParser",
    "build_ldba",
    "resolve_hoa",
    "LDBABuildError",
    "run_rabinizer",
    "RABINIZER_PATH",
    "ExhaustiveSearchSimple",
    "NoPathsException",
    # Runtime monitor
    "LTLMonitor",
    "MonitorState",
    # Task spec API
    "TaskLTLSpec",
    "TASK_SPECS",
    "TASK_LTL_SPECS",
    "PREBUILT_HOA",
    "OPENROUTER_API_KEY_ENV",
    "OpenRouterClient",
    "build_generation_messages",
    "build_task_generation_context",
    "compose_generation_record",
    "iter_suite_generation_contexts",
    "validate_generated_constraints",
    "validate_ltl_formula_syntax",
    "build_monitor_from_spec",
    "build_monitor_for_task",
    "get_task_ltl_spec",
    "infer_goal_props_from_info",
    "infer_safety_props_from_label",
]
