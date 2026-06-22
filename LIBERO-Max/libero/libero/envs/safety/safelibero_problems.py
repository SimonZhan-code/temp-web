"""
Per-arena SafeLIBERO domain resolution + dispatch.

SafeLIBERO bddls share their ``problem_name`` with regular LIBERO
(``libero_tabletop_manipulation``, ``libero_floor_manipulation``,
``libero_living_room_tabletop_manipulation``), so they can't be routed to a safety
domain by problem name alone. ``ControlEnv`` calls ``get_safety_domain_class`` only
when the bddl path is a SafeLIBERO task (``is_safelibero_bddl``); regular LIBERO
dispatch is untouched.

The arena problem classes are decorated with ``@register_problem``, which does NOT
return the class (the module-level name is ``None``) -- the real classes live only
in ``TASK_MAPPING``. So we resolve the arena from ``TASK_MAPPING`` and build a
safety subclass LAZILY (first use, after registration), each with its arena's own
metaclass, injecting the safety behavior from ``SafetyMixin``.
"""

from libero.libero.envs.safety.safelibero_base_domain import SafetyMixin

# problem_names (lowercased, as BDDLUtils returns them) that get safety treatment.
_SAFE_ARENA_PROBLEMS = {
    "libero_tabletop_manipulation",
    "libero_floor_manipulation",
    "libero_living_room_tabletop_manipulation",
}

_SAFETY_CLASS_CACHE = {}


def _add_safety(cls):
    """Inject SafeLIBERO behavior into a single-inheritance arena subclass."""

    def __init__(self, *args, safety_threshold: float = 0.001, **kwargs):
        self.safety_threshold = safety_threshold
        self._initial_obstacle_positions = {}
        self._cumulative_safety_violated = False
        super(cls, self).__init__(*args, **kwargs)

    def _reset_internal(self):
        super(cls, self)._reset_internal()
        self._cumulative_safety_violated = False
        self._initial_obstacle_positions = SafetyMixin._snapshot_object_positions(self)

    def step(self, action):
        obs, reward, done, info = super(cls, self).step(action)
        violations = SafetyMixin.get_safety_info(self)
        if violations:
            self._cumulative_safety_violated = True
        info["safety_violated"] = self._cumulative_safety_violated
        info["safety_violations"] = violations
        prop_set = self.get_ltl_propositions()
        info["safety_label"] = {
            p.name: p.evaluate(self)
            for p in prop_set.get_propositions_by_category("safety_violation")
        }
        return obs, reward, done, info

    def _assert_problem_name(self):
        # The safety subclass is named SafeLibero_* (differs from the bddl
        # problem_name); the correct arena was already chosen by dispatch.
        pass

    cls.__init__ = __init__
    cls._reset_internal = _reset_internal
    cls.step = step
    cls.get_safety_info = SafetyMixin.get_safety_info
    cls.get_ltl_propositions = SafetyMixin.get_ltl_propositions
    cls._assert_problem_name = _assert_problem_name
    return cls


def get_safety_domain_class(problem_name: str):
    """Return the SafeLIBERO domain class for ``problem_name``, or None.

    Built lazily from ``TASK_MAPPING`` (so the arena classes are registered) with
    the arena's own metaclass; cached per problem name.
    """
    pn = str(problem_name).lower()
    if pn not in _SAFE_ARENA_PROBLEMS:
        return None
    if pn in _SAFETY_CLASS_CACHE:
        return _SAFETY_CLASS_CACHE[pn]

    from libero.libero.envs.bddl_base_domain import TASK_MAPPING

    arena = TASK_MAPPING.get(pn)
    if arena is None:
        return None
    # Build with the arena's metaclass so single-inheritance class creation
    # (and robosuite env registration) behaves exactly like the arena's own.
    cls = type(arena)(f"SafeLibero_{arena.__name__}", (arena,), {})
    _add_safety(cls)
    _SAFETY_CLASS_CACHE[pn] = cls
    return cls


def is_safelibero_bddl(bddl_file_name) -> bool:
    """True if the BDDL path belongs to a SafeLIBERO suite."""
    return "safelibero" in str(bddl_file_name).lower()
