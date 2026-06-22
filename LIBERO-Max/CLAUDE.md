# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

LIBERO-Max unifies four LIBERO-family manipulation benchmarks (original LIBERO, LIBERO-Pro OOD, Safety LIBERO, LIBERO-10-R) behind one suite registry, and layers an LTL (linear temporal logic) specification + monitoring stack on top of the robosuite/MuJoCo environments. The Python package is named `libero` and lives under `libero/` (note the doubled path: `libero/libero/...`).

## Commands

```bash
# Install (robosuite/MuJoCo required for anything that builds an env)
pip install -r requirements.txt
pip install -e .

# Unit tests — pure logic, no MuJoCo needed (run from repo root)
PYTHONPATH=. pytest libero/libero/ltl_monitor/tests/test_monitor.py -v
PYTHONPATH=. pytest libero/libero/ltl_monitor/tests/test_llm_generation.py -v
PYTHONPATH=. pytest libero/libero/envs/ltl_utils/tests/test_proposition_feasibility.py -v
PYTHONPATH=. pytest libero/libero/benchmark/tests/test_family_taxonomy.py -v

# Single test
PYTHONPATH=. pytest path/to/test_file.py::TestClass::test_name -v

# Integration smoke test — builds real envs across all sources, needs MuJoCo + assets
python test_env.py

# Print the physically-feasible atomic-proposition set for one task (needs MuJoCo)
python scripts/print_feasible_propositions.py --task <TASK_NAME>

# Generate LTL constraints via LLM (needs OPENROUTER_API_KEY + OPENROUTER_MODEL)
python scripts/generate_ltl_constraints.py --all-suites --model <model> --output out.json
python scripts/generate_ltl_constraints.py --suite libero_10 --dry-run --output out.json  # prompts only
```

There is no `conftest.py`/`pytest.ini`; tests rely on `PYTHONPATH=.`. Many tests fake the env (monkeypatch / lightweight fake objects) precisely so they run without MuJoCo — prefer that pattern for new logic tests, since importing anything under `libero.libero.envs` triggers `envs/__init__.py`, which imports `robosuite` at module load.

## Architecture

### Suite registry (the unifying layer)
`libero/libero/benchmark/family.py` is the single entry point. `get_libero_suite(name)` returns a `LiberoFamilySuite`; `get_libero_suite_spec(name)` returns the underlying `LiberoSuiteSpec`. `_build_registry()` merges the four families into one `LIBERO_FAMILY_REGISTRY`, normalizing each suite to a common spec with: `source` (`original`/`safety`/`pro`/`libero_10_r`), `taxonomy_level_1` (`baseline_id`/`constraint_safety`/`distribution_shift_ood`), `evaluation_tags`, `max_steps`, and `bddl_root_fn`/`init_root_fn` callables. Filter helpers: `get_libero_suite_names_by_source(...)`, `..._by_eval_tags(...)`, `..._by_taxonomy_level_1(...)`. A suite's tasks map to `bddl_files/<suite>/<task>.bddl` plus a parallel init-state file under `init_states/` (SafeLIBERO uses `..._Level{I,II}.pruned_init` and `supports_level_init`).

### BDDL → runnable env
`ControlEnv`/`OffScreenRenderEnv` (`libero/libero/envs/env_wrapper.py`) take a `.bddl` path, call `BDDLUtils.get_problem_info` to read the problem name, then dispatch through `TASK_MAPPING[problem_name]` (populated by `@register_problem` in `bddl_base_domain.py`) to a concrete domain in `libero/libero/envs/problems/`. `BDDLBaseDomain` parses the file once into `self.parsed_problem` (`robosuite_parse_problem` in `bddl_utils.py`) with keys `objects`, `fixtures`, `regions`, `goal_state`, `initial_state`. The domain's `_load_objects/_fixtures/_sites_in_arena` instantiate MuJoCo objects, and `_check_success` evaluates `goal_state` predicates via `_eval_predicate`.

`OffScreenRenderEnv` forces offscreen rendering; for physics-only/headless use, construct `ControlEnv` directly with `has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False`. `env.env` is the underlying `BDDLBaseDomain` (the wrapper delegates).

### Object registry
BDDL declares instances as `<instance> - <category>` (e.g. `basket_1 - basket`). `get_object_fn(category)` (`envs/objects/__init__.py`) looks up `OBJECTS_DICT`, which `@register_object` (`envs/base_object.py`) populates by snake-casing each class name. Capability is encoded by base class, not flags: `ArticulatedObject` subclasses implement `is_open(qpos)`; toggleable objects (e.g. `FlatStove`) implement `turn_on(qpos)`; rigid `HopeBaseObject`/`GoogleScannedObject` implement neither. Probe the *underlying* object (`env.get_object(name)`) for affordances — the `ObjectState` wrapper in `envs/object_states/` always defines `is_open`/`turn_on` and will mislead you.

### Predicate evaluation
`envs/predicates/` holds the registry `VALIDATE_PREDICATE_FN_DICT` and `eval_predicate_fn(key, *states)`. Registry keys are `open/close/turnon/turnoff/on/in/...` — note these differ from the generator's verbose tokens (`is_open`, `turn_on`), which is why `ltl_utils/feasibility.py` maps between them. `On` calls `arg2.check_ontop(arg1)`; `In` calls `arg2.check_contain(arg1)` (region sites use `SiteObject.in_box`; plain objects lack `in_box`, so object-to-object `in` is not evaluable and containment is expressed via region sites).

### LTL atomic propositions
`env.get_ltl_propositions()` (`bddl_base_domain.py`) lazily runs `AtomicPropositionGenerator` (`envs/ltl_utils/proposition_generator.py`), producing a `PropositionSet` (`ltl_utils/__init__.py`) over four levels: unary state (L1), binary relation (L2), region containment (L3), and goal predicates from `:goal` (L4). `SafetyAtomicPropositionGenerator` adds `<obj>_displaced` safety props (L5). `ltl_utils/feasibility.py` applies physical-feasibility filtering (Layer 0 evaluability gate / Layer 1 type compatibility / Layer 2 region geometry) so the generator emits only evaluable, physically-feasible APs. `env.step()` attaches `ltl_label`, `ltl_label_array`, `ltl_goal_desc`, `ltl_task_spec` to `info`.

### LTL monitor
`libero/libero/ltl_monitor/` turns a task's goal/safety props into a runtime monitor. Flow: `get_task_ltl_spec(prop_set, task_id, bddl_file)` (`task_specs.py`) resolves a formula — from the hand-curated `TASK_LTL_SPECS` registry, or auto-generated as `F(goal_1 & ... )` (with `G(!safety)` for SafeLIBERO and ordered open→goal→close patterns when detected). `build_monitor_from_spec` / `build_ldba` (`builder.py`) produce an LDBA (`automata.py`) from a sidecar `.hoa` file, the in-code `PREBUILT_HOA` cache, or the external Rabinizer binary; `LTLMonitor.step(label_dict)` (`monitor.py`) advances the automaton and reports acceptance/violation/reach-avoid subgoals. `search.py` extracts the shortest reach-avoid sequence. Constraint *generation* lives in `llm_generation.py` (OpenRouter client), driven by `scripts/generate_ltl_constraints.py`, writing nested `benchmark → suite → task` JSON to `ltl_monitor/generated_task_ltl_constraints*.json`.

### Vectorized gym wrapper
`libero/env_wrapper/` (distinct from `envs/env_wrapper.py`) provides `LiberoEnv`, a vectorized `gym.Env` over subprocess workers, configured by OmegaConf (see `evaluation_config.yaml`): suite name, num envs, auto-reset, init-state sampling, video recording, and optional LTL monitoring (`enable_ltl_monitor: true` surfaces `ltl_accepted`/`ltl_violated`/`ltl_reach_reward`/`ltl_safety_cost` in `infos`).

## Conventions & gotchas

- **Import-path duality:** code/tests import `libero.libero.X` with a fallback to `libero.X` (try/except). Preserve both when adding cross-module imports in `ltl_monitor`/`ltl_utils`.
- **`LIBERO_LTL_SPEC_MODE`** selects spec resolution: default `goal_only` (always `F(goal_1 & ...)`); `auto` consults the registry and ordering heuristics. Tests set this via `monkeypatch`.
- **Other env vars:** `OPENROUTER_API_KEY` / `OPENROUTER_MODEL` (LLM generation), `RABINIZER_PATH` (external ltl2ldba; unset → `PREBUILT_HOA` fallback), `render_gpu_device_id` (`-1` = CPU offscreen).
- **Silent-False eval:** proposition eval closures swallow exceptions to `False`. A "False" label can mean "genuinely false" or "couldn't evaluate" — distinguish these when debugging (a bad predicate key, missing method, or missing site all read as False).
- **LIBERO_MAX_task_curation.md** documents the suite taxonomy/curation rationale; consult it before changing suite tags or taxonomy levels.
