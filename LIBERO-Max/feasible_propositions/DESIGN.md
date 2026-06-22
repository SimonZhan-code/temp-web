# `feasible_propositions/` — design & scene↔suite correlation

This folder holds the **atomic-proposition (AP) data, organized by scene**, for the
compositional-generalization training paradigm. It replaces the original
per-suite/per-task dumps. This doc explains the layout, the naming scheme, and how
each scene maps back to the **original LIBERO formulation**, so another agent can
follow the correlation without re-deriving it.

## TL;DR

- The unit of organization is a **scene** (a fixed environment = fixed object /
  region / fixture inventory), not a *suite* and not a *task*.
- Each scene folder has exactly two files:
  - `fixed_alphabet.json` — the scene's **state-AP alphabet** (the feasibility
    substrate), stored once; plus `safety_violation` *avoid* candidates for
    SafeLIBERO scenes.
  - `goal_aps.json` — the **goal-AP alphabet** (the reach-subgoal vocabulary) +
    the per-task anchor goals.
- **Composition = reach over the goal-AP alphabet** (`F(a & F(b & ...))`). The
  non-goal alphabet is the feasibility substrate (preconditions / mutex), not
  composed subgoals. SafeLIBERO will additionally use non-goal **avoid** clauses
  (`G(!ap)`) — those candidates live in `fixed_alphabet.json`.
- Current contents: **47 scenes, 190 tasks** (see `index.json`).

## Layout

```
feasible_propositions/
  <scene_id>/
    fixed_alphabet.json            # state-AP alphabet (once) + safety avoid APs
    goal_aps.json                  # reach goal-AP vocabulary + per-task anchor goals
    compositions_up_to_<d>.json    # (optional) enumerated feasible compositions for THIS scene
  scene_units.json          # cross-scene manifest (alphabet + anchor goals + derived analysis)
  index.json                # rich scene index (provenance + compositionality)
  DESIGN.md                 # this file
```

Each scene folder is self-contained: the two source files, plus (once generated)
its own enumerated compositions. The two cross-scene aggregates (`scene_units.json`,
`index.json`) live at the top level.

## Scene naming ↔ original LIBERO formulation

The original benchmark is a set of **suites** across families (Original LIBERO,
LIBERO-Object, SafeLIBERO, LIBERO-10-R, LIBERO-Pro). Scenes here are derived from
the kept suites as follows:

| Current scene id | Origin (original formulation) |
|------------------|-------------------------------|
| `KITCHEN_SCENE*`, `LIVING_ROOM_SCENE*`, `STUDY_SCENE*` | **Core**: `libero_90` (+ the long-horizon `libero_10`), grouped by the scene token. Where a `libero_10_r` scene is the *same environment* (alphabet subset), it is merged in too (see Merge rule). |
| `libero_spatial` | Original LIBERO **`libero_spatial`** suite — one fixed kitchen-table scene (2 black bowls, ramekin, cookies, plate + stove, cabinet). 10 tasks vary the bowl's *initial spatial relation*; the goal is always `on(bowl, plate)`. **Single reach goal → not reach-compositional.** |
| `libero_goal` | Original LIBERO **`libero_goal`** suite — one fixed kitchen scene (bowl, cream cheese, plate, wine bottle + stove, wine rack, cabinet). 10 distinct goals. **Reach-compositional.** |
| `libero_object` | Original LIBERO **`libero_object`** suite — one grocery-table scene, 10 "pick item → basket" tasks. Inventory **rotates per task** (`alphabet_fixed=false`) → object-identity generalization, **not reach-compositional**. |
| `r_<token>` | **LIBERO-10-R** OOD scene (robustness / composition / visual split over LIBERO-10) that was **not** merged into a core scene (different environment despite a shared token, or a token with no core counterpart). |
| `safe_<token>` / `safe_<suite>` | **SafeLIBERO** scene (obstacle-avoidance). Same base scene as a core one **plus obstacle objects**; carries `safety_violation` (`*_displaced`) **avoid** APs. Kept separate from core (different evaluation axis). |

`index.json` records, per scene, the exact `origin_suites` and a `source_family`
(`core`, `core+libero_10_r`, `libero_10_r`, `safelibero`, `object`).

### Removed from the original formulation
All **LIBERO-Pro** perturbation suites (`*_with_*` stickers/box/mug/book,
`*_temp_*`, episode/trigger variants) and the misc `libero_mine` /
`libero_study_table` were **dropped**: most are not focused on compositional
capability and many have missing custom assets. (Recoverable from git history.)

## Core ↔ LIBERO-10-R merge rule

A `libero_10_r` scene `r_<T>` is **merged into** the core scene `<T>` iff one's
non-goal alphabet is a **subset** of the other's (identical or subset) — i.e. they
are the *same environment*; merging unions their goal vocabularies. If the
alphabets are **incompatible** (neither is a subset → different object identities),
they are **kept separate** (`<T>` and `r_<T>` both exist), because union-merging
would fabricate compositions realizable in no real environment. SafeLIBERO is
never auto-merged into core even when subset-related (different eval axis).

## The two files

`fixed_alphabet.json`
```json
{
  "scene_id", "suites", "num_tasks", "alphabet_fixed",
  "objects", "fixtures",
  "alphabet": {
    "unary_state":        [{"name","args","init_value"}],
    "binary_relation":    [...],
    "region_containment": [...],
    "safety_violation":   [...]   // present only for safe_* scenes (avoid candidates)
  }
}
```
- `alphabet_fixed`: `true` if all the scene's tasks share an identical non-goal
  alphabet (true for the alphabet-clean compositional scenes).

`goal_aps.json`
```json
{
  "scene_id", "num_tasks",
  "goal_alphabet": [{"name","args"}],     // the reach-subgoal vocabulary (union of task goals)
  "tasks": [ {"task","suite","language_instruction","goals":[{"name","args","init_value"}]} ]
}
```
- `tasks[].goals` = each original task's **anchor goal** (in-distribution
  composition). Sampled compositions that equal an anchor set are flagged
  `is_held_out=false`.

## Composition semantics (what "compositional" means here)

- A **composition** is a feasible **ordered list of goal-AP subgoals** — achieve
  `subgoals[0]`, then `subgoals[1]`, … (order matters). It is deliberately **not**
  an LTL formula: training only needs sequential reach-tracking of the list, so no
  LDBA / Rabinizer is involved at train time. (`sampler.lift_to_ltl` can still
  produce an LTL form if eval-time monitoring wants one.)
- Feasibility is guaranteed **by construction** by the rule-based transition system
  (single-gripper; open-before-place into gated drawers; per-region init open-state
  read from open/close goal-AP `init_value`). No Rabinizer needed.
- `index.json.reach_compositional` is `true` when the scene is `alphabet_fixed`
  **and** the transition system yields ≥1 feasible **depth-2** composition.
  Of 47 scenes, 36 are reach-compositional. The three notable suite scenes:
  `libero_goal` ✅, `libero_spatial` ✗ (single goal), `libero_object` ✗ (rotating
  inventory; multi-object compositions over its union are *virtual*).

## index.json schema (v4)

Top: `version, layout, num_scenes, num_tasks, kept_suites,
removed_suite_families, source_family_counts, reach_compositional_count, scenes`.
Per scene: `source_family, origin_suites, num_tasks, alphabet_fixed,
num_state_aps, num_goal_aps, num_safety_avoid_aps, has_articulation,
feasible_compositions_depth2, reach_compositional, tasks`.

## Tooling (how this was built / how to regenerate)

Code lives in `libero/libero/envs/ltl_utils/composition/`
(`manifest.py`, `factored_state.py`, `transitions.py`, `sampler.py`, `filters.py`)
and `scripts/`. Run scripts in the **`libero-max` conda env**.

Pipeline (in order):
1. `scripts/dump_feasible_propositions.py` — per-task AP dumps (original suite layout; historical).
2. `scripts/restructure_feasible_by_scene.py` — suite→scene folders; drop LIBERO-Pro.
3. `scripts/split_scene_files.py` — per-task dumps → `fixed_alphabet.json` + `goal_aps.json`.
4. `scripts/merge_core_r_scenes.py` — merge same/subset core+`r_` scenes.
5. `scripts/build_scene_manifest.py` — write top-level `scene_units.json`.
6. `scripts/build_scene_index.py` — write top-level `index.json`.
7. `scripts/sample_ap_compositions.py --scene <id> --max-depth <d>` — enumerate feasible compositions into `<scene_id>/compositions_up_to_<d>.json`.

`manifest.build_scene_units()` reads this scene-keyed layout (one SceneUnit per
folder). `transitions.build_scene_model(unit)` + `sampler.enumerate_walks(model, d)`
produce the feasible compositions.
