---
name: vmpo-composition-verification
description: "Verified state of the V-MPO + KITCHEN_SCENE4 AP-composition pipeline (items checked, bugs found, open issues)"
metadata: 
  node_type: memory
  type: project
  originSessionId: fefb401b-2b2b-4690-a88d-ba4513e5a39b
---

Verification of the V-MPO + KITCHEN_SCENE4 AP-composition training (on V-MPO branch). See also [[local-run-environment]].

**Verified working:**
- AP sampling: `CompositionSampler` draws 132/132 distinct ordered compositions, all subgoals within the scene `goal_alphabet`, no adjacent repeats.
- Subgoal propagation: `LiberoCompositionEnv` advances the pointer when the current subgoal AP becomes true and switches the VLA prompt to the next AP (`render_subgoal_ap` → `pred(args)`); out-of-order satisfaction does NOT skip. Prompt is AP-only (no NL, no `reach_avoid_texts`).
- 10-epoch run completes on 2×A100-40GB (NVLink, disaggregated, NCCL), with eval at epoch 0 (`runner.eval_at_start`) and epoch 10 (`val_check_interval`), wandb logging to project `neuralsym-vla` (entity northwestern-ideas2). Config: `kitchen4_composition_vmpo_2gpu_10ep.yaml`.

**Bug found + fixed — best-of-N guided sampling (commit f5b08c4):** It was a no-op for two reasons. (1) `get_model` only applies `actor.model.openpi.*` overrides to `OpenPi0Config`, so a top-level `actor.model.best_of_n` is IGNORED (stays at default 1) — best_of_n MUST live under `actor.model.openpi`. (2) With `value_after_vlm: True` the value is read from the SHARED VLM prefix (identical for all N candidates → identical scores → always picks candidate 0). Fix: best_of_n under openpi + `value_after_vlm: False`. Verified: value_after_vlm True → score spread [0,0], idx [0,0]; False → spread ~0.1, idx [1,3]. Best-of-N is EVAL-ONLY (`sample_actions`: `mode=="eval" and best_of_n>1`), and `predict_action_batch` does NOT surface `best_of_n_scores/idx` — call `sample_actions` directly to inspect.

**LDBA multi-step subgoal-switch — FIXED (commit 88e7589).** The ap_prompt eval (`LiberoLTLEnv` + `ap_prompt`, on KITCHEN_SCENE3) now extracts each subgoal in turn via `immediate_reach_props(ldba, states)` in `libero_ltl_env.py`: a reverse-BFS distance-to-acceptance returning the positive guard (REQUIRED props = intersection across the transition's feasible assignments, so the safety prop left free by `G(!displaced)` is excluded) of the transition that makes the most progress from the CURRENT state, plus a "done" check once a finite spec has an accepting visit. Verified on `F(turn_on & F(moka_on))`: `turn_on(flat_stove_1)` → `on(moka_pot_1, flat_stove_1)` → `done`. Replaces the broken full-sequence search extraction (`ExhaustiveSearchSimple` + num_loops: 0 dropped the accepting-loop reach → premature done; 1 over-merged). Scoped to the ap_prompt path. NOTE still open: `_get_ltl_rewards` derives its DENSE reach reward from the old search sequence; eval success uses LDBA acceptance (correct), so prompt/subgoal extraction is fixed but the dense-reward path could be migrated to `immediate_reach_props` later.

**Eval limitation — RESOLVED via the LIBERO-Max canonical cache (commit 6bab35a, submodule bumped to 54af0d0).** LIBERO-Max now ships `ltl_monitor/canonical_task_ltl_labels.json` (per `<suite>/<task>`: resolved LTL formula + goal/safety atoms) and `ltl_monitor/hoa_store.json` (resolved_formula → prebuilt HOA), covering all 190 tasks including every KITCHEN_SCENE4 compositional task — no online Rabinizer. `rlinf/envs/libero/ltl_cache.py:build_ldba_from_cache(suite, task)` builds an `ltl_benchmark` LDBA from it (owl-format HOAs parse fine with `HOAParser`). `LiberoLTLEnv._build_ldba` prefers the cache (falls back to the 1-entry TASK_SPECS); `_reset_ldba_state` now takes the initial epsilon jump only if the automaton has one (owl HOAs don't; `get_next_states(take_epsilon=True)` asserts one exists). Same-scene eval config: `env/kitchen4_eval_ldba_samescene.yaml` (KITCHEN_SCENE4, min_goals=2). NOTE: the instance must have the updated submodule (`uv pip install -e LIBERO-Max` after bump) for the cache to be on the import path; the in-progress 50-epoch scaled run still uses the old KITCHEN_SCENE3 eval.
