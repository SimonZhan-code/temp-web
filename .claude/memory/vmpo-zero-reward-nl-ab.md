---
name: vmpo-zero-reward-nl-ab
description: "V-MPO composition runs get zero reward/success (AP-prompt OOD diagnosis); the fast NL-vs-AP A/B built to test it, plus wandb eval-video logging"
metadata: 
  node_type: memory
  type: project
  originSessionId: fefb401b-2b2b-4690-a88d-ba4513e5a39b
---

State of the V-MPO + KITCHEN_SCENE4 composition training as of 2026-07-07 (V-MPO branch). Builds on [[vmpo-composition-verification]] and [[local-run-environment]].

**wandb lives in project `northwestern-ideas2/neuralsym-vla`** (NOT `.../rlinf` — that project holds unrelated example runs like `libero_spatial_ppo_openpi_pi05`). Query runs with the `behavior` conda env (`~/anaconda3/envs/behavior/bin/python`, has `wandb`); the libero-max env does not. API key was provided by the user in-session (not stored here).

**Core finding — the reward never fires.** Every composition run (`kitchen4_composition_vmpo_2gpu_10ep`, `_10ep_best_of_n_active`, `_scaled_50ep`, and the full `kitchen4_composition_vmpo_8xh100` run id `286jzt12`) shows `env/success_once = 0`, `env/reward ≈ 0`, `rollout/rewards ≈ 0` throughout. The +1/subgoal reward never triggers because the frozen pi0.5 policy never achieves even the FIRST subgoal. Downstream V-MPO/critic metrics are just idling on a zero-reward stream: `grad_norm` collapses toward 0, `returns_mean`→0, `explained_variance` negative/meaningless (near-zero target variance), `eta` drifts up, `top_half_frac`≈0.5. Not an infra/algorithm/wiring bug — scalarization is `reach_rewards` when λ=0 (`fsdp_actor_worker.py`), reach is just always 0.

**Root-cause hypothesis (leading):** the AP-format prompt (`"Atomic-proposition subgoal to achieve: in(akita_black_bowl_1, white_cabinet_1_bottom_region)"`) is OUT-OF-DISTRIBUTION for the NL-pretrained SFT checkpoint, and the frozen VLM backbone (`train_expert_only`) can't adapt to it → ineffective actions → zero reward. Not yet proven; needs the NL control below.

**The 8xh100 run is also far too slow to tune:** ~3664 s/step (61 min) — `generate_rollouts` 66% (`rollout_epoch: 8` × 256 envs × 500 steps), `actor_training` 34%. At `max_epochs: 1000` that's ~42 days.

**Built this session (decisive A/B + fast loop):**
- `composition.prompt_style` knob in `libero_composition_env.py`: `ap` (default, unchanged) vs `nl` (new `render_subgoal_nl`/`_humanize` → e.g. `"put the akita black bowl in the white cabinet bottom region"`). Tracker/switch/reward identical; only prompt text changes. 21/21 unit tests pass.
- Env configs `env/kitchen4_ltl_composition_nl.yaml` (train) + `env/kitchen4_compositional_eval_nl.yaml` (eval, task_goals mode) — prompt-matched NL.
- Fast tuning top-level configs `kitchen4_composition_vmpo_fast_nl.yaml` and `kitchen4_composition_vmpo_fast_ap.yaml` (identical sizing, only prompt differs): rollout_epoch 1, 64 envs, horizon 200, update_epoch 2, global_batch 512, max_epochs 30, val_check 10, best_of_n 1 → target ~3-5 min/step. Run both, compare `env/success_once`: NL lifts off → prompt was the cause; both stay zero → env/label wiring bug instead. NOT yet run.
- wandb eval-video logging: `MetricLogger.log_video()` + `EmbodiedRunner._log_eval_videos()` upload eval rollout mp4s under `eval/video` (deduped, capped by `runner.logger.max_eval_videos`). Single-node/shared-FS only; never raises. Committed `d2e3768`.

**Collaborator external-compute copy:** repo `git@github.com:SimonZhan-code/temp-web.git` (branch `main`) at `/data/Projects/temp-web` mirrors Neuralsym-VLA for runs on rented compute; keep the two files (`metric_logger.py`, `embodied_runner.py`) in sync when pushing (last sync `c66d882`).
