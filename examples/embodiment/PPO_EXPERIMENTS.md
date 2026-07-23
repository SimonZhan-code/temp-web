# PPO experiments — KITCHEN_SCENE4 subgoal compositions (goal-conditioned)

Purpose: **isolate whether the subgoal-decomposition / goal-conditioned formulation is learnable** by
training the policy (PPO) instead of freezing it. V-MPO Option A (frozen policy, critic-only + eval-time
best-of-N) isn't steering at inference; PPO trains BOTH policy and critic by gradient on the *exact same*
env / tasks / prompts. If PPO learns the depth-1 goal-conditioned tasks, the decomposition is valid and the
bottleneck was V-MPO's frozen/BoN mechanism.

## Mode switch = which config you launch (no code changes)

Same composition env, same depth-1 KITCHEN_SCENE4 tasks, same NL/AP prompts. Only the algorithm differs:

| mode | selected by | configs |
|---|---|---|
| **V-MPO** (critic-only + best-of-N, frozen policy) | `adv_type: vmpo`, `loss_type: vmpo` | `kitchen4_composition_vmpo_*` |
| **PPO** (policy + critic, gradient) | `adv_type: gae`, `loss_type: actor_critic` | `kitchen4_composition_ppo_*` |

All V-MPO logic (temperature dual, λ, segmented GAE) is config-gated → absent in the PPO configs ⇒ no-op.
V-MPO configs/code are untouched.

## Pipeline validated ✅ (toy smoke, 2× H100)

The end-to-end PPO loop was validated on a 2× H100 (NVLink) node with `kitchen4_composition_ppo_nl_toy`
(2 envs, 50-step episodes, 3 epochs, frozen VLM), disaggregated placement (actor GPU0, rollout GPU1).
Run: `northwestern-ideas2/neuralsym-vla/runs/gniofiwu`. It confirmed:
- **rollout → NCCL weight sync → GAE → actor_critic loss → optimizer step → eval** all run
  (`time/time/sync_weights≈1.0s` — disaggregated NCCL over NVLink, **no ptrace / CUDA-IPC needed**);
- PPO trains **policy + critic** (real `approx_kl`, `clip_fraction`, `policy_loss`, `grad_norm` 58–119 —
  vs V-MPO's collapsed ~1e-3);
- the depth-1 `on_` single-goal eval task runs cleanly (per-env identity fix confirmed in production).

(success=0 in a 3-step toy is expected — mechanical smoke only, not a learning test.)

## RESULT ✅ — PPO learns (frozen), decomposition is valid

The frozen PPO runs (`l0ck8r6n` AP, `y4r3mfyc` NL) confirm the hypothesis: with only the action expert
trained, both **train and eval success rise** — NL `env/success_once` 0.21→0.34, `eval/success_once`
0.12→**0.44 peak**; AP 0.05→0.30, eval →**0.375 peak**; critic `explained_variance` up to 0.97. So V-MPO's
frozen-everything + best-of-N was the bottleneck, not the subgoal decomposition.

**Current focus: FROZEN configs only.** The unfrozen (full-unfreeze) variant runs mechanically but does
not train properly in practice — it is parked. Use `kitchen4_composition_ppo_{nl,ap}_frozen` for all
experiments (including the reach-avoid / depth≥2 work).

## Runs to train (all four)

Queue below. Start with the frozen NL variant (cheapest), then AP, then the unfrozen variants.

| # | config | prompt | freeze scope | FSDP | hardware | status |
|---|---|---|---|---|---|---|
| 1 | `kitchen4_composition_ppo_nl_frozen`   | NL | VLM frozen (expert+value) | no_shard | 8×H100 (or 2-GPU disagg) | **learning ✅** |
| 2 | `kitchen4_composition_ppo_ap_frozen`   | AP | VLM frozen (expert+value) | no_shard | 8×H100 (or 2-GPU disagg) | **learning ✅** |
| 3 | `kitchen4_composition_ppo_nl_unfrozen` | NL | **full unfreeze** (VLM+expert+value) | no_shard + use_orig_params | 8×H100 (or 2-GPU disagg) | **parked — not training properly, do not use** |
| 4 | `kitchen4_composition_ppo_ap_unfrozen` | AP | **full unfreeze** | no_shard + use_orig_params | 8×H100 (or 2-GPU disagg) | **parked — not training properly, do not use** |

Shared: `adv_type: gae`, `loss_type: actor_critic`, bare PPO (`kl_beta: 0`, `entropy_bonus: 0`),
`value_after_vlm: True` (state value), `add_value_head: True`, **no best-of-N**, `gae_lambda: 0.95`,
`gamma: 0.99`, depth-1 train + **blended depth-1/2 real-task eval** (`eval/*_d1`, `*_d2` per-depth metrics,
`ignore_terminations: False` so `eval/episode_len` = time-to-success). `gradient_checkpointing: False`
everywhere (openpi does not support it).

Frozen vs unfrozen differ in: `train_expert_only` (True/False); **unfrozen adds `use_orig_params: True`**
(required for a trainable VLM — avoids the FSDP FlatParameter view/in-place autograd error); and batch sizing
(frozen `micro 128` / `global 1024`, unfrozen `micro 16` to fit the trainable 3B). Both use `no_shard`
(`full_shard + cpu_offload` breaks the openpi VLM forward — do not use).

## Reach-avoid experiment (frozen VLM, avoid penalty on)

Same frozen recipe + `composition.avoid_beta` set **in the top-level config** (the shared env configs stay
at `0.0` = baseline; eval envs are always penalty-free). One un-commanded goal-AP toggle (e.g. the memorized
drawer-close no subgoal asked for) subtracts β from the reach reward; undoing an achieved subgoal counts too.

| # | config | prompt | avoid_beta | launch |
|---|---|---|---|---|
| 5 | `kitchen4_composition_ppo_nl_frozen_b0`   | NL | **0.0** (explicit baseline) | `bash examples/embodiment/run_embodiment.sh kitchen4_composition_ppo_nl_frozen_b0` |
| 6 | `kitchen4_composition_ppo_nl_frozen_b025` | NL | **0.25** | `... kitchen4_composition_ppo_nl_frozen_b025` |
| 7 | `kitchen4_composition_ppo_nl_frozen_b05`  | NL | **0.5**  | `... kitchen4_composition_ppo_nl_frozen_b05` |
| 8 | `kitchen4_composition_ppo_ap_frozen_b0`   | AP | **0.0** (explicit baseline) | `... kitchen4_composition_ppo_ap_frozen_b0` |
| 9 | `kitchen4_composition_ppo_ap_frozen_b025` | AP | **0.25** | `... kitchen4_composition_ppo_ap_frozen_b025` |
| 10 | `kitchen4_composition_ppo_ap_frozen_b05` | AP | **0.5**  | `... kitchen4_composition_ppo_ap_frozen_b05` |

The `_b0` configs are reward-identical to the plain `_frozen` configs (#1/#2) but use **250-step
train rollouts/episodes** (vs 500): depth-1 episodes finish well under 250 steps, so the shorter
window doubles update frequency per wall-clock. They anchor the {b0, b025, b05} sweep under parallel
wandb experiment names. (If comparing β variants head-to-head, note #6/#7/#9/#10 still use 500-step
rollouts — match them to 250 first if you want the sweep fully controlled.)

Same launch steps as below (model_path sed applies to these files too). What to compare vs the β=0
baselines (#1/#2): **`env/avoid_violations` should fall** (the penalty working) while **`env/success_once` /
`eval/success_once` should NOT collapse** — if success craters at β=0.5, the penalty is too harsh relative
to the +1 subgoal reward (violating trajectories are currently also the succeeding ones); fall back to 0.25
(or 0.1). `rollout/reach_rewards` can go **negative** early — expected under the penalty, not a bug.

## Launch

```bash
# 0) get the fixes + activate the openpi venv (run_embodiment.sh uses `python` from PATH):
git pull
source .venv/bin/activate            # or wherever the openpi venv lives

# 1) set the SFT checkpoint path IN the config (run_embodiment.sh forwards ONLY the config name,
#    so a hydra override on the command line will NOT apply). Edit both model_path lines:
#      rollout.model.model_path  and  actor.model.model_path
#    e.g.:  sed -i "s#/path/to/model/Pi05-LIBERO-SFT#$CKPT#g" \
#             examples/embodiment/config/kitchen4_composition_ppo_*_frozen*.yaml

# 2) env for headless MuJoCo render + logging:
ulimit -n 65535   # REQUIRED at >=64 envs: each MuJoCo env opens many asset files; the default
                  # soft limit (1024) crashes env creation with "MjModel.from_xml_string:
                  # Caught an unknown exception!"
# VAST.AI CONTAINERS ONLY: cap envs at <=8 PER ENV-WORKER PROCESS. Empirically (2-GPU vast
# nodes, driver 580.95): 2/8 envs per process work, 32+ crash in MuJoCo scene construction
# (libc++abi terminating) regardless of render backend (EGL and osmesa) and with ulimit
# raised — a container quirk, NOT memory. Real 8xH100 hardware runs the production layout
# (256 envs = 32/rank) fine. Also: global_batch must divide the rollout sample count
# (num_envs x rollout_steps / num_action_chunks), e.g. 8 envs x 25 chunks = 200 -> use 100.
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export WANDB_API_KEY=...

# --- 8x H100 (collocated, max throughput) — needs ptrace_scope=0 (root, once): ---
sudo sysctl -w kernel.yama.ptrace_scope=0
bash examples/embodiment/run_embodiment.sh kitchen4_composition_ppo_nl_frozen   # then _ap_frozen (unfrozen variants are parked)

# --- 2-GPU node (disaggregated, actor GPU0 / rollout GPU1) — NCCL, NO ptrace (works in vast): ---
# run_embodiment.sh only forwards the config name, so set the placement IN the config:
#   cluster:
#     component_placement:
#       actor: 0      # GPU 0
#       rollout: 1    # GPU 1  (NCCL weight sync, no ptrace/IPC)
#       env: 0
# (this is exactly what kitchen4_composition_ppo_nl_toy already uses). On NVLink omit NCCL_P2P_DISABLE.
bash examples/embodiment/run_embodiment.sh kitchen4_composition_ppo_nl_frozen
```
`kitchen4_composition_ppo_nl_toy` is the minimal 2-GPU disaggregated smoke (validated above). The four
training configs use collocated placement by default; edit their `cluster.component_placement` to the
disaggregated block above to run them on a 2-GPU node.

## What to watch

- **`env/success_once` (train)** should **rise** above the frozen-SFT baseline (confirmed for frozen).
- **`eval/success_once` + per-depth `eval/success_once_d1` / `eval/success_once_d2`**: blended depth-1/2 real
  tasks. d1 should track train; d2 is the harder generalization signal (see coverage caveat below).
- **`eval/episode_len` (+ `_d1`/`_d2`)**: now falls as the policy solves tasks faster (`ignore_terminations: False`).
- **`train/critic/explained_variance`**: PPO critic should be positive/stable (contrast V-MPO's noisy ≈0).
- **Unfrozen only**: watch for drift/collapse (bare PPO, no KL anchor). If it collapses, revisit adding a
  KL-to-SFT anchor (`kl_beta > 0`) + small `entropy_bonus`.

## Atomicity audit + canonical NL prompts (2026-07-22)

An audit of "are depth-1 subgoals really atomic given the initial state?" found the **task spec is
sound**: LIBERO-Max's generator enforces open-before-place when enumerating compositions; all 132
KITCHEN_SCENE4 chains are ordering-safe; and an empirical MuJoCo sweep of real init states shows every
goal AP false at init in every trial (bottom drawer physically open, top drawer closed) — no hidden
composites exist in the current setup (`tools/audit_init_atomicity.py` re-runs this for any scene).

The observed "asked to place into a closed drawer" behavior traces to **OOD prompt language**, now fixed:

- **Canonical NL prompts** (`composition.prompt_nl_canonical: True`, default): subgoals render in the
  ORIGINAL LIBERO task language ("put the black bowl in the bottom drawer of the cabinet", "open the top
  drawer", "put the black bowl on top of the cabinet") instead of mechanical region names ("...white
  cabinet bottom region", "...top side") the SFT never saw. ⚠ **NL runs before/after this change are not
  directly comparable** (prompt distribution changed); set `prompt_nl_canonical: False` to reproduce old
  prompts. AP mode unaffected.
- **Runtime precondition guard**: each episode's chain is verified against the ACTUAL first ltl_label.
  A hidden composite is **expanded** (the missing `open_*` subgoal is inserted → explicit ordered chain,
  `env/comp_expanded`); if not expressible in the alphabet, the chain is **resampled**
  (`env/comp_resampled`). `env/precond_broken` counts mid-episode gate breakage (policy closed the
  drawer its current subgoal needs). All three are zero in current KITCHEN_SCENE4 — the guard exists so
  new scenes / randomized inits cannot silently corrupt training.
- **Depth-scaled episode limits** (`composition.steps_per_subgoal`, e.g. 250): episode cap =
  `steps_per_subgoal × chain depth` instead of a fixed `max_episode_steps` — depth-2 chains get 500
  steps, and precondition expansion automatically lengthens the budget. Enabled in the `_b0` configs
  (== 250 at depth-1, so their behavior is unchanged until deeper curricula). `0` = legacy fixed cap.
  Episodes may span rollout epochs (auto-reset + bootstrap), so rollout tensor shapes are unaffected.

## V-MPO value-head best-of-N — sim verification (2026-07-23)

The PPO results above show the *decomposition* is learnable but leave the V-MPO question open: does
**value-head best-of-N** (V-MPO's actual policy-improvement mechanism) work if the critic is trained
long enough? The Wan world-model path is meant to score candidates by *imagined outcomes*, but the
pi0.5↔Wan gap is large (WM trained on OpenVLA-OFT actions, no proprio/wrist, binary-saturated reward
model, in-WM eval ≠ real sim). To **de-risk the idea before investing in WM fidelity**, run the exact
V-MPO mechanism with the **real MuJoCo sim as the rollout env** (perfect dynamics) and the **critic's
value head as the candidate scorer** — no world model in the loop. If `eval/success_once` climbs over a
full training session, the value head is learning to rank chunks and the idea holds; the remaining
problem is then *only* WM precision. If it stays flat even with perfect sim dynamics, the bottleneck is
the value-head/BoN scoring itself (consistent with the PPO-vs-V-MPO contrast above), not the WM.

Config: **`kitchen4_composition_vmpo_sim100`** — the 500-epoch NL reference
(`kitchen4_composition_vmpo_fast_nl`) with `max_epochs` cut to **100** and the parallelism scaled from
8 GPUs down to the proven 2-GPU disaggregated shape.

| knob | value | why |
|---|---|---|
| `adv_type` / `loss_type` | `vmpo` / `vmpo` | V-MPO critic-only (policy stays frozen; no actor gradient) |
| `best_of_n_scorer` | `value` (default) | the critic's value head ranks the N candidate chunks |
| `best_of_n_mode` | `eval` (default) | BoN applied at eval; training rollouts sample normally |
| `best_of_n` | **8** | candidates per decision (matches the fast_nl reference) |
| `value_after_vlm` | **False** | value is conditioned on the sampled action → BoN *can* discriminate |
| `detach_critic_input` | **True** | critic grad must NOT flow into the shared action-expert trunk (keeps the SFT policy frozen — critic-only V-MPO); default is False, must be set |
| rollout env | **real LIBERO sim** | `env/kitchen4_ltl_composition_nl`, **`composition.max_depth: 1`** (NOT the Wan WM) |
| eval env | `env/kitchen4_compositional_eval_nl_d12` | depth-1&2 blend, per-depth `eval/*_d1`/`*_d2`, 16 fixed init states |
| `max_epochs` / `val_check_interval` | 100 / 10 | eval every 10 epochs → 11-point learning curve incl. epoch-0 baseline |
| placement | actor GPU0, rollout GPU1, env GPU0 | NCCL weight sync (no ptrace/CUDA-IPC) — the unprivileged-container path |
| envs / batch | 16 train, 16 eval / micro 8, global 16 | 2-GPU shape (scale envs up on bigger boxes for denser success sampling) |

**Launch** (2-GPU node; `run_embodiment.sh` forwards ONLY the config name, so all knobs incl. placement
and `model_path` live in the config — CLI hydra overrides do NOT apply):

```bash
# model_path is already set in-config to ${REPO_PATH}/models/Pi05-LIBERO-SFT; if your checkpoint is
# elsewhere, sed both model_path lines (rollout.model + actor.model) as for the PPO configs above.
ulimit -n 65535
export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export WANDB_API_KEY=$(cat /root/.wandb_key)     # project: neuralsym-vla
export NCCL_P2P_DISABLE=1                          # omit on NVLink boxes
bash examples/embodiment/run_embodiment.sh kitchen4_composition_vmpo_sim100
```

**What to watch** (wandb `neuralsym-vla`, run `kitchen4_vmpo_sim_valuebon_100ep`):
- **`eval/success_once`** across the 11 eval points — the headline. Rising above the epoch-0 baseline ⇒
  the value head is learning to rank; flat ⇒ scoring is the bottleneck.
- **`env/success_once` + `rollout/rewards`** — must be **non-zero** (NL prompts fix the AP-OOD zero-reward
  failure; if zero, the prompt/label path regressed).
- **`train/critic/value_loss`, `explained_variance`** — the critic should actually fit (contrast V-MPO's
  historically collapsed ≈1e-3 / noisy-≈0). A useful value head is the precondition for value-BoN to work.
- **eval-vs-baseline gap** measures the *selection ceiling* of value-BoN, directly comparable to the
  Phase-I oracle-BoN ceiling (SFT 20.8% → oracle 23.1%) and to the PPO-frozen learning curve.

### Validated ✅ (2-GPU, 2×A100-80GB) — ready to scale on 8×H100

The pipeline was validated end-to-end on a 2-GPU node (`kitchen4_composition_vmpo_sim100`, run
`2y4akwbq`): model load → NCCL weight sync (`sync_weights≈0.8s`) → epoch-0 eval → V-MPO critic training,
all clean. **Epoch-0 baseline (frozen SFT + value-BoN, untrained critic): `eval/success_once_d1 = 0.50`,
`_d2 = 0.00`, blended 0.47.** Train rollout `env/success_once ≈ 0.12–0.19` (depth-1, reach rewards
firing), critic `explained_variance` climbing −19 → −0.4 in 5 steps, `value_loss ≈ 0.02–0.04`. This
confirms the mechanism runs and the critic fits; the full learning-curve run happens at scale.

**Large-scale run → `kitchen4_composition_vmpo_nl_8xh100`** (this validated recipe scaled to 8×H100:
256 train envs, `micro 128`/`global 2048`, `rollout_epoch 8`, `update_epoch 4`, 500 epochs, collocated
CUDA-IPC placement — needs `sudo sysctl -w kernel.yama.ptrace_scope=0`). Same NL + depth-1 + d12-eval +
`detach_critic_input: True` + value-BoN recipe. Launch:
`bash examples/embodiment/run_embodiment.sh kitchen4_composition_vmpo_nl_8xh100` (set both `model_path`s
first). Watch `eval/success_once_d1` across the eval points vs the 0.50 baseline — rising ⇒ the value
head learns to rank (idea holds; WM precision is the only remaining gap); flat ⇒ value-BoN scoring is
the bottleneck (matches the PPO-vs-V-MPO contrast).

## Notes

- **Reach channel is now LIVE (reach-avoid update).** Previously the per-subgoal
  `ltl_reach_rewards` were silently stripped at the env→rollout obs whitelist and PPO trained on the
  accept-only task reward via the `.get("reach_rewards", rewards)` fallback. At **depth-1** the two are
  semantically identical (+1 at the same sim step when the single subgoal == acceptance), so the frozen-run
  results above remain a valid baseline. At **depth≥2**, intermediate subgoal +1s now actually reach GAE —
  required for the composition curriculum. Reach/cost flow as first-class per-sim-step `[B, chunk]`
  channels through `chunk_step → EnvOutput → rollout`, so mid-chunk events and auto-reset chunks are
  captured. Watch `rollout/reach_rewards` (live training signal) alongside `rollout/rewards` (task reward).
- **Avoid penalty (optional)**: `composition.avoid_beta` in the train env configs (default `0.0` = old
  behavior). With `avoid_beta > 0`, each un-commanded goal-AP toggle (e.g. the memorized drawer-close no
  subgoal asked for) subtracts `avoid_beta` from the reach reward. `env/avoid_violations` (per-episode
  violation count) is logged regardless, so run with `0.0` first to see the baseline violation rate.
- The frozen variant trains the **action expert** by PPO (VLM frozen) — the real contrast vs V-MPO's
  frozen-everything + BoN. The unfrozen variant adds the VLM (`no_shard` + `use_orig_params: True`).
- **Depth-2/3 eval coverage caveat**: `libero_90` KITCHEN_SCENE4 has only **1 real depth-2 task** (close+open)
  and **no depth-3** tasks. So `eval/*_d2` is a single, sparsely-sampled task; depth-3 would need `libero_10`
  tasks (cross-suite) or synthetic sample-mode compositions.
- Depth-1 training (matched to the V-MPO experiments); deeper curriculum is a later step.
