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

## Runs to train (all four)

Queue below. Start with the frozen NL variant (cheapest), then AP, then the unfrozen variants.

| # | config | prompt | freeze scope | FSDP | hardware | status |
|---|---|---|---|---|---|---|
| 1 | `kitchen4_composition_ppo_nl_frozen`   | NL | VLM frozen (expert+value) | no_shard | 8×H100 (or 2-GPU disagg) | ready |
| 2 | `kitchen4_composition_ppo_ap_frozen`   | AP | VLM frozen (expert+value) | no_shard | 8×H100 (or 2-GPU disagg) | ready |
| 3 | `kitchen4_composition_ppo_nl_unfrozen` | NL | **full unfreeze** (VLM+expert+value) | full_shard + cpu_offload | 8×H100 | ready (memory-heavy) |
| 4 | `kitchen4_composition_ppo_ap_unfrozen` | AP | **full unfreeze** | full_shard + cpu_offload | 8×H100 | ready (memory-heavy) |

Shared: `adv_type: gae`, `loss_type: actor_critic`, bare PPO (`kl_beta: 0`, `entropy_bonus: 0`),
`value_after_vlm: True` (state value), `add_value_head: True`, **no best-of-N**, `gae_lambda: 0.95`,
`gamma: 0.99`, depth-1 train + depth-1 real-task eval. `gradient_checkpointing: False` everywhere (openpi
does not support it).

Frozen vs unfrozen differ only in: `train_expert_only` (True/False), `sharding_strategy`
(no_shard / full_shard + cpu_offload), `actor.enable_offload`, and batch/env sizing (256/64 vs 128/16).

## Launch

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export WANDB_API_KEY=...   # for wandb logging
# Set the SFT checkpoint path in the config (rollout/actor model_path) or via hydra override.

# --- 8x H100 (collocated, max throughput) — needs ptrace_scope=0 (root, once): ---
sudo sysctl -w kernel.yama.ptrace_scope=0
bash examples/embodiment/run_embodiment.sh kitchen4_composition_ppo_nl_frozen   # then _ap_frozen, _nl_unfrozen, _ap_unfrozen

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

- **`env/success_once` (train)**: under PPO this should **rise** above the frozen-SFT baseline (V-MPO held a
  flat ~17% NL / ~6% AP). A rising curve = the goal-conditioned formulation is learnable → decomposition valid.
- **`eval/success_once`**: on the depth-1 real single-goal tasks (matched difficulty). Should follow train.
- **`train/critic/explained_variance`**: PPO critic should be positive/stable (contrast V-MPO's noisy ≈0).
- **Unfrozen only**: watch for drift/collapse (bare PPO, no KL anchor). If it collapses, revisit adding a
  KL-to-SFT anchor (`kl_beta > 0`) + small `entropy_bonus`.

## Notes

- The frozen variant still trains the **action expert** (the part that outputs actions) by PPO — the real
  contrast vs V-MPO's frozen-everything + BoN — just with the VLM backbone frozen. The unfrozen variant adds
  the VLM.
- Depth-1 only (matched to the V-MPO experiments); deeper curriculum is a later step.
