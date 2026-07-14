# V-MPO value-head experiments — KITCHEN_SCENE4 (Option A)

Fast tuning loop testing value-head training levers, with a **depth-1-matched real-task eval**
so best-of-N's effect is finally measurable. All runs share:

- **Option A** (policy frozen): `detach_critic_input: True`, `best_of_n: 8` (eval-only), `value_after_vlm: False`.
- **Train**: depth-1 sampled compositions (`composition.max_depth: 1`), 256 envs, 500 epochs, `val_check_interval: 10`.
- **Eval**: REAL single-goal KITCHEN_SCENE4 tasks (`task_goals`, `min_goals: 1`, `max_goals: 1` — the 5 single-goal tasks), matched to training difficulty.
- **wandb project**: `neuralsym-vla`. Set `rollout.model.model_path` / `actor.model.model_path` (hydra override or edit) to the SFT checkpoint.

## Configs to run

| config | prompt | `update_epoch` | `normalize_returns` | isolates |
|---|---|---|---|---|
| `kitchen4_composition_vmpo_fast_nl_e2`    | NL | 2 | False | control (current epoch setting, now with visible eval) |
| `kitchen4_composition_vmpo_fast_nl_e4`    | NL | 4 | False | +2× critic epochs |
| `kitchen4_composition_vmpo_fast_nl_e4_rn` | NL | 4 | True  | +2× critic epochs + return normalization |
| `kitchen4_composition_vmpo_fast_ap_e2`    | AP | 2 | False | control |
| `kitchen4_composition_vmpo_fast_ap_e4`    | AP | 4 | False | +2× critic epochs |
| `kitchen4_composition_vmpo_fast_ap_e4_rn` | AP | 4 | True  | +2× critic epochs + return normalization |

Comparisons: `e2` vs `e4` = doubled-critic-epoch effect; `e4` vs `e4_rn` = return-normalization effect; NL vs AP throughout.

## Launch

```bash
# one-time on the 8x H100 box:
sudo sysctl -w kernel.yama.ptrace_scope=0
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

# single config:
bash examples/embodiment/run_embodiment.sh kitchen4_composition_vmpo_fast_nl_e4_rn

# all six sequentially:
for c in kitchen4_composition_vmpo_fast_nl_e2 kitchen4_composition_vmpo_fast_nl_e4 kitchen4_composition_vmpo_fast_nl_e4_rn \
         kitchen4_composition_vmpo_fast_ap_e2 kitchen4_composition_vmpo_fast_ap_e4 kitchen4_composition_vmpo_fast_ap_e4_rn; do
  bash examples/embodiment/run_embodiment.sh "$c"
done
```

## What to watch

- **`eval/success_once` should now be NON-ZERO** (eval difficulty matches training). Whether it exceeds the
  raw single-subgoal train baseline (~17% NL / ~6% AP) is the **best-of-N gain** — the core Option A signal.
- **`train/critic/explained_variance`** should be higher / more stable with `e4` and/or `_rn`
  (previously swung to −1.2 … +0.9, mean negative).
- **`train/critic/value_loss`** should stay stable under `_rn` (returns are standardized per batch; a brief
  transient at the value-clip scale switch is expected, then it should settle).

## Notes

- Depth-1 eval **replaces** the depth-2 eval for these configs; the original `kitchen4_compositional_eval*`
  (2-goal) and the unused LDBA eval configs are left untouched.
- These do not change the frozen-policy setup — no policy fine-tuning (that would be "Option B").
