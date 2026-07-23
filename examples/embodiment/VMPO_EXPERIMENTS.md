# V-MPO experiments — KITCHEN_SCENE4 subgoal compositions (frozen policy + best-of-N)

V-MPO here is **critic-only**: the pi0.5 policy is **frozen** (no actor gradient — `vmpo_actor_loss`
is never enabled), and the ONLY policy-improvement mechanism is **best-of-N selection** at eval. The
critic (a value head on the VLM) trains by the V-MPO loss; at eval the policy proposes N candidate
action chunks and a **scorer** ranks them, committing the argmax. The open question this doc's
experiments answer: **can the scorer actually rank candidates well enough to lift success?**

The contrast experiment (PPO trains policy+critic by gradient on the same tasks) lives in
[`PPO_EXPERIMENTS.md`](PPO_EXPERIMENTS.md); PPO-frozen lifted eval 0.12→0.44, proving the subgoal
decomposition is learnable and isolating the V-MPO best-of-N mechanism as the thing to validate.

## The defining axis: the best-of-N scorer

| scorer | knob | how candidates are ranked | rollout env |
|---|---|---|---|
| **value** | `best_of_n_scorer: value` (default) | the critic's value head scores each candidate | real sim OR WM |
| **oracle** | `best_of_n_scorer: external` + sim save/restore | MuJoCo rolls out every candidate, scores by true reach events (a **perfect** WM) | real sim |
| **wm** | `best_of_n_scorer: external` + Wan WM | the world model imagines every candidate, its ResNet reward model scores them | Wan world model |

The strategy is to validate the mechanism with the **perfect scorer first** (oracle / value-in-sim),
then move to the imperfect Wan WM — so a null result can be attributed to WM fidelity vs. the idea
itself. See memory `vmpo-sim-value-bon-verify`, `wan-world-model-rlinf`.

## Non-negotiable knobs (get these wrong → silent failure)

- **`env.train.composition.max_depth: 1`** — the base env samples depth 1–3, but the frozen SFT can
  only do single subgoals; without this override the unachievable depth-2/3 tasks **zero out reward**
  (`env/reward=0`, `reach_rewards=0`), which looks like the AP-OOD zero-reward bug but is a curriculum bug.
- **`actor.model.openpi.detach_critic_input: True`** — default is **False**. With it False the critic
  gradient flows into the shared action-expert trunk and **corrupts the frozen SFT policy** (no longer
  critic-only). Every production config sets it True; smoke/2gpu configs inherited the False default.
- **NL prompts** (`env/kitchen4_ltl_composition_nl`) — the AP-format prompt is OOD for the SFT and
  drives reward to ~0. Use the `_nl` train env and an `_nl` eval env.
- **`value_after_vlm: False`** — makes the value action-conditioned so best-of-N *can* discriminate
  candidates (with `True` the value is a pure state value, identical across candidates).

## Value-head best-of-N in the REAL sim (the main line)

Original V-MPO: value head ranks candidates, rollouts in the real MuJoCo sim (no world model). If
`eval/success_once_d1` climbs over training, the value head learns to rank ⇒ the idea holds and WM
precision becomes the only remaining gap; if flat even with perfect sim dynamics, the value-BoN scoring
is the bottleneck.

| config | scale | prompt | epochs | placement | status |
|---|---|---|---|---|---|
| `kitchen4_composition_vmpo_sim100` | 2-GPU (16 envs) | NL | 100 | disaggregated (NCCL) | **validated ✅** (2×A100-80GB) |
| `kitchen4_composition_vmpo_nl_8xh100` | 8×H100 (256 envs) | NL | 500 | collocated (CUDA-IPC) | **ready — large-scale run** |
| `kitchen4_composition_vmpo_8xh100` | 8×H100 (256 envs) | AP | 500 | collocated | AP variant (zero-reward risk) |
| `kitchen4_composition_vmpo_fast_nl` | 8-GPU (256 envs) | NL | 500 | collocated | fast-tuning (rollout_epoch 1) |

**Validated (2-GPU, 2×A100-80GB), run `2y4akwbq`:** pipeline runs end-to-end (model load → NCCL
`sync_weights≈0.8s` → epoch-0 eval → V-MPO critic training). Epoch-0 baseline (frozen SFT + value-BoN,
untrained critic): `eval/success_once_d1 = 0.50`, `_d2 = 0.00`; train `env/success_once ≈ 0.12–0.19`
(reach rewards firing); critic `explained_variance` −19 → −0.4 in 5 steps.

### Train commands

Shared env for headless MuJoCo render + logging (all configs):
```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export WANDB_API_KEY=...          # project: neuralsym-vla
ulimit -n 65535                    # REQUIRED at >=64 envs (MuJoCo opens many asset files)
# run_embodiment.sh forwards ONLY the config name — set model_path IN the config (both
# rollout.model.model_path and actor.model.model_path), CLI hydra overrides do NOT apply:
#   sed -i "s#/path/to/model/Pi05-LIBERO-SFT#$CKPT#g" examples/embodiment/config/<config>.yaml
```

**8×H100 large-scale (collocated, max throughput):**
```bash
sudo sysctl -w kernel.yama.ptrace_scope=0    # one-time, root — collocated CUDA-IPC weight sync
bash examples/embodiment/run_embodiment.sh kitchen4_composition_vmpo_nl_8xh100
```

**2-GPU node (disaggregated, actor GPU0 / rollout GPU1 — NCCL, NO ptrace, works in vast containers):**
```bash
export NCCL_P2P_DISABLE=1          # omit on NVLink boxes
bash examples/embodiment/run_embodiment.sh kitchen4_composition_vmpo_sim100
```
(`sim100` already has the disaggregated `cluster.component_placement` block; for the 8×H100 configs on
an unprivileged 2-GPU node, uncomment their disaggregated placement block instead of setting ptrace.)

**What to watch** (wandb `neuralsym-vla`): `eval/success_once_d1` vs the 0.50 baseline (rising ⇒ value
head learns to rank); `env/success_once` + `rollout/reach_rewards` must be **non-zero** (else NL/depth
regressed); `train/critic/value_loss` + `explained_variance` (the critic must fit — the precondition
for value-BoN to work). Eval metrics are NOT in stdout — read from tensorboard
(`logs/*/tensorboard/events.out.tfevents*`, `EventAccumulator`) or wandb.

## Oracle best-of-N (perfect-WM ceiling)

`kitchen4_composition_vmpo_oraclebon` — `best_of_n_scorer: external`; the env snapshots MuJoCo state,
rolls out every candidate, scores by true reach events, and commits the winner (the sim as a perfect
world model). Measures the **selection ceiling** for best-of-N before any WM fidelity loss. Same launch
as above (real sim). Result (Phase I short run): SFT 20.8% → oracle-BoN 23.1% (`bon_disc_frac ≈ 0.005`
— the event scorer is blind mid-episode; a graded/progress scorer is the open improvement).

## Wan world model as the rollout env (Phase II)

`wan_spatial_vmpo_pi05` — `best_of_n_scorer: external` with the pretrained RLinf Wan WM as the env
backend: rollouts happen **in imagination**, best-of-N branches the N futures and commits the
reward-model argmax. Requires the two WM rollout servers running first (wanspike venv):
```bash
# start servers (see tools/wan_spike/wan_rollout_server.py + env/wan_spatial_wm.yaml), then:
bash examples/embodiment/run_embodiment.sh wan_spatial_vmpo_pi05
```
This is the imagination path — code-complete but the pi0.5↔Wan gap is large (WM trained on OFT actions,
no proprio/wrist, binary-saturated RM). See memory `wan-world-model-rlinf` for the full state and the
node-rebuild recipe. **Left as-is** pending the sim value-BoN verdict.

## Smoke / dev configs (2-GPU)

`kitchen4_composition_vmpo_2gpu` (2ep), `_2gpu_scaled` (50ep), `_2gpu_10ep`, `_smoke` (1 env) — mechanical
smokes. NOTE these inherit `detach_critic_input: False` and lack the `max_depth: 1` override, so they are
for pipeline smoke only, NOT valid learning runs — use `sim100` for a correct 2-GPU experiment.
