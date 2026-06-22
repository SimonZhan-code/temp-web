# Neuralsym-VLA

Neuralsym-VLA extends [RLinf](https://github.com/RLinf/RLinf) with a BDDL Atomic Proposition Framework for LIBERO environments, enabling structured symbolic state representation for LTL-guided reinforcement learning of VLA policies.

For base framework documentation (training pipeline, rollout, actors, distributed setup), refer to the upstream RLinf README.

---

## What's Added: BDDL Atomic Proposition Framework

This project adds automatic extraction and evaluation of atomic propositions from BDDL environments, enabling LTL specification and verification in LIBERO tasks.

### Proposition Types

| Level | Type | Example | Description |
|-------|------|---------|-------------|
| 1 | `UNARY_STATE` | `moka_pot_1_is_open` | Intrinsic object state predicate |
| 2 | `BINARY_RELATION` | `moka_pot_1_on_flat_stove_1` | Object-object spatial relation |
| 3 | `REGION_CONTAINMENT` | `frypan_1_in_cook_region` | Object within a named BDDL region |
| 4 | `GOAL_PREDICATE` | `on_moka_pot_1_flat_stove_1` | Task goal state from BDDL `:goal` |

### Core Classes

- **`AtomicProposition`** — A single proposition with a name, type, arguments, and an `eval_fn(env) -> bool`.
- **`PropositionSet`** — Manages a collection of propositions; provides `get_label(env) -> np.ndarray` and `get_label_dict(env) -> dict`.
- **`LTLLabelingFunction`** — Wraps a `PropositionSet` and supports formula evaluation (`~`, `&`, `|`).
- **`AtomicPropositionGenerator`** — Automatically generates all four proposition levels from a `BDDLBaseDomain` instance.

Source: `LIBERO/libero/libero/envs/ltl_utils/`

### Safety Propositions (SAFETY_LIBERO)

| Level | Type | Example | Description |
|-------|------|---------|-------------|
| 5 | `SAFETY_DISPLACED` | `frypan_1_displaced` | Movable object displaced beyond safety threshold |

Source: `SAFETY_LIBERO/safelibero/envs/ltl_utils/safety_proposition_generator.py`

---

## LTL-to-Automaton Pipeline

The `ltl_benchmark/` module compiles LTL specifications into Limit Deterministic Büchi Automata (LDBA) and extracts reach-avoid subgoals at runtime. This follows the architecture from [GenZ-LTL](https://github.com/BU-DEPEND-Lab/GenZ-LTL) and [DeepLTL](https://github.com/mathiasj33/deep-ltl).

### Pipeline

```
LTL Formula → Rabinizer4 → HOA Format → HOAParser → LDBA → ExhaustiveSearch → Reach-Avoid Sequence
```

At each timestep, the env wrapper advances the LDBA state based on true atomic propositions, then the search extracts the current subgoal as a `(reach, avoid)` pair of assignment sets.

### Module Structure

```
ltl_benchmark/
├── logic/                    # Boolean expression evaluation
│   ├── boolean_lexer.py      # Tokenizer for guard labels (!, &, |, =>)
│   ├── boolean_parser.py     # Recursive descent parser → AST with eval()
│   └── assignment.py         # Assignment / FrozenAssignment for AP truth valuations
├── automata/                 # LDBA construction and manipulation
│   ├── ldba.py               # LDBA, LDBATransition, SCC (Tarjan's algorithm)
│   ├── ldba_sequence.py      # LDBASequence: list of (reach, avoid) pairs
│   ├── hoa_parser.py         # Parse Rabinizer4 HOA output into LDBA
│   └── rabinizer.py          # Subprocess wrapper for Rabinizer4
├── search/
│   └── exhaustive_search.py  # DFS reach-avoid extraction from LDBA
├── env_wrapper.py            # LDBAEnvWrapper for LIBERO environments
├── task_specs.py             # Task → LTL formula registry + prebuilt HOA
└── tests/
    └── test_automaton.py     # 28 unit tests (no MuJoCo required)
```

### Usage

```python
from ltl_benchmark.automata.hoa_parser import HOAParser
from ltl_benchmark.search import ExhaustiveSearchSimple
from ltl_benchmark.task_specs import PREBUILT_HOA

# Build LDBA from prebuilt HOA (no Java/Rabinizer needed)
formula_key = "F(flat_stove_1_turnon & F(moka_pot_1_on_flat_stove_1))"
hoa = PREBUILT_HOA[formula_key]
props = {'flat_stove_1_turnon', 'moka_pot_1_on_flat_stove_1'}
ldba = HOAParser(formula_key, hoa, props).parse_hoa()
ldba.complete_sink_state()
ldba.compute_sccs()

# Extract reach-avoid subgoals
search = ExhaustiveSearchSimple(props, num_loops=1)
sequence = search(ldba, [1])  # search from state 1 (after epsilon)
reach, avoid = sequence[0]    # current subgoal
```

With Rabinizer4 (requires Java 11+):

```python
from ltl_benchmark.automata import build_ldba

# Set RABINIZER_PATH env var to rabinizer4/bin/ltl2ldba
ldba = build_ldba("F(a & F(b))", {"a", "b"})
```

### Env Wrapper

```python
from ltl_benchmark.env_wrapper import LDBAEnvWrapper

wrapper = LDBAEnvWrapper(env, ldba, num_loops=1)
obs = wrapper.reset()
obs, reward, done, info = wrapper.step(action)

# info keys added by wrapper:
# info['ldba_state']          — current LDBA state(s)
# info['ldba_state_changed']  — whether state advanced this step
# info['ldba_violated']       — all paths led to violation
# info['ldba_accepted']       — reached accepting state
# info['reach_avoid_text']    — "Reach: stove_on\nAvoid: pan_displaced"
# info['reach_reward']        — 1.0 on state advance, 10.0 on acceptance
# info['safety_cost']         — 1.0 on violation
```

### Tests

```bash
pytest ltl_benchmark/tests/test_automaton.py -v
```

### Step Integration

`BDDLBaseDomain.step()` is augmented to append LTL state to the `info` dict on every step:

```python
info['ltl_label']       # dict: {prop_name: bool}
info['ltl_label_array'] # np.ndarray: shape (num_propositions,)
info['ltl_goal_desc']   # dict: {goal_prop_name: human_readable_description}
```

### Direct API Usage

```python
from libero.libero.envs.bddl_base_domain import BDDLBaseDomain

# env is any BDDLBaseDomain subclass (e.g., LIBERO task environment)
prop_set = env.get_ltl_propositions()
print(prop_set)  # PropositionSet(LIBERO_Kitchen_Tabletop_Manipulation, 87 propositions)

# Evaluate all propositions in current state
label_dict = env.get_ltl_label_dict()   # {prop_name: bool}
label_array = env.get_ltl_label()       # np.ndarray of bools

# Get goal descriptions for reward shaping
goal_desc = env.get_ltl_goal_desc_map()  # {goal_prop: description string}

# After env.step():
obs, reward, done, info = env.step(action)
print(info['ltl_label'])        # same as get_ltl_label_dict()
print(info['ltl_label_array'])  # same as get_ltl_label()
print(info['ltl_goal_desc'])    # same as get_ltl_goal_desc_map()
```

---

## Training Method: V-MPO (single-critic, on-policy latent steering)

Neuralsym-VLA steers a **frozen** pi0.5 backbone via best-of-N latent selection, conditioned
on the LTL reach-avoid subgoals produced by `ltl_benchmark`. The VL backbone and
flow-matching action expert stay frozen; only a single value head, the V-MPO temperature
`η*`, and a safety multiplier `λ` are learned. The method is selected through config
(`algorithm.adv_type: vmpo` + `algorithm.loss_type: vmpo`); the algorithm registry
(`rlinf/algorithms/registry.py`) dispatches on those strings.

### How it works

A single value head `V_{r̃}` is trained on a *scalarized* reward `r̃ = r_reach − λ·c_hazard`,
evaluated **on-policy** with λ-returns / GAE (κ→1) inside an iterative GPI loop. Trajectories
are **segmented at subgoal satisfaction** (each subgoal-completion step is a soft terminal),
so the σ-only-conditioned value never bootstraps across a subgoal switch. The temperature
`η*` comes from a single-variable V-MPO dual (top-half advantage filter, with per-subgoal
advantage normalization); the safety multiplier `λ` lives in the reward and is either a fixed
`β` or updated by slow PPO-Lagrangian dual ascent. Best-of-N candidates are scored by
`V_{r̃} / η*`. Reach-only tasks (no avoid set) degrade gracefully to reward-only
(`c_hazard ≡ 0 ⇒ r̃ = r_reach`), so the same algorithm block covers all LIBERO-Max LTL tasks.

### Config keys

```yaml
algorithm:
  adv_type: vmpo           # subgoal-segmented on-policy GAE on the scalarized reward
  loss_type: vmpo          # single critic; backbone stays frozen (critic-only)
  gae_lambda: 1.0          # κ=1: Monte-Carlo return-to-segment-terminal (sparse reach)
  entropy_bonus: 0         # no entropy gradient into the frozen policy
  vmpo:
    eta_init: 1.0
    eta_min: 0.01
    epsilon_eta: 0.1            # temperature trust-region budget
    per_subgoal_adv_norm: True  # normalize advantages within each subgoal (E-step)
    safety_mode: fixed          # fixed β (default) | adaptive dual ascent
    beta: 1.0                   # fixed-mode constant λ
    epsilon_1: 0.0              # adaptive-mode cost budget (J_c ≤ ε₁)
    alpha_lambda: 0.05          # adaptive-mode dual-ascent step
actor:
  model:
    add_value_head: True   # single critic V_{r̃}
    best_of_n: 8           # eval-time best-of-N improvement (score = V_{r̃}/η*)
    best_of_n_eta: 1.0     # η* init; synced from the actor's temperature dual
```

Implementation: solvers in `rlinf/algorithms/dual.py` (`VMPOTemperatureOptimizer`,
`SafetyLagrangeMultiplier`); advantage/loss registration in
`rlinf/algorithms/{advantages,losses}.py` (`vmpo`, `compute_segmented_gae`,
`per_subgoal_normalize`); orchestration in `rlinf/workers/actor/fsdp_actor_worker.py`.
Unit tests: `tests/unit_tests/test_vmpo.py`.

> The reach-avoid `cost_rewards` signal (signed safety margin) is produced by the env and
> consumed by the scalarized reward; the subgoal-satisfaction boundary is supplied by the
> LIBERO-Max env as `ltl_subgoal_advanced` (optional — segmentation is disabled gracefully
> when absent).

### Launching training

All embodied runs go through `run_embodiment.sh <config-name>` (Hydra configs under
`examples/embodiment/config/`). Two environment requirements first:

- **Headless MuJoCo render** needs the NVIDIA EGL vendor ICD.
- **Collocated** runs (`cluster.component_placement: actor,env,rollout: all`) use a
  same-device CUDA-IPC weight sync that requires `kernel.yama.ptrace_scope=0`.
  **Disaggregated** runs (actor and rollout on different GPUs) use NCCL instead and need
  no `ptrace_scope` change — use these on unprivileged containers where you can't set it.

```bash
source openpi/bin/activate

# Headless render (required) — point at the NVIDIA EGL vendor ICD:
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Collocated weight sync (single-node, all GPUs) — one-time, needs root:
sudo sysctl -w kernel.yama.ptrace_scope=0

# wandb logging is on in the kitchen4_* configs — authenticate first:
wandb login            # or: export WANDB_API_KEY=...
```

**V-MPO + KITCHEN_SCENE4 random ordered atomic-proposition compositions** (this fork's
focus — each episode samples a random ordered AP composition and prompts the policy with
the current subgoal in AP format; eval builds an LDBA from the LIBERO-Max canonical/HOA
cache and reports per-task success):

```bash
# Full-span production run, 8x H100 (collocated; needs ptrace_scope=0)
bash examples/embodiment/run_embodiment.sh kitchen4_composition_vmpo_8xh100

# 2-GPU disaggregated run (NCCL weight sync — no ptrace_scope change needed).
# On non-NVLink/PCIe nodes also set NCCL_P2P_DISABLE=1:
NCCL_P2P_DISABLE=1 bash examples/embodiment/run_embodiment.sh kitchen4_composition_vmpo_2gpu

# Single-GPU smoke test (fits 24 GB via offload; for a quick end-to-end check)
bash examples/embodiment/run_embodiment.sh kitchen4_composition_vmpo_smoke
```

**Reference LTL configs** (single hand-written spec, original LIBERO suites):

```bash
# V-MPO on SafeLIBERO-Spatial (reach-avoid: cost signal active)
bash examples/embodiment/run_embodiment.sh safelibero_spatial_vmpo_openpi_pi05

# V-MPO on LIBERO-10 LTL (reach-only: reward-only)
bash examples/embodiment/run_embodiment.sh libero_10_ltl_vmpo_openpi_pi05
```

(The base RLinf vanilla-PPO `actor_critic`/`gae` and GRPO paths remain available for
non-LTL training.)

---

## Installation

### Option 1: Python Virtual Environment (uv)

Prerequisites: Python 3.11, CUDA 12.4, `uv` (installed automatically if missing).

```bash
# Clone the repo (includes LIBERO submodule)
git clone --recurse-submodules <repo-url>
cd Neuralsym-VLA

# Install with openpi model + LIBERO environment
bash requirements/install.sh embodied \
    --venv openpi \
    --model openpi \
    --env libero

# Activate the environment
source openpi/bin/activate
```

Other supported models: `openvla`, `openvla-oft`, `gr00t`.

### Option 2: Docker

The Dockerfile defaults to building all four VLA venvs for LIBERO (`openvla`, `openvla-oft`, `openpi`, `gr00t`).

The image bakes in the four LIBERO VLA venvs (and the LIBERO/EGL system deps via
`requirements/embodied/sys_deps.sh`, which runs cleanly on the Ubuntu 22.04 base). It
contains the *environment* only — mount or clone this repo into the container to run.

```bash
# Build image (default target: embodied-libero)
docker build -f docker/Dockerfile -t neuralsym-vla .

# Or specify the target explicitly
docker build -f docker/Dockerfile \
    --build-arg BUILD_TARGET=embodied-libero \
    -t neuralsym-vla .

# Run container (mount the repo, models, and results)
docker run --gpus all \
    -v $(pwd):/workspace/Neuralsym-VLA \
    -v $(pwd)/models:/workspace/models \
    -v $(pwd)/results:/workspace/results \
    -it neuralsym-vla bash

# Inside container: pick the openpi venv, then launch as in "Launching training"
switch_env openpi
```

> Collocated runs still need `kernel.yama.ptrace_scope=0` on the **host** (it's a kernel
> setting, not bakeable into the image). Set it on the host, or use the disaggregated
> 2-GPU config. EGL render configs are installed in the image by `sys_deps.sh`.

---

## Downloading Models

Pre-trained models go in the `models/` directory. The example config expects:

```
models/
├── Pi05-LIBERO-SFT/                      # pi0.5 SFT checkpoint for LIBERO
└── Openvla-oft-SFT-libero10-trajall/     # OpenVLA-OFT SFT checkpoint
```

Download from Hugging Face (the pi0.5 LIBERO SFT checkpoint is public):

```bash
hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir models/Pi05-LIBERO-SFT
# (legacy `huggingface-cli download` is deprecated; use `hf download`)
```

---

## Example: Evaluating pi0.5 on LIBERO-10

Config: `examples/embodiment/config/libero_10_grpo_openpi_pi05.yaml`

Key settings:

```yaml
rollout:
  model:
    model_path: "models/Pi05-LIBERO-SFT"  # checkpoint to evaluate
    unnorm_key: libero_10

actor:
  model:
    model_path: "models/Pi05-LIBERO-SFT"
    num_steps: 4

env:
  eval:
    total_num_envs: 1
    auto_reset: True
    max_episode_steps: 480
    video_cfg:
      save_video: True
      video_base_dir: "../results/video/eval"
```

Launch evaluation:

```bash
source openpi/bin/activate

bash examples/embodiment/eval_embodiment.sh libero_10_grpo_openpi_pi05
```

Logs and eval videos are saved under `logs/<timestamp>/` in the repo root.

---

## Project Structure

```
Neuralsym-VLA/
├── LIBERO/                        # Modified LIBERO environment
│   └── libero/libero/envs/
│       ├── bddl_base_domain.py    # Extended with LTL proposition API
│       └── ltl_utils/
│           ├── __init__.py        # AtomicProposition, PropositionSet, LTLLabelingFunction
│           └── proposition_generator.py  # AtomicPropositionGenerator (4-level)
├── SAFETY_LIBERO/                 # Safety-constrained LIBERO variant
│   └── safelibero/envs/
│       └── ltl_utils/             # Level 5 safety propositions
├── ltl_benchmark/                 # LTL-to-automaton pipeline
│   ├── logic/                     # Boolean parser + AP assignments
│   ├── automata/                  # LDBA construction (Rabinizer/HOA)
│   ├── search/                    # DFS reach-avoid extraction
│   ├── env_wrapper.py             # LDBA state tracking wrapper
│   └── task_specs.py              # Task → LTL formula registry
├── examples/embodiment/config/
│   ├── libero_10_grpo_openpi_pi05.yaml         # pi0.5 GRPO config for LIBERO-10
│   ├── safelibero_spatial_vmpo_openpi_pi05.yaml # V-MPO, reach-avoid (cost active)
│   └── libero_10_ltl_vmpo_openpi_pi05.yaml      # V-MPO, reach-only (reward-only)
├── models/                        # Pre-trained model checkpoints
│   ├── Pi05-LIBERO-SFT/
│   └── Openvla-oft-SFT-libero10-trajall/
├── requirements/
│   └── install.sh                 # uv-based installer
└── docker/
    └── Dockerfile                 # Multi-venv Docker image
```
