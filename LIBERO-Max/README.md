# LIBERO-Max

A unified benchmark for LIBERO-family manipulation environments, combining
original LIBERO, LIBERO-Pro, Safety LIBERO, and LIBERO-10-R into a single
package with a common suite registry.

## Overview

LIBERO-Max provides:

- **60+ evaluation suites** spanning in-distribution, out-of-distribution,
  safety, and robustness evaluation
- **Unified suite registry** — one API to access all suites via
  `get_libero_suite(name)`
- **LTL proposition framework** — automatic extraction of atomic propositions
  from BDDL environments for LTL specification and verification
- **Safety extensions** — obstacle displacement tracking and Level 5 safety
  propositions
- **Environment wrapper** — a gym.Env wrapper with vectorized execution,
  auto-reset, and video recording

## Installation

```bash
# Clone the repo
git clone https://github.com/SimonZhan-code/LIBERO-Max.git
cd LIBERO-Max

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Supported Suites

### Original LIBERO (5 suites)
- `libero_spatial`, `libero_object`, `libero_goal`, `libero_90`, `libero_10`
- Aggregated: `libero_130`

### Safety LIBERO (4 suites)
- `safelibero_spatial`, `safelibero_object`, `safelibero_goal`, `safelibero_long`
- Support Level I / Level II init states for difficulty control

### LIBERO-10-R Robustness Splits (7 suites)
- `libero_10_r` (all 43 tasks)
- `libero_10_r_base` (original 10 in-distribution)
- `libero_10_r_ood` (all OOD)
- `libero_10_r_ood_composition` (novel compositions)
- `libero_10_r_ood_visual` (visual shifts)
- `libero_10_r_ood_visual_scene` (scene/background)
- `libero_10_r_ood_visual_distractor` (distractor objects)

### LIBERO-Pro OOD Suites (40+ suites)
Object, spatial, and goal variants with perturbations:
- `*_with_mug`, `*_with_red_stick`, `*_with_yellow_book`, ...
- `*_swap` (positional), `*_env` (environment), `*_lan` (language)
- `*_temp` (combined perturbations)

## Usage

### List available suites

```python
from libero.libero.benchmark.family import get_libero_suite_names

print(get_libero_suite_names())
```

### Load a suite and inspect tasks

```python
from libero.libero.benchmark.family import get_libero_suite

suite = get_libero_suite("libero_10")
print(f"Suite: {suite.name}, Tasks: {suite.n_tasks}, Max steps: {suite.max_steps}")

for i in range(suite.n_tasks):
    task = suite.get_task(i)
    print(f"  {task.name}: {task.language}")
```

### Filter suites by tags

```python
from libero.libero.benchmark.family import (
    get_libero_suite_names_by_eval_tags,
    get_libero_suite_names_by_source,
)

# All OOD suites
ood_suites = get_libero_suite_names_by_eval_tags("ood")

# All safety suites
safety_suites = get_libero_suite_names_by_source("safety")
```

### Use the environment wrapper

```python
from libero.env_wrapper import LiberoEnv
```

### LTL Proposition Framework

```python
from libero.libero.envs.ltl_utils import (
    AtomicProposition,
    PropositionSet,
    LTLLabelingFunction,
)
from libero.libero.envs.ltl_utils.proposition_generator import AtomicPropositionGenerator
```

### LTL Monitoring

LIBERO-Max also includes an automaton-style monitor for checking task formulas
against rollout proposition labels:

```python
from libero.libero.ltl_monitor import get_task_ltl_spec, build_monitor_from_spec

spec = get_task_ltl_spec(
    proposition_set=env.get_ltl_propositions(),
    task_id=task.name,
)
monitor = build_monitor_from_spec(spec)
monitor_info = monitor.step(info["ltl_label"])
```

Specs are selected from the task registry when available, otherwise generated
from BDDL goal propositions as `F(goal_1 & ...)`. SafeLIBERO propositions add
`G(!safety)` constraints, and simple open/turn-on / close/turn-off goal
patterns are converted into ordered formulas.

For vectorized evaluation through `LiberoEnv`, set
`enable_ltl_monitor: true` in the environment config. The wrapper then adds
`ltl_accepted`, `ltl_violated`, `ltl_reach_reward`, `ltl_safety_cost`, and
`ltl_reach_avoid_text` to `infos`.

## Project Structure

```
LIBERO-Max/
├── libero/
│   ├── libero/
│   │   ├── envs/           # Environment definitions
│   │   │   ├── ltl_utils/  # LTL proposition framework
│   │   │   └── safety/     # Safety LIBERO extensions
│   │   ├── benchmark/      # Suite definitions and unified registry
│   │   │   ├── family.py   # get_libero_suite() — unified API
│   │   │   └── task_maps.py
│   │   ├── assets/         # 3D models and textures
│   │   ├── bddl_files/     # BDDL task specifications
│   │   └── init_files/     # Initial state files
│   └── env_wrapper/        # Gym.Env wrapper with vectorized execution
├── libero_ood/             # OOD evaluation configs
├── benchmark_scripts/      # Benchmark utility scripts
├── scripts/                # Data collection and processing
├── notebooks/              # Jupyter notebooks and examples
├── setup.py
└── requirements.txt
```

## Acknowledgments

This benchmark builds on:
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) — the original benchmark for lifelong robot learning
- [LIBERO-Pro](https://github.com/Zxy-MLlab/LIBERO-PRO) — OOD evaluation extensions
- [Safety LIBERO](https://github.com/SimonZhan-code/LIBERO-Max) — safety-aware task variants
- [LIBERO-10-R](https://github.com/Max-Fu/LIBERO-10-R) — robustness evaluation splits

## License

See [LICENSE](LICENSE) for details.
