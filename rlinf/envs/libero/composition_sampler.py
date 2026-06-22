# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sampler over precomputed ordered AP compositions for a fixed LIBERO scene.

LIBERO-Max emits, per scene, an ``ordered_subgoal_list`` file
(``feasible_propositions/<scene>/compositions_up_to_<d>.json``). Each composition
is an ORDERED list of goal-style atomic-proposition names (achieve left to right).
This sampler simply draws one composition per episode; no LTL/LDBA is involved
(training tracks the ordered subgoals directly — see ``LiberoCompositionEnv``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def default_feasible_propositions_dir() -> Path:
    """Locate ``feasible_propositions`` for the *imported* ``libero`` package.

    Resolving relative to the live ``libero`` install guarantees the composition
    data matches the running scene-model/predicate code. The package layout varies
    (``libero/__init__.py`` vs the doubled ``libero/libero/__init__.py``), so walk
    up from the package file until a ``feasible_propositions`` dir is found.
    """
    import libero

    here = Path(libero.__file__).resolve()
    for parent in here.parents:
        candidate = parent / "feasible_propositions"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not locate 'feasible_propositions' above {here}. "
        "Pass data_dir explicitly."
    )


@dataclass
class Composition:
    """One ordered composition of subgoals (goal-style AP names)."""

    subgoals: List[str]
    primitives: List[dict]
    depth: int
    matches_anchor: bool
    is_held_out: bool
    raw: dict = field(default_factory=dict)


class CompositionSampler:
    """Draws ordered subgoal compositions for one scene from the precomputed JSON.

    Args:
        scene_id: e.g. ``"KITCHEN_SCENE4"``.
        max_depth: which ``compositions_up_to_<max_depth>.json`` to load.
        pool: ``"all"`` (default), ``"anchor"`` (matches_anchor), or
            ``"held_out"`` (is_held_out).
        min_depth: optional lower bound on composition depth (>=1).
        data_dir: override for the ``feasible_propositions`` directory.
    """

    def __init__(
        self,
        scene_id: str,
        max_depth: int = 3,
        pool: str = "all",
        min_depth: int = 1,
        data_dir: Optional[str] = None,
    ):
        self.scene_id = scene_id
        self.max_depth = max_depth
        self.pool = pool
        self.min_depth = min_depth

        base = Path(data_dir) if data_dir else default_feasible_propositions_dir()
        self._scene_dir = base / scene_id
        comp_path = self._scene_dir / f"compositions_up_to_{max_depth}.json"
        if not comp_path.exists():
            raise FileNotFoundError(
                f"No composition file for scene {scene_id} at {comp_path}"
            )
        data = json.loads(comp_path.read_text())
        fmt = data.get("format")
        if fmt != "ordered_subgoal_list":
            raise ValueError(
                f"{comp_path} has format={fmt!r}, expected 'ordered_subgoal_list'. "
                "Regenerate compositions with LIBERO-Max scripts/sample_ap_compositions.py."
            )

        comps: List[Composition] = []
        for c in data["compositions"]:
            if c["depth"] < min_depth:
                continue
            if pool == "anchor" and not c.get("matches_anchor", False):
                continue
            if pool == "held_out" and not c.get("is_held_out", False):
                continue
            comps.append(
                Composition(
                    subgoals=list(c["subgoals"]),
                    primitives=list(c.get("primitives", [])),
                    depth=int(c["depth"]),
                    matches_anchor=bool(c.get("matches_anchor", False)),
                    is_held_out=bool(c.get("is_held_out", False)),
                    raw=c,
                )
            )
        if not comps:
            raise ValueError(
                f"No compositions left for scene={scene_id} pool={pool} "
                f"min_depth={min_depth} max_depth={max_depth}."
            )
        self.compositions = comps

    # ---- goal alphabet (universe of subgoal APs for identity validation) ----
    @property
    def goal_alphabet(self) -> List[dict]:
        """The scene's goal-AP records (name + args), from goal_aps.json."""
        ga_path = self._scene_dir / "goal_aps.json"
        return json.loads(ga_path.read_text())["goal_alphabet"]

    def goal_alphabet_names(self) -> List[str]:
        return [g["name"] for g in self.goal_alphabet]

    def __len__(self) -> int:
        return len(self.compositions)

    def sample(self, rng: np.random.Generator) -> Composition:
        """Uniformly draw one composition."""
        idx = int(rng.integers(0, len(self.compositions)))
        return self.compositions[idx]

    def counts_by_depth(self) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for c in self.compositions:
            out[c.depth] = out.get(c.depth, 0) + 1
        return out
