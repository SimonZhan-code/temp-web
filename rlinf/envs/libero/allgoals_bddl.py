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

"""Generate an "all-goals" BDDL for a scene so the env emits every goal-style AP.

LIBERO only emits the predicate-first, region-specific goal propositions (L4) for
predicates that appear in the loaded BDDL's ``:goal``. By splicing the union of a
scene's ``goal_alphabet`` predicates into ``:goal``, the env emits **all** goal-style
``ltl_label`` keys at once — making the composition-subgoal -> label map the identity
(verified: e.g. ``open_..._top_region`` and ``close_..._bottom_region`` coexist as
distinct region-specific keys). The physical scene (objects/regions/fixtures/init
regions) is untouched, so init states from any task of the same scene stay compatible.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List


def _repo_root() -> Path:
    # this file: <repo>/rlinf/envs/libero/allgoals_bddl.py
    return Path(__file__).resolve().parents[3]


def _goal_clause(args: List[str]) -> str:
    """``["close", "white_cabinet_1_bottom_region"]`` -> ``(Close white_cabinet_1_bottom_region)``."""
    pred = args[0]
    pred_bddl = pred[:1].upper() + pred[1:]
    operands = " ".join(args[1:])
    return f"({pred_bddl} {operands})" if operands else f"({pred_bddl})"


def _replace_goal_block(text: str, new_goal_inner: str) -> str:
    """Replace the ``(:goal ...)`` s-expression with ``(:goal <new_goal_inner>)``.

    Uses paren matching (robust to indentation/newlines) rather than a regex.
    """
    marker = "(:goal"
    start = text.find(marker)
    if start == -1:
        raise ValueError("BDDL has no (:goal ...) block to replace.")
    depth = 0
    end = None
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise ValueError("Unbalanced parentheses in (:goal ...) block.")
    replacement = f"(:goal\n    {new_goal_inner}\n  )"
    return text[:start] + replacement + text[end:]


def build_all_goals_bddl(
    source_bddl: str,
    goal_alphabet: List[dict],
    scene_id: str,
    out_dir: str | None = None,
) -> str:
    """Write an all-goals variant of ``source_bddl`` and return its path.

    Args:
        source_bddl: path to any canonical BDDL of the target scene (objects/regions
            identical across the scene's tasks).
        goal_alphabet: list of ``{"name", "args"}`` records (from ``goal_aps.json``).
        scene_id: e.g. ``"KITCHEN_SCENE4"`` (used in the output filename).
        out_dir: cache directory (default ``<repo>/.cache/allgoals_bddl``).

    The output filename includes a short hash of the goal set so changing the goal
    alphabet regenerates a distinct file (no stale reuse).
    """
    src_text = Path(source_bddl).read_text()
    clauses = [_goal_clause(g["args"]) for g in goal_alphabet]
    new_goal_inner = "(And " + " ".join(clauses) + ")"
    out_text = _replace_goal_block(src_text, new_goal_inner)

    digest = hashlib.sha1(new_goal_inner.encode()).hexdigest()[:8]
    cache = Path(out_dir) if out_dir else _repo_root() / ".cache" / "allgoals_bddl"
    cache.mkdir(parents=True, exist_ok=True)
    out_path = cache / f"{scene_id}_allgoals_{digest}.bddl"
    out_path.write_text(out_text)
    return str(out_path)
