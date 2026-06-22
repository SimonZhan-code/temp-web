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

"""Build ``ltl_benchmark`` LDBAs from the LIBERO-Max canonical LTL label + HOA cache.

LIBERO-Max (>= the "Canonical LTL labels + prebuilt LDBA cache" commit) ships:
  - ``ltl_monitor/canonical_task_ltl_labels.json`` — per task ``<suite>/<task>``: the
    resolved LTL formula and goal/safety atoms,
  - ``ltl_monitor/hoa_store.json`` — ``resolved_formula -> HOA`` (prebuilt, no online
    Rabinizer).

This lets the eval env build an LDBA for ANY task (every KITCHEN_SCENE* including the
compositional ones), not just the single hand-written ``ltl_benchmark.TASK_SPECS``
entry. The owl-format HOAs parse with ``ltl_benchmark``'s ``HOAParser`` (verified), so
the resulting LDBA is the same object type the eval env already advances.
"""

import functools
import json
import os

from ltl_benchmark.automata.hoa_parser import HOAParser


def _ltl_monitor_dir():
    """Locate LIBERO-Max's ``ltl_monitor`` dir via the imported ``libero`` package."""
    import libero

    base = os.path.dirname(os.path.abspath(libero.__file__))
    for cand in (
        os.path.join(base, "libero", "ltl_monitor"),  # doubled layout libero/libero/...
        os.path.join(base, "ltl_monitor"),
    ):
        if os.path.isdir(cand):
            return cand
    return None


@functools.lru_cache(maxsize=1)
def _load_cache():
    d = _ltl_monitor_dir()
    if d is None:
        return None, None
    labels_path = os.path.join(d, "canonical_task_ltl_labels.json")
    hoa_path = os.path.join(d, "hoa_store.json")
    if not (os.path.isfile(labels_path) and os.path.isfile(hoa_path)):
        return None, None
    labels = json.load(open(labels_path)).get("labels", {})
    hoa = json.load(open(hoa_path))
    return labels, hoa


def has_cache():
    labels, _ = _load_cache()
    return bool(labels)


def build_ldba_from_cache(suite, task_name):
    """Return ``(ldba, propositions)`` for ``<suite>/<task_name>`` from the canonical
    cache, or ``(None, None)`` if the cache or this task/HOA is unavailable."""
    labels, hoa_store = _load_cache()
    if not labels:
        return None, None
    entry = labels.get(f"{suite}/{task_name}")
    if entry is None:  # fall back to a suffix match if suite differs
        cand = [v for k, v in labels.items() if k.endswith("/" + task_name)]
        entry = cand[0] if cand else None
    if entry is None:
        return None, None
    formula = entry.get("resolved_formula")
    atoms = set(entry.get("goal_atoms", [])) | set(entry.get("safety_atoms", []))
    hoa_text = hoa_store.get(formula) if formula else None
    if not hoa_text or not atoms:
        return None, None
    ldba = HOAParser(formula, hoa_text, atoms).parse_hoa()
    ldba.complete_sink_state()
    ldba.compute_sccs()
    return ldba, atoms
