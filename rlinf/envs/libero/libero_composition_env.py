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

"""Train-only LIBERO env: fixed scene, per-episode random ordered AP composition.

Each episode samples one *ordered* composition of subgoals (goal-style atomic
propositions) for a fixed scene (e.g. ``KITCHEN_SCENE4``) and rewards the agent for
achieving them left to right. No LDBA/Rabinizer is used: because the env loads an
"all-goals" BDDL (so every ``goal_alphabet`` proposition is emitted in ``ltl_label``),
the subgoal -> label map is the identity and a simple progress pointer reproduces the
exact reward contract the V-MPO actor consumes (``obs["ltl_reach_rewards"]`` /
``obs["ltl_cost_rewards"]``).

Reward is reach-only for now; ``ltl_cost_rewards`` is a constant "safe" signal
(``-1.0``) that keeps the safety/Lagrangian port open but inert (see plan).

Evaluation does NOT use this env — it runs the stock ``LiberoLTLEnv`` on real,
LTL-labeled compositional LIBERO tasks.
"""

import numpy as np
import torch

from rlinf.envs.libero.allgoals_bddl import build_all_goals_bddl
from rlinf.envs.libero.composition_sampler import CompositionSampler
from rlinf.envs.libero.libero_env import LiberoEnv
from rlinf.envs.utils import (
    list_of_dict_to_dict_of_list,
    put_info_on_image,
    tile_images,
    to_tensor,
)

# cost convention follows LiberoLTLEnv: -1.0 == safe, +1.0 == hazard.
_SAFE_MARGIN = -1.0


def advance_ordered_subgoals(subgoals, ptr, label):
    """Advance an ordered-composition pointer; reward +1 per subgoal achieved this step.

    Achieve subgoals strictly left to right: while the subgoal at ``ptr`` is true in
    ``label``, advance the pointer. The per-step reward is the NUMBER of subgoals newly
    completed this step (each worth 1.0) — an event reward, not the old progress *level*.
    The pointer is monotonic, so each subgoal is rewarded exactly once and the episode
    return equals the count of subgoals achieved (max = ``len(subgoals)``). ``accepted``
    is True once all subgoals are done.

    Pure function (no env/MuJoCo) so it is unit-testable.

    Args:
        subgoals: ordered list of goal-style AP names.
        ptr: current integer pointer (number already achieved).
        label: dict mapping AP name -> bool for the current step (or None).

    Returns:
        (new_ptr, reach_reward, accepted)
    """
    k = len(subgoals)
    if k == 0:
        return ptr, 0.0, False
    if not isinstance(label, dict):
        # cannot evaluate this step; no new subgoal completed -> no reward
        return ptr, 0.0, ptr >= k
    p = int(ptr)
    old = p
    while p < k and bool(label.get(subgoals[p], False)):
        p += 1
    return p, float(p - old), p >= k


# Default prompt preamble: tells the VLA, from the start, that the goal is delivered
# as a single atomic-proposition subgoal (no natural language, no overall-task LTL).
DEFAULT_PROMPT_PREAMBLE = "Atomic-proposition subgoal to achieve:"
# Marker shown once all subgoals are achieved (episode terminates immediately after).
_DONE_TOKEN = "done"


def render_subgoal_ap(ap_name, primitive=None, fmt="predicate"):
    """Render one subgoal atomic proposition for the prompt.

    ``fmt="predicate"`` (default) -> ``pred(arg, ...)`` using the composition primitive
    (e.g. ``on(akita_black_bowl_1, white_cabinet_1_top_side)``, ``open(white_cabinet_1_top_region)``).
    ``fmt="raw"`` -> the bare AP name (e.g. ``on_akita_black_bowl_1_white_cabinet_1_top_side``).
    No natural language is produced in either case.
    """
    if fmt == "raw" or primitive is None:
        return str(ap_name)
    kind = primitive.get("kind")
    obj = primitive.get("obj")
    target = primitive.get("target")
    if kind == "move":
        rel = "in" if str(ap_name).startswith("in_") else "on"
        return f"{rel}({obj}, {target})"
    if kind in ("open", "close"):
        return f"{kind}({target})"
    if kind in ("turn_on", "turn_off"):
        return f"{kind}({obj})"
    return str(ap_name)


def _humanize(token):
    """``white_cabinet_1_bottom_region`` -> ``white cabinet bottom region``.

    Drops numeric instance ids (``_1``) wherever they appear and turns underscores
    into spaces, so region/object tokens read as natural language for the NL VLA.
    """
    if token is None:
        return ""
    parts = [p for p in str(token).split("_") if not p.isdigit()]
    return " ".join(parts)


def render_subgoal_nl(ap_name, primitive=None):
    """Render one subgoal as a NATURAL-LANGUAGE instruction (in-distribution for the
    NL-pretrained SFT checkpoint), e.g.
    ``put the akita black bowl in the white cabinet bottom region``,
    ``open the white cabinet top region``.

    This is the counterpart to :func:`render_subgoal_ap` for ``prompt_style="nl"``.
    Falls back to a de-underscored AP name if no primitive is available.
    """
    if primitive is None:
        return str(ap_name).replace("_", " ")
    kind = primitive.get("kind")
    obj = _humanize(primitive.get("obj"))
    target = _humanize(primitive.get("target"))
    if kind == "move":
        rel = "in" if str(ap_name).startswith("in_") else "on"
        return f"put the {obj} {rel} the {target}"
    if kind in ("open", "close"):
        return f"{kind} the {target}"
    if kind == "turn_on":
        return f"turn on the {obj}"
    if kind == "turn_off":
        return f"turn off the {obj}"
    return str(ap_name).replace("_", " ")


def render_proposition_ap(name, fmt="predicate"):
    """Render a generator-style proposition NAME (from an LDBA trace) as AP-format.

    Used at eval time to feed the LDBA's current reach proposition to the VLA in the
    SAME ``pred(args)`` format the policy trained on (no primitive available here, so
    we parse the name). Generator naming (see proposition_generator.py):
      L1 unary:   ``{obj}_{is_open|is_close|turn_on|turn_off}`` -> ``open|close|turn_on|turn_off(obj)``
      L2 binary:  ``{obj1}_on_{obj2}``                          -> ``on(obj1, obj2)``
      L3 region:  ``{obj}_in_{region}``                         -> ``in(obj, region)``
    ``fmt="raw"`` returns the bare name.
    """
    if fmt == "raw":
        return str(name)
    for suf, pred in (
        ("_turn_on", "turn_on"),
        ("_turn_off", "turn_off"),
        ("_is_open", "open"),
        ("_is_close", "close"),
    ):
        if name.endswith(suf):
            return f"{pred}({name[: -len(suf)]})"
    if "_on_" in name:
        a, b = name.split("_on_", 1)
        return f"on({a}, {b})"
    if "_in_" in name:
        a, b = name.split("_in_", 1)
        return f"in({a}, {b})"
    return str(name)


def _goal_block(text):
    """Return the inner s-expression text of the BDDL ``(:goal ...)`` block."""
    start = text.find("(:goal")
    if start == -1:
        return ""
    depth = 0
    end = len(text)
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    return text[start:end]


def _split_clauses(s):
    """Yield top-level ``(...)`` s-expressions found in ``s``."""
    depth = 0
    buf = []
    for ch in s:
        if ch == "(":
            if depth == 0:
                buf = []
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
            if depth == 0:
                yield "".join(buf)
        elif depth > 0:
            buf.append(ch)


def parse_goal_subgoals(bddl_path):
    """Parse a real task's BDDL ``:goal`` into ordered (subgoal_aps, primitives).

    ``(And (Close R) (In A R) ...)`` -> subgoal AP names ``close_R``, ``in_A_R`` and
    primitives compatible with :func:`render_subgoal_ap`. Order follows the BDDL.
    """
    text = open(bddl_path).read()
    block = _goal_block(text)
    # strip the leading "(:goal" and trailing ")"
    inner = block[len("(:goal") : -1].strip() if block else ""
    # unwrap a single (And ...) if present
    and_clauses = list(_split_clauses(inner))
    if len(and_clauses) == 1 and and_clauses[0][1:4].lower() == "and":
        inner = and_clauses[0][1:-1]
        inner = inner.strip()[3:]  # drop the "And" token
    subgoals, primitives = [], []
    for clause in _split_clauses(inner):
        toks = clause.strip()[1:-1].split()
        if not toks:
            continue
        pred = toks[0].lower()
        args = toks[1:]
        ap_name = "_".join([pred] + args)
        if pred in ("in", "on"):
            prim = {"kind": "move", "obj": args[0], "target": args[1],
                    "achieved_ap": ap_name}
        elif pred in ("open", "close"):
            prim = {"kind": pred, "obj": args[0], "target": args[0],
                    "achieved_ap": ap_name}
        elif pred in ("turnon", "turn_on"):
            prim = {"kind": "turn_on", "obj": args[0], "target": args[0],
                    "achieved_ap": ap_name}
        elif pred in ("turnoff", "turn_off"):
            prim = {"kind": "turn_off", "obj": args[0], "target": args[0],
                    "achieved_ap": ap_name}
        else:
            prim = {"kind": pred, "obj": args[0] if args else "",
                    "target": args[-1] if args else "", "achieved_ap": ap_name}
        subgoals.append(ap_name)
        primitives.append(prim)
    return subgoals, primitives


class LiberoCompositionEnv(LiberoEnv):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        comp = cfg.get("composition", None)
        if comp is None or comp.get("scene_id", None) is None:
            raise ValueError(
                "libero_ltl_composition requires cfg.composition.scene_id to be set."
            )
        self.scene_id = comp.get("scene_id")
        # mode="sample": train on random ordered compositions (all-goals BDDL).
        # mode="task_goals": eval on a real task's own ordered goal predicates
        #   (real per-task BDDL), with the SAME AP-format prompting + tracker so the
        #   prompt distribution matches training.
        self._mode = str(comp.get("mode", "sample"))
        self._sampler = None
        if self._mode == "sample":
            self._sampler = CompositionSampler(
                scene_id=self.scene_id,
                max_depth=int(comp.get("max_depth", 3)),
                pool=str(comp.get("pool", "all")),
                min_depth=int(comp.get("min_depth", 1)),
                data_dir=comp.get("data_dir", None),
            )
        self._task_subgoal_cache = {}  # task_id -> (subgoals, primitives)
        self._allgoals_bddl_path = None  # built lazily inside get_env_fn_params
        self._comp_rng = np.random.default_rng(cfg.seed + seed_offset + 8191)
        # prompt style:
        #   "ap" (default) -> "<preamble> <current AP>" (atomic-proposition format);
        #   "nl"           -> a natural-language instruction for the current subgoal
        #                     (in-distribution for the NL-pretrained SFT checkpoint).
        # The ordered-subgoal tracker/switching is identical either way — only the
        # rendered text the VLA sees changes.
        self._prompt_style = str(comp.get("prompt_style", "ap"))
        self._prompt_preamble = str(
            comp.get("prompt_preamble", DEFAULT_PROMPT_PREAMBLE)
        )
        self._ap_format = str(comp.get("prompt_ap_format", "predicate"))
        # per-env tracker state
        self._subgoals = [None] * num_envs
        self._primitives = [None] * num_envs
        self._ptr = np.zeros(num_envs, dtype=np.int32)
        self._subgoal_aps = [None] * num_envs  # rendered AP string per subgoal
        self._identity_checked = False

        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

    # ---- load the all-goals BDDL for every env (so all goal keys are emitted) ----
    def _get_allgoals_bddl(self):
        if self._allgoals_bddl_path is None:
            # Any task of the locked scene shares identical objects/regions.
            src_task_id = self.allowed_task_ids[0]
            source_bddl = self.task_suite.get_task_bddl_file_path(src_task_id)
            self._allgoals_bddl_path = build_all_goals_bddl(
                source_bddl=source_bddl,
                goal_alphabet=self._sampler.goal_alphabet,
                scene_id=self.scene_id,
            )
        return self._allgoals_bddl_path

    def get_env_fn_params(self, env_idx=None):
        params = super().get_env_fn_params(env_idx)
        if self._mode == "sample":
            # All envs load the all-goals BDDL so every goal-style key is emitted.
            allgoals = self._get_allgoals_bddl()
            for p in params:
                p["bddl_file_name"] = allgoals
        # task_goals mode keeps each task's real BDDL (so success == real task).
        return params

    def _task_subgoals(self, task_id):
        """Ordered (subgoals, primitives) from a real task's BDDL ``:goal``."""
        if task_id not in self._task_subgoal_cache:
            bddl = self.task_suite.get_task_bddl_file_path(task_id)
            self._task_subgoal_cache[task_id] = parse_goal_subgoals(bddl)
        return self._task_subgoal_cache[task_id]

    # ---- composition sampling / tracker ----
    def _set_subgoals_for_env(self, env_id, subgoals, primitives):
        self._subgoals[env_id] = list(subgoals)
        self._primitives[env_id] = list(primitives)
        self._ptr[env_id] = 0
        if self._prompt_style == "nl":
            self._subgoal_aps[env_id] = [
                render_subgoal_nl(ap, primitives[i] if i < len(primitives) else None)
                for i, ap in enumerate(subgoals)
            ]
        else:
            self._subgoal_aps[env_id] = [
                render_subgoal_ap(
                    ap,
                    primitives[i] if i < len(primitives) else None,
                    fmt=self._ap_format,
                )
                for i, ap in enumerate(subgoals)
            ]

    def _sample_composition_for(self, env_idx):
        for env_id in env_idx:
            if self._mode == "sample":
                comp = self._sampler.sample(self._comp_rng)
                self._set_subgoals_for_env(env_id, comp.subgoals, comp.primitives)
            else:  # task_goals: use the real task's own ordered goal predicates
                subgoals, prims = self._task_subgoals(int(self.task_ids[env_id]))
                self._set_subgoals_for_env(env_id, subgoals, prims)

    def _current_subgoal_ap(self, env_id):
        """The current (unachieved) subgoal AP string, or the done token."""
        rendered = self._subgoal_aps[env_id]
        if not rendered:
            return ""
        p = int(self._ptr[env_id])
        if p >= len(rendered):
            return _DONE_TOKEN
        return rendered[p]

    def _current_prompt_texts(self):
        """Per-env prompt for the current subgoal.

        ``prompt_style="ap"``  -> ``"<preamble> <current AP>"``.
        ``prompt_style="nl"``  -> the natural-language instruction itself (no preamble),
        so the string is in-distribution for the NL-pretrained checkpoint.
        """
        nl = self._prompt_style == "nl"
        texts = []
        for env_id in range(self.num_envs):
            ap = self._current_subgoal_ap(env_id)
            if not ap:
                # no composition yet (should not happen post-reset); fall back safe
                texts.append("" if nl else self._prompt_preamble)
            elif nl or ap == _DONE_TOKEN:
                texts.append(ap)
            else:
                texts.append(f"{self._prompt_preamble} {ap}")
        return texts

    def _validate_identity_once(self, ltl_labels):
        """Assert every active subgoal AP is an emitted ltl_label key (both modes)."""
        if self._identity_checked or not ltl_labels:
            return
        label = next((x for x in ltl_labels if isinstance(x, dict)), None)
        if label is None:
            return
        needed = set()
        for sg in self._subgoals:
            if sg:
                needed.update(sg)
        missing = sorted(k for k in needed if k not in label)
        if missing:
            raise RuntimeError(
                f"Scene {self.scene_id} ({self._mode} mode): subgoal APs {missing} "
                f"are not emitted in ltl_label. Present keys: {sorted(label)[:40]}"
            )
        self._identity_checked = True

    def _tracker_rewards(self, ltl_labels):
        """Advance each env's pointer; return (reach_rewards, safety_margins).

        Reward = +1 per subgoal achieved this step (event reward); episode return equals
        the number of subgoals completed (max == len(subgoals)).
        """
        reach = np.zeros(self.num_envs, dtype=np.float32)
        safety = np.full(self.num_envs, _SAFE_MARGIN, dtype=np.float32)
        accepted = np.zeros(self.num_envs, dtype=bool)

        for env_id in range(self.num_envs):
            subgoals = self._subgoals[env_id]
            if not subgoals:
                continue
            label = (
                ltl_labels[env_id]
                if (ltl_labels is not None and env_id < len(ltl_labels))
                else None
            )
            new_ptr, reach_r, acc = advance_ordered_subgoals(
                subgoals, int(self._ptr[env_id]), label
            )
            self._ptr[env_id] = new_ptr
            reach[env_id] = reach_r
            accepted[env_id] = acc
        self._last_accepted = accepted
        return torch.from_numpy(reach), torch.from_numpy(safety)

    # ---- obs / reset / step ----
    def _wrap_obs(self, obs_list):
        obs = super()._wrap_obs(obs_list)
        # Prompt = AP-format CURRENT subgoal only (advances with the pointer). No NL,
        # no overall-task LTL. Deliberately do NOT set obs["reach_avoid_texts"] so the
        # model does not append anything else to the prompt (openpi_action_model).
        obs["task_descriptions"] = self._current_prompt_texts()
        return obs

    def reset(self, env_idx=None, reset_state_ids=None):
        obs, infos = super().reset(env_idx=env_idx, reset_state_ids=reset_state_ids)
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        self._sample_composition_for(env_idx)
        obs["task_descriptions"] = self._current_prompt_texts()
        return obs, infos

    def step(self, actions=None, auto_reset=True):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, info_lists = self.env.step(actions)
        self.current_raw_obs = raw_obs
        infos = list_of_dict_to_dict_of_list(info_lists)
        truncations = self.elapsed_steps >= self.cfg.max_episode_steps

        ltl_labels = infos.get("ltl_label", None)
        self._validate_identity_once(ltl_labels)

        # ordered-tracker reward (advances pointers, may flag acceptance)
        ltl_reach_rewards, ltl_cost_signals = self._tracker_rewards(ltl_labels)

        obs = self._wrap_obs(raw_obs)
        obs["ltl_reach_rewards"] = ltl_reach_rewards
        obs["ltl_cost_rewards"] = ltl_cost_signals

        # composition success drives termination (not BDDL all-goals success)
        terminations = np.asarray(self._last_accepted, dtype=bool)
        step_reward = self._calc_step_reward(terminations)

        if self.video_cfg.save_video:
            plot_infos = {
                "rewards": step_reward,
                "terminations": terminations,
                "subgoal": [self._current_subgoal_ap(i) for i in range(self.num_envs)],
            }
            self.add_new_frames(raw_obs, plot_infos)

        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def add_new_frames(self, raw_obs, plot_infos):
        images = []
        for env_id, raw_single_obs in enumerate(raw_obs):
            info_item = {
                k: v if np.size(v) == 1 else v[env_id]
                for k, v in plot_infos.items()
            }
            subgoals = self._subgoals[env_id] or []
            p = int(self._ptr[env_id])
            info_item["progress"] = f"{p}/{len(subgoals)}"
            img = raw_single_obs["agentview_image"][::-1, ::-1]
            ap = self._current_subgoal_ap(env_id)
            extras = [f"subgoal: {ap}"] if ap else []
            img = put_info_on_image(img, info_item, extras=extras)
            images.append(img)
        full_image = tile_images(images, nrows=int(np.sqrt(self.num_envs)))
        self.render_images.append(full_image)
