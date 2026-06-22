"""LiberoEnv subclass with LDBA state tracking and LTL overlay on video frames."""

import numpy as np
import torch

from ltl_benchmark.automata.hoa_parser import HOAParser
from ltl_benchmark.env_wrapper import CurrentState, format_avoid_text, format_reach_text
from ltl_benchmark.search import ExhaustiveSearchSimple, NoPathsException
from ltl_benchmark.task_specs import PREBUILT_HOA, TASK_SPECS
from rlinf.envs.libero.ltl_cache import build_ldba_from_cache
from rlinf.envs.libero.libero_env import LiberoEnv
from rlinf.envs.utils import (
    list_of_dict_to_dict_of_list,
    put_info_on_image,
    tile_images,
    to_tensor,
)


def immediate_reach_props(ldba, states):
    """Immediate next reach proposition(s) from the current LDBA state(s).

    The "current subgoal" of a (possibly nested-F) spec is the positive guard of the
    transition that makes the most progress toward acceptance from the current state,
    found via a reverse-BFS distance-to-acceptance over the LDBA. This is correct for
    sequential nested-F specs where the full-sequence search (ExhaustiveSearchSimple +
    num_loops + min-length) is not: num_loops=0 drops the accepting self-loop's reach
    (premature "done"), num_loops=1 over-merges the next subgoal with later ones.

    Returns the sorted positive proposition names of the chosen progress transition
    (the subgoal to achieve now), or [] if the automaton is already accepting / stuck.
    """
    transitions = ldba.state_to_transitions
    all_states = list(transitions.keys())
    INF = float("inf")

    # distance[s] = min # of reach steps from s to taking an accepting transition
    dist = {s: INF for s in all_states}
    for s in all_states:
        for t in transitions[s]:
            if t.accepting and not t.is_epsilon() and t.feasible_assignments:
                dist[s] = 1
    changed = True
    while changed:  # small graphs -> Bellman-style relaxation to fixpoint
        changed = False
        for s in all_states:
            best = dist[s]
            for t in transitions[s]:
                if t.is_epsilon():
                    cand = dist.get(t.target, INF)  # epsilon consumes no reach step
                elif not t.feasible_assignments:
                    continue
                elif t.accepting:
                    cand = 1
                else:
                    tgt = dist.get(t.target, INF)
                    cand = INF if tgt == INF else 1 + tgt
                if cand < best:
                    best = cand
            if best < dist[s]:
                dist[s] = best
                changed = True

    def required_props(t):
        # Props that are TRUE in EVERY feasible assignment of the guard = the positive
        # literals the guard requires (the subgoal). Intersection, not union: a guard
        # like `a` leaves other props (e.g. the safety `displaced`) free, so they appear
        # true in SOME assignment but are not part of the subgoal.
        fas = list(t.feasible_assignments)
        if not fas:
            return set()
        req = set(fas[0].get_true_propositions())
        for fa in fas[1:]:
            req &= set(fa.get_true_propositions())
        return req

    names = set()
    for s in states:
        best_d, best_props = INF, None
        for t in transitions.get(s, []):
            if t.is_epsilon() or not t.feasible_assignments:
                continue
            tgt = dist.get(t.target, INF)
            d = 1 if t.accepting else (INF if tgt == INF else 1 + tgt)
            req = required_props(t)
            if d < best_d and req:  # only positive-reach transitions are subgoals
                best_d, best_props = d, req
        if best_props:
            names |= best_props
    return sorted(names)


def _build_ldba(task_name, suite=None):
    """Build LDBA for a task. Prefer the LIBERO-Max canonical LTL/HOA cache (covers all
    tasks incl. KITCHEN_SCENE* compositions); fall back to the hand-written
    ltl_benchmark.TASK_SPECS for older checkouts."""
    if suite is not None:
        ldba, props = build_ldba_from_cache(suite, task_name)
        if ldba is not None:
            return ldba, props
    if task_name not in TASK_SPECS:
        return None, None
    spec = TASK_SPECS[task_name]
    formula = spec["formula"]
    propositions = spec["propositions"]
    hoa_text = PREBUILT_HOA[formula]
    ldba = HOAParser(formula, hoa_text, propositions).parse_hoa()
    ldba.complete_sink_state()
    ldba.compute_sccs()
    return ldba, propositions


def _task_to_spec_key(task):
    """Convert a LIBERO task object to the TASK_SPECS key format."""
    # TASK_SPECS keys match task.name directly, e.g.
    # "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
    return task.name.split("_demo")[0] if "_demo" in task.name else task.name


class LiberoLTLEnv(LiberoEnv):
    """LiberoEnv with per-env LDBA state tracking and LTL video overlay.

    Exposes current reach-avoid subgoal text in obs["reach_avoid_texts"]
    so that downstream models can condition on it (e.g., appended to prompt).
    """

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)
        # When ap_prompt is set, feed the VLA the LDBA's CURRENT reach proposition in the
        # same atomic-proposition format used at training time (matches the composition
        # env's prompt), instead of the default "Reach: ... | Avoid: ..." text. The LDBA
        # still drives subgoal tracking and reach/avoid extraction underneath.
        self._ap_prompt = bool(cfg.get("ap_prompt", False))
        self._ap_preamble = str(
            cfg.get("ap_prompt_preamble", "Atomic-proposition subgoal to achieve:")
        )
        self._init_ldba()

    def _init_ldba(self):
        """Build LDBA for each env's task and initialize tracking state."""
        self.ldba_per_env = [None] * self.num_envs
        self.props_per_env = [None] * self.num_envs
        self.ldba_states = [None] * self.num_envs
        self.ldba_sequences = [None] * self.num_envs
        self.search_per_env = [None] * self.num_envs

        for env_id in range(self.num_envs):
            task = self.task_suite.get_task(self.task_ids[env_id])
            spec_key = _task_to_spec_key(task)
            ldba, props = _build_ldba(spec_key, suite=self.cfg.task_suite_name)
            if ldba is not None:
                self.ldba_per_env[env_id] = ldba
                self.props_per_env[env_id] = props
                self.search_per_env[env_id] = ExhaustiveSearchSimple(props, num_loops=1)
                self._reset_ldba_state(env_id)

    def _reset_ldba_state(self, env_id):
        """Reset LDBA state for a single env."""
        ldba = self.ldba_per_env[env_id]
        if ldba is None:
            self.ldba_states[env_id] = None
            self.ldba_sequences[env_id] = None
            return

        states = [CurrentState(state=ldba.initial_state, accepting=False)]
        # Take initial epsilon transitions IF the automaton has them. Rabinizer-style
        # LDBAs (ltl_benchmark.PREBUILT_HOA) start with an epsilon jump; owl-generated
        # HOAs from the LIBERO-Max hoa_store cache do not, and get_next_states(
        # take_epsilon=True) asserts epsilon exists -> guard it.
        new_states = []
        for cs in states:
            has_eps = any(
                t.is_epsilon() for t in ldba.state_to_transitions[cs.state]
            )
            if has_eps:
                for target, accepting in ldba.get_next_states(
                    cs.state, set(), take_epsilon=True
                ):
                    new_states.append(cs.get_successor(target, accepting))
            else:
                new_states.append(cs)
        self.ldba_states[env_id] = new_states

        # Run search for initial reach-avoid sequence
        search = self.search_per_env[env_id]
        try:
            self.ldba_sequences[env_id] = search(
                ldba, [s.state for s in new_states]
            )
        except NoPathsException:
            self.ldba_sequences[env_id] = None


    def _get_reach_avoid_texts(self):
        """Return a list of reach-avoid text strings, one per env.

        Each string has the format "Reach: <props> | Avoid: <props>".
        Returns empty string for envs without LDBA or without a valid sequence.
        """
        texts = []
        for env_id in range(self.num_envs):
            props = self.props_per_env[env_id]
            seq = self.ldba_sequences[env_id]
            if seq is not None and len(seq) > 0 and props is not None:
                reach, avoid = seq[0]
                reach_text = format_reach_text(reach, props)
                avoid_text = format_avoid_text(avoid, props)
                texts.append(f"Reach: {reach_text} | Avoid: {avoid_text}")
            else:
                texts.append("")
        return texts

    def _current_reach_props(self, env_id):
        """The LDBA's current reach proposition name(s) for one env (or []).

        Uses the per-state immediate-next-reach (reverse-BFS distance to acceptance),
        NOT the full-sequence search: the search's num_loops/min-length heuristic
        mis-handles nested-F finite specs (drops the accepting loop's reach -> premature
        "done", or over-merges future subgoals into the current one).
        """
        ldba = self.ldba_per_env[env_id]
        states = self.ldba_states[env_id]
        if ldba is None or not states:
            return []
        # A finite reachability spec is satisfied once an accepting transition is taken;
        # no remaining subgoal -> "done" (the accepting self-loop would otherwise keep
        # reporting the last reach prop forever).
        if ldba.is_finite_specification() and any(
            getattr(s, "num_accepting_visits", 0) >= 1 for s in states
        ):
            return []
        return immediate_reach_props(ldba, [s.state for s in states])

    def _get_ap_prompt_texts(self):
        """Per-env prompt: the LDBA's current reach proposition in AP format.

        Bridges the LDBA trace into the training-time prompt ("<preamble> pred(args)"),
        so the policy sees the same format at eval. Falls back to "done" when the
        automaton has accepted (no remaining reach prop).
        """
        from rlinf.envs.libero.libero_composition_env import render_proposition_ap

        texts = []
        for env_id in range(self.num_envs):
            reach_props = self._current_reach_props(env_id)
            if not reach_props:
                texts.append(f"{self._ap_preamble} done")
            else:
                ap = " & ".join(render_proposition_ap(p) for p in reach_props)
                texts.append(f"{self._ap_preamble} {ap}")
        return texts

    def _get_ltl_rewards(self, pre_advance_sequences, ltl_labels, ldba_infos):
        """Compute subgoal-aware reach rewards and signed safety margins.

        Evaluates the current AP labels against the reach-avoid subgoal that was
        active *before* the LDBA advance (the subgoal the agent was pursuing).

        Reach reward (dense, [0, 1] + bonus):
            For each FrozenAssignment in the reach set (disjunction), compute the
            fraction of its required true propositions that are satisfied by
            ltl_label. Take max over all alternatives. Add +1.0 bonus on
            automaton acceptance.

        Safety margin h(s) (signed, following Fisac et al. HJ reachability):
            +1.0 if any avoid condition is fully triggered (L(s) ∈ A⁻)
            -1.0 otherwise (safe)
            V_h(s) ≤ 0 certifies the trajectory never enters the avoid set.

        Args:
            pre_advance_sequences: list of LDBASequence snapshots before advance.
            ltl_labels: list of ltl_label dicts (one per env).
            ldba_infos: list of ldba_info dicts from _advance_ldba().

        Returns:
            reach_rewards: [num_envs] tensor.
            safety_margins: [num_envs] tensor, values in {-1, +1}.
        """
        from ltl_benchmark.automata.ldba_sequence import LDBASequence

        reach_rewards = np.zeros(self.num_envs, dtype=np.float32)
        safety_margins = np.full(self.num_envs, -1.0, dtype=np.float32)

        for env_id in range(self.num_envs):
            seq = pre_advance_sequences[env_id]
            label = ltl_labels[env_id] if ltl_labels is not None else None
            info = ldba_infos[env_id] if ldba_infos is not None else {}

            if seq is None or label is None or not isinstance(label, dict):
                continue

            if len(seq) == 0:
                continue

            reach, avoid = seq[0]
            true_props = {k for k, v in label.items() if v}

            # --- Reach reward: dense progress toward reach set ---
            if reach is not None and reach != LDBASequence.EPSILON:
                best_progress = 0.0
                for fa in reach:
                    required = fa.get_true_propositions()
                    if len(required) == 0:
                        best_progress = 1.0
                        break
                    satisfied = len(required & true_props)
                    progress = satisfied / len(required)
                    best_progress = max(best_progress, progress)
                reach_rewards[env_id] = best_progress

            # Bonus for automaton acceptance
            if info.get("ldba_accepted", False):
                reach_rewards[env_id] = min(reach_rewards[env_id] + 1.0, 2.0)

            # --- Signed safety margin: h(s) ∈ {+1, -1} ---
            if avoid:
                for fa in avoid:
                    required = fa.get_true_propositions()
                    if len(required) > 0 and required.issubset(true_props):
                        safety_margins[env_id] = 1.0
                        break

        return torch.from_numpy(reach_rewards), torch.from_numpy(safety_margins)

    def _wrap_obs(self, obs_list):
        """Override to inject reach-avoid subgoal text into observations."""
        obs = super()._wrap_obs(obs_list)
        if self._ap_prompt:
            # Eval-time AP-format prompting: feed the LDBA's current reach proposition
            # as the prompt (matching training), and do NOT set reach_avoid_texts so the
            # model appends nothing extra.
            obs["task_descriptions"] = self._get_ap_prompt_texts()
        else:
            obs["reach_avoid_texts"] = self._get_reach_avoid_texts()
        return obs

    def _advance_ldba(self, env_id, ltl_label):
        """Advance LDBA state for one env based on current proposition truth values."""
        ldba = self.ldba_per_env[env_id]
        if ldba is None or self.ldba_states[env_id] is None:
            return {}

        true_props = {k for k, v in ltl_label.items() if v}

        new_states = []
        for cs in self.ldba_states[env_id]:
            next_states = ldba.get_next_states(cs.state, true_props)
            for target, accepting in next_states:
                new_states.append(cs.get_successor(target, accepting))

        # Take epsilon transitions on new states (only if they have epsilon transitions)
        eps_expanded = []
        for cs in new_states:
            eps_transitions = [t for t in ldba.state_to_transitions[cs.state] if t.is_epsilon()]
            if eps_transitions:
                for t in eps_transitions:
                    eps_expanded.append(cs.get_successor(t.target, t.accepting))
            else:
                eps_expanded.append(cs)
        new_states = eps_expanded

        # Deduplicate by state
        seen = {}
        for cs in new_states:
            if cs.state not in seen or cs.num_accepting_visits > seen[cs.state].num_accepting_visits:
                seen[cs.state] = cs
        new_states = list(seen.values())

        # Filter out violating states
        new_states = [cs for cs in new_states if not ldba.is_state_violating(cs.state)]

        old_state_ids = {s.state for s in self.ldba_states[env_id]}
        self.ldba_states[env_id] = new_states

        accepted = any(s.accepting for s in new_states) if new_states else False
        violated = len(new_states) == 0
        state_changed = {s.state for s in new_states} != old_state_ids

        # Update reach-avoid sequence
        if state_changed and new_states and not violated:
            search = self.search_per_env[env_id]
            try:
                self.ldba_sequences[env_id] = search(
                    ldba, [s.state for s in new_states]
                )
            except NoPathsException:
                self.ldba_sequences[env_id] = None

        # Build info
        props = self.props_per_env[env_id]
        reach_avoid_text = ""
        seq = self.ldba_sequences[env_id]
        if seq is not None and len(seq) > 0:
            reach, avoid = seq[0]
            reach_avoid_text = (
                f"Reach: {format_reach_text(reach, props)}\n"
                f"Avoid: {format_avoid_text(avoid, props)}"
            )

        return {
            "ldba_state": [s.state for s in new_states] if new_states else [],
            "ldba_accepted": accepted,
            "ldba_violated": violated,
            "ldba_state_changed": state_changed,
            "reach_avoid_text": reach_avoid_text,
        }

    def reset(self, env_idx=None, reset_state_ids=None):
        obs, infos = super().reset(env_idx=env_idx, reset_state_ids=reset_state_ids)
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        for idx in env_idx:
            # Task may have changed after reconfigure
            task = self.task_suite.get_task(self.task_ids[idx])
            spec_key = _task_to_spec_key(task)
            ldba, props = _build_ldba(spec_key, suite=self.cfg.task_suite_name)
            if ldba is not None:
                self.ldba_per_env[idx] = ldba
                self.props_per_env[idx] = props
                self.search_per_env[idx] = ExhaustiveSearchSimple(props, num_loops=1)
            else:
                self.ldba_per_env[idx] = None
                self.props_per_env[idx] = None
                self.search_per_env[idx] = None
            self._reset_ldba_state(idx)
        # Refresh prompts now that LDBA states are initialized
        if self._ap_prompt:
            obs["task_descriptions"] = self._get_ap_prompt_texts()
        else:
            obs["reach_avoid_texts"] = self._get_reach_avoid_texts()
        return obs, infos

    def step(self, actions=None, auto_reset=True):
        """Override parent step to advance LDBA before video rendering.

        The LDBA is advanced using s_{t+1}'s AP labels, then the updated
        reach-avoid subgoal is injected into the returned observation so
        that the next action is conditioned on the correct subgoal.
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, info_lists = self.env.step(actions)
        self.current_raw_obs = raw_obs
        infos = list_of_dict_to_dict_of_list(info_lists)
        truncations = self.elapsed_steps >= self.cfg.max_episode_steps

        step_reward = self._calc_step_reward(terminations)

        # Snapshot current reach-avoid subgoals BEFORE advancing the LDBA,
        # so rewards evaluate the subgoal the agent was actually pursuing
        pre_advance_sequences = list(self.ldba_sequences)

        # Advance LDBA states BEFORE wrapping obs so that reach-avoid text
        # reflects the automaton state after processing s_{t+1}'s AP labels
        ltl_labels = infos.get("ltl_label", None)
        ldba_infos = None
        if ltl_labels is not None and isinstance(ltl_labels, list):
            ldba_infos = []
            for env_id in range(self.num_envs):
                if env_id < len(ltl_labels) and isinstance(ltl_labels[env_id], dict):
                    ldba_info = self._advance_ldba(env_id, ltl_labels[env_id])
                else:
                    ldba_info = {}
                ldba_infos.append(ldba_info)
            self._last_ldba_infos = ldba_infos

        # Wrap obs AFTER LDBA advance so reach_avoid_texts is up to date
        obs = self._wrap_obs(raw_obs)

        # Compute subgoal-aware reward signals using pre-advance subgoals
        ltl_reach_rewards, ltl_cost_signals = self._get_ltl_rewards(
            pre_advance_sequences, ltl_labels, ldba_infos
        )
        obs["ltl_reach_rewards"] = ltl_reach_rewards
        obs["ltl_cost_rewards"] = ltl_cost_signals

        if self.video_cfg.save_video:
            plot_infos = {
                "rewards": step_reward,
                "terminations": terminations,
                "task": self.task_descriptions,
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
        """Override to add LTL state overlay on video frames."""
        ldba_infos = getattr(self, "_last_ldba_infos", None)
        images = []
        for env_id, raw_single_obs in enumerate(raw_obs):
            info_item = {
                k: v if np.size(v) == 1 else v[env_id]
                for k, v in plot_infos.items()
            }

            # Add LTL info
            if ldba_infos and env_id < len(ldba_infos) and ldba_infos[env_id]:
                li = ldba_infos[env_id]
                info_item["LDBA"] = str(li.get("ldba_state", []))
                if li.get("ldba_accepted"):
                    info_item["LTL"] = "ACCEPTED"
                elif li.get("ldba_violated"):
                    info_item["LTL"] = "VIOLATED"

            img = raw_single_obs["agentview_image"][::-1, ::-1]
            extras = []
            if ldba_infos and env_id < len(ldba_infos) and ldba_infos[env_id]:
                ra_text = ldba_infos[env_id].get("reach_avoid_text", "")
                if ra_text:
                    extras = ra_text.split("\n")

            img = put_info_on_image(img, info_item, extras=extras)
            images.append(img)

        full_image = tile_images(images, nrows=int(np.sqrt(self.num_envs)))
        self.render_images.append(full_image)
