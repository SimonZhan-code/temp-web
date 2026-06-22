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

from typing import Optional

import torch

from rlinf.algorithms.registry import register_advantage
from rlinf.algorithms.utils import kl_penalty, safe_normalize
from rlinf.utils.utils import masked_mean


@register_advantage("gae")
def compute_gae_advantages_and_returns(
    rewards: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    values: Optional[torch.Tensor] = None,
    normalize_advantages: bool = True,
    normalize_returns: bool = False,
    loss_mask: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate advantages and returns for Proximal Policy Optimization (PPO).
    NOTE: currently this function does not support auto-reset.

    This function implements Generalized Advantage Estimation (GAE) to compute
    advantages and returns for PPO training. The advantages are normalized
    using mean and standard deviation for stable training.

    Args:
        rewards (torch.Tensor): Rewards per timestep. Shape: [seq_len, bsz].
        values (torch.Tensor): Value function estimates. Shape: [seq_len, bsz].
        dones (torch.Tensor): Done flags (1 if episode ended, else 0).
        gamma (float, optional): Discount factor. Defaults to 1.0.
        gae_lambda (float, optional): GAE smoothing factor. Defaults to 1.0.
        normalize_advantages (bool, optional): Whether to normalize advantages. Defaults to True.
        normalize_returns (bool, optional): Whether to normalize returns. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (advantages, returns)
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0

    critic_free = values is None
    if critic_free:
        gae_lambda = 1
        gamma = 1

    for step in reversed(range(T)):
        if critic_free:
            delta = rewards[step]
        else:
            delta = (
                rewards[step]
                + gamma * values[step + 1] * (~dones[step + 1])
                - values[step]
            )

        gae = delta + gamma * gae_lambda * (~dones[step + 1]) * gae
        returns[step] = gae if critic_free else gae + values[step]

    advantages = returns - values[:-1] if not critic_free else returns

    if normalize_advantages:
        advantages = safe_normalize(advantages, loss_mask=loss_mask)
    if normalize_returns:
        returns = safe_normalize(returns, loss_mask=loss_mask)

    return advantages, returns


@register_advantage("vmpo")
def compute_vmpo_advantages_and_returns(
    rewards: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    values: Optional[torch.Tensor] = None,
    normalize_advantages: bool = True,
    normalize_returns: bool = False,
    loss_mask: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """On-policy λ-returns / GAE for the single-critic V-MPO method.

    ``rewards`` is the already-scalarized reach-avoid reward r̃ = r_reach − λ·c_hazard
    (computed in the actor where the live λ scalar is available). With gae_lambda≈1
    (κ→1) this propagates the sparse reach signal back along the trajectory without
    bootstrapping bias. Structurally identical to GAE — reuses the same backup.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (advantages, returns)
    """
    return compute_gae_advantages_and_returns(
        rewards=rewards,
        gamma=gamma,
        gae_lambda=gae_lambda,
        values=values,
        normalize_advantages=normalize_advantages,
        normalize_returns=normalize_returns,
        loss_mask=loss_mask,
        dones=dones,
        **kwargs,
    )


def compute_segmented_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    segment_boundary: Optional[torch.Tensor],
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    loss_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Subgoal-segmented GAE for goal-conditioned V-MPO (flattened time).

    Because the V-MPO critic conditions on the active subgoal σ *only* (not the
    automaton state q), the value of (state, σ) must be the discounted reward to
    *complete σ*, terminal at subgoal satisfaction — otherwise the bootstrap across a
    subgoal switch asks a σ-only head to represent downstream return it cannot encode,
    making the regression target ill-defined. We enforce this by treating each subgoal
    boundary as a soft terminal: ``segment_dones = dones | segment_boundary``.

    With ``gae_lambda → 1`` this is Monte-Carlo return-to-segment-terminal; segmentation
    keeps each segment short so MC variance stays controlled.

    Args:
        rewards: Scalarized reward r̃. Shape [T, bsz].
        values: Value estimates V_{r̃}. Shape [T+1, bsz].
        dones: Episode done flags. Shape [T+1, bsz].
        segment_boundary: 1 at steps where the active subgoal was satisfied this step
            (the segment terminates after this step). Shape [T, bsz]. None disables
            segmentation (falls back to plain episode-level GAE).
        gamma: Discount factor.
        gae_lambda: GAE smoothing (use ≈1.0 for sparse reach).
        loss_mask: Optional validity mask. Shape [T, bsz].

    Returns:
        (advantages [T, bsz], returns [T, bsz], segment_dones [T+1, bsz]).
        Advantages are NOT normalized here — callers apply (per-subgoal) normalization.
    """
    segment_dones = dones.clone().bool()
    if segment_boundary is not None:
        # A boundary at step t terminates the segment, so the *next* state (t+1) must
        # not bootstrap across it — set dones[t+1].
        segment_dones[1:] = segment_dones[1:] | segment_boundary.bool()

    advantages, returns = compute_gae_advantages_and_returns(
        rewards=rewards,
        gamma=gamma,
        gae_lambda=gae_lambda,
        values=values,
        normalize_advantages=False,
        normalize_returns=False,
        loss_mask=loss_mask,
        dones=segment_dones,
    )
    return advantages, returns, segment_dones


def per_subgoal_normalize(
    advantages: torch.Tensor,
    group_ids: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    min_count: int = 2,
) -> torch.Tensor:
    """Standardize advantages within each subgoal group (for the V-MPO E-step).

    V-MPO's temperature dual keeps the top half of the batch by advantage. If subgoals
    have different value scales, that filter is dominated by high-scale subgoals and
    stops being per-subgoal. Standardizing advantages within each group restores the
    intended per-(s, q, σ) reweighting. Groups with fewer than ``min_count`` valid
    samples fall back to the global (masked) mean/std.

    Args:
        advantages: Shape [T, bsz] (any shape; flattened internally).
        group_ids: Integer group id per entry, same shape as ``advantages``. Typically
            the subgoal id (preferred) or a derived (column, segment) id.
        loss_mask: Optional validity mask, same shape.
        eps: Numerical floor for the std.
        min_count: Minimum valid samples for a group to use its own statistics.

    Returns:
        Normalized advantages, same shape as input (invalid entries zeroed).
    """
    shape = advantages.shape
    adv = advantages.reshape(-1).float()
    gid = group_ids.reshape(-1).long()
    if loss_mask is not None:
        m = loss_mask.reshape(-1).bool()
    else:
        m = torch.ones_like(adv, dtype=torch.bool)

    if m.sum() == 0:
        return torch.zeros_like(advantages)

    # Remap arbitrary group ids to a contiguous [0, G) range.
    uniq, inv = torch.unique(gid, return_inverse=True)
    G = uniq.numel()
    mf = m.float()
    count = torch.zeros(G, device=adv.device).index_add_(0, inv, mf)
    ssum = torch.zeros(G, device=adv.device).index_add_(0, inv, adv * mf)
    sqsum = torch.zeros(G, device=adv.device).index_add_(0, inv, (adv * adv) * mf)
    denom = count.clamp(min=1.0)
    g_mean = ssum / denom
    g_var = (sqsum / denom) - g_mean * g_mean
    g_std = g_var.clamp(min=0.0).sqrt()

    # Global (masked) fallback for tiny groups.
    glob_mean = adv[m].mean()
    glob_std = adv[m].std().clamp(min=eps)

    use_group = count[inv] >= float(min_count)
    grp_norm = (adv - g_mean[inv]) / (g_std[inv] + eps)
    glob_norm = (adv - glob_mean) / glob_std
    norm = torch.where(use_group, grp_norm, glob_norm)
    norm = norm * mf  # zero out invalid entries
    return norm.reshape(shape)


@register_advantage("grpo")
def compute_grpo_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    **kwargs,
):
    """
    Compute GRPO advantages.

    Args:
        rewards (torch.Tensor): Reward or score values. Shape: [num_groups, group_size]
        loss_mask (torch.Tensor): Loss mask for valid entries. Shape: [num_groups, group_size]
        group_size (int): Group size for advantage computation.

    Returns:
        torch.Tensor: advantages
    """
    grouped_rewards = rewards.view(-1, group_size)

    grouped_reward_mean = grouped_rewards.mean(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )
    grouped_reward_std = grouped_rewards.std(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )

    advantages = grouped_rewards - grouped_reward_mean
    advantages = advantages / (grouped_reward_std + 1e-6)

    advantages = (torch.zeros_like(loss_mask) + advantages.view(1, -1)) * loss_mask

    return advantages, None


@register_advantage("reinpp")
def compute_reinpp_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    use_reinpp_baseline: bool = False,
    kl_beta: float = 0.0,
    logprob=None,
    ref_logprob=None,
    kl_penalty_type: str = "",
    **kwargs,
):
    """
    Compute advantages for reinforce++ and reinforce++ baseline.

    Args:
        rewards (torch.Tensor): The reward or score values.
        loss_mask (torch.Tensor): The loss mask for valid entries.
        group_size (int): The group size for advantage computation.
        use_reinpp_baseline (bool, optional): Whether to use reinforce++ baseline.
        kl_beta (float, optional): KL penalty coefficient.
        logprob (optional): Log probability of current policy.
        ref_logprob (optional): Log probability of reference policy.
        kl_penalty_type (str, optional): Type of KL penalty.

    Returns:
        torch.Tensor: advantages
    """
    # first group baseline for reinforce++ baseline
    if use_reinpp_baseline:
        grouped_rewards = rewards.view(-1, group_size)  # [num_prompt, group_size]
        grouped_rewards -= grouped_rewards.mean(dim=1, keepdims=True)
        rewards = grouped_rewards.view(-1)  # [B]

    # build the reward matrix
    r_matrix = torch.zeros_like(loss_mask).float()  # [L, B]
    seq_length = loss_mask.size(0)
    mask_flipped = loss_mask.long().fliplr()
    eos_positions = mask_flipped.argmax(
        dim=0, keepdim=True
    )  # position of last True in original mask
    eos_indices = seq_length - 1 - eos_positions  # [1, B]

    r_matrix = r_matrix.scatter_(dim=0, index=eos_indices, src=rewards)  # [L, B]

    # add kl penalty
    if kl_beta > 0:
        kld = kl_penalty(logprob, ref_logprob, kl_penalty=kl_penalty_type)  # [L, B]
        r_matrix -= kl_beta * kld

    # compute return
    ret_matrix = torch.cumsum(r_matrix.flip(dims=[0]), dim=0).flip(dims=[0])

    # normalize
    advantages = ret_matrix.clone()

    mean = masked_mean(advantages, loss_mask)
    var = masked_mean((advantages - mean).pow(2), loss_mask)
    rstd = var.clamp(min=1e-8).rsqrt()

    advantages = (advantages - mean) * rstd

    return advantages, None
