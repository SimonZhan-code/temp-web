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

"""Unit tests for the single-critic V-MPO method (no MuJoCo / GPU required)."""

import pytest

torch = pytest.importorskip("torch")

from rlinf.algorithms.dual import (  # noqa: E402
    SafetyLagrangeMultiplier,
    VMPOTemperatureOptimizer,
)
from rlinf.algorithms.registry import ADV_REGISTRY, LOSS_REGISTRY  # noqa: E402


# --------------------------------------------------------------------------- #
# VMPOTemperatureOptimizer
# --------------------------------------------------------------------------- #
def test_vmpo_temperature_respects_eta_min():
    opt = VMPOTemperatureOptimizer(eta_init=1.0, eta_min=0.05)
    adv = torch.randn(256)
    opt.step(adv)
    assert opt.eta.item() >= 0.05


def test_vmpo_temperature_top_half_fraction():
    opt = VMPOTemperatureOptimizer()
    adv = torch.arange(100, dtype=torch.float32)
    metrics = opt.step(adv)
    # median split keeps roughly half the (unmasked) entries
    assert 0.45 <= metrics["vmpo/top_half_frac"] <= 0.55


def test_vmpo_temperature_dual_loss_decreases():
    opt = VMPOTemperatureOptimizer(eta_init=3.0, lr_eta=0.1, epsilon_eta=0.1)
    adv = torch.randn(512) * 2.0
    first = opt.step(adv)["vmpo/dual_loss"]
    last = first
    for _ in range(50):
        last = opt.step(adv)["vmpo/dual_loss"]
    assert last <= first + 1e-4  # convex 1-D dual should not increase


def test_vmpo_temperature_respects_mask():
    opt = VMPOTemperatureOptimizer()
    adv = torch.randn(64)
    mask = torch.zeros(64, dtype=torch.bool)
    mask[:8] = True
    metrics = opt.step(adv, mask=mask)
    assert torch.isfinite(torch.tensor(metrics["vmpo/dual_loss"]))


# --------------------------------------------------------------------------- #
# SafetyLagrangeMultiplier
# --------------------------------------------------------------------------- #
def test_safety_fixed_mode_is_constant():
    m = SafetyLagrangeMultiplier(mode="fixed", beta=2.5)
    assert m.lam.item() == pytest.approx(2.5)
    m.step(measured_cost_return=10.0)  # no-op
    assert m.lam.item() == pytest.approx(2.5)


def test_safety_adaptive_rises_on_violation():
    m = SafetyLagrangeMultiplier(
        mode="adaptive", lambda_init=0.0, epsilon_1=0.0, alpha_lambda=0.1
    )
    m.step(measured_cost_return=1.0)  # J_c > ε₁ ⇒ λ rises
    assert m.lam.item() > 0.0


def test_safety_adaptive_relaxes_and_stays_nonnegative():
    m = SafetyLagrangeMultiplier(
        mode="adaptive", lambda_init=1.0, epsilon_1=0.0, alpha_lambda=0.5
    )
    for _ in range(20):
        m.step(measured_cost_return=0.0)  # J_c < ε₁ ⇒ λ relaxes
    assert m.lam.item() >= 0.0
    assert m.lam.item() < 1.0


def test_safety_adaptive_clamped_to_lambda_max():
    m = SafetyLagrangeMultiplier(
        mode="adaptive", lambda_init=0.0, epsilon_1=0.0,
        alpha_lambda=10.0, lambda_max=3.0,
    )
    for _ in range(10):
        m.step(measured_cost_return=5.0)
    assert m.lam.item() <= 3.0


# --------------------------------------------------------------------------- #
# Registry dispatch
# --------------------------------------------------------------------------- #
def test_vmpo_registered():
    assert "vmpo" in ADV_REGISTRY
    assert "vmpo" in LOSS_REGISTRY


def test_vmpo_advantage_matches_gae():
    """The vmpo advantage is a thin GAE wrapper — identical given the same inputs."""
    from rlinf.algorithms.advantages import (
        compute_gae_advantages_and_returns,
        compute_vmpo_advantages_and_returns,
    )

    T, B = 8, 4
    rewards = torch.randn(T, B)
    values = torch.randn(T + 1, B)
    dones = torch.zeros(T + 1, B, dtype=torch.bool)
    loss_mask = torch.ones(T, B, dtype=torch.bool)

    a_gae, r_gae = compute_gae_advantages_and_returns(
        rewards=rewards.clone(), gamma=0.99, gae_lambda=0.97,
        values=values.clone(), normalize_advantages=False,
        loss_mask=loss_mask, dones=dones,
    )
    a_vmpo, r_vmpo = compute_vmpo_advantages_and_returns(
        rewards=rewards.clone(), gamma=0.99, gae_lambda=0.97,
        values=values.clone(), normalize_advantages=False,
        loss_mask=loss_mask, dones=dones,
    )
    assert torch.allclose(a_gae, a_vmpo)
    assert torch.allclose(r_gae, r_vmpo)
    assert torch.isfinite(a_vmpo).all()
    assert torch.isfinite(r_vmpo).all()


# --------------------------------------------------------------------------- #
# Subgoal segmentation + per-subgoal normalization
# --------------------------------------------------------------------------- #
def test_segmented_gae_blocks_bootstrap_across_subgoal():
    """A subgoal boundary must stop value bootstrapping across it."""
    from rlinf.algorithms.advantages import compute_segmented_gae

    T, B = 6, 1
    rewards = torch.zeros(T, B)
    rewards[2, 0] = 1.0  # subgoal-1 completion reward at t=2
    rewards[5, 0] = 1.0  # subgoal-2 completion reward at t=5
    values = torch.zeros(T + 1, B)
    dones = torch.zeros(T + 1, B, dtype=torch.bool)
    boundary = torch.zeros(T, B)
    boundary[2, 0] = 1.0  # segment terminates after t=2

    _, ret_seg, seg_dones = compute_segmented_gae(
        rewards, values, dones, boundary, gamma=0.99, gae_lambda=1.0
    )
    _, ret_no, _ = compute_segmented_gae(
        rewards, values, dones, None, gamma=0.99, gae_lambda=1.0
    )
    # segment_dones marks the step AFTER the boundary (index 3)
    assert bool(seg_dones[3, 0])
    # Without segmentation, t=0 return sees both rewards; with segmentation only the first.
    assert ret_seg[0, 0] < ret_no[0, 0] - 1e-6
    # The second segment's reward must not leak into the first segment's return.
    assert ret_seg[0, 0] == pytest.approx((0.99**2) * 1.0, abs=1e-5)


def test_segmented_gae_none_boundary_equals_plain_gae():
    from rlinf.algorithms.advantages import (
        compute_gae_advantages_and_returns,
        compute_segmented_gae,
    )

    T, B = 5, 3
    rewards = torch.randn(T, B)
    values = torch.randn(T + 1, B)
    dones = torch.zeros(T + 1, B, dtype=torch.bool)
    a_seg, r_seg, _ = compute_segmented_gae(
        rewards.clone(), values.clone(), dones, None, gamma=0.99, gae_lambda=0.95
    )
    a_ref, r_ref = compute_gae_advantages_and_returns(
        rewards=rewards.clone(), gamma=0.99, gae_lambda=0.95,
        values=values.clone(), normalize_advantages=False, dones=dones,
    )
    assert torch.allclose(a_seg, a_ref)
    assert torch.allclose(r_seg, r_ref)


def test_per_subgoal_normalize_zero_means_per_group():
    from rlinf.algorithms.advantages import per_subgoal_normalize

    T, B = 4, 2
    # group A (id 0) has large scale, group B (id 1) small scale
    adv = torch.tensor(
        [[10.0, 0.1], [20.0, 0.2], [30.0, 0.3], [40.0, 0.4]]
    )
    group_ids = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
    out = per_subgoal_normalize(adv, group_ids, loss_mask=None)
    # each column is its own group → standardized to ~zero mean, unit-ish std
    for c in range(B):
        col = out[:, c]
        assert abs(col.mean().item()) < 1e-5
        assert col.std().item() == pytest.approx(1.0, abs=0.2)


def test_per_subgoal_normalize_respects_mask():
    from rlinf.algorithms.advantages import per_subgoal_normalize

    adv = torch.randn(8, 2)
    group_ids = torch.randint(0, 3, (8, 2))
    mask = torch.ones(8, 2, dtype=torch.bool)
    mask[0, 0] = False
    out = per_subgoal_normalize(adv, group_ids, loss_mask=mask)
    assert out[0, 0].item() == 0.0  # masked entry zeroed
    assert torch.isfinite(out).all()


def test_vmpo_loss_is_critic_only_by_default():
    """compute_vmpo_loss returns finite critic loss and no actor term by default."""
    from rlinf.algorithms.registry import get_policy_loss

    loss_fn = get_policy_loss("vmpo")
    B = 16
    values = torch.randn(B, requires_grad=True)
    returns = torch.randn(B)
    prev_values = torch.randn(B)
    loss, metrics = loss_fn(
        values=values,
        returns=returns,
        prev_values=prev_values,
        value_clip=0.2,
        huber_delta=10.0,
        loss_mask=torch.ones(B, dtype=torch.bool),
    )
    assert torch.isfinite(loss)
    assert "critic/value_loss" in metrics
    assert "actor/policy_loss" not in metrics  # actor loss OFF by default
