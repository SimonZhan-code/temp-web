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

import torch
import torch.nn as nn


class VMPOTemperatureOptimizer(nn.Module):
    """Single-variable V-MPO temperature dual for on-policy latent steering.

    Used by the single-critic V-MPO method (scalarized reach-avoid reward). The safety
    multiplier λ lives in the reward via scalarization and is updated by a separate slow
    dual-ascent step (``SafetyLagrangeMultiplier``), so this temperature dual is
    one-dimensional.

    On the fresh on-policy batch we keep the **top half by advantage** (V-MPO's
    variance-control filter), 𝒟̃ = {t : Ã_t ≥ median(Ã)}, and solve:

        η* = argmin_{η ≥ η_min}  η·ε_η + η · log( (1/|𝒟̃|) Σ_{t∈𝒟̃} exp(Ã_t / η) )

    The nonparametric optimum is the Boltzmann weighting ψ(z_t) ∝ exp(Ã_t/η*), which
    sets the temperature for best-of-N selection at rollout time.

    Args:
        eta_init: Initial temperature η.
        eta_min: Lower bound for η (prevents degenerate sharp selection).
        epsilon_eta: Temperature trust-region budget ε_η.
        lr_eta: Learning rate for η updates.
    """

    def __init__(
        self,
        eta_init: float = 1.0,
        eta_min: float = 0.01,
        epsilon_eta: float = 0.1,
        lr_eta: float = 0.01,
    ):
        super().__init__()
        self.eta_min = eta_min
        self.epsilon_eta = epsilon_eta
        self.lr_eta = lr_eta

        # log-space parameter enforces positivity via exp()
        self._log_eta = nn.Parameter(torch.tensor(eta_init).clamp(min=eta_min).log())
        self.optimizer = torch.optim.Adam([self._log_eta], lr=lr_eta)

    @property
    def eta(self) -> torch.Tensor:
        return self._log_eta.exp().clamp(min=self.eta_min)

    def _top_half_mask(
        self,
        advantages: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Boolean mask selecting valid entries with Ã ≥ median(Ã)."""
        valid = (
            mask.flatten().bool()
            if mask is not None
            else torch.ones_like(advantages.flatten(), dtype=torch.bool)
        )
        adv = advantages.flatten()
        if valid.sum() == 0:
            return valid
        median = adv[valid].median()
        return valid & (adv >= median)

    def dual_loss(
        self,
        advantages: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the single-variable temperature dual g(η) over the top half.

        g(η) = η·ε_η + η · log( (1/N) Σ exp(Ã_t / η) ),  N = |top half|.
        """
        # The η parameter lives on CPU but advantages arrive on the actor's GPU; align
        # the scalar to the advantages' device (grad still flows back to the CPU leaf).
        eta = self.eta.to(advantages.device)
        if mask is not None:
            mask = mask.to(advantages.device)
        adv = advantages.detach().flatten()
        top = self._top_half_mask(advantages, mask)
        adv = adv[top]

        if adv.numel() == 0:
            return torch.tensor(0.0, device=eta.device)

        # log-mean-exp for numerical stability
        scaled = adv / eta
        log_mean_exp = torch.logsumexp(scaled, dim=0) - torch.log(
            torch.tensor(float(adv.numel()), device=eta.device)
        )
        return eta * self.epsilon_eta + eta * log_mean_exp

    @torch.no_grad()
    def step(
        self,
        advantages: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Run one temperature dual optimization step.

        Args:
            advantages: On-policy advantages Ã_t. Any shape (flattened internally).
            mask: Optional validity mask matching ``advantages``.

        Returns:
            Dict with updated eta, dual loss, top-half fraction, advantage median.
        """
        self.optimizer.zero_grad()
        with torch.enable_grad():
            loss = self.dual_loss(advantages, mask)
            loss.backward()
        self.optimizer.step()

        # diagnostics
        top = self._top_half_mask(advantages, mask)
        adv_flat = advantages.detach().flatten()
        if mask is not None:
            valid = mask.flatten().bool()
            n_valid = int(valid.sum().item())
            adv_median = (
                adv_flat[valid].median().item() if n_valid > 0 else 0.0
            )
        else:
            n_valid = adv_flat.numel()
            adv_median = adv_flat.median().item() if n_valid > 0 else 0.0
        top_half_frac = (top.sum().item() / n_valid) if n_valid > 0 else 0.0

        return {
            "vmpo/eta": self.eta.item(),
            "vmpo/dual_loss": loss.item(),
            "vmpo/top_half_frac": top_half_frac,
            "vmpo/adv_median": adv_median,
        }

    def state_dict_scalars(self) -> dict[str, float]:
        """Return current η* as a plain float for syncing to model config."""
        return {"eta": self.eta.item()}


class SafetyLagrangeMultiplier(nn.Module):
    """Scalar safety multiplier λ in the scalarized reward r̃ = r_reach − λ·c_hazard.

    Two modes (from the single-critic method's safety-penalty design):

    - ``"fixed"``: λ ≡ β, a tuned constant. Stationary shaped reward, no dual
      dynamics — at the cost of no formal constraint satisfaction.
    - ``"adaptive"``: PPO-Lagrangian dual ascent on the *measured* cost return of the
      current best-of-N policy,
          λ ← [ λ + α_λ·(J_c − ε_1) ]_+ .
      λ rises when the policy violates the budget ε_1 and relaxes when it is safe.
      Kept on a slower timescale than the critic so V_{r̃} can track it.

    Args:
        mode: ``"fixed"`` or ``"adaptive"``.
        beta: Constant λ for fixed mode.
        lambda_init: Initial λ for adaptive mode.
        epsilon_1: Cost budget ε_1 (J_c ≤ ε_1) for adaptive mode.
        alpha_lambda: Dual-ascent step size α_λ for adaptive mode.
        lambda_max: Upper clamp for λ (projection bound).
    """

    def __init__(
        self,
        mode: str = "fixed",
        beta: float = 0.0,
        lambda_init: float = 0.0,
        epsilon_1: float = 0.0,
        alpha_lambda: float = 0.05,
        lambda_max: float = 100.0,
    ):
        super().__init__()
        assert mode in ("fixed", "adaptive"), f"unknown safety mode {mode}"
        self.mode = mode
        self.epsilon_1 = epsilon_1
        self.alpha_lambda = alpha_lambda
        self.lambda_max = lambda_max

        # Plain buffers (projected ascent — no autograd needed).
        self.register_buffer("_beta", torch.tensor(float(beta)))
        self.register_buffer("_lambda", torch.tensor(float(lambda_init)))

    @property
    def lam(self) -> torch.Tensor:
        return self._beta if self.mode == "fixed" else self._lambda

    @torch.no_grad()
    def step(self, measured_cost_return: float) -> dict[str, float]:
        """Update λ from the measured discounted cost return J_c.

        Fixed mode is a no-op (λ = β). Adaptive mode performs one projected
        dual-ascent step.
        """
        if self.mode == "adaptive":
            new_lambda = self._lambda + self.alpha_lambda * (
                float(measured_cost_return) - self.epsilon_1
            )
            self._lambda.fill_(float(new_lambda.clamp(min=0.0, max=self.lambda_max)))
        return {
            "vmpo/lambda": self.lam.item(),
            "vmpo/Jc": float(measured_cost_return),
        }
