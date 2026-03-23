"""
UnconditionalEnergy — scalar energy E(x) → ℝ for Energy Matching training.

Unlike SimpleEnergy (pairwise, σ-conditioned), this model takes a single
embedding x ∈ ℝ^d and outputs a scalar energy. No time conditioning, no
auxiliary inputs — the energy landscape must encode the full data distribution.

Architecture:
    x (1024d) → OrthoLinear(1024, 2048) → GroupSort(2)
              → OrthoLinear(2048, 1024) → GroupSort(2)
              → OrthoLinear(1024, 512)  → GroupSort(2)
              → Linear(512, 1)          → scalar E

The hidden layers are 1-Lipschitz (OrthoLinear + GroupSort) for smooth energy
landscapes. The final layer is unconstrained so energy magnitude can adapt freely.

Mathematical basis:
    Energy Matching (Balcerak et al., NeurIPS 2025, arXiv:2504.10612) trains
    E_θ such that -∇_x E_θ(x_t) ≈ u_t(x_t | x_1), where u_t is the conditional
    OT velocity field. The learned energy is time-invariant — one landscape
    simultaneously encodes transport directions at all positions.

Spec reference: §10.5, tasks/todo.md §0
"""

import torch
import torch.nn as nn
from torch import Tensor

from cebcm.models.activations import GroupSort, LipschitzLinearSpline
from cebcm.models.normalization import OrthoLinear


class UnconditionalEnergy(nn.Module):
    """
    E_θ(x) → scalar energy. Lower energy = higher data likelihood.

    Args:
        dim: Input embedding dimension (1024 for SONAR).
        hidden_dims: Hidden layer sizes. Default: [2048, 1024, 512].
        norm_mode: Weight normalization — "orthonorm", "spectral_norm", or "none".
        activation: Activation — "groupsort", "lipschitz_spline", or "relu".
        ortho_n_iters: Bjorck iterations for orthonormalization.
        groupsort_size: Group size for GroupSort (2 = MaxMin).
        spline_num_knots: Knots for LipschitzLinearSpline.
    """

    def __init__(
        self,
        dim: int = 1024,
        hidden_dims: list[int] | None = None,
        norm_mode: str = "orthonorm",
        activation: str = "groupsort",
        ortho_n_iters: int = 15,
        groupsort_size: int = 2,
        spline_num_knots: int = 4,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [2048, 1024, 512]

        self.norm_mode = norm_mode
        self.activation_name = activation

        layers: list[nn.Module] = []
        prev_dim = dim

        for h_dim in hidden_dims:
            # Ensure hidden dim is divisible by groupsort_size
            if activation == "groupsort" and h_dim % groupsort_size != 0:
                h_dim = (h_dim // groupsort_size) * groupsort_size

            linear = self._make_linear(prev_dim, h_dim, norm_mode, ortho_n_iters)
            layers.append(linear)

            act = self._make_activation(activation, h_dim, groupsort_size, spline_num_knots)
            layers.append(act)

            prev_dim = h_dim

        # Final projection: unconstrained linear → scalar
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

        # Learnable energy scale (log-parameterized)
        self.log_energy_scale = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _make_linear(
        in_dim: int,
        out_dim: int,
        norm_mode: str,
        ortho_n_iters: int,
    ) -> nn.Module:
        if norm_mode == "orthonorm":
            return OrthoLinear(in_dim, out_dim, bias=True, n_iters=ortho_n_iters)
        elif norm_mode == "spectral_norm":
            linear = nn.Linear(in_dim, out_dim)
            return nn.utils.parametrizations.spectral_norm(linear)
        else:
            return nn.Linear(in_dim, out_dim)

    @staticmethod
    def _make_activation(
        activation: str,
        dim: int,
        groupsort_size: int,
        spline_num_knots: int,
    ) -> nn.Module:
        if activation == "groupsort":
            return GroupSort(group_size=groupsort_size)
        elif activation == "lipschitz_spline":
            return LipschitzLinearSpline(num_features=dim, num_knots=spline_num_knots)
        elif activation == "relu":
            return nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute energy for a batch of embeddings.

        Args:
            x: [B, D] embeddings.

        Returns:
            [B] scalar energies.
        """
        return self.log_energy_scale.exp() * self.net(x).squeeze(-1)

    def energy_and_grad(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute energy and its gradient ∇_x E(x) in one pass.

        Used for:
        - Langevin sampling: x_{k+1} = x_k - η·∇E + noise
        - ODE integration: dx/dt = -∇E

        Args:
            x: [B, D] embeddings.

        Returns:
            (energy [B], grad [B, D])
        """
        x = x.detach().requires_grad_(True)
        energy = self.forward(x)
        grad = torch.autograd.grad(
            energy.sum(), x, create_graph=False
        )[0]
        return energy.detach(), grad.detach()

    def sample_ode(
        self,
        x_init: Tensor,
        num_steps: int = 100,
        target_norm: float | None = None,
    ) -> Tensor:
        """
        Generate samples via Euler ODE integration: dx/dt = -∇E.

        Args:
            x_init: [B, D] initial points (from prior distribution).
            num_steps: Number of Euler steps.
            target_norm: If set, project to sphere after each step.

        Returns:
            [B, D] generated samples.
        """
        dt = 1.0 / num_steps
        x = x_init.clone()

        for _ in range(num_steps):
            _, grad = self.energy_and_grad(x)
            x = x - dt * grad  # dx/dt = -∇E → x += dt * (-∇E)

            if target_norm is not None:
                x = torch.nn.functional.normalize(x, dim=-1) * target_norm

        return x

    def sample_langevin(
        self,
        x_init: Tensor,
        num_steps: int = 100,
        lr: float = 0.01,
        noise_scale: float = 0.005,
        target_norm: float | None = None,
    ) -> Tensor:
        """
        Generate samples via Langevin dynamics: x_{k+1} = x_k - η∇E + √(2η)ε.

        Args:
            x_init: [B, D] initial points.
            num_steps: Number of Langevin steps.
            lr: Step size η.
            noise_scale: Noise multiplier.
            target_norm: If set, project to sphere after each step.

        Returns:
            [B, D] generated samples.
        """
        x = x_init.clone()

        for _ in range(num_steps):
            _, grad = self.energy_and_grad(x)
            noise = torch.randn_like(x) * noise_scale
            x = x - lr * grad + (2 * lr) ** 0.5 * noise

            if target_norm is not None:
                x = torch.nn.functional.normalize(x, dim=-1) * target_norm

        return x
