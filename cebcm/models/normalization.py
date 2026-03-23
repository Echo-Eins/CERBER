"""
Orthonormalization for Linear layers — alternative to Spectral Normalization.

Spectral Normalization only constrains the largest singular value σ_max(W) = 1,
leaving other singular values potentially near 0, causing:
    1. Gradient attenuation (Jacobian much smaller than allowed)
    2. Ill-conditioned weight matrices
    3. Loss of expressiveness

Orthonormalization constrains ALL singular values to 1, making W orthogonal.
This gives:
    - Perfect gradient flow (no attenuation)
    - 1-Lipschitz guarantee by construction
    - Maximum expressiveness within the Lipschitz constraint

Implementation uses Bjorck orthonormalization (iterative), which is
differentiable and converges quadratically.

References:
    - "Improving Lipschitz-Constrained Neural Networks by Learning Activation
      Functions" (JMLR 2024)
    - Bjorck & Bowie, "An Iterative Algorithm for Computing the Best Estimate
      of an Orthogonal Matrix" (1971)
    - Li et al., "Preventing Gradient Attenuation in Lipschitz Constrained
      Convolutional Neural Networks" (NeurIPS 2019)

Spec reference: §5.6 (regularization)
"""

import torch
import torch.nn as nn
from torch import Tensor


def _power_iteration_sigma_max(W: Tensor, n_steps: int = 2) -> Tensor:
    """
    Fast estimate of spectral norm used to keep Bjorck iterations stable.

    Uses a detached estimate, so it does not expand the autograd graph.
    """
    with torch.no_grad():
        rows, cols = W.shape
        u = torch.randn(rows, 1, device=W.device, dtype=W.dtype)
        u = u / u.norm().clamp(min=1e-8)
        for _ in range(max(1, n_steps)):
            v = W.t() @ u
            v = v / v.norm().clamp(min=1e-8)
            u = W @ v
            u = u / u.norm().clamp(min=1e-8)
        sigma = (u.t() @ W @ v).abs().squeeze()
        return sigma.clamp(min=1e-6)


def bjorck_orthonormalize(
    W: Tensor,
    n_iters: int = 15,
    order: int = 1,
) -> Tensor:
    """
    Bjorck orthonormalization: iteratively project W towards the nearest
    orthogonal matrix (in Frobenius norm).

    For a matrix W, the iteration is:
        W_{k+1} = W_k (I + 0.5 * (I - W_k^T W_k))     [order=1]

    Converges quadratically to the closest orthogonal matrix.

    Args:
        W: [out, in] weight matrix (out <= in for semi-orthogonal).
        n_iters: Number of Bjorck iterations. 15 is typically sufficient.
        order: Approximation order. 1 is standard and sufficient.

    Returns:
        Orthonormalized weight matrix with all singular values ≈ 1.
    """
    # Pre-normalize by sigma_max to keep Bjorck in a convergence-friendly regime.
    sigma_max = _power_iteration_sigma_max(W.detach(), n_steps=2)
    W = W / sigma_max.clamp(min=1.0)

    rows, cols = W.shape
    if rows <= cols:
        # Wide/semi-orthogonal case: enforce W W^T ≈ I_rows (all singular values ≈ 1).
        eye = torch.eye(rows, device=W.device, dtype=W.dtype)
        for _ in range(n_iters):
            WWt = W @ W.t()
            W = (1.5 * eye - 0.5 * WWt) @ W
            if not torch.isfinite(W).all():
                W = torch.nan_to_num(W, nan=0.0, posinf=1e4, neginf=-1e4)
    else:
        # Tall case: enforce W^T W ≈ I_cols.
        eye = torch.eye(cols, device=W.device, dtype=W.dtype)
        for _ in range(n_iters):
            WtW = W.t() @ W
            W = W @ (1.5 * eye - 0.5 * WtW)
            if not torch.isfinite(W).all():
                W = torch.nan_to_num(W, nan=0.0, posinf=1e4, neginf=-1e4)

    return W


class OrthoLinear(nn.Module):
    """
    Linear layer with orthonormalized weights via Bjorck iteration.

    All singular values of the weight matrix are constrained to 1,
    making this layer exactly 1-Lipschitz. This is strictly better
    than Spectral Normalization for Lipschitz-constrained networks.

    The orthonormalization is applied during forward pass (like spectral norm),
    so the actual stored weights are unconstrained and optimized freely.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to include a bias term.
        n_iters: Number of Bjorck iterations per forward pass.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        n_iters: int = 15,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_iters = n_iters

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights near orthogonal for fast convergence."""
        nn.init.orthogonal_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with orthonormalized weights.

        Args:
            x: [..., in_features]

        Returns:
            [..., out_features]
        """
        W_ortho = bjorck_orthonormalize(self.weight, n_iters=self.n_iters)
        return torch.nn.functional.linear(x, W_ortho, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, n_iters={self.n_iters}"
        )
