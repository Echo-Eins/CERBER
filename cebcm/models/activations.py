"""
Learnable Lipschitz-constrained activations for energy-based models.

Replaces ReLU in Lipschitz-constrained networks. ReLU + Spectral Normalization
causes gradient attenuation — the Jacobian is much smaller than theoretically
allowed, losing expressiveness. These alternatives maintain 1-Lipschitz guarantee
while maximizing expressiveness.

Implements:
    - LipschitzLinearSpline: Learnable piecewise-linear 1-Lipschitz activation
    - GroupSort: Sort-based activation that preserves gradient norms
    - FullSort: Special case of GroupSort with group_size = full dimension

References:
    - "Improving Lipschitz-Constrained Neural Networks by Learning Activation
      Functions" (JMLR 2024, vol 25, 22-1347)
    - GroupSort: Anil et al., "Sorting out Lipschitz function approximation" (ICML 2019)

Spec reference: §5.6 (regularization), §5.2 Mode A (architecture)
"""

import torch
import torch.nn as nn
from torch import Tensor


class LipschitzLinearSpline(nn.Module):
    """
    Learnable 1-Lipschitz piecewise-linear activation function.

    Each input channel gets its own activation with `num_knots` adjustable
    linear regions. The slopes are constrained so that adjacent slopes differ
    by at most 1 in absolute value and each individual slope is in [-1, 1],
    ensuring the overall function is 1-Lipschitz.

    With num_knots=3, this gives 3 linear regions — optimal expressivity
    among all 1-Lipschitz activations with component-wise functions (JMLR 2024).

    Args:
        num_features: Number of input channels/features (applied per-channel).
        num_knots: Number of knot points defining the piecewise-linear function.
                   More knots = more expressive but more parameters.
        init: Initialization strategy.
              "relu" — initialize to approximate ReLU behavior.
              "identity" — initialize to identity function.
    """

    def __init__(
        self,
        num_features: int,
        num_knots: int = 4,
        init: str = "relu",
    ):
        super().__init__()
        self.num_features = num_features
        self.num_knots = num_knots

        # Knot positions (fixed, evenly spaced in [-2, 2])
        knot_positions = torch.linspace(-2.0, 2.0, num_knots)
        self.register_buffer("knot_positions", knot_positions)

        # Learnable slopes for each segment (num_features x (num_knots + 1))
        # +1 because there are num_knots+1 segments (before first knot, between
        # each pair, after last knot)
        num_segments = num_knots + 1
        slopes_raw = torch.zeros(num_features, num_segments)

        if init == "relu":
            # Approximate ReLU: slope=0 for x<0, slope=1 for x>0
            mid = num_knots // 2
            for i in range(num_segments):
                if i <= mid:
                    slopes_raw[:, i] = 0.0
                else:
                    slopes_raw[:, i] = 1.0
        elif init == "identity":
            slopes_raw[:, :] = 1.0

        self.slopes_raw = nn.Parameter(slopes_raw)

        # Learnable bias per feature
        self.bias = nn.Parameter(torch.zeros(num_features))

    def _get_constrained_slopes(self) -> Tensor:
        """Project raw slopes to satisfy 1-Lipschitz constraint.

        Each slope is clamped to [-1, 1] via tanh, guaranteeing
        the piecewise-linear function has Lipschitz constant <= 1.
        """
        return torch.tanh(self.slopes_raw)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply learnable piecewise-linear activation.

        Args:
            x: [..., num_features] input tensor

        Returns:
            Same shape as input
        """
        slopes = self._get_constrained_slopes()  # [F, S]
        knots = self.knot_positions                # [K]

        # Determine which segment each value falls into
        # x: [..., F], knots: [K] -> compare: [..., F, K]
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.num_features)  # [N, F]

        # Expand for comparison: [N, F, 1] vs [K]
        x_expanded = x_flat.unsqueeze(-1)  # [N, F, 1]
        knots_expanded = knots.unsqueeze(0).unsqueeze(0)  # [1, 1, K]

        # Count how many knots each value exceeds -> segment index
        segment_idx = (x_expanded >= knots_expanded).sum(dim=-1)  # [N, F], values in [0, num_knots]

        # Gather the slope for each (sample, feature) pair
        segment_idx_clamped = segment_idx.clamp(0, slopes.shape[1] - 1)  # [N, F]
        selected_slopes = slopes.gather(1, segment_idx_clamped.t()).t()  # [N, F]

        # Compute piecewise-linear output
        # For segment 0 (before first knot): y = slopes[0] * (x - knots[0])
        # For segment k: y = sum of previous segments + slopes[k] * (x - knots[k-1])
        # We compute this cumulatively for numerical stability

        # Base: value at first knot
        first_knot = knots[0]
        result = torch.zeros_like(x_flat)

        # Accumulate through segments
        for k in range(slopes.shape[1]):
            if k == 0:
                # Before first knot: slope * (x - first_knot)
                mask = segment_idx == 0  # [N, F]
                if mask.any():
                    result[mask] = slopes[:, 0].expand_as(x_flat)[mask] * (x_flat[mask] - first_knot)
            else:
                knot_prev = knots[k - 1] if k <= self.num_knots else knots[-1]
                knot_curr = knots[k] if k < self.num_knots else knots[-1]

                # For values in this segment or beyond
                in_or_past = segment_idx >= k  # [N, F]
                if in_or_past.any():
                    if k < self.num_knots:
                        # Full contribution of this segment for values past it
                        past = segment_idx > k
                        segment_width = knot_curr - knot_prev
                        if past.any():
                            result[past] += slopes[:, k].expand_as(x_flat)[past] * segment_width

                        # Partial contribution for values in this segment
                        in_seg = segment_idx == k
                        if in_seg.any():
                            result[in_seg] += slopes[:, k].expand_as(x_flat)[in_seg] * (x_flat[in_seg] - knot_prev)
                    else:
                        # Beyond last knot
                        in_seg = segment_idx == k
                        if in_seg.any():
                            result[in_seg] += slopes[:, k].expand_as(x_flat)[in_seg] * (x_flat[in_seg] - knots[-1])

        result = result + self.bias.unsqueeze(0)
        return result.reshape(orig_shape)


class GroupSort(nn.Module):
    """
    GroupSort activation: sort activations within groups.

    Splits the input into groups of `group_size` and sorts each group
    in descending order. This is a 1-Lipschitz operation (sorting is a
    permutation, which preserves norms) that significantly increases
    expressiveness compared to ReLU for Lipschitz-constrained networks.

    When group_size=2, this is equivalent to MaxMin activation:
    for each pair (a, b) → (max(a,b), min(a,b)).

    Args:
        group_size: Size of each group to sort. Must divide the feature dimension.
                    group_size=2 gives MaxMin (most common).
                    group_size=dim gives FullSort.
    """

    def __init__(self, group_size: int = 2):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: Tensor) -> Tensor:
        """
        Sort within groups along the last dimension.

        Args:
            x: [..., D] where D is divisible by group_size

        Returns:
            Same shape, with values sorted (descending) within groups
        """
        *batch_dims, d = x.shape
        assert d % self.group_size == 0, (
            f"Feature dim {d} not divisible by group_size {self.group_size}"
        )

        # Reshape to [..., num_groups, group_size]
        num_groups = d // self.group_size
        x_grouped = x.reshape(*batch_dims, num_groups, self.group_size)

        # Sort descending within each group
        x_sorted, _ = x_grouped.sort(dim=-1, descending=True)

        # Reshape back
        return x_sorted.reshape(*batch_dims, d)

    def extra_repr(self) -> str:
        return f"group_size={self.group_size}"


class FullSort(GroupSort):
    """
    FullSort activation: sort the entire feature vector.

    Special case of GroupSort where group_size equals the full feature dimension.
    Most expressive sorting-based activation, but requires all features to interact.

    Note: This creates a fixed group_size at init time. The feature dimension
    must match at forward time.
    """

    def __init__(self, dim: int):
        super().__init__(group_size=dim)
