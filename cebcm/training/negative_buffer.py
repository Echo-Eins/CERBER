"""
Negative sample buffer and NCE loss for EBM warmstart training.

The replay buffer maintains a pool of "negative" samples (points where the
energy should be high) that are continuously refreshed via Langevin steps.
This provides hard negatives that track the evolving energy landscape.

NCE (Noise Contrastive Estimation) uses the buffer to train the energy
to correctly distinguish data from noise, fixing the mode proportion
blindness of pure score matching (Fisher divergence doesn't capture
relative mode weights — see Yang Song's blog on score-based models).

Mathematical basis:
    NCE estimates the data density ratio p_data(x) / p_noise(x) via:

    L_NCE = -E_{x~data}[log σ(f(x))] - E_{x~noise}[log σ(-f(x))]

    where f(x) = -E(x) - log p_noise(x) is the log-ratio.
    At optimum: f*(x) = log p_data(x) - log p_noise(x)
    Therefore: E*(x) = -log p_data(x) + const (up to partition function)

    This is the key advantage over DSM: NCE learns actual density ratios,
    so mode proportions are captured correctly.

Spec reference: tasks/todo.md §1.4
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor


class NegativeBuffer:
    """
    Persistent replay buffer for EBM training negatives.

    Maintains a pool of vectors that are periodically refreshed via
    Langevin dynamics on the current energy model. This gives negatives
    that are "hard" — they lie in regions where the energy is low but
    shouldn't be (false low-energy regions).

    Args:
        buffer_size: Number of vectors to store.
        dim: Embedding dimension (1024 for SONAR).
        init_std: Std for initial Gaussian fill.
        refresh_fraction: Fraction of buffer to replace each update.
        langevin_steps: Steps of Langevin dynamics per refresh.
        langevin_lr: Langevin step size.
        langevin_noise: Langevin noise scale.
        target_norm: If set, project samples to this norm.
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        dim: int = 1024,
        init_std: float = 0.00641,
        refresh_fraction: float = 0.05,
        langevin_steps: int = 20,
        langevin_lr: float = 0.01,
        langevin_noise: float = 0.005,
        target_norm: float | None = 0.2051,
    ):
        self.buffer_size = buffer_size
        self.dim = dim
        self.refresh_fraction = refresh_fraction
        self.langevin_steps = langevin_steps
        self.langevin_lr = langevin_lr
        self.langevin_noise = langevin_noise
        self.target_norm = target_norm

        # Initialize buffer from prior
        self.buffer = torch.randn(buffer_size, dim) * init_std
        if target_norm is not None:
            self.buffer = F.normalize(self.buffer, dim=-1) * target_norm

        self._insert_idx = 0

    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        """
        Sample a batch of negatives from the buffer.

        Args:
            batch_size: Number of negatives to sample.
            device: Target device.

        Returns:
            [batch_size, dim] negative samples.
        """
        indices = torch.randint(0, self.buffer_size, (batch_size,))
        return self.buffer[indices].to(device)

    @torch.no_grad()
    def refresh(self, energy_fn: torch.nn.Module, device: torch.device) -> None:
        """
        Refresh a fraction of the buffer via Langevin dynamics.

        Takes the oldest entries, runs Langevin to find low-energy regions,
        and replaces them. This ensures negatives track the energy landscape.

        Args:
            energy_fn: Current energy model E(x) → scalar.
            device: Compute device.
        """
        num_refresh = max(1, int(self.buffer_size * self.refresh_fraction))

        # Take a slice of buffer to refresh
        start_idx = self._insert_idx
        indices = [(start_idx + i) % self.buffer_size for i in range(num_refresh)]
        x = self.buffer[indices].to(device)

        # Run Langevin dynamics to find low-energy regions.
        # We need enable_grad() because refresh() is decorated with no_grad()
        # to avoid tracking the outer update, but autograd.grad requires a graph.
        for _ in range(self.langevin_steps):
            x_grad = x.detach().requires_grad_(True)
            with torch.enable_grad():
                energy = energy_fn(x_grad)
                grad = torch.autograd.grad(energy.sum(), x_grad)[0]

            noise = torch.randn_like(x) * self.langevin_noise
            x = x - self.langevin_lr * grad + (2 * self.langevin_lr) ** 0.5 * noise

            if self.target_norm is not None:
                x = F.normalize(x, dim=-1) * self.target_norm

        # 5% chance of reinitializing from random (prevents staleness)
        reinit_mask = torch.rand(num_refresh) < 0.05
        if reinit_mask.any():
            n_reinit = reinit_mask.sum().item()
            fresh = torch.randn(n_reinit, self.dim, device=device)
            if self.target_norm is not None:
                fresh = F.normalize(fresh, dim=-1) * self.target_norm
            x[reinit_mask] = fresh

        # Write back
        self.buffer[indices] = x.cpu()
        self._insert_idx = (start_idx + num_refresh) % self.buffer_size

    def seed_from_data(self, data: Tensor, noise_scale: float = 0.3) -> None:
        """
        Seed buffer with noisy versions of data (better initialization than pure random).

        Args:
            data: [N, D] data samples.
            noise_scale: Relative noise level.
        """
        n = min(self.buffer_size, data.shape[0])
        indices = torch.randperm(data.shape[0])[:n]
        x = data[indices].clone()

        # Add noise
        norms = x.norm(dim=-1, keepdim=True)
        x = x + torch.randn_like(x) * noise_scale * norms

        if self.target_norm is not None:
            x = F.normalize(x, dim=-1) * self.target_norm

        self.buffer[:n] = x
        # Fill remaining with random if buffer > data
        if n < self.buffer_size:
            remaining = torch.randn(self.buffer_size - n, self.dim)
            if self.target_norm is not None:
                remaining = F.normalize(remaining, dim=-1) * self.target_norm
            self.buffer[n:] = remaining


def nce_loss(
    energy_fn: torch.nn.Module,
    x_data: Tensor,
    x_noise: Tensor,
    prior_std: float = 0.00641,
) -> Tensor:
    """
    Noise Contrastive Estimation loss.

    L = -E_data[log σ(f(x))] - E_noise[log σ(-f(x))]

    where f(x) = -E(x) - log p_noise(x)
    and p_noise(x) = N(0, σ²I) → log p_noise(x) = -||x||²/(2σ²) - d/2·log(2πσ²)

    At optimum, E*(x) ∝ -log p_data(x), correctly capturing mode proportions.

    Args:
        energy_fn: Energy model E(x) → [B] scalar.
        x_data: [B, D] real data samples.
        x_noise: [B, D] noise samples (from buffer or prior).
        prior_std: Std of noise distribution (for log p_noise computation).

    Returns:
        Scalar NCE loss.
    """
    # Compute energies
    e_data = energy_fn(x_data)   # [B] — should be low
    e_noise = energy_fn(x_noise)  # [B] — should be high

    # Log-density of noise distribution: log N(x; 0, σ²I)
    # = -||x||²/(2σ²) - D/2·log(2πσ²)
    D = x_data.shape[-1]
    log_norm_const = -0.5 * D * math.log(2 * math.pi * prior_std ** 2)

    log_p_noise_data = -x_data.pow(2).sum(dim=-1) / (2 * prior_std ** 2) + log_norm_const
    log_p_noise_noise = -x_noise.pow(2).sum(dim=-1) / (2 * prior_std ** 2) + log_norm_const

    # Log-ratio: f(x) = -E(x) - log p_noise(x)
    f_data = -e_data - log_p_noise_data
    f_noise = -e_noise - log_p_noise_noise

    # NCE loss: -E_data[log σ(f)] - E_noise[log σ(-f)]
    loss_data = -F.logsigmoid(f_data).mean()
    loss_noise = -F.logsigmoid(-f_noise).mean()

    return loss_data + loss_noise


def nce_loss_simple(
    energy_fn: torch.nn.Module,
    x_data: Tensor,
    x_noise: Tensor,
) -> Tensor:
    """
    Simplified NCE loss (without explicit log p_noise).

    L = -E_data[log σ(-E(x))] - E_noise[log σ(E(x))]

    This version omits the log p_noise term, treating all noise samples
    as equally likely. Simpler but less principled than full NCE.
    Works well when noise distribution is approximately uniform relative
    to the data distribution (e.g., when buffer is well-mixed).

    Args:
        energy_fn: Energy model E(x) → [B].
        x_data: [B, D] real data.
        x_noise: [B, D] noise samples.

    Returns:
        Scalar loss.
    """
    e_data = energy_fn(x_data)
    e_noise = energy_fn(x_noise)

    loss_data = -F.logsigmoid(-e_data).mean()
    loss_noise = -F.logsigmoid(e_noise).mean()

    return loss_data + loss_noise
