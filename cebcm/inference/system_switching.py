"""
System 1/2 switching for inference and training.

System 1 (Fast Shot):
  - Pairwise energy scoring only
  - 10-50 Langevin steps with aggressive momentum (cruise_ratio 0.5-0.7)
  - For simple queries, fast response

System 2 (Deep Thinking):
  - Chain Head energy scoring
  - 100-200+ Langevin steps with low momentum (cruise_ratio 0.0-0.3)
  - Chain validation every chain_eval_every steps
  - Backtracking: revert to best point if no improvement for N steps

Spec reference: §8.2 (Deep Thinking), IMPLEMENTATION_PLAN.md §6.1 Phase B
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from cebcm.inference.langevin import LangevinResult, run_langevin


@dataclass
class System1Config:
    """Fast Shot configuration."""
    max_steps_choices: list[int] = field(default_factory=lambda: [10, 20, 50])
    cruise_ratio_choices: list[float] = field(default_factory=lambda: [0.5, 0.7])


@dataclass
class System2Config:
    """Deep Thinking configuration."""
    max_steps_choices: list[int] = field(default_factory=lambda: [100, 200])
    cruise_ratio_choices: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.3])
    chain_eval_every: int = 5       # Evaluate chain every N Langevin steps
    backtrack_patience: int = 30    # Steps without improvement before reverting
    max_chain_len: int = 20         # Max vectors in reasoning chain


@dataclass
class ThinkingResult:
    """Result of a System 1 or System 2 inference run."""
    v_final: Tensor               # [B, D] refined output vectors
    mode: str                     # "system1" or "system2"
    num_steps: int                # Actual Langevin steps taken
    energy_trajectory: list[float]  # Energy at each step
    chain_energies: list[float] = field(default_factory=list)  # Chain Head energies (System 2)
    backtrack_count: int = 0      # Number of backtracks performed
    cos_trajectory: list[float] = field(default_factory=list)  # Optional cos to target


def select_thinking_mode(
    system1_weight: float = 0.3,
    system2_weight: float = 0.7,
) -> str:
    """
    Randomly select thinking mode for training.

    Bias toward System 2 (more complex, needs more training exposure).
    """
    return random.choices(
        ["system1", "system2"],
        weights=[system1_weight, system2_weight],
    )[0]


def run_system1(
    energy_fn: torch.nn.Module,
    v_query: Tensor,
    v_init: Tensor,
    cfg: System1Config,
    langevin_kwargs: dict | None = None,
    v_target: Tensor | None = None,
) -> ThinkingResult:
    """
    System 1 (Fast Shot): pairwise energy + short Langevin.

    Args:
        energy_fn: Pairwise energy function (SimpleEnergy or decomposed)
        v_query: [B, D] query vectors
        v_init: [B, D] initial point (from IPP)
        cfg: System 1 configuration
        langevin_kwargs: Additional Langevin parameters
        v_target: [B, D] optional ground truth for eval metrics

    Returns:
        ThinkingResult
    """
    max_steps = random.choice(cfg.max_steps_choices)
    # cruise_ratio not directly used in Langevin (it's for inertial navigation)
    # but we vary max_steps as proxy for fast vs slower

    kwargs = dict(
        method="pid",
        energy_fn=energy_fn,
        v_query=v_query,
        v_init=v_init,
        max_steps=max_steps,
        v_target=v_target,
    )
    if langevin_kwargs:
        kwargs.update(langevin_kwargs)

    result = run_langevin(**kwargs)

    return ThinkingResult(
        v_final=result.v_final,
        mode="system1",
        num_steps=result.num_steps,
        energy_trajectory=result.trajectory,
        cos_trajectory=result.cos_trajectory,
    )


def run_system2(
    pairwise_fn: torch.nn.Module,
    chain_head: torch.nn.Module,
    v_query: Tensor,
    v_init: Tensor,
    cfg: System2Config,
    langevin_kwargs: dict | None = None,
    v_target: Tensor | None = None,
) -> ThinkingResult:
    """
    System 2 (Deep Thinking): chain-validated Langevin with backtracking.

    Runs extended Langevin dynamics using pairwise energy for gradient steps,
    but validates the accumulated reasoning chain every chain_eval_every steps
    using the Chain Head. If chain energy degrades, backtracks to best point.

    Args:
        pairwise_fn: Pairwise energy function for Langevin gradients
        chain_head: EBTChainHead for chain quality evaluation
        v_query: [B, D] query vectors
        v_init: [B, D] initial point (from IPP)
        cfg: System 2 configuration
        langevin_kwargs: Additional Langevin parameters (lr, noise, target_norm, etc.)
        v_target: [B, D] optional ground truth for eval metrics

    Returns:
        ThinkingResult with chain energies and backtrack info
    """
    max_steps = random.choice(cfg.max_steps_choices)
    device = v_init.device
    B, D = v_init.shape

    # Default Langevin params
    lk = langevin_kwargs or {}
    lr = lk.get("lr", 0.01)
    noise_scale = lk.get("noise_scale", 0.005)
    target_norm = lk.get("target_norm", 0.2051)

    # State
    v_current = v_init.clone().requires_grad_(False)
    v_best = v_current.clone()
    best_chain_energy = float("inf")

    # Reasoning chain: accumulate intermediate points
    chain_buffer: list[Tensor] = [v_init.detach().clone()]
    energy_trajectory: list[float] = []
    chain_energies: list[float] = []
    cos_trajectory: list[float] = []
    backtrack_count = 0
    steps_since_improve = 0
    just_backtracked = False  # Prevent duplicate chain entry after backtrack

    for step in range(max_steps):
        # Langevin step using pairwise energy
        v_current = v_current.detach().requires_grad_(True)
        E_pair = pairwise_fn(v_query, v_current)
        grad = torch.autograd.grad(E_pair.sum(), v_current)[0]

        with torch.no_grad():
            noise = noise_scale * torch.randn_like(v_current)
            v_current = v_current - lr * grad + noise

            # Project to SONAR sphere (OOD protection)
            if target_norm is not None:
                v_current = F.normalize(v_current, dim=-1) * target_norm

        energy_trajectory.append(E_pair.mean().item())

        # Track cosine to target if available
        if v_target is not None:
            with torch.no_grad():
                cos = F.cosine_similarity(v_current, v_target, dim=-1).mean().item()
                cos_trajectory.append(cos)

        # Chain validation every N steps
        if (step + 1) % cfg.chain_eval_every == 0:
            # Add current point to chain (skip if just backtracked — already added)
            if not just_backtracked:
                chain_buffer.append(v_current.detach().clone())
            just_backtracked = False

            # Keep chain within max length (sliding window)
            if len(chain_buffer) > cfg.max_chain_len:
                chain_buffer = chain_buffer[-cfg.max_chain_len:]

            # Evaluate chain quality
            if len(chain_buffer) >= 3:
                chain_tensor = torch.stack(chain_buffer, dim=1)  # [B, chain_len, D]
                with torch.no_grad():
                    E_chain = chain_head(chain_tensor).mean().item()
                chain_energies.append(E_chain)

                if E_chain < best_chain_energy:
                    best_chain_energy = E_chain
                    v_best = v_current.detach().clone()
                    steps_since_improve = 0
                else:
                    steps_since_improve += cfg.chain_eval_every

                # Backtracking: revert if no improvement for too long
                if steps_since_improve >= cfg.backtrack_patience:
                    v_current = v_best.clone()
                    backtrack_count += 1
                    steps_since_improve = 0
                    just_backtracked = True
                    # Trim chain to remove degraded portion, append revert point
                    chain_buffer = chain_buffer[:max(3, len(chain_buffer) // 2)]
                    chain_buffer.append(v_best.detach().clone())

    return ThinkingResult(
        v_final=v_best,
        mode="system2",
        num_steps=max_steps,
        energy_trajectory=energy_trajectory,
        chain_energies=chain_energies,
        backtrack_count=backtrack_count,
        cos_trajectory=cos_trajectory,
    )


def run_thinking(
    mode: str,
    pairwise_fn: torch.nn.Module,
    chain_head: torch.nn.Module | None,
    v_query: Tensor,
    v_init: Tensor,
    sys1_cfg: System1Config | None = None,
    sys2_cfg: System2Config | None = None,
    langevin_kwargs: dict | None = None,
    v_target: Tensor | None = None,
) -> ThinkingResult:
    """
    Unified entry point for System 1/2 thinking.

    Args:
        mode: "system1" or "system2"
        pairwise_fn: Pairwise energy function
        chain_head: Chain Head (required for system2)
        v_query: [B, D] query vectors
        v_init: [B, D] initial points
        sys1_cfg: System 1 config
        sys2_cfg: System 2 config
        langevin_kwargs: Shared Langevin parameters
        v_target: Optional ground truth

    Returns:
        ThinkingResult
    """
    if mode == "system1":
        return run_system1(
            energy_fn=pairwise_fn,
            v_query=v_query,
            v_init=v_init,
            cfg=sys1_cfg or System1Config(),
            langevin_kwargs=langevin_kwargs,
            v_target=v_target,
        )
    elif mode == "system2":
        assert chain_head is not None, "Chain Head required for System 2"
        return run_system2(
            pairwise_fn=pairwise_fn,
            chain_head=chain_head,
            v_query=v_query,
            v_init=v_init,
            cfg=sys2_cfg or System2Config(),
            langevin_kwargs=langevin_kwargs,
            v_target=v_target,
        )
    else:
        raise ValueError(f"Unknown thinking mode: {mode}")
