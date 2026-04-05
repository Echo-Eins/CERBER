"""
System 1/2 switching for inference and training.

System 1 (Fast Shot):
  - Pairwise energy scoring only
  - 10 Langevin steps (PID) for fast convergence
  - For simple queries, fast response

System 2 (Deep Thinking):
  - Chain Head energy scoring
  - <=50 Langevin steps (PID) with chain-guided gradients
  - Chain validation every chain_eval_every steps
  - Backtracking: revert to best point if no improvement for N steps
  - Chain energy contributes to update direction (not only monitoring)

Spec reference: §8.2 (Deep Thinking), IMPLEMENTATION_PLAN.md §6.1 Phase B
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from cebcm.inference.langevin import LangevinResult, run_langevin


@dataclass
class System1Config:
    """Fast Shot configuration."""
    max_steps_choices: list[int] = field(default_factory=lambda: [10])
    cruise_ratio_choices: list[float] = field(default_factory=lambda: [0.5, 0.7])


@dataclass
class System2Config:
    """Deep Thinking configuration."""
    max_steps_choices: list[int] = field(default_factory=lambda: [50])
    cruise_ratio_choices: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.3])
    chain_eval_every: int = 5       # Evaluate chain every N Langevin steps
    backtrack_patience: int = 30    # Steps without improvement before reverting
    max_chain_len: int = 20         # Max vectors in reasoning chain
    chain_guidance_weight: float = 0.35  # Weight of chain gradient in total update
    min_chain_guidance_len: int = 5      # Avoid OOD chain lengths for chain-head guidance


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
    v_trajectory: list[Tensor] = field(default_factory=list)  # Per-step vectors (if tracked)
    grad_norms: list[float] = field(default_factory=list)  # Gradient norms per step
    attention_snapshots: list[tuple[int, list[Tensor]]] = field(default_factory=list)  # (step, attn_maps)


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
    track_vectors: bool = False,
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
        track_vectors: Store per-step vectors for trajectory visualization

    Returns:
        ThinkingResult
    """
    max_steps = random.choice(cfg.max_steps_choices)

    kwargs = dict(
        method="pid",
        energy_fn=energy_fn,
        v_query=v_query,
        v_init=v_init,
        max_steps=max_steps,
        v_target=v_target,
        track_vectors=track_vectors,
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
        v_trajectory=result.v_trajectory if track_vectors else [],
    )


def run_system2(
    pairwise_fn: torch.nn.Module,
    chain_head: torch.nn.Module,
    v_query: Tensor,
    v_init: Tensor,
    cfg: System2Config,
    langevin_kwargs: dict | None = None,
    v_target: Tensor | None = None,
    track_vectors: bool = False,
    attention_callback: Callable[[int, Tensor], list[Tensor]] | None = None,
) -> ThinkingResult:
    """
    System 2 (Deep Thinking): chain-guided PID Langevin with backtracking.

    Runs PID Langevin updates where the gradient is:
      grad_total = grad_pairwise + w_chain * grad_chain
    and grad_chain is computed from Chain Head energy over recent trajectory.
    Chain quality is additionally validated every chain_eval_every steps,
    with backtracking to the best chain state on stagnation.

    Args:
        pairwise_fn: Pairwise energy function for Langevin gradients
        chain_head: EBTChainHead for chain quality evaluation
        v_query: [B, D] query vectors
        v_init: [B, D] initial point (from IPP)
        cfg: System 2 configuration
        langevin_kwargs: Additional Langevin parameters (lr, noise, target_norm, etc.)
        v_target: [B, D] optional ground truth for eval metrics
        track_vectors: Store per-step vectors for trajectory visualization
        attention_callback: If provided, called at each chain eval with (step, chain_tensor).
            Should return list of attention maps [H, L, L] per layer.

    Returns:
        ThinkingResult with chain energies, backtrack info, and optional diagnostics
    """
    max_steps = min(random.choice(cfg.max_steps_choices), 50)

    lk = langevin_kwargs or {}
    lr = float(lk.get("lr", 0.01))
    noise_scale = float(lk.get("noise_scale", 0.005))
    target_norm = lk.get("target_norm", 0.2051)
    if target_norm is not None:
        target_norm = float(target_norm)
    chain_eval_every = max(1, int(cfg.chain_eval_every))
    max_chain_len = min(int(cfg.max_chain_len), 20)
    backtrack_patience = max(chain_eval_every, int(cfg.backtrack_patience))
    chain_guidance_weight = float(getattr(cfg, "chain_guidance_weight", 0.35))
    min_chain_guidance_len = max(3, int(getattr(cfg, "min_chain_guidance_len", 5)))

    # PID + early-stop parameters (align with run_langevin defaults)
    kp = float(lk.get("kp", 1.0))
    ki = float(lk.get("ki", 0.3))
    kd = float(lk.get("kd", 0.1))
    integral_decay = float(lk.get("integral_decay", 0.95))
    plateau_patience = int(lk.get("plateau_patience", 10))
    plateau_delta = float(lk.get("plateau_delta", 1e-4))
    energy_threshold = lk.get("energy_threshold", None)
    cosine_early_stop = bool(lk.get("cosine_early_stop", False)) and (v_target is not None)
    cosine_patience = int(lk.get("cosine_patience", 20))
    cosine_delta = float(lk.get("cosine_delta", 0.001))
    tamed = bool(lk.get("tamed", False))

    v_current = v_init.clone().detach()
    v_best_chain = v_current.clone()
    v_best_energy = v_current.clone()
    v_best_cos = v_current.clone()
    best_chain_energy = float("inf")
    best_pairwise_energy = float("inf")
    best_cosine = float("-inf")
    plateau_counter = 0
    cosine_plateau_counter = 0

    # PID state
    integral = torch.zeros_like(v_current)
    prev_grad: Tensor | None = None

    # chain_history stores accepted states and is used both for guidance and eval
    chain_history: list[Tensor] = [v_init.detach().clone()]
    energy_trajectory: list[float] = []
    chain_energies: list[float] = []
    cos_trajectory: list[float] = []
    grad_norms: list[float] = []
    v_trajectory: list[Tensor] = []
    attention_snapshots: list[tuple[int, list[Tensor]]] = []
    backtrack_count = 0
    steps_since_improve = 0

    if track_vectors:
        v_trajectory.append(v_current.detach().cpu().clone())

    for step in range(max_steps):
        if hasattr(pairwise_fn, "set_step"):
            pairwise_fn.set_step(step)

        v_current_req = v_current.detach().requires_grad_(True)
        e_pair = pairwise_fn(v_query, v_current_req)
        pairwise_mean = e_pair.mean().item()

        # Chain-guided gradient: include chain_head energy on recent trajectory + current candidate.
        use_chain_guidance = len(chain_history) >= (min_chain_guidance_len - 1)
        total_objective = e_pair.sum()
        if use_chain_guidance and chain_guidance_weight > 0:
            history = chain_history[-max(1, max_chain_len - 1):]
            chain_tensor = torch.stack([*history, v_current_req], dim=1)  # [B, L, D]
            e_chain_guidance = chain_head(chain_tensor)
            total_objective = total_objective + chain_guidance_weight * e_chain_guidance.sum()

        total_grad = torch.autograd.grad(total_objective, v_current_req)[0]

        if tamed:
            g_norm = total_grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            total_grad = total_grad / (1.0 + lr * g_norm)

        grad_norms.append(total_grad.norm().item())
        energy_trajectory.append(pairwise_mean)

        prev_best_pairwise = best_pairwise_energy
        # Track best pairwise-energy point
        if pairwise_mean < best_pairwise_energy:
            best_pairwise_energy = pairwise_mean
            v_best_energy = v_current.detach().clone()

        # Energy-based early stop (pairwise objective)
        if pairwise_mean < (prev_best_pairwise - plateau_delta):
            plateau_counter = 0
        else:
            plateau_counter += 1

        if energy_threshold is not None and pairwise_mean < float(energy_threshold):
            break
        if plateau_counter >= plateau_patience:
            break

        # Optional cosine tracking / stopping
        if v_target is not None:
            with torch.no_grad():
                cos = F.cosine_similarity(v_current, v_target, dim=-1).mean().item()
                cos_trajectory.append(cos)
                if cos > best_cosine:
                    best_cosine = cos
                    v_best_cos = v_current.detach().clone()
                    cosine_plateau_counter = 0
                elif cos > best_cosine - cosine_delta:
                    cosine_plateau_counter += 1
                else:
                    cosine_plateau_counter += 1

            if cosine_early_stop and cosine_plateau_counter >= cosine_patience:
                break

        # PID update components
        p_term = total_grad
        integral = integral_decay * integral + total_grad
        i_term = integral
        if prev_grad is None:
            d_term = torch.zeros_like(total_grad)
        else:
            d_term = total_grad - prev_grad
        prev_grad = total_grad.detach().clone()
        update = kp * p_term + ki * i_term + kd * d_term

        # Tangent projection to keep update on sphere
        if target_norm is not None:
            v_hat = F.normalize(v_current, dim=-1)
            radial = (update * v_hat).sum(dim=-1, keepdim=True) * v_hat
            update = update - radial

        with torch.no_grad():
            step_noise_scale = float(noise_scale)
            get_step_noise_scale = getattr(pairwise_fn, "get_step_noise_scale", None)
            if callable(get_step_noise_scale):
                try:
                    step_noise_scale = float(
                        get_step_noise_scale(
                            v_query=v_query,
                            v_candidate=v_current,
                            base_noise_scale=noise_scale,
                        )
                    )
                except Exception:
                    step_noise_scale = float(noise_scale)
            step_noise_scale = max(0.0, step_noise_scale)

            noise = torch.randn_like(v_current) * (2.0 * lr * step_noise_scale) ** 0.5
            v_current = v_current - lr * update + noise
            if target_norm is not None:
                v_current = F.normalize(v_current, dim=-1) * target_norm

        chain_history.append(v_current.detach().clone())
        if len(chain_history) > max_chain_len:
            chain_history = chain_history[-max_chain_len:]

        if track_vectors:
            v_trajectory.append(v_current.detach().cpu().clone())

        if (step + 1) % chain_eval_every == 0 and len(chain_history) >= 3:
            chain_tensor_eval = torch.stack(chain_history, dim=1)  # [B, L, D]
            with torch.no_grad():
                e_chain = chain_head(chain_tensor_eval).mean().item()
            chain_energies.append(e_chain)

            if attention_callback is not None:
                try:
                    attn_maps = attention_callback(step + 1, chain_tensor_eval)
                    attention_snapshots.append((step + 1, attn_maps))
                except Exception:
                    pass

            if e_chain < best_chain_energy:
                best_chain_energy = e_chain
                v_best_chain = v_current.detach().clone()
                steps_since_improve = 0
            else:
                steps_since_improve += chain_eval_every

            # Backtracking to best chain state if chain quality stalls
            if steps_since_improve >= backtrack_patience:
                v_current = v_best_chain.detach().clone()
                backtrack_count += 1
                steps_since_improve = 0
                # Keep trajectory context short and re-anchor at best state
                chain_history = chain_history[-max(3, max_chain_len // 2):]
                chain_history.append(v_best_chain.detach().clone())

    # Final decision: prioritize chain-best for System 2, else cosine-best, else pairwise-best.
    if chain_energies:
        v_final = v_best_chain
    elif v_target is not None and best_cosine > float("-inf"):
        v_final = v_best_cos
    else:
        v_final = v_best_energy

    return ThinkingResult(
        v_final=v_final,
        mode="system2",
        num_steps=len(energy_trajectory),
        energy_trajectory=energy_trajectory,
        chain_energies=chain_energies,
        backtrack_count=backtrack_count,
        cos_trajectory=cos_trajectory,
        v_trajectory=v_trajectory,
        grad_norms=grad_norms,
        attention_snapshots=attention_snapshots,
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
    track_vectors: bool = False,
    attention_callback: Callable[[int, Tensor], list[Tensor]] | None = None,
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
        track_vectors: Store per-step vectors for trajectory visualization
        attention_callback: Called at each System 2 chain eval for attention capture

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
            track_vectors=track_vectors,
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
            track_vectors=track_vectors,
            attention_callback=attention_callback,
        )
    else:
        raise ValueError(f"Unknown thinking mode: {mode}")
