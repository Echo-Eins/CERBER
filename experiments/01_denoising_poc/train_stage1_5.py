"""
Stage1.5: Hybrid Actor-Critic with SOTA Stabilization

Key changes from Stage1:
1. P0: Removed actor_energy_loss contradiction - actor no longer pushed E(actor) < E(clean)
2. P0: Added MDSM to critic for gradient field validity
3. P1: Hybrid critic E_total = E_cond(q,v) + λ*E_prior(v)
4. P1: Alternating training (2 critic steps : 1 actor step)
5. P2: CQL regularization for OOD prevention
6. P2: BC regularization for embedding anchor
7. P2: Added gradient penalty for smooth landscape
8. P2: Shell barrier penalty for norm control
9. P3: Comprehensive metrics telemetry (energy stats, grad norms, etc.)
10. Full SOTA eval metrics + stability guards

Implements all items from research3.md Phase P0-P2 and todo.md Stage1.5 Pass 11.
"""

import argparse
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.base import Stage1_5Config
from cebcm.models.actor import LatentDenoiseActor
from cebcm.models.energy import SimpleEnergy
from cebcm.models.energy_unconditional import UnconditionalEnergy
from cebcm.models.normalization import OrthoLinear
from cebcm.training.losses import (
    multiscale_dsm_loss,
    margin_contrastive_loss,
    gradient_penalty,
)
from cebcm.training.kill_criteria import (
    summarize_conditional_eval,
    check_conditional_kill,
)
from cebcm.inference.langevin import run_langevin
from cebcm.data.dataset import SONARVectorDataset


def add_relative_noise(v: torch.Tensor, scale: float) -> torch.Tensor:
    """Add Gaussian noise relative to embedding norm."""
    norms = v.norm(dim=-1, keepdim=True)
    noise = torch.randn_like(v) * scale * norms
    return v + noise


def cosine_to_geodesic(cos_value: float) -> float:
    return math.acos(max(-1.0, min(1.0, float(cos_value))))


def sample_sigma(
    batch_size: int,
    device: torch.device,
    sigma_min: float,
    sigma_max: float,
    sigma_sampling: str,
    edm_p_mean: float,
    edm_p_std: float,
) -> torch.Tensor:
    if sigma_sampling == "loguniform":
        log_sigma = torch.rand(batch_size, 1, device=device) * (
            math.log(sigma_max) - math.log(sigma_min)
        ) + math.log(sigma_min)
        return log_sigma.exp()
    if sigma_sampling == "edm":
        return torch.exp(
            torch.randn(batch_size, 1, device=device) * edm_p_std + edm_p_mean
        ).clamp(min=sigma_min, max=sigma_max)
    raise ValueError(f"Unknown sigma_sampling: {sigma_sampling}")


def project_tangent(update: torch.Tensor, v_current: torch.Tensor) -> torch.Tensor:
    v_hat = F.normalize(v_current, dim=-1)
    return update - (update * v_hat).sum(dim=-1, keepdim=True) * v_hat


def project_sphere(v: torch.Tensor, target_norm: float | None) -> torch.Tensor:
    if target_norm is None:
        return v
    return F.normalize(v, dim=-1) * target_norm


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_amp_dtype(amp_dtype: str) -> torch.dtype:
    if amp_dtype == "bf16":
        return torch.bfloat16
    if amp_dtype == "fp16":
        return torch.float16
    return torch.float32


def apply_non_finite_backoff(optimizer, backoff_factor: float = 0.5) -> None:
    for group in optimizer.param_groups:
        group["lr"] = max(group["lr"] * backoff_factor, 1e-7)


# =============================================================================
# Stage1.5 Training: Hybrid Critic + Actor with Alternating Updates
# =============================================================================

def train_epoch_stage1_5(
    critic: SimpleEnergy,
    prior_critic: UnconditionalEnergy | None,
    actor: LatentDenoiseActor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    config: Stage1Config,
    device: torch.device,
    epoch: int,
    global_step: int,
) -> tuple[dict, int]:
    """
    Stage1.5 training loop with:
    - Hybrid critic: E_total = E_cond(q,v) + λ*E_prior(v)
    - MDSM + ranking loss for critic
    - Actor with final-state geometry loss
    - Alternating updates (2 critic : 1 actor)
    - CQL regularization
    - BC regularization
    """
    critic.train()
    actor.train()
    if prior_critic is not None:
        prior_critic.train()

    total_loss = 0.0
    total_critic_loss = 0.0
    total_actor_loss = 0.0
    total_mdsm_loss = 0.0
    total_rank_loss = 0.0
    total_cql_loss = 0.0
    total_bc_loss = 0.0

    e_clean_sum = 0.0
    e_actor_sum = 0.0
    e_noisy_sum = 0.0

    total_gp_loss = 0.0
    total_shell_loss = 0.0

    num_batches = 0
    skipped_batches = 0
    non_finite_streak = 0

    sigma_min = config.sigma_curriculum_start
    sigma_max = config.sigma_curriculum_end

    for batch_idx, batch in enumerate(dataloader):
        v_clean = batch["v"].to(device)  # [B, D]
        batch_size = v_clean.shape[0]
        weights = batch.get("weight", torch.ones(batch_size, device=device))

        # Sample sigma for this batch
        sigma = sample_sigma(
            batch_size,
            device,
            sigma_min,
            sigma_max,
            config.sigma_sampling,
            config.edm_p_mean,
            config.edm_p_std,
        )

        # Create noisy samples with RELATIVE noise (SONAR convention)
        v_noisy = add_relative_noise(v_clean, sigma.squeeze(-1))

        # ============ CRITIC TRAINING (2 steps per 1 actor step) ============
        critic_steps = config.get("critic_steps_per_actor", 2)

        for _ in range(critic_steps):
            with torch.autocast(device_type="cuda", dtype=resolve_amp_dtype(config.amp_dtype), enabled=config.amp_enabled):
                # MDSM loss for gradient field validity
                v_noisy_mdsm = v_noisy.detach().requires_grad_(True)
                mdsm_loss_val = multiscale_dsm_loss(
                    critic,
                    v_clean,
                    v_noisy=v_noisy_mdsm,
                    sigma=sigma.detach(),
                    sigma_min=config.sigma_min,
                    sigma_max=config.sigma_max,
                    tangent_projection=config.mdsm_tangent_projection,
                    directional=config.mdsm_directional,
                    magnitude_aux_weight=config.mdsm_magnitude_aux_weight,
                    sigma_weighting=config.sigma_weighting,
                    cosine_eps=config.mdsm_cosine_eps,
                    norm_floor=config.mdsm_norm_floor,
                    force_fp32=config.mdsm_force_fp32,
                )

                # Ranking loss for scalar ordering
                v_actor = actor(
                    v_query=v_clean,
                    v_current=v_noisy,
                    sigma=sigma.detach(),
                )
                v_actor = project_sphere(
                    v_noisy + actor.predict_step(
                        v_query=v_clean,
                        v_current=v_noisy,
                        sigma=sigma.detach(),
                        step_size=config.actor_step_size,
                        target_norm=config.langevin.target_norm,
                        tangent_projection=config.actor_tangent_projection,
                    )[0],
                    config.langevin.target_norm,
                )

                e_clean = critic(v_clean, v_clean, sigma=sigma.detach())
                e_actor_detached = critic(v_clean, v_actor.detach(), sigma=sigma.detach())
                e_noisy = critic(v_clean, v_noisy.detach(), sigma=sigma.detach())

                # Add prior critic if available (hybrid critic)
                if prior_critic is not None:
                    lambda_prior = config.get("lambda_prior", 0.1)
                    e_clean = e_clean + lambda_prior * prior_critic(v_clean, sigma=sigma.detach())
                    e_actor_detached = e_actor_detached + lambda_prior * prior_critic(v_actor.detach(), sigma=sigma.detach())
                    e_noisy = e_noisy + lambda_prior * prior_critic(v_noisy.detach(), sigma=sigma.detach())

                ranking_loss_val = (
                    F.relu(e_clean - e_actor_detached + config.critic_margin_clean_actor)
                    + F.relu(e_actor_detached - e_noisy + config.critic_margin_actor_noisy)
                    + F.relu(e_clean - e_noisy + config.critic_margin_clean_noisy)
                ).mean()

                # CQL regularization (prevent OOD embeddings)
                cql_loss_val = torch.tensor(0.0, device=device)
                if config.get("use_cql", False):
                    # Sample OOD negatives
                    ood_noise = torch.randn_like(v_clean) * config.get("cql_noise_scale", 0.5)
                    v_ood = project_sphere(v_clean + ood_noise, config.langevin.target_norm)

                    e_ood = critic(v_clean, v_ood, sigma=sigma.detach())
                    if prior_critic is not None:
                        e_ood = e_ood + lambda_prior * prior_critic(v_ood, sigma=sigma.detach())

                    # CQL term: penalize low energy for OOD
                    cql_loss_val = F.softplus(e_ood).mean()

                # Gradient penalty for smooth landscape (research3.md P1)
                gp_loss_val = torch.tensor(0.0, device=device)
                if config.get("use_gradient_penalty", False) and config.get("gradient_penalty_lambda", 0.0) > 0:
                    gp_loss_val = gradient_penalty(
                        lambda x: critic(v_clean, x, sigma=sigma.detach()).sum(),
                        v_actor,
                        lambda_gp=config.get("gradient_penalty_lambda", 0.05)
                    )

                # Shell barrier penalty for norm control (research3.md P2)
                shell_loss_val = torch.tensor(0.0, device=device)
                if config.get("use_shell_barrier", False):
                    target_norm = config.langevin.target_norm or 0.2051
                    margin = config.get("shell_barrier_margin", 0.1)
                    actor_norm = v_actor.norm(dim=-1)
                    lower = target_norm * (1 - margin)
                    upper = target_norm * (1 + margin)
                    shell_loss_val = (
                        F.relu(lower - actor_norm) ** 2 +
                        F.relu(actor_norm - upper) ** 2
                    ).mean()

                # Total critic loss
                lambda_mdsm = config.get("lambda_mdsm", 1.0)
                lambda_rank = config.get("lambda_rank", 0.25)
                lambda_cql = config.get("lambda_cql", 0.1)
                lambda_gp = config.get("gradient_penalty_lambda", 0.05) if config.get("use_gradient_penalty", False) else 0.0
                lambda_shell = config.get("lambda_shell", 0.1) if config.get("use_shell_barrier", False) else 0.0

                critic_loss_val = (
                    lambda_mdsm * mdsm_loss_val
                    + lambda_rank * ranking_loss_val
                    + lambda_cql * cql_loss_val
                    + lambda_gp * gp_loss_val
                    + lambda_shell * shell_loss_val
                )

            # Critic optimizer step
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(critic_loss_val).backward()
                scaler.unscale_(optimizer)
                for p in critic.parameters():
                    if p.grad is not None:
                        torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0, out=p.grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                critic_loss_val.backward()
                for p in critic.parameters():
                    if p.grad is not None:
                        torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0, out=p.grad)
                optimizer.step()

            total_mdsm_loss += mdsm_loss_val.item()
            total_rank_loss += ranking_loss_val.item()
            total_cql_loss += cql_loss_val.item()

        # ============ ACTOR TRAINING (1 step) ============
        with torch.autocast(device_type="cuda", dtype=resolve_amp_dtype(config.amp_dtype), enabled=config.amp_enabled):
            # Actor forward
            v_actor = actor(
                v_query=v_clean,
                v_current=v_noisy,
                sigma=sigma.detach(),
            )

            # Actor step with projection
            delta, v_actor_projected = actor.predict_step(
                v_query=v_clean,
                v_current=v_noisy,
                sigma=sigma.detach(),
                step_size=config.actor_step_size,
                target_norm=config.langevin.target_norm,
                tangent_projection=config.actor_tangent_projection,
            )

            # BC Regularization (anchor to clean)
            bc_loss_val = torch.tensor(0.0, device=device)
            if config.get("use_bc", False):
                lambda_bc = config.get("lambda_bc", 0.5)
                bc_loss_val = lambda_bc * F.mse_loss(v_actor_projected, v_clean)

            # Final-state geometry loss (cosine/geodesic on projected state)
            cosine_final = F.cosine_similarity(v_actor_projected, v_clean, dim=-1)
            geo_loss_val = (1.0 - cosine_final.clamp(min=-1.0, max=1.0)).mean()

            # Gradient alignment loss (optional)
            align_loss_val = torch.tensor(0.0, device=device)
            if config.get("use_grad_align", False):
                with torch.enable_grad():
                    v_actor.requires_grad_(True)
                    _, grad = critic(v_clean, v_actor, sigma=sigma.detach())
                    grad_direction = -grad.detach()
                    actual_direction = (v_actor_projected - v_noisy).detach()
                    align_loss_val = (1.0 - F.cosine_similarity(
                        actual_direction, grad_direction, dim=-1
                    )).mean()

            # Total actor loss
            lambda_geo = config.get("lambda_geo", 1.0)
            lambda_align = config.get("lambda_align", 0.1)
            lambda_bc_reg = config.get("lambda_bc_reg", 0.5)

            actor_loss_val = (
                lambda_geo * geo_loss_val
                + lambda_align * align_loss_val
                + lambda_bc_reg * bc_loss_val
            )

        # Actor optimizer step
        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(actor_loss_val).backward()
            scaler.unscale_(optimizer)
            for p in actor.parameters():
                if p.grad is not None:
                    torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0, out=p.grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            actor_loss_val.backward()
            for p in actor.parameters():
                if p.grad is not None:
                    torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0, out=p.grad)
            optimizer.step()

        # ============ METRICS TRACKING (research3.md P3) ============
        e_clean_sum += e_clean.mean().item()
        e_actor_sum += e_actor_detached.mean().item()
        e_noisy_sum += e_noisy.mean().item()

        total_loss += critic_loss_val.item() + actor_loss_val.item()
        total_critic_loss += critic_loss_val.item()
        total_actor_loss += actor_loss_val.item()
        total_gp_loss += gp_loss_val.item()
        total_shell_loss += shell_loss_val.item()

        # Energy statistics (research3.md §7)
        energy_stats = {
            "mean": e_actor_detached.mean().item(),
            "std": e_actor_detached.std().item(),
            "min": e_actor_detached.min().item(),
            "max": e_actor_detached.max().item(),
        }

        # Gradient norm statistics (research3.md P3)
        grad_norms = []
        for p in critic.parameters():
            if p.grad is not None:
                grad_norms.append(p.grad.norm().item())
        grad_norm_mean = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        grad_norm_max = max(grad_norms) if grad_norms else 0.0

        # Clean-minimum violation rate (research3.md §7)
        clean_min_violation = (e_actor_detached < e_clean).float().mean().item()

        num_batches += 1
        global_step += 1
        non_finite_streak = 0

        if config.log_every > 0 and (batch_idx + 1) % config.log_every == 0:
            e_scale = torch.exp(critic.log_energy_scale.detach().clamp(min=-8.0, max=8.0)).item()
            actor_step_scale = torch.exp(actor.log_step_scale.detach().clamp(min=-6.0, max=3.0)).item()
            print(
                f"  [{batch_idx + 1}/{len(dataloader)}] "
                f"loss={total_loss / num_batches:.4f} "
                f"critic={total_critic_loss / num_batches:.4f} "
                f"actor={total_actor_loss / num_batches:.4f} "
                f"MDSM={total_mdsm_loss / num_batches:.4f} "
                f"Rank={total_rank_loss / num_batches:.4f} "
                f"CQL={total_cql_loss / num_batches:.4f} "
                f"GP={total_gp_loss / num_batches:.4f} "
                f"Shell={total_shell_loss / num_batches:.4f} "
                f"E(clean/actor/noisy)=({e_clean_sum / num_batches:.3f}/"
                f"{e_actor_sum / num_batches:.3f}/{e_noisy_sum / num_batches:.3f}) "
                f"clean_min_viol={clean_min_violation:.3f} "
                f"grad_norm(mean/max)={grad_norm_mean:.4f}/{grad_norm_max:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f} "
                f"E_scale={e_scale:.2f} actor_scale={actor_step_scale:.3f}"
            )

    return {
        # Core losses
        "loss": total_loss / max(num_batches, 1),
        "critic_loss": total_critic_loss / max(num_batches, 1),
        "actor_loss": total_actor_loss / max(num_batches, 1),
        "mdsm_loss": total_mdsm_loss / max(num_batches, 1),
        "ranking_loss": total_rank_loss / max(num_batches, 1),
        "cql_loss": total_cql_loss / max(num_batches, 1),
        "gp_loss": total_gp_loss / max(num_batches, 1),
        "shell_loss": total_shell_loss / max(num_batches, 1),
        "bc_loss": total_bc_loss / max(num_batches, 1),
        # Energy statistics (research3.md §7)
        "e_clean_mean": e_clean_sum / max(num_batches, 1),
        "e_actor_mean": e_actor_sum / max(num_batches, 1),
        "e_noisy_mean": e_noisy_sum / max(num_batches, 1),
        "critic_gap_clean_actor": (e_clean_sum - e_actor_sum) / max(num_batches, 1),
        "critic_gap_clean_noisy": (e_clean_sum - e_noisy_sum) / max(num_batches, 1),
        # Stability metrics
        "clean_min_violation_rate": clean_min_violation,
        "grad_norm_mean": grad_norm_mean,
        "grad_norm_max": grad_norm_max,
        "skipped_batches": float(skipped_batches),
    }, global_step


# =============================================================================
# Main training orchestration
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage1.5 Hybrid Actor-Critic Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--output", type=str, default="output/stage1_5", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config_dict = json.load(f)
    config = Stage1_5Config(**config_dict)

    # Set seed
    set_seed(config.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory structure
    output_dir = Path(config_dict.get("output_dir", args.output))
    checkpoint_dir = Path(config_dict.get("checkpoint_dir", output_dir / "checkpoints"))
    logs_dir = Path(config_dict.get("logs_dir", output_dir / "logs"))

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Stage1.5 Training ===")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Checkpoints: {checkpoint_dir.resolve()}")
    print(f"Logs: {logs_dir.resolve()}\n")

    # Initialize models
    critic = SimpleEnergy(
        dim=config.energy_dim,
        hidden_dims=config.energy_hidden_dims,
        norm_mode=config.norm_mode,
        activation=config.activation,
    ).to(device)

    prior_critic = None
    if config.get("use_prior_critic", False):
        prior_critic = UnconditionalEnergy(
            dim=config.energy_dim,
            hidden_dims=config.energy_hidden_dims,
            norm_mode=config.norm_mode,
            activation=config.activation,
        ).to(device)

    actor = LatentDenoiseActor(
        dim=config.energy_dim,
        hidden_dims=config.actor_hidden_dims,
        norm_mode=config.norm_mode,
        activation=config.activation,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW([
        {"params": critic.parameters(), "lr": config.critic_lr},
        {"params": actor.parameters(), "lr": config.actor_lr},
    ], weight_decay=config.weight_decay)

    if prior_critic is not None:
        optimizer.add_param_group({"params": prior_critic.parameters(), "lr": config.get("prior_critic_lr", config.critic_lr)})

    # Scaler for AMP
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp_enabled)

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        critic.load_state_dict(checkpoint["critic"])
        actor.load_state_dict(checkpoint["actor"])
        if prior_critic is not None and "prior_critic" in checkpoint:
            prior_critic.load_state_dict(checkpoint["prior_critic"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("global_step", 0)
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    # Data
    train_dataset = SONARVectorDataset(config.train_data_path)
    val_dataset = SONARVectorDataset(config.val_data_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Training loop
    best_composite_score = float("-inf")

    for epoch in range(start_epoch, config.num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{config.num_epochs} ===")
        start_time = time.time()

        train_metrics, global_step = train_epoch_stage1_5(
            critic, prior_critic, actor, train_loader,
            optimizer, scaler, config, device, epoch, global_step,
        )

        # Evaluation
        eval_metrics = evaluate_denoising(
            critic, actor, val_dataset, config, device,
            num_samples=config.eval_num_samples,
        )

        # Composite score for checkpoint selection
        composite_score = (
            eval_metrics.get("cosine_improvement", 0) * 0.4
            + eval_metrics.get("geodesic_improvement", 0) * 0.3
            + (1.0 if eval_metrics.get("clean_min_violation_rate", 1.0) < 0.05 else 0.0) * 0.2
            + eval_metrics.get("energy_success_rate", 0) * 0.1
        )

        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch + 1} completed in {epoch_time:.2f}s")
        print(f"  Train loss: {train_metrics['loss']:.4f}")
        print(f"  Cosine improvement: {eval_metrics.get('cosine_improvement', 0):.4f}")
        print(f"  Composite score: {composite_score:.4f}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "critic": critic.state_dict(),
            "actor": actor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": asdict(config),
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "composite_score": composite_score,
        }

        # Save epoch checkpoint
        epoch_ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, epoch_ckpt_path)
        print(f"  Checkpoint saved: {epoch_ckpt_path.name}")

        # Save best checkpoint
        if composite_score > best_composite_score:
            best_composite_score = composite_score
            best_ckpt_path = checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_ckpt_path)
            print(f"  [BEST] New best checkpoint saved (score: {composite_score:.4f}): {best_ckpt_path.name}")

        # Save training metrics to logs
        metrics_log = {
            "epoch": epoch + 1,
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "composite_score": composite_score,
            "epoch_time_sec": epoch_time,
        }
        with open(logs_dir / "training_metrics.jsonl", "a") as f:
            f.write(json.dumps(metrics_log) + "\n")

        # Kill criteria check
        kill_verdict = check_conditional_kill(eval_metrics, config)
        if kill_verdict["should_stop"]:
            print(f"\n!!! Kill criteria triggered: {kill_verdict['reason']}")
            # Save final checkpoint before stopping
            final_ckpt_path = checkpoint_dir / "checkpoint_killed.pt"
            torch.save(checkpoint, final_ckpt_path)
            print(f"  Final checkpoint saved: {final_ckpt_path.name}")
            break

    print(f"\n=== Training completed ===")
    print(f"Best composite score: {best_composite_score:.4f}")
    print(f"Checkpoints directory: {checkpoint_dir}")
    print(f"Logs directory: {logs_dir}")

    # Save final summary
    summary = {
        "best_composite_score": best_composite_score,
        "total_epochs": epoch + 1,
        "final_global_step": global_step,
        "config": asdict(config),
    }
    with open(logs_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
