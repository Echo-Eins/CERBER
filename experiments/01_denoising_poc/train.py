"""
Stage 1: Denoising PoC - Train SimpleEnergy on SONAR vectors.

Improvements in this revision:
- deterministic seeds and fixed eval subsets,
- full Stage1 config saved to checkpoints,
- optional AMP and torch.compile,
- dataloader throughput tuning,
- Bjorck iteration scheduling (15 -> 8 -> 4 by default),
- sigma curriculum + configurable DSM geometry options,
- no unnecessary gradient-penalty computation when lambda=0.
"""

import argparse
import json
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.base import Stage1Config
from cebcm.models.energy import SimpleEnergy
from cebcm.models.normalization import OrthoLinear
from cebcm.training.losses import (
    multiscale_dsm_loss,
    margin_contrastive_loss,
    gradient_penalty,
)
from cebcm.inference.langevin import run_langevin
from cebcm.data.dataset import SONARVectorDataset


def add_relative_noise(v: torch.Tensor, scale: float) -> torch.Tensor:
    """Add Gaussian noise relative to embedding norm."""
    norms = v.norm(dim=-1, keepdim=True)
    noise = torch.randn_like(v) * scale * norms
    return v + noise


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
    raise ValueError(f"Unknown amp dtype: {amp_dtype}")


def get_current_sigma_range(config: Stage1Config, epoch: int) -> tuple[float, float]:
    """Large->small sigma curriculum: start with larger minimum sigma."""
    if not config.mdsm_sigma_curriculum:
        return config.mdsm_sigma_min, config.mdsm_sigma_max

    total = max(1, config.num_epochs - 1)
    progress = max(0.0, min(1.0, epoch / total))
    sigma_min = (
        config.mdsm_sigma_curriculum_start_min * (1.0 - progress)
        + config.mdsm_sigma_min * progress
    )
    sigma_min = float(min(max(sigma_min, config.mdsm_sigma_min), config.mdsm_sigma_max))
    return sigma_min, config.mdsm_sigma_max


def get_ortho_iters(config: Stage1Config, epoch: int) -> int:
    """Piecewise schedule for Bjorck iterations."""
    if not config.ortho_schedule_enabled:
        return config.ortho_n_iters

    schedule = config.ortho_schedule_iters
    boundaries = config.ortho_schedule_boundaries
    if len(schedule) != 3 or len(boundaries) != 2:
        return config.ortho_n_iters

    total = max(1, config.num_epochs)
    progress = epoch / total
    if progress < boundaries[0]:
        return int(schedule[0])
    if progress < boundaries[1]:
        return int(schedule[1])
    return int(schedule[2])


def set_ortho_n_iters(model: torch.nn.Module, n_iters: int) -> int:
    """Update n_iters for all OrthoLinear modules."""
    count = 0
    for module in model.modules():
        if isinstance(module, OrthoLinear):
            module.n_iters = int(n_iters)
            count += 1
    return count


def gradients_are_finite(model: torch.nn.Module) -> bool:
    """Return True if all available gradients are finite."""
    for p in model.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return False
    return True


def parameters_are_finite(model: torch.nn.Module) -> bool:
    """Return True if all model parameters are finite."""
    for p in model.parameters():
        if not torch.isfinite(p).all():
            return False
    return True


def apply_non_finite_backoff(optimizer: torch.optim.Optimizer, factor: float) -> None:
    """Reduce LR after non-finite events to stabilize optimization."""
    if not (0.0 < factor < 1.0):
        return
    for pg in optimizer.param_groups:
        pg["lr"] = max(pg["lr"] * factor, 1e-8)


def train_epoch_mdsm(
    model: SimpleEnergy,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: Stage1Config,
    device: torch.device,
    epoch: int,
    global_step: int,
    scaler: torch.amp.GradScaler,
    autocast_ctx,
) -> tuple[dict, int]:
    model.train()
    total_loss = 0.0
    total_dsm = 0.0
    total_gp = 0.0
    num_batches = 0
    skipped_batches = 0

    sigma_min, sigma_max = get_current_sigma_range(config, epoch)
    non_finite_streak = 0

    for batch_idx, v_clean in enumerate(dataloader):
        if (batch_idx % 16 == 0) and not parameters_are_finite(model):
            raise RuntimeError(
                f"Model parameters became non-finite before batch {batch_idx + 1}"
            )

        v_clean = v_clean.to(device, non_blocking=True)

        if global_step < config.warmup_steps:
            warmup_factor = (global_step + 1) / max(1, config.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = pg["initial_lr"] * warmup_factor

        dsm_ctx = nullcontext if config.mdsm_force_fp32 else autocast_ctx
        with dsm_ctx():
            loss_dsm = multiscale_dsm_loss(
                energy_fn=model,
                v_clean=v_clean,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                relative_noise=True,
                sigma_sampling=config.mdsm_sigma_sampling,
                sigma_weighting=config.mdsm_sigma_weighting,
                directional=config.mdsm_directional,
                magnitude_aux_weight=config.mdsm_magnitude_aux_weight,
                tangent_projection=(
                    config.mdsm_tangent_projection and config.langevin.target_norm is not None
                ),
                edm_p_mean=config.mdsm_edm_p_mean,
                edm_p_std=config.mdsm_edm_p_std,
                cosine_eps=config.mdsm_cosine_eps,
                norm_floor=config.mdsm_norm_floor,
            )

            loss_gp = torch.tensor(0.0, device=device)
            if config.gradient_penalty_lambda > 0:
                v_noisy = add_relative_noise(v_clean, scale=0.1)
                loss_gp = gradient_penalty(model, v_clean, v_noisy)

            loss = loss_dsm + config.gradient_penalty_lambda * loss_gp

        if not torch.isfinite(loss):
            if config.skip_non_finite_batches:
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                non_finite_streak += 1
                skipped_batches += 1
                if non_finite_streak >= config.non_finite_backoff_streak_trigger:
                    apply_non_finite_backoff(optimizer, config.non_finite_lr_backoff)
                if non_finite_streak <= 5 or non_finite_streak % 25 == 0:
                    print(
                        f"  [WARN] non-finite MDSM loss at batch {batch_idx + 1}, "
                        f"streak={non_finite_streak}, lr={optimizer.param_groups[0]['lr']:.6g}; "
                        "batch skipped"
                    )
                if non_finite_streak >= config.max_consecutive_non_finite_batches:
                    raise RuntimeError(
                        "Too many consecutive non-finite MDSM losses; stopping to avoid silent stall."
                    )
                continue
            raise RuntimeError("Non-finite loss encountered in train_epoch_mdsm")

        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if not gradients_are_finite(model):
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                non_finite_streak += 1
                skipped_batches += 1
                if non_finite_streak >= config.non_finite_backoff_streak_trigger:
                    apply_non_finite_backoff(optimizer, config.non_finite_lr_backoff)
                scaler.update()
                if non_finite_streak <= 5 or non_finite_streak % 25 == 0:
                    print(
                        f"  [WARN] non-finite MDSM gradients at batch {batch_idx + 1}, "
                        f"streak={non_finite_streak}, lr={optimizer.param_groups[0]['lr']:.6g}; "
                        "step skipped"
                    )
                if non_finite_streak >= config.max_consecutive_non_finite_batches:
                    raise RuntimeError(
                        "Too many consecutive non-finite MDSM gradients; stopping to avoid silent stall."
                    )
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                non_finite_streak += 1
                skipped_batches += 1
                if non_finite_streak >= config.non_finite_backoff_streak_trigger:
                    apply_non_finite_backoff(optimizer, config.non_finite_lr_backoff)
                scaler.update()
                if non_finite_streak <= 5 or non_finite_streak % 25 == 0:
                    print(
                        f"  [WARN] non-finite clipped grad norm at batch {batch_idx + 1}, "
                        f"streak={non_finite_streak}, lr={optimizer.param_groups[0]['lr']:.6g}; "
                        "step skipped"
                    )
                if non_finite_streak >= config.max_consecutive_non_finite_batches:
                    raise RuntimeError(
                        "Too many consecutive non-finite clipped grad norms; stopping to avoid silent stall."
                    )
                continue
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if not gradients_are_finite(model):
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                non_finite_streak += 1
                skipped_batches += 1
                if non_finite_streak >= config.non_finite_backoff_streak_trigger:
                    apply_non_finite_backoff(optimizer, config.non_finite_lr_backoff)
                if non_finite_streak <= 5 or non_finite_streak % 25 == 0:
                    print(
                        f"  [WARN] non-finite MDSM gradients at batch {batch_idx + 1}, "
                        f"streak={non_finite_streak}, lr={optimizer.param_groups[0]['lr']:.6g}; "
                        "step skipped"
                    )
                if non_finite_streak >= config.max_consecutive_non_finite_batches:
                    raise RuntimeError(
                        "Too many consecutive non-finite MDSM gradients; stopping to avoid silent stall."
                    )
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                non_finite_streak += 1
                skipped_batches += 1
                if non_finite_streak >= config.non_finite_backoff_streak_trigger:
                    apply_non_finite_backoff(optimizer, config.non_finite_lr_backoff)
                if non_finite_streak <= 5 or non_finite_streak % 25 == 0:
                    print(
                        f"  [WARN] non-finite clipped grad norm at batch {batch_idx + 1}, "
                        f"streak={non_finite_streak}, lr={optimizer.param_groups[0]['lr']:.6g}; "
                        "step skipped"
                    )
                if non_finite_streak >= config.max_consecutive_non_finite_batches:
                    raise RuntimeError(
                        "Too many consecutive non-finite clipped grad norms; stopping to avoid silent stall."
                    )
                continue
            optimizer.step()

        non_finite_streak = 0
        total_loss += loss.item()
        total_dsm += loss_dsm.item()
        total_gp += loss_gp.item()
        num_batches += 1
        global_step += 1

        if config.log_every > 0 and (batch_idx + 1) % config.log_every == 0:
            e_scale = torch.exp(model.log_energy_scale.detach().clamp(min=-8.0, max=8.0)).item()
            print(
                f"  [{batch_idx + 1}/{len(dataloader)}] "
                f"loss={loss.item():.4f} DSM={loss_dsm.item():.4f} "
                f"GP={loss_gp.item():.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f} "
                f"E_scale={e_scale:.2f} sigma=[{sigma_min:.4f},{sigma_max:.4f}]"
            )

    return {
        "loss": total_loss / max(num_batches, 1),
        "dsm_loss": total_dsm / max(num_batches, 1),
        "gradient_penalty": total_gp / max(num_batches, 1),
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "skipped_batches": float(skipped_batches),
        "skip_rate": skipped_batches / max(len(dataloader), 1),
    }, global_step


def train_epoch_contrastive(
    model: SimpleEnergy,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: Stage1Config,
    device: torch.device,
    global_step: int,
    scaler: torch.amp.GradScaler,
    autocast_ctx,
) -> tuple[dict, int]:
    model.train()
    total_loss = 0.0
    total_e_pos = 0.0
    total_e_neg = 0.0
    total_gp = 0.0
    num_batches = 0
    skipped_batches = 0
    non_finite_streak = 0

    for batch_idx, v_orig in enumerate(dataloader):
        if (batch_idx % 16 == 0) and not parameters_are_finite(model):
            raise RuntimeError(
                f"Model parameters became non-finite before batch {batch_idx + 1}"
            )

        v_orig = v_orig.to(device, non_blocking=True)

        if global_step < config.warmup_steps:
            warmup_factor = (global_step + 1) / max(1, config.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = pg["initial_lr"] * warmup_factor

        scale_idx = torch.randint(0, len(config.train_noise_scales), (1,)).item()
        noise_scale = config.train_noise_scales[scale_idx]

        with autocast_ctx():
            e_pos = model(v_orig, v_orig)
            v_noisy = add_relative_noise(v_orig, noise_scale)
            e_neg = model(v_orig, v_noisy)

            loss_contrastive = margin_contrastive_loss(e_pos, e_neg, margin=config.margin)
            gp = torch.tensor(0.0, device=device)
            if config.gradient_penalty_lambda > 0:
                gp = gradient_penalty(model, v_orig, v_noisy)
            loss = loss_contrastive + config.gradient_penalty_lambda * gp

        if not torch.isfinite(loss):
            if config.skip_non_finite_batches:
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                non_finite_streak += 1
                skipped_batches += 1
                if non_finite_streak >= config.non_finite_backoff_streak_trigger:
                    apply_non_finite_backoff(optimizer, config.non_finite_lr_backoff)
                if non_finite_streak <= 5 or non_finite_streak % 25 == 0:
                    print(
                        f"  [WARN] non-finite contrastive loss at batch {batch_idx + 1}, "
                        f"streak={non_finite_streak}, lr={optimizer.param_groups[0]['lr']:.6g}; "
                        "batch skipped"
                    )
                if non_finite_streak >= config.max_consecutive_non_finite_batches:
                    raise RuntimeError(
                        "Too many consecutive non-finite contrastive losses; stopping to avoid silent stall."
                    )
                continue
            raise RuntimeError("Non-finite loss encountered in train_epoch_contrastive")

        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if not gradients_are_finite(model):
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                non_finite_streak += 1
                skipped_batches += 1
                if non_finite_streak >= config.non_finite_backoff_streak_trigger:
                    apply_non_finite_backoff(optimizer, config.non_finite_lr_backoff)
                scaler.update()
                if non_finite_streak <= 5 or non_finite_streak % 25 == 0:
                    print(
                        f"  [WARN] non-finite contrastive gradients at batch {batch_idx + 1}, "
                        f"streak={non_finite_streak}, lr={optimizer.param_groups[0]['lr']:.6g}; "
                        "step skipped"
                    )
                if non_finite_streak >= config.max_consecutive_non_finite_batches:
                    raise RuntimeError(
                        "Too many consecutive non-finite contrastive gradients; stopping to avoid silent stall."
                    )
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                non_finite_streak += 1
                skipped_batches += 1
                if non_finite_streak >= config.non_finite_backoff_streak_trigger:
                    apply_non_finite_backoff(optimizer, config.non_finite_lr_backoff)
                scaler.update()
                if non_finite_streak <= 5 or non_finite_streak % 25 == 0:
                    print(
                        f"  [WARN] non-finite clipped grad norm at batch {batch_idx + 1}, "
                        f"streak={non_finite_streak}, lr={optimizer.param_groups[0]['lr']:.6g}; "
                        "step skipped"
                    )
                if non_finite_streak >= config.max_consecutive_non_finite_batches:
                    raise RuntimeError(
                        "Too many consecutive non-finite clipped grad norms; stopping to avoid silent stall."
                    )
                continue
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if not gradients_are_finite(model):
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                non_finite_streak += 1
                skipped_batches += 1
                if non_finite_streak >= config.non_finite_backoff_streak_trigger:
                    apply_non_finite_backoff(optimizer, config.non_finite_lr_backoff)
                if non_finite_streak <= 5 or non_finite_streak % 25 == 0:
                    print(
                        f"  [WARN] non-finite contrastive gradients at batch {batch_idx + 1}, "
                        f"streak={non_finite_streak}, lr={optimizer.param_groups[0]['lr']:.6g}; "
                        "step skipped"
                    )
                if non_finite_streak >= config.max_consecutive_non_finite_batches:
                    raise RuntimeError(
                        "Too many consecutive non-finite contrastive gradients; stopping to avoid silent stall."
                    )
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                non_finite_streak += 1
                skipped_batches += 1
                if non_finite_streak >= config.non_finite_backoff_streak_trigger:
                    apply_non_finite_backoff(optimizer, config.non_finite_lr_backoff)
                if non_finite_streak <= 5 or non_finite_streak % 25 == 0:
                    print(
                        f"  [WARN] non-finite clipped grad norm at batch {batch_idx + 1}, "
                        f"streak={non_finite_streak}, lr={optimizer.param_groups[0]['lr']:.6g}; "
                        "step skipped"
                    )
                if non_finite_streak >= config.max_consecutive_non_finite_batches:
                    raise RuntimeError(
                        "Too many consecutive non-finite clipped grad norms; stopping to avoid silent stall."
                    )
                continue
            optimizer.step()

        non_finite_streak = 0
        total_loss += loss.item()
        total_e_pos += e_pos.mean().item()
        total_e_neg += e_neg.mean().item()
        total_gp += gp.item()
        num_batches += 1
        global_step += 1

        if config.log_every > 0 and (batch_idx + 1) % config.log_every == 0:
            print(
                f"  [{batch_idx + 1}/{len(dataloader)}] "
                f"loss={loss.item():.4f} E_pos={e_pos.mean().item():.4f} "
                f"E_neg={e_neg.mean().item():.4f} GP={gp.item():.4f}"
            )

    return {
        "loss": total_loss / max(num_batches, 1),
        "e_pos_mean": total_e_pos / max(num_batches, 1),
        "e_neg_mean": total_e_neg / max(num_batches, 1),
        "gradient_penalty": total_gp / max(num_batches, 1),
        "energy_gap": (total_e_neg - total_e_pos) / max(num_batches, 1),
        "skipped_batches": float(skipped_batches),
        "skip_rate": skipped_batches / max(len(dataloader), 1),
    }, global_step


def evaluate_denoising(
    model: SimpleEnergy,
    dataset: SONARVectorDataset,
    config: Stage1Config,
    device: torch.device,
    num_samples: int = 100,
    eval_indices: torch.Tensor | None = None,
) -> dict:
    """Evaluate denoising quality with configured Langevin method."""
    model.eval()
    results: dict[str, dict] = {}

    if eval_indices is None:
        eval_indices = torch.randperm(len(dataset))[:num_samples]
    else:
        eval_indices = eval_indices[:num_samples]

    method = config.langevin.method
    method_kwargs = {}
    if method == "pid":
        method_kwargs = dict(
            kp=config.langevin.pid_kp,
            ki=config.langevin.pid_ki,
            kd=config.langevin.pid_kd,
            integral_decay=config.langevin.pid_integral_decay,
        )
    elif method == "underdamped":
        method_kwargs = dict(
            friction=config.langevin.underdamped_friction,
            mass=config.langevin.underdamped_mass,
        )
    elif method == "overdamped":
        method_kwargs = dict(momentum_beta=config.langevin.momentum_beta)

    for noise_scale in config.eval_noise_scales:
        cos_before_list = []
        cos_after_list = []
        energy_before_list = []
        energy_after_list = []

        for idx in eval_indices:
            v_orig = dataset[int(idx.item())].unsqueeze(0).to(device)
            v_noisy = add_relative_noise(v_orig, noise_scale)

            cos_before = F.cosine_similarity(v_orig, v_noisy, dim=-1).item()

            result = run_langevin(
                method=method,
                energy_fn=model,
                v_query=v_orig,
                v_init=v_noisy,
                lr=config.langevin.lr,
                noise_scale=config.langevin.noise_scale,
                max_steps=config.langevin.max_steps,
                target_norm=config.langevin.target_norm,
                energy_threshold=config.langevin.energy_threshold,
                plateau_patience=config.langevin.plateau_patience,
                plateau_delta=config.langevin.plateau_delta,
                v_target=v_orig,
                **method_kwargs,
            )

            cos_after = F.cosine_similarity(v_orig, result.v_final, dim=-1).item()

            with torch.no_grad():
                e_before = model(v_orig, v_noisy).item()
                e_after = model(v_orig, result.v_final).item()

            cos_before_list.append(cos_before)
            cos_after_list.append(cos_after)
            energy_before_list.append(e_before)
            energy_after_list.append(e_after)

        cos_before_mean = sum(cos_before_list) / len(cos_before_list)
        cos_after_mean = sum(cos_after_list) / len(cos_after_list)
        improvement = cos_after_mean - cos_before_mean

        results[f"noise_{noise_scale}"] = {
            "cos_before_mean": cos_before_mean,
            "cos_after_mean": cos_after_mean,
            "improvement": improvement,
            "energy_before_mean": sum(energy_before_list) / len(energy_before_list),
            "energy_after_mean": sum(energy_after_list) / len(energy_after_list),
            "success_rate": sum(
                1 for a, b in zip(cos_after_list, cos_before_list) if a > b
            ) / len(cos_after_list),
        }

    return results


def build_autocast_context(
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
):
    def _ctx():
        if use_amp:
            return torch.amp.autocast(device_type=device.type, dtype=amp_dtype)
        return nullcontext()

    return _ctx


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Train SimpleEnergy for denoising")
    parser.add_argument("--data", type=str, required=True, help="Path to encoded .pt dataset")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gp_lambda", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP")
    parser.add_argument("--amp_dtype", type=str, default=None, choices=["bf16", "fp16"])
    # Architecture flags
    parser.add_argument("--loss", type=str, default=None,
                        choices=["mdsm", "margin_contrastive"],
                        help="Loss type (default: mdsm)")
    parser.add_argument("--norm", type=str, default=None,
                        choices=["orthonorm", "spectral_norm", "none"],
                        help="Normalization (default: orthonorm)")
    parser.add_argument("--activation", type=str, default=None,
                        choices=["groupsort", "lipschitz_spline", "relu"],
                        help="Activation (default: groupsort)")
    parser.add_argument("--langevin_method", type=str, default=None,
                        choices=["overdamped", "pid", "underdamped"],
                        help="Langevin method (default: pid)")
    args = parser.parse_args()

    config = Stage1Config()
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.loss is not None:
        config.loss_type = args.loss
    if args.norm is not None:
        config.norm_mode = args.norm
    if args.activation is not None:
        config.activation = args.activation
    if args.langevin_method is not None:
        config.langevin.method = args.langevin_method
    if args.seed is not None:
        config.seed = args.seed
    if args.gp_lambda is not None:
        config.gradient_penalty_lambda = args.gp_lambda
    if args.num_workers is not None:
        config.dataloader_num_workers = args.num_workers
    if args.compile:
        config.enable_compile = True
    if args.no_amp:
        config.enable_amp = False
    if args.amp_dtype is not None:
        config.amp_dtype = args.amp_dtype
    config.use_wandb = args.wandb

    set_seed(config.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    use_amp = config.enable_amp and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(config.amp_dtype)
    autocast_ctx = build_autocast_context(device, use_amp, amp_dtype)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=(use_amp and amp_dtype == torch.float16),
    )

    print(
        f"Mixed precision: {'on' if use_amp else 'off'} "
        f"(dtype={config.amp_dtype if use_amp else 'n/a'})"
    )

    print(f"Loading dataset from {args.data}...")
    full_dataset = SONARVectorDataset(args.data)
    print(f"  Total vectors: {len(full_dataset)}, dim: {full_dataset.embeddings.shape[1]}")

    n_test = min(config.num_test_sentences, len(full_dataset) // 10)
    n_train = len(full_dataset) - n_test
    train_dataset = full_dataset.subset(0, n_train)
    test_dataset = full_dataset.subset(n_train, len(full_dataset))
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    train_gen = torch.Generator()
    train_gen.manual_seed(config.seed)

    loader_kwargs = dict(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.dataloader_num_workers,
        pin_memory=(config.dataloader_pin_memory and device.type == "cuda"),
        generator=train_gen,
    )
    if config.dataloader_num_workers > 0:
        loader_kwargs["persistent_workers"] = config.dataloader_persistent_workers
        loader_kwargs["prefetch_factor"] = config.dataloader_prefetch_factor

    train_loader = DataLoader(**loader_kwargs)

    raw_model = SimpleEnergy(
        dim=config.energy_dim,
        hidden_dims=config.energy_hidden_dims,
        norm_mode=config.norm_mode,
        activation=config.activation,
        ortho_n_iters=config.ortho_n_iters,
        groupsort_size=config.groupsort_size,
        spline_num_knots=config.spline_num_knots,
    ).to(device)

    model = raw_model
    if config.enable_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(raw_model, mode=config.compile_mode)
            print(f"torch.compile enabled (mode={config.compile_mode})")
        except Exception as exc:
            print(f"torch.compile failed, fallback to eager: {type(exc).__name__}")
            model = raw_model

    num_params = sum(p.numel() for p in raw_model.parameters())
    print(f"SimpleEnergy: {num_params:,} parameters")
    print(f"  Architecture: norm={config.norm_mode}, activation={config.activation}")
    print(f"  Hidden dims: {config.energy_hidden_dims}")
    print(f"  Loss: {config.loss_type}")
    print(f"  Langevin: {config.langevin.method}")

    scale_params = [raw_model.log_energy_scale]
    other_params = [
        p for n, p in raw_model.named_parameters() if "log_energy_scale" not in n
    ]
    scale_lr = config.lr * config.energy_scale_lr_multiplier
    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": config.lr, "weight_decay": config.weight_decay},
        {"params": scale_params, "lr": scale_lr, "weight_decay": 0.0},
    ])
    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=config.lr * 0.1,
    )

    start_epoch = 0
    global_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        raw_model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        print(f"  Resumed at epoch {start_epoch}")

    wandb_run = None
    if config.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=config.wandb_project,
            config={
                "stage1_config": asdict(config),
                "num_params": num_params,
            },
        )

    output_dir = Path(config.output_dir)
    checkpoint_dir = Path(config.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    eval_gen = torch.Generator()
    eval_gen.manual_seed(config.seed + 2026)
    eval_size = min(100, len(test_dataset))
    eval_indices = torch.randperm(len(test_dataset), generator=eval_gen)[:eval_size]

    print(f"\n{'='*60}")
    print(f"Training SimpleEnergy for {config.num_epochs} epochs")
    print(f"  Batch size: {config.batch_size}")
    print(f"  LR: {config.lr} (warmup: {config.warmup_steps} steps)")
    print(f"  Energy-scale LR multiplier: {config.energy_scale_lr_multiplier}")
    print(f"  Loss: {config.loss_type}")
    if config.loss_type == "mdsm":
        print(f"  MDSM sigma range: [{config.mdsm_sigma_min}, {config.mdsm_sigma_max}]")
        print(f"  MDSM sigma sampling: {config.mdsm_sigma_sampling}")
        print(f"  MDSM sigma weighting: {config.mdsm_sigma_weighting}")
        print(f"  MDSM directional: {config.mdsm_directional}")
        print(f"  MDSM magnitude aux weight: {config.mdsm_magnitude_aux_weight}")
        print(f"  MDSM tangent projection: {config.mdsm_tangent_projection}")
        print(f"  MDSM cosine eps: {config.mdsm_cosine_eps}")
        print(f"  MDSM norm floor: {config.mdsm_norm_floor}")
        print(f"  MDSM force FP32 path: {config.mdsm_force_fp32}")
    else:
        print(f"  Margin: {config.margin}")
        print(f"  Train noise scales: {config.train_noise_scales}")
    print(f"  Eval noise scales: {config.eval_noise_scales}")
    print(f"  Langevin: method={config.langevin.method}, lr={config.langevin.lr}, steps={config.langevin.max_steps}")
    print(f"  Target norm: {config.langevin.target_norm}")
    print(f"  GP lambda: {config.gradient_penalty_lambda}")
    print(
        "  Non-finite handling: "
        f"skip={config.skip_non_finite_batches}, "
        f"max_streak={config.max_consecutive_non_finite_batches}, "
        f"lr_backoff={config.non_finite_lr_backoff}, "
        f"backoff_trigger={config.non_finite_backoff_streak_trigger}"
    )
    print(f"{'='*60}\n")

    all_metrics: list[dict] = []
    best_improvement = -float("inf")
    start_time = time.time()

    for epoch in range(start_epoch, config.num_epochs):
        epoch_start = time.time()
        print(f"Epoch {epoch + 1}/{config.num_epochs}")

        if config.norm_mode == "orthonorm":
            n_iters = get_ortho_iters(config, epoch)
            layers_updated = set_ortho_n_iters(raw_model, n_iters)
            if layers_updated > 0:
                print(f"  Ortho schedule: n_iters={n_iters} on {layers_updated} layers")

        if config.loss_type == "mdsm":
            train_metrics, global_step = train_epoch_mdsm(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                config=config,
                device=device,
                epoch=epoch,
                global_step=global_step,
                scaler=scaler,
                autocast_ctx=autocast_ctx,
            )
        else:
            train_metrics, global_step = train_epoch_contrastive(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                config=config,
                device=device,
                global_step=global_step,
                scaler=scaler,
                autocast_ctx=autocast_ctx,
            )

        if global_step >= config.warmup_steps:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        metrics_str = " ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
        print(f"  {metrics_str} lr={optimizer.param_groups[0]['lr']:.6f} ({epoch_time:.1f}s)")

        eval_metrics = None
        if (epoch + 1) % config.eval_every_epoch == 0 or epoch == config.num_epochs - 1:
            print("  Evaluating denoising...")
            eval_metrics = evaluate_denoising(
                model=model,
                dataset=test_dataset,
                config=config,
                device=device,
                num_samples=eval_size,
                eval_indices=eval_indices,
            )
            for noise_key, metrics in eval_metrics.items():
                print(
                    f"    {noise_key}: "
                    f"cos_before={metrics['cos_before_mean']:.4f} -> "
                    f"cos_after={metrics['cos_after_mean']:.4f} "
                    f"(d={metrics['improvement']:+.4f}, "
                    f"success={metrics['success_rate']:.0%})"
                )

            avg_improvement = sum(
                m["improvement"] for m in eval_metrics.values()
            ) / len(eval_metrics)

            if avg_improvement > best_improvement:
                best_improvement = avg_improvement
                torch.save(
                    {
                        "model_state": raw_model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "metrics": eval_metrics,
                        "stage1_config": asdict(config),
                        "config": {
                            "energy_dim": config.energy_dim,
                            "energy_hidden_dims": config.energy_hidden_dims,
                            "norm_mode": config.norm_mode,
                            "activation": config.activation,
                            "loss_type": config.loss_type,
                        },
                    },
                    checkpoint_dir / "best.pt",
                )
                print(f"    * New best (avg improvement: {avg_improvement:+.4f})")

        epoch_record = {"epoch": epoch + 1, "train": train_metrics}
        if eval_metrics is not None:
            epoch_record["eval"] = eval_metrics
        all_metrics.append(epoch_record)

        if wandb_run is not None:
            log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
            log_dict["train/lr"] = optimizer.param_groups[0]["lr"]
            if eval_metrics is not None:
                for noise_key, metrics in eval_metrics.items():
                    for mk, mv in metrics.items():
                        log_dict[f"eval/{noise_key}/{mk}"] = mv
            wandb_run.log(log_dict, step=epoch + 1)

        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "model_state": raw_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "stage1_config": asdict(config),
                },
                checkpoint_dir / f"epoch_{epoch + 1:03d}.pt",
            )

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")

    print("\nFinal evaluation on test set...")
    final_size = min(200, len(test_dataset))
    final_gen = torch.Generator()
    final_gen.manual_seed(config.seed + 4096)
    final_indices = torch.randperm(len(test_dataset), generator=final_gen)[:final_size]

    final_eval = evaluate_denoising(
        model=model,
        dataset=test_dataset,
        config=config,
        device=device,
        num_samples=final_size,
        eval_indices=final_indices,
    )
    for noise_key, metrics in final_eval.items():
        print(
            f"  {noise_key}: "
            f"cos_before={metrics['cos_before_mean']:.4f} -> "
            f"cos_after={metrics['cos_after_mean']:.4f} "
            f"(d={metrics['improvement']:+.4f}, "
            f"success={metrics['success_rate']:.0%})"
        )

    torch.save(
        {
            "model_state": raw_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": config.num_epochs - 1,
            "global_step": global_step,
            "final_eval": final_eval,
            "stage1_config": asdict(config),
            "config": {
                "energy_dim": config.energy_dim,
                "energy_hidden_dims": config.energy_hidden_dims,
                "norm_mode": config.norm_mode,
                "activation": config.activation,
                "loss_type": config.loss_type,
            },
        },
        checkpoint_dir / "final.pt",
    )

    summary = {
        "total_time_seconds": total_time,
        "num_epochs": config.num_epochs,
        "best_improvement": best_improvement,
        "final_eval": final_eval,
        "stage1_config": asdict(config),
        "epoch_metrics": all_metrics,
    }
    with open(output_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved to {output_dir / 'training_metrics.json'}")

    print(f"\n{'='*60}")
    print("KILL CRITERION CHECK")
    print(f"{'='*60}")
    all_pass = True
    for noise_key, metrics in final_eval.items():
        passed = metrics["improvement"] > 0 and metrics["success_rate"] > 0.5
        status = "PASS" if passed else "FAIL"
        print(
            f"  {noise_key}: improvement={metrics['improvement']:+.4f}, "
            f"success={metrics['success_rate']:.0%} -> {status}"
        )
        if not passed:
            all_pass = False

    if all_pass:
        print("\n  VERDICT: Stage 1 PASSED - energy function works for denoising")
    else:
        print("\n  VERDICT: Stage 1 FAILED - denoised vectors not closer to originals")
        print("    -> Energy function does not work. Review architecture or training.")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
