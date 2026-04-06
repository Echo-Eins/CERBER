#!/usr/bin/env python3
"""
Phase 1B: CompositeCritic training on HotpotQA.

Trains the CompositeCritic (ConditionalAngularCritic + AnalyticalRadialGuard)
with a cleaner loss design than Phase 1:

  1. Focal-InfoNCE contrastive: E(q, answer, ctx) < E(q, distractor, ctx)
     — Primary loss, always on.
  2. Path-contrastive loss (warmup): monotonically decreasing energy along
     the geodesic from v_query → v_answer.
     — 1st-order only (no create_graph/Hessian).  Covers the ENTIRE
       navigation path, not just near-answer neighborhood.
     — NOT the same as interp_gp (lessons Phase 2e: GP flattens gradients).
       This enforces energy ORDERING → gradients naturally point toward answer.

Removed from Phase 1:
  • Hinge loss (lessons: died at epoch 1, E_target < E_predicted trivially)
  • REINFORCE text loss (lessons: always 0, add only after ranking+cosine work)
  • Direction loss (lessons: 2nd-order dominance, only covers near-answer area)

Architecture improvements:
  • Angular critic operates on unit sphere — gradient tangential by construction
  • Radial guard (analytical, no training) keeps ‖v‖ on SONAR manifold
  • Path-contrastive is 1st-order — no 2nd/1st-order gradient conflict

Usage:
    python experiments/12_composite_critic/train_composite.py
    python experiments/12_composite_critic/train_composite.py --config configs/composite_critic_config.json
    python experiments/12_composite_critic/train_composite.py --config configs/composite_critic_config.json --max-epochs 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cebcm.models.composite_critic import CompositeCritic, CompositeCriticConfig
from cebcm.models.conditional_angular_critic import ConditionalAngularCriticConfig
from cebcm.models.radial_guard import RadialGuardConfig
from cebcm.training.stage2_utils import (
    MetricTracker,
    get_cosine_schedule_with_warmup,
    load_config,
    resolve_device,
    save_checkpoint,
    setup_amp,
    setup_seed,
)


# ── Dataset ──────────────────────────────────────────────────────

class CriticDataset(Dataset):
    """Dataset for critic training.  Returns QA pairs with negatives."""

    def __init__(self, samples: list[dict], num_negatives: int = 7):
        self.samples = samples
        self.num_negatives = num_negatives
        self._all_answers = torch.stack([s["v_answer"] for s in samples])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        v_q = s["v_question"]
        v_a = s["v_answer"]
        v_steps = s["v_steps"]

        # Context = mean of reasoning steps
        v_context = v_steps.mean(dim=0) if v_steps.shape[0] > 0 else v_q

        # Sample negatives (random answers from other samples)
        neg_indices = []
        while len(neg_indices) < self.num_negatives:
            j = random.randint(0, len(self.samples) - 1)
            if j != idx:
                neg_indices.append(j)
        v_negatives = self._all_answers[neg_indices]

        return {
            "v_question": v_q,
            "v_answer": v_a,
            "v_context": v_context,
            "v_negatives": v_negatives,
        }


def collate_critic(batch: list[dict]) -> dict:
    return {
        "v_questions": torch.stack([b["v_question"] for b in batch]),
        "v_answers": torch.stack([b["v_answer"] for b in batch]),
        "v_contexts": torch.stack([b["v_context"] for b in batch]),
        "v_negatives": torch.stack([b["v_negatives"] for b in batch]),
    }


# ── Training ─────────────────────────────────────────────────────

def add_noise(v: torch.Tensor, noise_scale: float) -> torch.Tensor:
    """Add isotropic Gaussian noise scaled relative to vector norm."""
    noise = torch.randn_like(v)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return v + noise_scale * v_norm * noise


def train_step(
    critic: CompositeCritic,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    cfg: dict,
    epoch: int,
) -> dict[str, float]:
    """
    Single training step.

    Losses:
      1. Focal-InfoNCE contrastive (always)
      2. Direction loss (after warmup epoch, with linear ramp-up)
    """
    v_q = batch["v_questions"].to(device)
    v_a = batch["v_answers"].to(device)
    v_ctx = batch["v_contexts"].to(device)
    v_neg = batch["v_negatives"].to(device)

    clip_grad = cfg.get("clip_grad_norm", 1.0)
    w_rank = cfg.get("w_rank", 1.0)
    noise_scale = cfg.get("noise_scale", 0.01)

    # Path-contrastive loss: ramps from w_path_max/ramp_epochs to w_path_max
    # Replaces direction_loss — 1st-order only (no create_graph/Hessian),
    # covers the ENTIRE navigation path (not just near-answer neighborhood).
    path_warmup = cfg.get("path_warmup_epoch", 0)
    w_path_max = cfg.get("w_path_max", 0.3)
    ramp_epochs = cfg.get("path_ramp_epochs", 3)
    if epoch < path_warmup:
        w_path = 0.0
    else:
        ramp = min(1.0, (epoch - path_warmup + 1) / ramp_epochs)
        w_path = w_path_max * ramp

    optimizer.zero_grad()

    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
        # ── LOSS 1: Focal-InfoNCE contrastive ──
        loss_rank, rank_metrics = critic.compute_contrastive_loss(
            v_q, v_a, v_neg, v_context=v_ctx,
        )

        total_loss = w_rank * loss_rank

        # ── LOSS 2: Path-contrastive (1st-order, no Hessian) ──
        path_metrics: dict[str, float] = {}
        if w_path > 0:
            loss_path, path_metrics = critic.compute_path_contrastive_loss(
                v_q, v_a, v_context=v_ctx,
                num_waypoints=cfg.get("path_num_waypoints", 5),
                waypoint_noise=cfg.get("path_waypoint_noise", 0.02),
                margin=cfg.get("path_margin", 0.1),
            )
            total_loss = total_loss + w_path * loss_path

    # ── Langevin assessment (detached, for monitoring only) ──
    langevin_steps = cfg.get("langevin_steps", 30)
    langevin_lr = cfg.get("langevin_lr", 0.1)

    target_norm = critic.radial.target_norm

    with torch.no_grad():
        v_current = add_noise(v_q, noise_scale * 2)  # start from noisy query
        for _ in range(langevin_steps):
            v_current = v_current.detach().requires_grad_(True)
            with torch.enable_grad():
                e = critic(v_q, v_current, v_context=v_ctx)
                grad_ld = torch.autograd.grad(e.sum(), v_current)[0]
            # Tamed gradient: grad / (1 + lr * ||grad||)
            grad_norm = grad_ld.norm(dim=-1, keepdim=True)
            tamed = grad_ld / (1.0 + langevin_lr * grad_norm)
            # Tangent projection (remove radial component before step)
            v_hat = F.normalize(v_current.detach(), dim=-1)
            radial = (tamed * v_hat).sum(dim=-1, keepdim=True) * v_hat
            tamed = tamed - radial
            v_current = (v_current - langevin_lr * tamed).detach()
            # Sphere projection (lessons: "taming before projection, target_norm still applies")
            v_current = F.normalize(v_current, dim=-1) * target_norm

    v_predicted = v_current.detach()
    cos_sim = F.cosine_similarity(v_predicted, v_a, dim=-1)
    v_pred_norm = v_predicted.norm(dim=-1)

    # ── Backward ──
    scaler.scale(total_loss).backward()

    if clip_grad > 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(critic.parameters(), clip_grad)

    scaler.step(optimizer)
    scaler.update()

    metrics = {
        **rank_metrics,
        **path_metrics,
        "total_loss": total_loss.item(),
        "loss_rank": loss_rank.item(),
        "w_path": w_path,
        "cos_sim_mean": cos_sim.mean().item(),
        "cos_sim_min": cos_sim.min().item(),
        "cos_sim_max": cos_sim.max().item(),
        "v_pred_norm_mean": v_pred_norm.mean().item(),
    }
    return metrics


@torch.no_grad()
def eval_step(
    critic: CompositeCritic,
    batch: dict,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    langevin_steps: int = 30,
    langevin_lr: float = 0.1,
    noise_scale: float = 0.01,
) -> dict[str, float]:
    """Evaluation: ranking accuracy + Langevin cosine assessment."""
    v_q = batch["v_questions"].to(device)
    v_a = batch["v_answers"].to(device)
    v_ctx = batch["v_contexts"].to(device)
    v_neg = batch["v_negatives"].to(device)

    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
        E_pos = critic(v_q, v_a, v_context=v_ctx)
        B, N, D = v_neg.shape
        v_q_exp = v_q.unsqueeze(1).expand_as(v_neg).reshape(B * N, D)
        v_ctx_exp = v_ctx.unsqueeze(1).expand(B, N, D).reshape(B * N, D)
        v_neg_flat = v_neg.reshape(B * N, D)
        E_neg = critic(v_q_exp, v_neg_flat, v_context=v_ctx_exp).reshape(B, N)

        rank_acc = (E_pos.unsqueeze(1) < E_neg).float().mean().item()
        energy_gap = (E_neg.mean(dim=1) - E_pos).mean().item()

    # Mini-Langevin for cosine assessment
    target_norm = critic.radial.target_norm
    with torch.enable_grad():
        v_current = add_noise(v_q, noise_scale * 2)
        for _ in range(langevin_steps):
            v_current = v_current.detach().requires_grad_(True)
            e = critic(v_q, v_current, v_context=v_ctx)
            grad = torch.autograd.grad(e.sum(), v_current)[0]
            grad_norm = grad.norm(dim=-1, keepdim=True)
            tamed = grad / (1.0 + langevin_lr * grad_norm)
            # Tangent projection + sphere projection (match langevin.py)
            v_hat = F.normalize(v_current.detach(), dim=-1)
            radial = (tamed * v_hat).sum(dim=-1, keepdim=True) * v_hat
            tamed = tamed - radial
            v_current = (v_current - langevin_lr * tamed).detach()
            v_current = F.normalize(v_current, dim=-1) * target_norm

    cos_sim = F.cosine_similarity(v_current.detach(), v_a, dim=-1)
    v_pred_norm = v_current.detach().norm(dim=-1)

    return {
        "val_rank_acc": rank_acc,
        "val_energy_gap": energy_gap,
        "val_E_pos_mean": E_pos.mean().item(),
        "val_E_neg_mean": E_neg.mean().item(),
        "val_cos_sim_mean": cos_sim.mean().item(),
        "val_cos_sim_min": cos_sim.min().item(),
        "val_cos_sim_max": cos_sim.max().item(),
        "val_v_pred_norm_mean": v_pred_norm.mean().item(),
    }


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1B: CompositeCritic Training (Angular + Radial Guard)"
    )
    parser.add_argument("--config", default="configs/composite_critic_config.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    setup_seed(config.get("seed", 42), device)

    out_cfg = config["output"]
    for d in [out_cfg["output_dir"], out_cfg["checkpoint_dir"], out_cfg["logs_dir"]]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 1B: CompositeCritic Training")
    print(f"  Angular: ConditionalAngularCritic (learned, tangential)")
    print(f"  Radial:  AnalyticalRadialGuard (analytical, no params)")
    print(f"  Device: {device}")
    print("=" * 60)

    # ── Build composite critic ──
    critic_raw = config.get("critic", {})
    ang_cfg = ConditionalAngularCriticConfig(**critic_raw.get("angular", {}))
    rad_cfg = RadialGuardConfig(**critic_raw.get("radial", {}))
    comp_cfg = CompositeCriticConfig(
        angular=ang_cfg,
        radial=rad_cfg,
        lambda_radial=critic_raw.get("lambda_radial", 5.0),
    )
    critic = CompositeCritic(comp_cfg).to(device)

    print(f"  Angular params: {critic.angular.num_params:,}")
    print(f"  Radial params:  0 (analytical)")
    print(f"  λ_radial: {comp_cfg.lambda_radial}")
    print(f"  target_norm: {rad_cfg.target_norm}")

    # ── Load data ──
    data_path = config["data"]["path"]
    print(f"  Loading data from {data_path}")
    data = torch.load(data_path, map_location="cpu", weights_only=False)

    num_neg = ang_cfg.num_negatives
    train_ds = CriticDataset(data["train"], num_negatives=num_neg)
    val_ds = CriticDataset(data["val"], num_negatives=num_neg)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_cfg = config["training"]
    batch_size = train_cfg["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=config["data"].get("num_workers", 4),
        collate_fn=collate_critic, pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        collate_fn=collate_critic, pin_memory=device.type == "cuda",
    )

    # ── Optimizer ──
    lr = train_cfg.get("lr", 3e-4)
    optimizer = torch.optim.AdamW(
        critic.parameters(), lr=lr,
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    num_epochs = args.max_epochs or train_cfg["num_epochs"]
    total_steps = num_epochs * len(train_loader)
    warmup_steps = train_cfg.get("warmup_epochs", 2) * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps,
        min_factor=train_cfg.get("lr_min_factor", 0.01),
    )

    amp_enabled, amp_dtype, scaler = setup_amp(config.get("amp", {}), device)

    # ── Resume ──
    start_epoch = 0
    best_metric = 0.0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        critic.load_state_dict(ckpt["critic"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metric = ckpt.get("best_metric", 0.0)
        print(f"  Resumed from epoch {start_epoch}, best_metric={best_metric:.4f}")
        print(f"  Scheduler restored: {'yes' if 'scheduler' in ckpt else 'NO (will restart)'}")
        print(f"  Current lr: {optimizer.param_groups[0]['lr']:.2e}")

    # ── Training loop ──
    log_every = train_cfg.get("log_every", 50)
    patience = train_cfg.get("early_stop_patience", 10)
    no_improve = 0

    print(f"\nStarting training: {num_epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"Losses: InfoNCE (w={train_cfg.get('w_rank', 1.0)}) "
          f"+ PathContrastive (warmup@E{train_cfg.get('path_warmup_epoch', 0)}, "
          f"max={train_cfg.get('w_path_max', 0.3)}, "
          f"waypoints={train_cfg.get('path_num_waypoints', 5)})")
    print(f"Langevin: {train_cfg.get('langevin_steps', 30)} steps, "
          f"lr={train_cfg.get('langevin_lr', 0.1)}, tamed=True")

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        critic.train()
        tracker = MetricTracker()

        for batch_idx, batch in enumerate(train_loader):
            metrics = train_step(
                critic, batch, optimizer, scaler,
                device, amp_enabled, amp_dtype, train_cfg,
                epoch=epoch,
            )
            tracker.update(metrics)
            scheduler.step()

            if (batch_idx + 1) % log_every == 0:
                avg = tracker.get()
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  [E{epoch} S{batch_idx+1}] "
                    f"total={avg.get('total_loss', 0):.4f} "
                    f"rank_acc={avg.get('rank_acc', 0):.4f} "
                    f"cos_sim={avg.get('cos_sim_mean', 0):.4f} "
                    f"E_gap={avg.get('energy_gap', 0):.4f} "
                    f"path_viol={avg.get('path_violations', 0):.3f} "
                    f"w_path={avg.get('w_path', 0):.3f} "
                    f"lr={current_lr:.2e}"
                )

        elapsed = time.time() - t0
        train_avg = tracker.get()
        print(f"\n  Epoch {epoch} train ({elapsed:.1f}s):")
        for k, v in sorted(train_avg.items()):
            print(f"    {k}: {v:.4f}")

        # ── Evaluation ──
        critic.eval()
        val_tracker = MetricTracker()
        for batch in val_loader:
            vm = eval_step(
                critic, batch, device, amp_enabled, amp_dtype,
                langevin_steps=train_cfg.get("langevin_steps", 30),
                langevin_lr=train_cfg.get("langevin_lr", 0.1),
                noise_scale=train_cfg.get("noise_scale", 0.01),
            )
            val_tracker.update(vm)

        val_avg = val_tracker.get()
        print(f"  Epoch {epoch} val:")
        for k, v in sorted(val_avg.items()):
            print(f"    {k}: {v:.4f}")

        # Primary metric: rank_acc × cos_sim
        combined = val_avg.get("val_rank_acc", 0) * max(0, val_avg.get("val_cos_sim_mean", 0))

        if combined > best_metric:
            best_metric = combined
            no_improve = 0
            save_checkpoint(
                Path(out_cfg["checkpoint_dir"]) / "best_composite.pt",
                {
                    "critic": critic.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "val_metrics": val_avg,
                    "config": config,
                },
            )
            print(f"  >> New best: combined={combined:.4f} "
                  f"(rank_acc={val_avg.get('val_rank_acc', 0):.4f} "
                  f"× cos_sim={val_avg.get('val_cos_sim_mean', 0):.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping: no improvement for {patience} epochs")
                break

        # Periodic checkpoint
        if (epoch + 1) % train_cfg.get("checkpoint_every", 5) == 0:
            save_checkpoint(
                Path(out_cfg["checkpoint_dir"]) / f"composite_epoch_{epoch}.pt",
                {
                    "critic": critic.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                },
            )

    print(f"\nPhase 1B finished. Best combined metric: {best_metric:.4f}")
    print(f"Best checkpoint: {out_cfg['checkpoint_dir']}/best_composite.pt")
    print("\nNext steps:")
    print("  1. Verify rank_acc >= 0.95 and cos_sim >= 0.25")
    print("  2. Proceed to Phase 2 (Chain Head v2 on frozen composite critic)")


if __name__ == "__main__":
    main()
