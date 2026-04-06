#!/usr/bin/env python3
"""
Phase 1: ConditionalCritic solo training on HotpotQA.

Trains ONLY the ConditionalCritic (no Chain Head) with 3 losses:
  1. Focal-InfoNCE contrastive: E(q, answer, ctx) < E(q, distractor, ctx)
  2. Energy hinge surrogate: relu(E_target - E_predicted + margin)
  3. REINFORCE text reward (periodic): decode→re-encode→cosine reward

After this phase achieves rank_acc ≥ 0.95 and cos_sim ≥ 0.25,
proceed to Phase 2 (Chain Head v2 training on frozen critic).

Usage:
    python experiments/11_critic/train_critic.py --config configs/critic_config.json
    python experiments/11_critic/train_critic.py --config configs/critic_config.json --max-epochs 5
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

from cebcm.models.conditional_critic import ConditionalCritic, ConditionalCriticConfig
from cebcm.training.text_reward import (
    compute_text_reward,
    compute_reinforce_loss,
    RewardBaseline,
)
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
    """Dataset for critic-only training. Returns QA pairs with negatives."""

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

        # Context = mean of reasoning steps (simple aggregation)
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
            "question_text": s["question"],
            "answer_text": s["answer"],
        }


def collate_critic(batch: list[dict]) -> dict:
    """Simple collation for QA pairs."""
    return {
        "v_questions": torch.stack([b["v_question"] for b in batch]),
        "v_answers": torch.stack([b["v_answer"] for b in batch]),
        "v_contexts": torch.stack([b["v_context"] for b in batch]),
        "v_negatives": torch.stack([b["v_negatives"] for b in batch]),
        "question_texts": [b["question_text"] for b in batch],
        "answer_texts": [b["answer_text"] for b in batch],
    }


# ── Training ─────────────────────────────────────────────────────

def train_step(
    critic: ConditionalCritic,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    cfg: dict,
    reward_baseline: RewardBaseline,
    sonar=None,
    step_global: int = 0,
) -> dict[str, float]:
    """Single training step: 3 losses (contrastive + hinge + REINFORCE)."""
    v_q = batch["v_questions"].to(device)
    v_a = batch["v_answers"].to(device)
    v_ctx = batch["v_contexts"].to(device)
    v_neg = batch["v_negatives"].to(device)

    clip_grad = cfg.get("clip_grad_norm", 1.0)
    w_rank = cfg.get("w_rank", 1.0)
    w_cos = cfg.get("w_cos", 0.5)
    w_text = cfg.get("w_text", 0.2)
    text_every = cfg.get("text_loss_every", 20)
    langevin_steps = cfg.get("langevin_steps", 30)
    langevin_lr = cfg.get("langevin_lr", 0.3)
    hinge_margin = cfg.get("hinge_margin", 0.3)

    optimizer.zero_grad()

    # Cache v_init for REINFORCE re-run
    v_init = (v_q + torch.randn_like(v_q) * 0.02).detach()

    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
        # LOSS 1: Focal-InfoNCE contrastive ranking
        loss_rank, rank_metrics = critic.compute_contrastive_loss(
            v_q, v_a, v_neg, v_context=v_ctx,
        )

        # LOSS 2: Energy hinge surrogate
        # Run Langevin (detached) to see where critic currently converges
        with torch.enable_grad():
            v_current = v_init.clone()
            for _ in range(langevin_steps):
                v_current = v_current.detach().requires_grad_(True)
                e = critic(v_q, v_current, v_context=v_ctx)
                grad_ld = torch.autograd.grad(e.sum(), v_current)[0]
                v_current = (v_current - langevin_lr * grad_ld).detach()

        v_predicted = v_current.detach()

        # Cosine metric (info only)
        cos_sim = F.cosine_similarity(v_predicted, v_a, dim=-1)
        cos_metrics = {
            "cos_sim_mean": cos_sim.mean().item(),
            "cos_sim_min": cos_sim.min().item(),
            "cos_sim_max": cos_sim.max().item(),
        }

        # Hinge: push E(answer) below E(predicted) by margin
        E_target = critic(v_q, v_a, v_context=v_ctx)
        E_predicted = critic(v_q, v_predicted, v_context=v_ctx)
        loss_hinge = F.relu(E_target - E_predicted + hinge_margin).mean()

        # LOSS 3: REINFORCE text reward (periodic)
        text_metrics: dict[str, float] = {"text_reward_mean": 0.0}
        reinforce_loss = torch.tensor(0.0, device=device)
        do_text = sonar is not None and step_global % text_every == 0 and w_text > 0

        if do_text:
            with torch.no_grad():
                reward, text_metrics = compute_text_reward(
                    v_predicted, v_a,
                    target_texts=batch["answer_texts"],
                    sonar=sonar,
                )
                baseline_val = reward_baseline.update(reward.mean().item())

    # Differentiable Langevin for REINFORCE (outside autocast for memory)
    if do_text:
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
            v_g = v_init.clone()
            for _ in range(langevin_steps):
                v_g = v_g.detach().requires_grad_(True)
                e = critic(v_q, v_g, v_context=v_ctx)
                grad_g = torch.autograd.grad(e.sum(), v_g, create_graph=True)[0]
                v_g = v_g - langevin_lr * grad_g

            reinforce_loss, reinforce_metrics = compute_reinforce_loss(
                v_g, v_a, reward, baseline=baseline_val,
            )
            text_metrics.update(reinforce_metrics)

    # Combined loss — single backward
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
        total_loss = w_rank * loss_rank + w_cos * loss_hinge
        if do_text:
            total_loss = total_loss + w_text * reinforce_loss

    scaler.scale(total_loss).backward()

    if clip_grad > 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(critic.parameters(), clip_grad)

    scaler.step(optimizer)
    scaler.update()

    metrics = {
        **rank_metrics,
        **cos_metrics,
        **text_metrics,
        "total_loss": total_loss.item(),
        "loss_rank": loss_rank.item(),
        "loss_hinge": loss_hinge.item(),
        "E_target_mean": E_target.mean().item(),
        "E_predicted_mean": E_predicted.mean().item(),
    }
    return metrics


@torch.no_grad()
def eval_step(
    critic: ConditionalCritic,
    batch: dict,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    langevin_steps: int = 30,
    langevin_lr: float = 0.3,
) -> dict[str, float]:
    """Evaluation step — ranking accuracy + Langevin cosine assessment."""
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
        E_pos_mean = E_pos.mean().item()
        E_neg_mean = E_neg.mean().item()

    # Mini-Langevin for cosine assessment
    with torch.enable_grad():
        v_current = (v_q + torch.randn_like(v_q) * 0.02).detach()
        for _ in range(langevin_steps):
            v_current = v_current.detach().requires_grad_(True)
            e = critic(v_q, v_current, v_context=v_ctx)
            grad = torch.autograd.grad(e.sum(), v_current)[0]
            v_current = (v_current - langevin_lr * grad).detach()

    cos_sim = F.cosine_similarity(v_current.detach(), v_a, dim=-1)

    return {
        "val_rank_acc": rank_acc,
        "val_energy_gap": energy_gap,
        "val_E_pos_mean": E_pos_mean,
        "val_E_neg_mean": E_neg_mean,
        "val_cos_sim_mean": cos_sim.mean().item(),
        "val_cos_sim_min": cos_sim.min().item(),
        "val_cos_sim_max": cos_sim.max().item(),
    }


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 1: ConditionalCritic Solo Training")
    parser.add_argument("--config", default="configs/critic_config.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--skip-text-loss", action="store_true",
                        help="Skip REINFORCE text loss (no SONAR decoder needed)")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    setup_seed(config.get("seed", 42), device)

    out_cfg = config["output"]
    for d in [out_cfg["output_dir"], out_cfg["checkpoint_dir"], out_cfg["logs_dir"]]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 1: ConditionalCritic Solo Training")
    print(f"Device: {device}")
    print("=" * 60)

    # ── Build critic ──
    critic_cfg = ConditionalCriticConfig(**config.get("critic", {}))
    critic = ConditionalCritic(critic_cfg).to(device)
    print(f"  ConditionalCritic: {critic.num_params:,} parameters")

    # ── Load data ──
    data_path = config["data"]["path"]
    print(f"  Loading data from {data_path}")
    data = torch.load(data_path, map_location="cpu", weights_only=False)

    num_neg = config.get("critic", {}).get("num_negatives", 7)
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

    # ── SONAR (for REINFORCE text loss) ──
    sonar = None
    if not args.skip_text_loss and train_cfg.get("w_text", 0) > 0:
        try:
            from cebcm.models.sonar_wrapper import SONARWrapper
            sonar = SONARWrapper(device=str(device))
            print("  SONAR loaded for text loss")
        except Exception as e:
            print(f"  WARNING: Could not load SONAR ({e}). Text loss disabled.")
            sonar = None

    reward_baseline = RewardBaseline(decay=0.99)

    # ── Resume ──
    start_epoch = 0
    best_metric = 0.0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        critic.load_state_dict(ckpt["critic"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metric = ckpt.get("best_metric", 0.0)
        print(f"  Resumed from epoch {start_epoch}, best_metric={best_metric:.4f}")

    # ── Training loop ──
    log_every = train_cfg.get("log_every", 50)
    patience = train_cfg.get("early_stop_patience", 10)
    no_improve = 0
    langevin_steps = train_cfg.get("langevin_steps", 30)
    langevin_lr = train_cfg.get("langevin_lr", 0.3)

    print(f"\nStarting training: {num_epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"Loss weights: rank={train_cfg.get('w_rank', 1.0)}, "
          f"hinge={train_cfg.get('w_cos', 0.5)}, text={train_cfg.get('w_text', 0.2)}")
    print(f"Langevin: {langevin_steps} steps, lr={langevin_lr}, "
          f"hinge_margin={train_cfg.get('hinge_margin', 0.3)}")

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        critic.train()
        tracker = MetricTracker()

        for batch_idx, batch in enumerate(train_loader):
            step_global = epoch * len(train_loader) + batch_idx
            metrics = train_step(
                critic, batch, optimizer, scaler,
                device, amp_enabled, amp_dtype, train_cfg,
                reward_baseline, sonar, step_global,
            )
            tracker.update(metrics)

            # scheduler.step() AFTER optimizer.step() (fix PyTorch warning)
            scheduler.step()

            if (batch_idx + 1) % log_every == 0:
                avg = tracker.get()
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  [E{epoch} S{batch_idx+1}] "
                    f"total={avg.get('total_loss', 0):.4f} "
                    f"rank={avg.get('loss_rank', 0):.4f} "
                    f"hinge={avg.get('loss_hinge', 0):.4f} "
                    f"rank_acc={avg.get('rank_acc', 0):.4f} "
                    f"cos_sim={avg.get('cos_sim_mean', 0):.4f} "
                    f"text_r={avg.get('text_reward_mean', 0):.4f} "
                    f"E_gap={avg.get('energy_gap', 0):.4f} "
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
                langevin_steps=langevin_steps, langevin_lr=langevin_lr,
            )
            val_tracker.update(vm)

        val_avg = val_tracker.get()
        print(f"  Epoch {epoch} val:")
        for k, v in sorted(val_avg.items()):
            print(f"    {k}: {v:.4f}")

        # Primary metric: rank_acc × cos_sim (both must improve)
        combined = val_avg.get("val_rank_acc", 0) * val_avg.get("val_cos_sim_mean", 0)

        if combined > best_metric:
            best_metric = combined
            no_improve = 0
            save_checkpoint(
                Path(out_cfg["checkpoint_dir"]) / "best_critic.pt",
                {
                    "critic": critic.state_dict(),
                    "optimizer": optimizer.state_dict(),
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
                Path(out_cfg["checkpoint_dir"]) / f"critic_epoch_{epoch}.pt",
                {
                    "critic": critic.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                },
            )

    print(f"\nPhase 1 finished. Best combined metric: {best_metric:.4f}")
    print(f"Best checkpoint: {out_cfg['checkpoint_dir']}/best_critic.pt")
    print("\nNext: run Phase 2 (Chain Head v2 training) with this checkpoint.")


if __name__ == "__main__":
    main()
