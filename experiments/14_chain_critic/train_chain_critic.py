#!/usr/bin/env python3
"""
ChainCritic training — pure reranker for ChainGenerator pipeline.

Strictly follows lessons.md:
  - L1242: "NEVER activate all losses simultaneously — start with ranking only"
  - L1682: "NEVER use iterative navigation (Langevin, Flow, ODE)" → no Langevin even for monitoring
  - L1700: "Keep the trained critic as a RERANKER only"
  - L1225: "Unconstrained MLP + SiLU" for ranking
  - L62:   "Self-denoise critic CANNOT solve QA" → trained on (q, answer) pairs

Architecture: CompositeCritic = ConditionalAngularCritic + AnalyticalRadialGuard
Loss: Focal-InfoNCE ONLY — E(q, correct_answer) < E(q, distractor)
No path-contrastive, no direction_loss, no Langevin, no auxiliary losses.

The critic's SOLE purpose is best-of-N reranking of ChainGenerator candidates.

Usage:
    python experiments/14_chain_critic/train_chain_critic.py
    python experiments/14_chain_critic/train_chain_critic.py --config configs/chain_critic_config.json
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
    """
    Dataset for critic reranker training.

    Each sample has (v_question, v_answer, v_steps).
    We sample random answers from other samples as negatives.
    Context = mean of reasoning steps (or v_question if no steps).
    """

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

        # Context = mean of reasoning steps (semantic grounding)
        v_context = v_steps.mean(dim=0) if v_steps.shape[0] > 0 else v_q

        # Sample negatives: random answers from other samples
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


# ── Training step ────────────────────────────────────────────────

def train_step(
    critic: CompositeCritic,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    cfg: dict,
) -> dict[str, float]:
    """
    Single training step — Focal-InfoNCE only.

    No path-contrastive (lessons L1682: navigation is broken).
    No direction loss (lessons L20: 2nd-order dominance).
    No Langevin (lessons L1700: critic is reranker only).
    """
    v_q = batch["v_questions"].to(device)
    v_a = batch["v_answers"].to(device)
    v_ctx = batch["v_contexts"].to(device)
    v_neg = batch["v_negatives"].to(device)

    clip_grad = cfg.get("clip_grad_norm", 1.0)

    optimizer.zero_grad()

    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
        loss, rank_metrics = critic.compute_contrastive_loss(
            v_q, v_a, v_neg, v_context=v_ctx,
        )

    scaler.scale(loss).backward()

    if clip_grad > 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(critic.parameters(), clip_grad)

    scaler.step(optimizer)
    scaler.update()

    return {**rank_metrics, "loss": loss.item()}


# ── Eval step ────────────────────────────────────────────────────

@torch.no_grad()
def eval_step(
    critic: CompositeCritic,
    batch: dict,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    """
    Evaluation: ranking accuracy + energy gap.

    No Langevin — critic is a reranker, not a navigator.
    """
    v_q = batch["v_questions"].to(device)
    v_a = batch["v_answers"].to(device)
    v_ctx = batch["v_contexts"].to(device)
    v_neg = batch["v_negatives"].to(device)

    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
        # Energy for positive (correct answer)
        E_pos = critic(v_q, v_a, v_context=v_ctx)  # [B]

        # Energy for negatives
        B, N, D = v_neg.shape
        v_q_exp = v_q.unsqueeze(1).expand(B, N, D).reshape(B * N, D)
        v_ctx_exp = v_ctx.unsqueeze(1).expand(B, N, D).reshape(B * N, D)
        v_neg_flat = v_neg.reshape(B * N, D)
        E_neg = critic(v_q_exp, v_neg_flat, v_context=v_ctx_exp).reshape(B, N)

        # Ranking accuracy: E(positive) < E(negative) for all negatives
        rank_acc = (E_pos.unsqueeze(1) < E_neg).float().mean().item()
        # Per-sample rank accuracy (correct < ALL negatives)
        perfect_rank = (E_pos.unsqueeze(1) < E_neg).all(dim=1).float().mean().item()
        energy_gap = (E_neg.mean(dim=1) - E_pos).mean().item()

    return {
        "val_rank_acc": rank_acc,
        "val_perfect_rank": perfect_rank,
        "val_energy_gap": energy_gap,
        "val_E_pos_mean": E_pos.mean().item(),
        "val_E_neg_mean": E_neg.mean().item(),
        "val_E_pos_std": E_pos.std().item(),
        "val_E_neg_std": E_neg.std().item(),
    }


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ChainCritic Training — Pure Reranker for ChainGenerator"
    )
    parser.add_argument("--config", default="configs/chain_critic_config.json")
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
    print("ChainCritic Training — Pure Reranker for ChainGenerator")
    print(f"  Loss: Focal-InfoNCE ONLY (no path, no direction, no Langevin)")
    print(f"  Purpose: best-of-N reranking of generator candidates")
    print(f"  Device: {device}")
    print("=" * 60)

    # ── Build critic ──
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
    print(f"  Radial: analytical (0 params)")
    print(f"  lambda_radial: {comp_cfg.lambda_radial}")
    print(f"  target_norm: {rad_cfg.target_norm}")
    print(f"  temperature: {ang_cfg.temperature}")
    print(f"  focal_gamma: {ang_cfg.focal_gamma}")
    print(f"  num_negatives: {ang_cfg.num_negatives}")

    # ── Load data ──
    data_path = config["data"]["path"]
    print(f"\n  Loading data from {data_path}")
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
        if "critic" in ckpt:
            critic.load_state_dict(ckpt["critic"])
        elif "model" in ckpt:
            critic.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metric = ckpt.get("best_metric", 0.0)
        print(f"  Resumed from epoch {start_epoch}, best rank_acc={best_metric:.4f}")

    # ── Training loop ──
    log_every = train_cfg.get("log_every", 50)
    ckpt_every = train_cfg.get("checkpoint_every", 5)
    patience = train_cfg.get("early_stop_patience", 15)
    no_improve = 0

    print(f"\n  Epochs: {num_epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"  Loss: Focal-InfoNCE (tau={ang_cfg.temperature}, gamma={ang_cfg.focal_gamma})")
    print(f"  Negatives: {num_neg} per sample")
    print("=" * 60)

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        critic.train()
        tracker = MetricTracker()

        for step, batch in enumerate(train_loader):
            metrics = train_step(
                critic, batch, optimizer, scaler,
                device, amp_enabled, amp_dtype, train_cfg,
            )
            tracker.update(metrics)
            scheduler.step()

            if (step + 1) % log_every == 0:
                avg = tracker.get()
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"  [E{epoch} S{step+1}] "
                    f"loss={avg.get('loss', 0):.4f} "
                    f"rank_acc={avg.get('rank_acc', 0):.4f} "
                    f"E_gap={avg.get('energy_gap', 0):.4f} "
                    f"E_pos={avg.get('E_pos_mean', 0):.3f} "
                    f"E_neg={avg.get('E_neg_mean', 0):.3f} "
                    f"lr={lr_now:.2e}"
                )
                tracker.reset()

        elapsed = time.time() - t0

        # ── Validation ──
        critic.eval()
        val_tracker = MetricTracker()
        for batch in val_loader:
            vm = eval_step(critic, batch, device, amp_enabled, amp_dtype)
            val_tracker.update(vm)

        val_avg = val_tracker.get()
        val_rank = val_avg.get("val_rank_acc", 0)
        val_perfect = val_avg.get("val_perfect_rank", 0)

        print(
            f"\n  [E{epoch} VAL] "
            f"rank_acc={val_rank:.4f} "
            f"perfect_rank={val_perfect:.4f} "
            f"E_gap={val_avg.get('val_energy_gap', 0):.4f} "
            f"E_pos={val_avg.get('val_E_pos_mean', 0):.3f}±{val_avg.get('val_E_pos_std', 0):.3f} "
            f"E_neg={val_avg.get('val_E_neg_mean', 0):.3f}±{val_avg.get('val_E_neg_std', 0):.3f} "
            f"({elapsed:.1f}s)"
        )

        # ── Checkpointing (metric = rank_acc) ──
        improved = val_rank > best_metric
        if improved:
            best_metric = val_rank
            no_improve = 0
            save_checkpoint(
                Path(out_cfg["checkpoint_dir"]) / "best_chain_critic.pt",
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
            print(f"  ** New best: rank_acc={best_metric:.4f}")
        else:
            no_improve += 1

        if (epoch + 1) % ckpt_every == 0:
            save_checkpoint(
                Path(out_cfg["checkpoint_dir"]) / f"chain_critic_epoch_{epoch}.pt",
                {
                    "critic": critic.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "config": config,
                },
            )

        if no_improve >= patience:
            print(f"\n  Early stopping after {patience} epochs without improvement")
            break

    print(f"\nTraining complete. Best rank_acc={best_metric:.4f}")
    print(f"Checkpoints in: {out_cfg['checkpoint_dir']}")
    print(f"\nUsage: load best_chain_critic.pt in ChainGenerator GUI for reranking")


if __name__ == "__main__":
    main()
