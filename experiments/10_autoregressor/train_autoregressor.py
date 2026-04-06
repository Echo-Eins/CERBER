#!/usr/bin/env python3
"""
CEBM Autoregressor: Joint training of ConditionalCritic + Chain Head.

This replaces the old Stage 3 Phase B pipeline. Key differences:
  - ConditionalCritic (not self-denoise pairwise) — learns E(query, answer, context)
  - 4 training losses: contrastive + cosine + chain + REINFORCE text
  - HotpotQA dataset with (question, answer, reasoning_steps) triplets
  - System 1/2 switching with strict gating (Phase B semantics preserved)

Usage:
    py -3 experiments/10_autoregressor/train_autoregressor.py \
        --config configs/autoregressor_config.json

    # Quick test:
    py -3 experiments/10_autoregressor/train_autoregressor.py \
        --config configs/autoregressor_config.json --max-epochs 1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cebcm.models.conditional_critic import ConditionalCritic, ConditionalCriticConfig
from cebcm.models.chain_head import ChainHeadConfig, EBTChainHead
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

class AutoregressorDataset(Dataset):
    """
    Dataset for autoregressor training.

    Each sample = (v_question, v_answer, v_steps, question_text, answer_text, type).
    """

    def __init__(self, samples: list[dict], num_negatives: int = 7):
        self.samples = samples
        self.num_negatives = num_negatives

        # Pre-collect all answer vectors for negative sampling
        self._all_answers = torch.stack([s["v_answer"] for s in samples])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        v_q = s["v_question"]       # [D]
        v_a = s["v_answer"]         # [D]
        v_steps = s["v_steps"]      # [N_steps, D]

        # Context = mean of reasoning steps
        v_context = v_steps.mean(dim=0) if v_steps.shape[0] > 0 else v_q

        # Sample negatives (random answers from other samples)
        neg_indices = []
        while len(neg_indices) < self.num_negatives:
            j = random.randint(0, len(self.samples) - 1)
            if j != idx:
                neg_indices.append(j)
        v_negatives = self._all_answers[neg_indices]  # [N_neg, D]

        return {
            "v_question": v_q,
            "v_answer": v_a,
            "v_context": v_context,
            "v_negatives": v_negatives,
            "v_steps": v_steps,
            "question_text": s["question"],
            "answer_text": s["answer"],
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate with padding for variable-length reasoning steps."""
    v_questions = torch.stack([b["v_question"] for b in batch])
    v_answers = torch.stack([b["v_answer"] for b in batch])
    v_contexts = torch.stack([b["v_context"] for b in batch])
    v_negatives = torch.stack([b["v_negatives"] for b in batch])

    # Pad reasoning steps to max length in batch
    max_steps = max(b["v_steps"].shape[0] for b in batch)
    D = batch[0]["v_steps"].shape[1]
    v_steps_padded = torch.zeros(len(batch), max_steps, D)
    step_lengths = []
    for i, b in enumerate(batch):
        n = b["v_steps"].shape[0]
        v_steps_padded[i, :n] = b["v_steps"]
        step_lengths.append(n)

    return {
        "v_questions": v_questions,
        "v_answers": v_answers,
        "v_contexts": v_contexts,
        "v_negatives": v_negatives,
        "v_steps": v_steps_padded,
        "step_lengths": torch.tensor(step_lengths),
        "question_texts": [b["question_text"] for b in batch],
        "answer_texts": [b["answer_text"] for b in batch],
    }


# ── Chain data builder ───────────────────────────────────────────

def build_chain_batch_from_steps(
    v_steps: torch.Tensor,
    step_lengths: torch.Tensor,
    num_negatives: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build positive and negative chains from reasoning steps for Chain Head training.

    Returns:
        positives: [B, max_len, D]
        pos_lengths: [B]
        negatives: [B, N_neg, max_len, D]
        neg_lengths: [B, N_neg]
    """
    B, max_len, D = v_steps.shape
    device = v_steps.device

    positives = v_steps  # [B, max_len, D]
    pos_lengths = step_lengths  # [B]

    # Build negative chains: shuffle step order within each sample
    neg_list = []
    neg_length_list = []
    for _ in range(num_negatives):
        neg = v_steps.clone()
        for i in range(B):
            n = int(step_lengths[i].item())
            if n > 1:
                perm = torch.randperm(n)
                neg[i, :n] = neg[i, perm]
        neg_list.append(neg)
        neg_length_list.append(step_lengths.clone())

    negatives = torch.stack(neg_list, dim=1)  # [B, N_neg, max_len, D]
    neg_lengths = torch.stack(neg_length_list, dim=1)  # [B, N_neg]

    return positives, pos_lengths, negatives, neg_lengths


# ── Training functions ───────────────────────────────────────────

def train_step(
    critic: ConditionalCritic,
    chain_head: EBTChainHead,
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
    """Single training step with 4 losses."""
    v_q = batch["v_questions"].to(device)
    v_a = batch["v_answers"].to(device)
    v_ctx = batch["v_contexts"].to(device)
    v_neg = batch["v_negatives"].to(device)
    v_steps = batch["v_steps"].to(device)
    step_lengths = batch["step_lengths"].to(device)

    clip_grad = cfg.get("clip_grad_norm", 1.0)
    w_rank = cfg.get("w_rank", 1.0)
    w_cos = cfg.get("w_cos", 0.5)
    w_chain = cfg.get("w_chain", 0.3)
    w_text = cfg.get("w_text", 0.2)
    text_every = cfg.get("text_loss_every", 20)
    langevin_steps = cfg.get("langevin_steps", 10)
    langevin_lr = cfg.get("langevin_lr", 0.01)

    optimizer.zero_grad()

    # Cache v_init for potential REINFORCE re-run (BUG 7 fix)
    v_init_cached = (v_q + torch.randn_like(v_q) * 0.02).detach()

    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
        # LOSS 1: Contrastive ranking
        loss_rank, rank_metrics = critic.compute_contrastive_loss(
            v_q, v_a, v_neg, v_context=v_ctx,
        )

        # LOSS 2: Energy-based cosine surrogate
        # Run Langevin (detached) to assess current critic quality,
        # then add a differentiable surrogate that pushes E(target) < E(predicted).
        v_current = v_init_cached.clone()
        with torch.enable_grad():
            for _ in range(langevin_steps):
                v_current = v_current.detach().requires_grad_(True)
                e = critic(v_q, v_current, v_context=v_ctx)
                grad_ld = torch.autograd.grad(e.sum(), v_current)[0]
                v_current = (v_current - langevin_lr * grad_ld).detach()

        v_predicted = v_current.detach()

        # Cosine evaluation metric (info only, no gradient)
        cos_metrics_raw = critic.compute_cosine_loss(v_predicted, v_a)
        _, cos_metrics = cos_metrics_raw

        # Differentiable surrogate: critic should assign lower energy to
        # the correct answer than to what Langevin converged on.
        # This trains the critic's energy landscape, not the Langevin trajectory.
        E_target = critic(v_q, v_a, v_context=v_ctx)            # [B]
        E_predicted = critic(v_q, v_predicted, v_context=v_ctx)  # [B] (detached candidate)
        # Hinge: E_target should be at least margin below E_predicted
        loss_cos = F.relu(E_target - E_predicted + 0.1).mean()

        # LOSS 3: Chain Head InfoNCE (keep chain ordering quality)
        if v_steps.shape[1] >= 3 and (step_lengths >= 3).any():
            positives, pos_lens, negatives, neg_lens = build_chain_batch_from_steps(
                v_steps, step_lengths, num_negatives=3,
            )
            # Filter to samples with enough steps
            mask = pos_lens >= 3
            if mask.any():
                loss_chain_raw, chain_metrics = chain_head.compute_infonce_loss(
                    positives[mask], negatives[mask],
                    pos_lengths=pos_lens[mask],
                    neg_lengths=neg_lens[mask],
                )
                loss_chain = loss_chain_raw
            else:
                loss_chain = torch.tensor(0.0, device=device)
                chain_metrics = {"chain_loss": 0.0, "chain_rank_acc": 0.0}
        else:
            loss_chain = torch.tensor(0.0, device=device)
            chain_metrics = {"chain_loss": 0.0, "chain_rank_acc": 0.0}

        # LOSS 4: REINFORCE text loss (periodic, requires SONAR decoder)
        # Integrated into main loss for proper AMP scaler handling (BUG 3 fix)
        text_metrics = {"text_reward_mean": 0.0}
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

    # If text loss is active, run differentiable Langevin and compute REINFORCE
    if do_text:
        # Re-run Langevin with create_graph=True for gradient flow
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
            v_current_g = v_init_cached.clone()
            for _ in range(langevin_steps):
                v_current_g = v_current_g.detach().requires_grad_(True)
                e = critic(v_q, v_current_g, v_context=v_ctx)
                grad_g = torch.autograd.grad(e.sum(), v_current_g, create_graph=True)[0]
                v_current_g = v_current_g - langevin_lr * grad_g

            reinforce_loss, reinforce_metrics = compute_reinforce_loss(
                v_current_g, v_a, reward, baseline=baseline_val,
            )
            text_metrics.update(reinforce_metrics)

    # Combined loss with proper AMP (BUG 3 fix: single backward+step)
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
        total_loss = w_rank * loss_rank + w_cos * loss_cos + w_chain * loss_chain
        if do_text:
            total_loss = total_loss + w_text * reinforce_loss

    scaler.scale(total_loss).backward()

    if clip_grad > 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            list(critic.parameters()) + list(chain_head.parameters()),
            clip_grad,
        )

    scaler.step(optimizer)
    scaler.update()

    # Aggregate metrics
    metrics = {
        **rank_metrics,
        **cos_metrics,
        **{f"chain_{k}" if not k.startswith("chain_") else k: v for k, v in chain_metrics.items()},
        **text_metrics,
        "total_loss": total_loss.item(),
        "loss_rank": loss_rank.item(),
        "loss_cos": loss_cos.item(),
        "loss_chain": loss_chain.item() if isinstance(loss_chain, torch.Tensor) else 0.0,
    }
    return metrics


def eval_step(
    critic: ConditionalCritic,
    chain_head: EBTChainHead,
    batch: dict,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    langevin_steps: int = 10,
    langevin_lr: float = 0.01,
) -> dict[str, float]:
    """Evaluation step — compute metrics without gradients."""
    v_q = batch["v_questions"].to(device)
    v_a = batch["v_answers"].to(device)
    v_ctx = batch["v_contexts"].to(device)
    v_neg = batch["v_negatives"].to(device)

    with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
        # Ranking accuracy
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

    # Mini-langevin for cosine assessment (BUG 5 fix: enable_grad for autograd.grad)
    with torch.enable_grad():
        v_current = (v_q + torch.randn_like(v_q) * 0.02).detach()
        for _ in range(langevin_steps):
            v_current = v_current.detach().requires_grad_(True)
            e = critic(v_q, v_current, v_context=v_ctx)
            grad = torch.autograd.grad(e.sum(), v_current)[0]
            v_current = (v_current - langevin_lr * grad).detach()

    with torch.no_grad():
        cos_sim = F.cosine_similarity(v_current, v_a, dim=-1)

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
    parser = argparse.ArgumentParser(description="CEBM Autoregressor Training")
    parser.add_argument("--config", default="configs/autoregressor_config.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs")
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
    print("CEBM Autoregressor: ConditionalCritic + Chain Head Training")
    print(f"Device: {device}")
    print("=" * 60)

    # ── Build models ──
    critic_cfg = ConditionalCriticConfig(**config.get("critic", {}))
    critic = ConditionalCritic(critic_cfg).to(device)
    print(f"  ConditionalCritic: {critic.num_params:,} parameters")

    chain_cfg = ChainHeadConfig(**config.get("chain_head", {}))
    chain_head = EBTChainHead(chain_cfg).to(device)
    print(f"  Chain Head: {chain_head.num_params:,} parameters")

    # Load Chain Head checkpoint if available
    chain_ckpt = config.get("chain_head_checkpoint")
    if chain_ckpt and Path(chain_ckpt).exists():
        ckpt = torch.load(chain_ckpt, map_location=device, weights_only=False)
        chain_head.load_state_dict(ckpt["model"])
        print(f"  Chain Head loaded from {chain_ckpt}")

    total_params = critic.num_params + chain_head.num_params
    print(f"  Total trainable: {total_params:,} parameters")

    # ── Load data ──
    data_path = config["data"]["path"]
    print(f"  Loading data from {data_path}")
    data = torch.load(data_path, map_location="cpu", weights_only=False)

    num_neg = config.get("critic", {}).get("num_negatives", 7)
    train_ds = AutoregressorDataset(data["train"], num_negatives=num_neg)
    val_ds = AutoregressorDataset(data["val"], num_negatives=num_neg)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_cfg = config["training"]
    batch_size = train_cfg["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=config["data"].get("num_workers", 4),
        collate_fn=collate_fn, pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        collate_fn=collate_fn, pin_memory=device.type == "cuda",
    )

    # ── Optimizer ──
    optimizer = torch.optim.AdamW([
        {"params": critic.parameters(), "lr": train_cfg["critic_lr"]},
        {"params": chain_head.parameters(), "lr": train_cfg["chain_head_lr"]},
    ], weight_decay=train_cfg.get("weight_decay", 1e-4))

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
        chain_head.load_state_dict(ckpt["chain_head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metric = ckpt.get("best_metric", 0.0)
        print(f"  Resumed from epoch {start_epoch}, best_metric={best_metric:.4f}")

    # ── Training loop ──
    log_every = train_cfg.get("log_every", 50)
    patience = train_cfg.get("early_stop_patience", 10)
    no_improve = 0

    print(f"\nStarting training: {num_epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"Loss weights: rank={train_cfg.get('w_rank', 1.0)}, cos={train_cfg.get('w_cos', 0.5)}, "
          f"chain={train_cfg.get('w_chain', 0.3)}, text={train_cfg.get('w_text', 0.2)}")

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        critic.train()
        chain_head.train()
        tracker = MetricTracker()

        for batch_idx, batch in enumerate(train_loader):
            step_global = epoch * len(train_loader) + batch_idx
            metrics = train_step(
                critic, chain_head, batch, optimizer, scaler,
                device, amp_enabled, amp_dtype, train_cfg,
                reward_baseline, sonar, step_global,
            )
            tracker.update(metrics)
            scheduler.step()

            if (batch_idx + 1) % log_every == 0:
                avg = tracker.get()
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  [E{epoch} S{batch_idx+1}] "
                    f"total={avg.get('total_loss', 0):.4f} "
                    f"rank={avg.get('loss_rank', 0):.4f} "
                    f"cos={avg.get('loss_cos', 0):.4f} "
                    f"chain={avg.get('loss_chain', 0):.4f} "
                    f"rank_acc={avg.get('rank_acc', 0):.4f} "
                    f"cos_sim={avg.get('cos_sim_mean', 0):.4f} "
                    f"text_r={avg.get('text_reward_mean', 0):.4f} "
                    f"lr={lr:.2e}"
                )

        elapsed = time.time() - t0
        train_avg = tracker.get()
        print(f"\n  Epoch {epoch} train ({elapsed:.1f}s):")
        for k, v in sorted(train_avg.items()):
            print(f"    {k}: {v:.4f}")

        # ── Evaluation ──
        critic.eval()
        chain_head.eval()
        val_tracker = MetricTracker()
        for batch in val_loader:
            vm = eval_step(
                critic, chain_head, batch,
                device, amp_enabled, amp_dtype,
                langevin_steps=train_cfg.get("langevin_steps", 10),
                langevin_lr=train_cfg.get("langevin_lr", 0.01),
            )
            val_tracker.update(vm)

        val_avg = val_tracker.get()
        print(f"  Epoch {epoch} val:")
        for k, v in sorted(val_avg.items()):
            print(f"    {k}: {v:.4f}")

        # Combined metric: rank_acc * cos_sim
        combined = val_avg.get("val_rank_acc", 0) * val_avg.get("val_cos_sim_mean", 0)

        if combined > best_metric:
            best_metric = combined
            no_improve = 0
            save_checkpoint(
                Path(out_cfg["checkpoint_dir"]) / "best_autoregressor.pt",
                {
                    "critic": critic.state_dict(),
                    "chain_head": chain_head.state_dict(),
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
                Path(out_cfg["checkpoint_dir"]) / f"autoregressor_epoch_{epoch}.pt",
                {
                    "critic": critic.state_dict(),
                    "chain_head": chain_head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                },
            )

    print(f"\nTraining finished. Best combined metric: {best_metric:.4f}")
    print(f"Best checkpoint: {out_cfg['checkpoint_dir']}/best_autoregressor.pt")


if __name__ == "__main__":
    main()
