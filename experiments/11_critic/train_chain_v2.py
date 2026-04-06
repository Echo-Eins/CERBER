#!/usr/bin/env python3
"""
Phase 2: Chain Head v2 training on frozen ConditionalCritic.

Trains Chain Head FROM SCRATCH using the new critic's energy landscape.
The old Chain Head was trained on self-denoise critic (minimum at query),
which is incompatible with the new conditional critic (minimum at answer).

2 losses:
  1. Chain InfoNCE: rank positive chains (correct order) above
     shuffled/corrupted chains — same as before but fresh weights
  2. Critic-guided chain quality bonus: for each chain context,
     run mini-Langevin via frozen critic → cos(v_out, v_answer) as reward.
     Chains whose context helps the critic find the answer get rewarded.

Usage:
    python experiments/11_critic/train_chain_v2.py --config configs/chain_v2_config.json
    python experiments/11_critic/train_chain_v2.py --config configs/chain_v2_config.json \\
        --critic-checkpoint experiments/11_critic/output/checkpoints/best_critic.pt
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cebcm.models.chain_head import ChainHeadConfig, EBTChainHead
from cebcm.models.conditional_critic import ConditionalCritic, ConditionalCriticConfig
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

class ChainDataset(Dataset):
    """Dataset for chain head training with QA reasoning chains."""

    def __init__(self, samples: list[dict], num_negatives: int = 7):
        self.samples = [s for s in samples if s["v_steps"].shape[0] >= 3]
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        v_q = s["v_question"]
        v_a = s["v_answer"]
        v_steps = s["v_steps"]  # [N_steps, D]

        return {
            "v_question": v_q,
            "v_answer": v_a,
            "v_steps": v_steps,
            "question_text": s["question"],
            "answer_text": s["answer"],
        }


def collate_chain(batch: list[dict]) -> dict:
    """Collation with step padding."""
    max_steps = max(b["v_steps"].shape[0] for b in batch)
    D = batch[0]["v_steps"].shape[1]
    v_steps_padded = torch.zeros(len(batch), max_steps, D)
    step_lengths = []
    for i, b in enumerate(batch):
        n = b["v_steps"].shape[0]
        v_steps_padded[i, :n] = b["v_steps"]
        step_lengths.append(n)

    return {
        "v_questions": torch.stack([b["v_question"] for b in batch]),
        "v_answers": torch.stack([b["v_answer"] for b in batch]),
        "v_steps": v_steps_padded,
        "step_lengths": torch.tensor(step_lengths),
        "question_texts": [b["question_text"] for b in batch],
        "answer_texts": [b["answer_text"] for b in batch],
    }


# ── Negative chain generation ────────────────────────────────────

def build_chain_negatives(
    v_steps: torch.Tensor,
    step_lengths: torch.Tensor,
    num_negatives: int = 7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build positive and negative chains from reasoning steps.

    Negative types (harder mix than old Stage 3):
      - Adjacent swap (2): swap 1-2 neighboring pairs
      - Block shuffle (2): divide into 2-3 blocks, shuffle blocks
      - Full shuffle (2): full random permutation
      - Truncated (1): remove last 1-2 steps

    Returns:
        positives: [B, max_len, D]
        pos_lengths: [B]
        negatives: [B, N_neg, max_len, D]
        neg_lengths: [B, N_neg]
    """
    B, max_len, D = v_steps.shape
    device = v_steps.device

    positives = v_steps
    pos_lengths = step_lengths

    neg_list = []
    neg_len_list = []

    def _make_neg(perm_fn):
        """Helper to create a batch of negatives with given permutation."""
        neg = v_steps.clone()
        n_lens = step_lengths.clone()
        for i in range(B):
            n = int(step_lengths[i].item())
            if n > 1:
                new_order, new_n = perm_fn(n)
                neg[i, :new_n] = v_steps[i, new_order[:new_n]]
                n_lens[i] = new_n
        return neg, n_lens

    # Type 1: Adjacent swaps (2 negatives)
    for _ in range(2):
        def adj_swap(n):
            order = list(range(n))
            # Swap 1-2 adjacent pairs
            num_swaps = min(2, n - 1)
            for _ in range(num_swaps):
                j = random.randint(0, n - 2)
                order[j], order[j + 1] = order[j + 1], order[j]
            return order, n
        neg, n_lens = _make_neg(adj_swap)
        neg_list.append(neg)
        neg_len_list.append(n_lens)

    # Type 2: Block shuffle (2 negatives)
    for _ in range(2):
        def block_shuffle(n):
            if n < 4:
                order = list(range(n))
                random.shuffle(order)
                return order, n
            mid = n // 2
            # Swap the two halves
            order = list(range(mid, n)) + list(range(mid))
            return order, n
        neg, n_lens = _make_neg(block_shuffle)
        neg_list.append(neg)
        neg_len_list.append(n_lens)

    # Type 3: Full shuffle (2 negatives)
    for _ in range(2):
        def full_shuffle(n):
            order = list(range(n))
            random.shuffle(order)
            return order, n
        neg, n_lens = _make_neg(full_shuffle)
        neg_list.append(neg)
        neg_len_list.append(n_lens)

    # Type 4: Truncated (1 negative)
    def truncate(n):
        new_n = max(2, n - random.randint(1, min(2, n - 2)))
        return list(range(new_n)), new_n
    neg, n_lens = _make_neg(truncate)
    neg_list.append(neg)
    neg_len_list.append(n_lens)

    negatives = torch.stack(neg_list, dim=1)  # [B, N_neg, max_len, D]
    neg_lengths = torch.stack(neg_len_list, dim=1)  # [B, N_neg]

    return positives, pos_lengths, negatives, neg_lengths


# ── Critic-guided chain quality ──────────────────────────────────

@torch.no_grad()
def compute_critic_chain_bonus(
    v_questions: torch.Tensor,
    v_answers: torch.Tensor,
    v_steps: torch.Tensor,
    step_lengths: torch.Tensor,
    critic: ConditionalCritic,
    langevin_steps: int = 30,
    langevin_lr: float = 0.3,
) -> torch.Tensor:
    """
    Compute how well each chain's context helps the critic find the answer.

    For each sample:
    1. v_context = mean(chain steps)
    2. Run Langevin with critic(v_q, v, v_context)
    3. cos_bonus = cos(v_langevin_output, v_answer)

    Returns:
        bonus: [B] cosine similarity scores (higher = better chain context)
    """
    B, max_len, D = v_steps.shape
    device = v_steps.device

    # Compute context from chains
    v_context = torch.zeros(B, D, device=device)
    for i in range(B):
        n = int(step_lengths[i].item())
        v_context[i] = v_steps[i, :n].mean(dim=0)

    # Run mini-Langevin with critic
    with torch.enable_grad():
        v_current = (v_questions + torch.randn_like(v_questions) * 0.02).detach()
        for _ in range(langevin_steps):
            v_current = v_current.detach().requires_grad_(True)
            e = critic(v_questions, v_current, v_context=v_context)
            grad = torch.autograd.grad(e.sum(), v_current)[0]
            v_current = (v_current - langevin_lr * grad).detach()

    # Cosine to answer = quality of this chain's context
    bonus = F.cosine_similarity(v_current.detach(), v_answers, dim=-1)
    return bonus


# ── Training ─────────────────────────────────────────────────────

def train_step(
    chain_head: EBTChainHead,
    critic: ConditionalCritic,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    cfg: dict,
    step_global: int = 0,
) -> dict[str, float]:
    """Single training step: Chain InfoNCE + critic bonus."""
    v_q = batch["v_questions"].to(device)
    v_a = batch["v_answers"].to(device)
    v_steps = batch["v_steps"].to(device)
    step_lengths = batch["step_lengths"].to(device)

    clip_grad = cfg.get("clip_grad_norm", 1.0)
    w_nce = cfg.get("w_chain_nce", 1.0)
    w_bonus = cfg.get("w_critic_bonus", 0.3)
    bonus_every = cfg.get("critic_bonus_every", 5)
    num_neg = cfg.get("num_chain_negatives", 7)

    optimizer.zero_grad()

    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
        # Build chain negatives
        positives, pos_lens, negatives, neg_lengths = build_chain_negatives(
            v_steps, step_lengths, num_negatives=num_neg,
        )

        # LOSS 1: Chain InfoNCE
        loss_nce, chain_metrics = chain_head.compute_infonce_loss(
            positives, negatives,
            pos_lengths=pos_lens,
            neg_lengths=neg_lengths,
        )

        # LOSS 2: Critic-guided bonus (periodic to save compute)
        bonus_metrics: dict[str, float] = {"critic_cos_bonus": 0.0}
        loss_bonus = torch.tensor(0.0, device=device)

        if step_global % bonus_every == 0 and w_bonus > 0:
            # Get chain energy for positives and use critic cos as supervision
            bonus = compute_critic_chain_bonus(
                v_q, v_a, v_steps, step_lengths, critic,
                langevin_steps=cfg.get("langevin_steps", 30),
                langevin_lr=cfg.get("langevin_lr", 0.3),
            )

            # Chain head energy for positive chains
            pos_energy = chain_head(positives, lengths=pos_lens)  # [B]

            # Lower chain energy should correlate with higher critic bonus
            # Use Spearman-style loss: chain_E should be low when bonus is high
            # Proxy: -(bonus * (-chain_E)) = bonus * chain_E → minimize
            # When bonus is high (good context) → push chain_E lower
            loss_bonus = (bonus.detach() * pos_energy).mean()

            bonus_metrics = {
                "critic_cos_bonus": bonus.mean().item(),
                "critic_cos_bonus_std": bonus.std().item(),
                "chain_E_pos_with_bonus": pos_energy.mean().item(),
            }

        total_loss = w_nce * loss_nce + w_bonus * loss_bonus

    scaler.scale(total_loss).backward()

    if clip_grad > 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(chain_head.parameters(), clip_grad)

    scaler.step(optimizer)
    scaler.update()

    metrics = {
        **chain_metrics,
        **bonus_metrics,
        "total_loss": total_loss.item(),
        "loss_nce": loss_nce.item(),
        "loss_bonus": loss_bonus.item() if isinstance(loss_bonus, torch.Tensor) else 0.0,
    }
    return metrics


@torch.no_grad()
def eval_step(
    chain_head: EBTChainHead,
    critic: ConditionalCritic,
    batch: dict,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    cfg: dict,
) -> dict[str, float]:
    """Evaluation: chain ranking + critic cosine bonus."""
    v_q = batch["v_questions"].to(device)
    v_a = batch["v_answers"].to(device)
    v_steps = batch["v_steps"].to(device)
    step_lengths = batch["step_lengths"].to(device)

    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
        positives, pos_lens, negatives, neg_lengths = build_chain_negatives(
            v_steps, step_lengths, num_negatives=7,
        )
        _, chain_metrics = chain_head.compute_infonce_loss(
            positives, negatives,
            pos_lengths=pos_lens,
            neg_lengths=neg_lengths,
        )

    # Critic cosine bonus on val set
    bonus = compute_critic_chain_bonus(
        v_q, v_a, v_steps, step_lengths, critic,
        langevin_steps=cfg.get("langevin_steps", 30),
        langevin_lr=cfg.get("langevin_lr", 0.3),
    )

    return {
        **{f"val_{k}": v for k, v in chain_metrics.items()},
        "val_critic_cos_bonus": bonus.mean().item(),
    }


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Chain Head v2 Training")
    parser.add_argument("--config", default="configs/chain_v2_config.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--critic-checkpoint", default=None,
                        help="Override critic checkpoint path from config")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    setup_seed(config.get("seed", 42), device)

    out_cfg = config["output"]
    for d in [out_cfg["output_dir"], out_cfg["checkpoint_dir"], out_cfg["logs_dir"]]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 2: Chain Head v2 Training (on frozen ConditionalCritic)")
    print(f"Device: {device}")
    print("=" * 60)

    # ── Load frozen critic ──
    critic_ckpt_path = args.critic_checkpoint or config.get("critic_checkpoint")
    if not critic_ckpt_path or not Path(critic_ckpt_path).exists():
        print(f"ERROR: Critic checkpoint not found: {critic_ckpt_path}")
        print("  Run Phase 1 first: python experiments/11_critic/train_critic.py")
        sys.exit(1)

    critic_cfg = ConditionalCriticConfig(**config.get("critic", {}))
    critic = ConditionalCritic(critic_cfg).to(device)
    ckpt = torch.load(critic_ckpt_path, map_location=device, weights_only=False)
    critic.load_state_dict(ckpt["critic"])
    critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)

    critic_metrics = ckpt.get("val_metrics", {})
    print(f"  Critic loaded from {critic_ckpt_path}")
    print(f"    rank_acc={critic_metrics.get('val_rank_acc', '?')}, "
          f"cos_sim={critic_metrics.get('val_cos_sim_mean', '?')}")
    print(f"  Critic frozen: {critic.num_params:,} params (no grad)")

    # ── Build chain head (from scratch) ──
    chain_cfg = ChainHeadConfig(**config.get("chain_head", {}))
    chain_head = EBTChainHead(chain_cfg).to(device)
    print(f"  Chain Head v2: {chain_head.num_params:,} parameters (NEW, from scratch)")

    # ── Load data ──
    data_path = config["data"]["path"]
    print(f"  Loading data from {data_path}")
    data = torch.load(data_path, map_location="cpu", weights_only=False)

    train_ds = ChainDataset(data["train"])
    val_ds = ChainDataset(data["val"])
    print(f"  Train: {len(train_ds)} (filtered ≥3 steps), Val: {len(val_ds)}")

    train_cfg = config["training"]
    batch_size = train_cfg["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=config["data"].get("num_workers", 4),
        collate_fn=collate_chain, pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        collate_fn=collate_chain, pin_memory=device.type == "cuda",
    )

    # ── Optimizer (chain head only) ──
    lr = train_cfg.get("lr", 1e-4)
    optimizer = torch.optim.AdamW(
        chain_head.parameters(), lr=lr,
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
        ckpt_r = torch.load(args.resume, map_location=device, weights_only=False)
        chain_head.load_state_dict(ckpt_r["chain_head"])
        optimizer.load_state_dict(ckpt_r["optimizer"])
        start_epoch = ckpt_r.get("epoch", 0) + 1
        best_metric = ckpt_r.get("best_metric", 0.0)
        print(f"  Resumed from epoch {start_epoch}, best_metric={best_metric:.4f}")

    # ── Training loop ──
    log_every = train_cfg.get("log_every", 50)
    patience = train_cfg.get("early_stop_patience", 10)
    no_improve = 0

    print(f"\nStarting training: {num_epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"Loss weights: nce={train_cfg.get('w_chain_nce', 1.0)}, "
          f"critic_bonus={train_cfg.get('w_critic_bonus', 0.3)}")

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        chain_head.train()
        tracker = MetricTracker()

        for batch_idx, batch in enumerate(train_loader):
            step_global = epoch * len(train_loader) + batch_idx
            metrics = train_step(
                chain_head, critic, batch, optimizer, scaler,
                device, amp_enabled, amp_dtype, train_cfg, step_global,
            )
            tracker.update(metrics)
            scheduler.step()

            if (batch_idx + 1) % log_every == 0:
                avg = tracker.get()
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  [E{epoch} S{batch_idx+1}] "
                    f"total={avg.get('total_loss', 0):.4f} "
                    f"nce={avg.get('loss_nce', 0):.4f} "
                    f"rank_acc={avg.get('chain_rank_acc', 0):.4f} "
                    f"E_gap={avg.get('chain_energy_gap', 0):.4f} "
                    f"cos_bonus={avg.get('critic_cos_bonus', 0):.4f} "
                    f"lr={current_lr:.2e}"
                )

        elapsed = time.time() - t0
        train_avg = tracker.get()
        print(f"\n  Epoch {epoch} train ({elapsed:.1f}s):")
        for k, v in sorted(train_avg.items()):
            print(f"    {k}: {v:.4f}")

        # ── Evaluation ──
        chain_head.eval()
        val_tracker = MetricTracker()
        for batch in val_loader:
            vm = eval_step(
                chain_head, critic, batch, device, amp_enabled, amp_dtype, train_cfg,
            )
            val_tracker.update(vm)

        val_avg = val_tracker.get()
        print(f"  Epoch {epoch} val:")
        for k, v in sorted(val_avg.items()):
            print(f"    {k}: {v:.4f}")

        # Metric: chain_rank_acc × critic_cos_bonus
        chain_acc = val_avg.get("val_chain_rank_acc", 0)
        cos_bonus = max(0.01, val_avg.get("val_critic_cos_bonus", 0.01))
        combined = chain_acc * cos_bonus

        if combined > best_metric:
            best_metric = combined
            no_improve = 0
            save_checkpoint(
                Path(out_cfg["checkpoint_dir"]) / "best_chain_v2.pt",
                {
                    "chain_head": chain_head.state_dict(),  # ← same key as old for compat
                    "model": chain_head.state_dict(),       # ← legacy key
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "val_metrics": val_avg,
                    "config": config,
                    "critic_checkpoint": str(critic_ckpt_path),
                },
            )
            print(f"  >> New best: combined={combined:.4f} "
                  f"(rank_acc={chain_acc:.4f} × cos_bonus={cos_bonus:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping: no improvement for {patience} epochs")
                break

        if (epoch + 1) % train_cfg.get("checkpoint_every", 5) == 0:
            save_checkpoint(
                Path(out_cfg["checkpoint_dir"]) / f"chain_v2_epoch_{epoch}.pt",
                {
                    "chain_head": chain_head.state_dict(),
                    "model": chain_head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                },
            )

    print(f"\nPhase 2 finished. Best combined metric: {best_metric:.4f}")
    print(f"Best checkpoint: {out_cfg['checkpoint_dir']}/best_chain_v2.pt")
    print("\nNext: run Phase 3 (joint fine-tuning) with both checkpoints:")
    print(f"  python experiments/10_autoregressor/train_autoregressor.py \\")
    print(f"    --config configs/autoregressor_config.json")


if __name__ == "__main__":
    main()
