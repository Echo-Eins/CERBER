#!/usr/bin/env python3
"""
ChainGenerator training — autoregressive Transformer decoder in SONAR space.

Pure QA neural network. NOT a denoiser. Direct generation of answer embeddings.

Training:
  - Teacher forcing on ground-truth reasoning chains (v_steps + v_answer)
  - Loss: cosine similarity + MSE per chain step
  - Curriculum: System 1 → System 2 (gradually increase chain length)

Eval:
  - Teacher-forced metrics (cos_sim per step)
  - Autoregressive generation metrics (cos_sim of final answer)
  - Optional: CompositeCritic reranking (best-of-N)

Usage:
    python experiments/13_chain_generator/train_chain_generator.py
    python experiments/13_chain_generator/train_chain_generator.py --config configs/chain_generator_config.json
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

from cebcm.models.chain_generator import ChainGenerator, ChainGeneratorConfig
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
    """
    Dataset for chain generator training.

    Each sample has:
      - v_question: [D] question embedding
      - v_answer:   [D] answer embedding
      - v_steps:    [S, D] reasoning step embeddings (variable length)

    The training chain is: [v_steps..., v_answer] (answer is the last step).
    """

    def __init__(self, samples: list[dict], max_chain_len: int = 20):
        self.samples = samples
        self.max_chain_len = max_chain_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        v_q = s["v_question"]        # [D]
        v_a = s["v_answer"]          # [D]
        v_steps = s["v_steps"]       # [S, D]

        # Build chain: [reasoning_steps..., answer]
        # v_answer is always the final step
        if v_steps.shape[0] > 0:
            chain = torch.cat([v_steps, v_a.unsqueeze(0)], dim=0)
        else:
            chain = v_a.unsqueeze(0)  # [1, D] — just the answer

        # Truncate to max length
        if chain.shape[0] > self.max_chain_len:
            chain = chain[:self.max_chain_len]

        return {
            "v_question": v_q,
            "chain": chain,
            "chain_len": chain.shape[0],
        }


def collate_chains(batch: list[dict]) -> dict:
    """Pad chains to max length in batch."""
    v_questions = torch.stack([b["v_question"] for b in batch])
    chain_lens = torch.tensor([b["chain_len"] for b in batch])
    max_len = chain_lens.max().item()
    D = batch[0]["chain"].shape[-1]

    # Pad chains
    chains = torch.zeros(len(batch), max_len, D)
    for i, b in enumerate(batch):
        L = b["chain_len"]
        chains[i, :L] = b["chain"]

    return {
        "v_questions": v_questions,
        "chains": chains,
        "chain_lens": chain_lens,
    }


# ── Curriculum: chain length scheduling ──────────────────────────

def get_chain_steps(epoch: int, cfg: dict) -> int:
    """
    Curriculum: gradually increase chain length.

    System 1 phase: train with 1-step chains (direct answer)
    System 2 ramp: gradually increase to max_chain_steps
    """
    s1_epochs = cfg.get("system1_epochs", 10)
    ramp_epochs = cfg.get("system2_ramp_epochs", 10)
    max_steps = cfg.get("max_chain_steps", 5)

    if epoch < s1_epochs:
        return 1  # System 1: direct answer
    ramp_progress = min(1.0, (epoch - s1_epochs) / max(ramp_epochs, 1))
    return max(1, int(1 + ramp_progress * (max_steps - 1)))


# ── Training step ────────────────────────────────────────────────

def train_step(
    model: ChainGenerator,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    cfg: dict,
    target_steps: int,
) -> dict[str, float]:
    """Single training step with teacher forcing."""
    v_q = batch["v_questions"].to(device)      # [B, D]
    chains = batch["chains"].to(device)         # [B, max_L, D]
    chain_lens = batch["chain_lens"].to(device) # [B]

    clip_grad = cfg.get("clip_grad_norm", 1.0)

    # Truncate chains to target_steps for curriculum
    effective_len = min(target_steps, chains.shape[1])
    chains_trunc = chains[:, :effective_len, :]  # [B, T, D]

    # Mask: only compute loss on valid positions
    mask = torch.arange(effective_len, device=device).unsqueeze(0) < chain_lens.unsqueeze(1).clamp(max=effective_len)
    # [B, T] bool

    optimizer.zero_grad()

    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
        loss, metrics = model.compute_loss(v_q, chains_trunc)

    scaler.scale(loss).backward()

    if clip_grad > 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

    scaler.step(optimizer)
    scaler.update()

    metrics["target_steps"] = target_steps
    return metrics


# ── Eval step ────────────────────────────────────────────────────

@torch.no_grad()
def eval_step(
    model: ChainGenerator,
    batch: dict,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    gen_steps: int = 1,
) -> dict[str, float]:
    """
    Evaluation: teacher-forced metrics + autoregressive generation.

    Reports:
      - Teacher-forced cos_sim (how well does model predict next step?)
      - Autoregressive cos_sim (how good is the generated answer?)
    """
    v_q = batch["v_questions"].to(device)
    chains = batch["chains"].to(device)
    chain_lens = batch["chain_lens"].to(device)

    # Teacher-forced metrics
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
        _, tf_metrics = model.compute_loss(v_q, chains)

    # Autoregressive generation
    v_generated = model.generate(v_q, num_steps=gen_steps)  # [B, gen_steps, D]

    # Compare last generated step to answer (last step of chain)
    v_gen_answer = v_generated[:, -1, :]  # [B, D]

    # Get actual answer from chains (last valid position)
    answer_idx = (chain_lens - 1).clamp(min=0)  # [B]
    v_answer = chains[torch.arange(chains.shape[0], device=device), answer_idx]  # [B, D]

    gen_cos = F.cosine_similarity(v_gen_answer, v_answer, dim=-1)

    return {
        "val_tf_cos_sim": tf_metrics["cos_sim_mean"],
        "val_tf_cos_last": tf_metrics["cos_sim_last"],
        "val_tf_loss": tf_metrics["loss"],
        "val_gen_cos_mean": gen_cos.mean().item(),
        "val_gen_cos_min": gen_cos.min().item(),
        "val_gen_cos_max": gen_cos.max().item(),
        "val_gen_norm_mean": v_gen_answer.norm(dim=-1).mean().item(),
    }


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ChainGenerator Training — Autoregressive QA in SONAR space"
    )
    parser.add_argument("--config", default="configs/chain_generator_config.json")
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
    print("ChainGenerator Training — Autoregressive QA in SONAR Space")
    print(f"  Architecture: Transformer Decoder (RoPE + Cross-Attention)")
    print(f"  NOT a denoiser. Direct prediction of answer embeddings.")
    print(f"  Device: {device}")
    print("=" * 60)

    # ── Build model ──
    gen_raw = config.get("generator", {})
    gen_cfg = ChainGeneratorConfig(**gen_raw)
    model = ChainGenerator(gen_cfg).to(device)

    print(f"  Parameters: {model.num_params:,}")
    print(f"  Layers: {gen_cfg.n_layers}")
    print(f"  Heads: {gen_cfg.n_heads}")
    print(f"  FFN dim: {gen_cfg.dim_feedforward}")
    print(f"  Target norm: {gen_cfg.target_norm}")
    print(f"  Max chain: {gen_cfg.max_chain_len}")

    # ── Load data ──
    data_path = config["data"]["path"]
    print(f"  Loading data from {data_path}")
    data = torch.load(data_path, map_location="cpu", weights_only=False)

    train_ds = ChainDataset(data["train"], max_chain_len=gen_cfg.max_chain_len)
    val_ds = ChainDataset(data["val"], max_chain_len=gen_cfg.max_chain_len)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_cfg = config["training"]
    batch_size = train_cfg["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=config["data"].get("num_workers", 4),
        collate_fn=collate_chains, pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        collate_fn=collate_chains, pin_memory=device.type == "cuda",
    )

    # ── Optimizer ──
    lr = train_cfg.get("lr", 1e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    num_epochs = args.max_epochs or train_cfg["num_epochs"]
    total_steps = num_epochs * len(train_loader)
    warmup_steps = train_cfg.get("warmup_epochs", 3) * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps,
        min_factor=train_cfg.get("lr_min_factor", 0.01),
    )

    amp_enabled, amp_dtype, scaler = setup_amp(config.get("amp", {}), device)

    # ── Resume ──
    start_epoch = 0
    best_metric = 0.0
    if args.resume:
        print(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metric = ckpt.get("best_metric", 0.0)
        print(f"  Resumed at epoch {start_epoch}, best cos_sim={best_metric:.4f}")

    # ── Training loop ──
    log_every = train_cfg.get("log_every", 50)
    ckpt_every = train_cfg.get("checkpoint_every", 5)
    patience = train_cfg.get("early_stop_patience", 15)
    no_improve = 0

    tracker = MetricTracker()

    print(f"\n  Epochs: {num_epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"  System 1 epochs: {train_cfg.get('system1_epochs', 10)}")
    print(f"  System 2 ramp epochs: {train_cfg.get('system2_ramp_epochs', 10)}")
    print(f"  Max chain steps: {train_cfg.get('max_chain_steps', 5)}")
    print("=" * 60)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        target_steps = get_chain_steps(epoch, train_cfg)
        epoch_start = time.time()

        print(f"\n[E{epoch}] target_steps={target_steps} ({'System 1' if target_steps == 1 else f'System 2 ({target_steps} steps)'})")

        for step, batch in enumerate(train_loader):
            metrics = train_step(
                model, batch, optimizer, scaler,
                device, amp_enabled, amp_dtype, train_cfg,
                target_steps=target_steps,
            )
            scheduler.step()
            tracker.update(metrics)

            if (step + 1) % log_every == 0:
                avg = tracker.average()
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"  [E{epoch} S{step+1}] "
                    f"loss={avg.get('loss', 0):.4f} "
                    f"cos={avg.get('cos_sim_mean', 0):.4f} "
                    f"cos_last={avg.get('cos_sim_last', 0):.4f} "
                    f"norm={avg.get('pred_norm_mean', 0):.4f} "
                    f"lr={lr_now:.2e}"
                )
                tracker.reset()

        epoch_time = time.time() - epoch_start

        # ── Validation ──
        model.eval()
        val_tracker = MetricTracker()
        gen_steps = target_steps  # match curriculum

        for batch in val_loader:
            vm = eval_step(
                model, batch, device, amp_enabled, amp_dtype,
                gen_steps=gen_steps,
            )
            val_tracker.update(vm)

        val_avg = val_tracker.average()
        val_cos = val_avg.get("val_gen_cos_mean", 0)
        val_tf_cos = val_avg.get("val_tf_cos_sim", 0)

        print(
            f"  [E{epoch} VAL] "
            f"tf_cos={val_tf_cos:.4f} "
            f"gen_cos={val_cos:.4f} "
            f"gen_cos_min={val_avg.get('val_gen_cos_min', 0):.4f} "
            f"gen_cos_max={val_avg.get('val_gen_cos_max', 0):.4f} "
            f"gen_norm={val_avg.get('val_gen_norm_mean', 0):.4f} "
            f"({epoch_time:.1f}s)"
        )

        # ── Checkpointing ──
        improved = val_cos > best_metric
        if improved:
            best_metric = val_cos
            no_improve = 0
            save_checkpoint(
                Path(out_cfg["checkpoint_dir"]) / "best.pt",
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": config,
                    "best_metric": best_metric,
                },
            )
            print(f"  ** New best: gen_cos={best_metric:.4f}")
        else:
            no_improve += 1

        if (epoch + 1) % ckpt_every == 0:
            save_checkpoint(
                Path(out_cfg["checkpoint_dir"]) / f"epoch_{epoch}.pt",
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": config,
                    "best_metric": best_metric,
                },
            )

        if no_improve >= patience:
            print(f"\n  Early stopping after {patience} epochs without improvement")
            break

    print(f"\nTraining complete. Best gen_cos={best_metric:.4f}")
    print(f"Checkpoints in: {out_cfg['checkpoint_dir']}")


if __name__ == "__main__":
    main()
