#!/usr/bin/env python3
"""
Stage 3 Phase A: Chain Head training (Pairwise frozen).

Trains the EBT Chain Head to distinguish coherent reasoning chains from
4 types of negatives: shuffled, truncated, corrupted, wrong conclusion.

Pairwise critic is FROZEN — only Chain Head parameters are updated.

Success criterion: chain_rank_acc > 0.85

Spec reference: IMPLEMENTATION_PLAN.md §6.1 Phase A
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cebcm.models.chain_head import ChainHeadConfig, EBTChainHead
from cebcm.training.chain_data import (
    ChainDataConfig,
    ChainDataset,
    apply_curriculum,
    chain_collate_fn,
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


def build_chain_head(cfg: dict, device: torch.device) -> EBTChainHead:
    """Build Chain Head from config dict."""
    chain_cfg = ChainHeadConfig(
        d_model=cfg.get("d_model", 1024),
        n_heads=cfg.get("n_heads", 8),
        n_layers=cfg.get("n_layers", 2),
        dim_feedforward=cfg.get("dim_feedforward", 2048),
        max_chain_len=cfg.get("max_chain_len", 20),
        dropout=cfg.get("dropout", 0.1),
        activation=cfg.get("activation", "gelu"),
        energy_hidden=cfg.get("energy_hidden", 512),
        temperature=cfg.get("temperature", 0.07),
        focal_gamma=cfg.get("focal_gamma", 2.0),
    )
    model = EBTChainHead(chain_cfg).to(device)
    print(f"  Chain Head: {model.num_params:,} parameters")
    return model


def build_chain_datasets(
    data_cfg: dict,
    chain_data_cfg: dict,
    seed: int,
) -> tuple[ChainDataset, ChainDataset]:
    """Load SONAR sequences and create chain train/val datasets."""
    data_path = data_cfg["train_data_path"]
    print(f"  Loading data from {data_path}")
    raw = torch.load(data_path, map_location="cpu", weights_only=True)

    if isinstance(raw, dict):
        vectors = raw["vectors"]
        lengths = raw["lengths"]
    else:
        vectors = raw
        lengths = torch.tensor([v.shape[0] for v in vectors])

    # Train/val split
    split = data_cfg.get("train_val_split", 0.9)
    n = len(lengths)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_train = int(n * split)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    cfg = ChainDataConfig(
        min_chain_len=chain_data_cfg.get("min_chain_len", 5),
        max_chain_len=chain_data_cfg.get("max_chain_len", 15),
        num_negatives=chain_data_cfg.get("num_negatives", 7),
        neg_ratio_shuffled=chain_data_cfg.get("neg_ratio_shuffled", 0.25),
        neg_ratio_truncated=chain_data_cfg.get("neg_ratio_truncated", 0.25),
        neg_ratio_corrupted=chain_data_cfg.get("neg_ratio_corrupted", 0.25),
        neg_ratio_wrong_conclusion=chain_data_cfg.get("neg_ratio_wrong_conclusion", 0.25),
        noise_std=chain_data_cfg.get("noise_std", 0.2),
        target_norm=chain_data_cfg.get("target_norm", 0.2051),
    )

    train_ds = ChainDataset(vectors[train_idx], lengths[train_idx], cfg)
    val_ds = ChainDataset(vectors[val_idx], lengths[val_idx], cfg)

    print(f"  Train chains: {len(train_ds)}, Val chains: {len(val_ds)}")
    return train_ds, val_ds


def train_epoch(
    model: EBTChainHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    clip_grad: float,
    log_every: int,
    epoch: int,
) -> dict[str, float]:
    model.train()
    tracker = MetricTracker()
    step = 0

    for batch in loader:
        positives = batch["positives"].to(device)
        pos_lengths = batch["pos_lengths"].to(device)
        negatives = batch["negatives"].to(device)
        neg_lengths = batch["neg_lengths"].to(device)

        optimizer.zero_grad()

        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
            loss, metrics = model.compute_infonce_loss(
                positives, negatives,
                pos_lengths=pos_lengths,
                neg_lengths=neg_lengths,
            )

        scaler.scale(loss).backward()
        if clip_grad > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        tracker.update(metrics)
        step += 1

        if step % log_every == 0:
            avg = tracker.get()
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [Epoch {epoch} Step {step}] "
                f"loss={avg.get('chain_loss', 0):.4f} "
                f"rank_acc={avg.get('chain_rank_acc', 0):.4f} "
                f"E_gap={avg.get('chain_energy_gap', 0):.4f} "
                f"lr={lr:.2e}"
            )

    return tracker.get()


@torch.no_grad()
def eval_epoch(
    model: EBTChainHead,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    tracker = MetricTracker()

    for batch in loader:
        positives = batch["positives"].to(device)
        pos_lengths = batch["pos_lengths"].to(device)
        negatives = batch["negatives"].to(device)
        neg_lengths = batch["neg_lengths"].to(device)

        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
            _, metrics = model.compute_infonce_loss(
                positives, negatives,
                pos_lengths=pos_lengths,
                neg_lengths=neg_lengths,
            )

        tracker.update(metrics)

    return tracker.get()


def main():
    parser = argparse.ArgumentParser(description="Stage 3 Phase A: Chain Head Training")
    parser.add_argument("--config", default="configs/stage3_config.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    setup_seed(config.get("seed", 42), device)

    # Setup output dirs
    out_cfg = config["output"]
    for d in [out_cfg["output_dir"], out_cfg["checkpoint_dir"], out_cfg["logs_dir"]]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Stage 3 Phase A: Chain Head Training")
    print(f"Device: {device}")
    print("=" * 60)

    # Build model
    model = build_chain_head(config["chain_head"], device)

    # Build data
    train_ds, val_ds = build_chain_datasets(
        config["data"], config["chain_data"], config.get("seed", 42)
    )

    phase_cfg = config["phase_a"]
    train_loader = DataLoader(
        train_ds,
        batch_size=phase_cfg["batch_size"],
        shuffle=True,
        num_workers=config["data"].get("num_workers", 4),
        collate_fn=chain_collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=phase_cfg["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        collate_fn=chain_collate_fn,
        pin_memory=device.type == "cuda",
    )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=phase_cfg["lr"],
        weight_decay=phase_cfg["weight_decay"],
    )
    total_steps = phase_cfg["num_epochs"] * len(train_loader)
    warmup_steps = phase_cfg["warmup_epochs"] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps,
        min_factor=phase_cfg.get("lr_min_factor", 0.01),
    )

    # AMP
    amp_enabled, amp_dtype, scaler = setup_amp(config.get("amp", {}), device)

    # Resume
    start_epoch = 0
    best_rank_acc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_rank_acc = ckpt.get("best_rank_acc", 0.0)
        print(f"  Resumed from epoch {start_epoch}, best_rank_acc={best_rank_acc:.4f}")

    # Training loop
    no_improve_count = 0
    patience = phase_cfg.get("early_stop_patience", 7)
    target_acc = phase_cfg.get("target_rank_acc", 0.85)

    for epoch in range(start_epoch, phase_cfg["num_epochs"]):
        t0 = time.time()

        # Curriculum: adjust negative ratios based on training progress
        progress = epoch / max(1, phase_cfg["num_epochs"] - 1)
        train_ds.set_curriculum_progress(progress)

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, amp_enabled, amp_dtype,
            clip_grad=phase_cfg["clip_grad_norm"],
            log_every=phase_cfg["log_every"],
            epoch=epoch,
        )

        elapsed = time.time() - t0
        print(f"\n  Epoch {epoch} train ({elapsed:.1f}s):")
        for k, v in sorted(train_metrics.items()):
            print(f"    {k}: {v:.4f}")

        # Eval
        if (epoch + 1) % phase_cfg.get("eval_every_epochs", 1) == 0:
            val_metrics = eval_epoch(model, val_loader, device, amp_enabled, amp_dtype)
            print(f"  Epoch {epoch} val:")
            for k, v in sorted(val_metrics.items()):
                print(f"    {k}: {v:.4f}")

            rank_acc = val_metrics.get("chain_rank_acc", 0.0)

            if rank_acc > best_rank_acc:
                best_rank_acc = rank_acc
                no_improve_count = 0
                save_checkpoint(
                    Path(out_cfg["checkpoint_dir"]) / "best_chain_head.pt",
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_rank_acc": best_rank_acc,
                        "config": config["chain_head"],
                        "val_metrics": val_metrics,
                    },
                )
                print(f"  >> New best rank_acc={best_rank_acc:.4f}")
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"  Early stopping: no improvement for {patience} evals")
                    break

            # Check if target achieved
            if rank_acc >= target_acc:
                print(f"\n  TARGET ACHIEVED: rank_acc={rank_acc:.4f} >= {target_acc}")
                print("  Phase A complete. Ready for Phase B.")
                break

        # Periodic checkpoint
        if (epoch + 1) % phase_cfg.get("checkpoint_every_epochs", 5) == 0:
            save_checkpoint(
                Path(out_cfg["checkpoint_dir"]) / f"chain_head_epoch_{epoch}.pt",
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_rank_acc": best_rank_acc,
                    "config": config["chain_head"],
                },
            )

    print(f"\nPhase A finished. Best rank_acc={best_rank_acc:.4f}")
    print(f"Best checkpoint: {out_cfg['checkpoint_dir']}/best_chain_head.pt")


if __name__ == "__main__":
    main()
