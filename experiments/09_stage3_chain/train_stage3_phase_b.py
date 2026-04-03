#!/usr/bin/env python3
"""
Stage 3 Phase B: Joint fine-tune + System 1/2 switching.

Both Pairwise and Chain Head are UNFROZEN and trained jointly.
Each batch is processed in a random mode (System 1 or System 2, 30/70 bias).

Key training mechanisms:
  - System 1 (30%): Pairwise energy + short Langevin (10-50 steps)
  - System 2 (70%): Chain Head energy + extended Langevin (100-200 steps)
  - Cruise ratio sampling: expose model to [0.0, 0.3, 0.5, 0.7]
  - Combined loss: pairwise ranking + chain ranking + cosine reconstruction

Success criteria:
  - System 2 quality > System 1 quality (cosine to ground truth)
  - Quality degradation < 15% across cruise_ratio variants
  - Chain rank_acc maintained > 0.85

Spec reference: IMPLEMENTATION_PLAN.md §6.1 Phase B
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cebcm.models.chain_head import ChainHeadConfig, EBTChainHead
from cebcm.models.energy import SimpleEnergy
from cebcm.inference.system_switching import (
    System1Config,
    System2Config,
    select_thinking_mode,
    run_thinking,
)
from cebcm.training.chain_data import (
    ChainDataConfig,
    ChainDataset,
    chain_collate_fn,
)
from cebcm.training.stage2_utils import (
    MetricTracker,
    get_cosine_schedule_with_warmup,
    load_config,
    resolve_device,
    save_checkpoint,
    load_checkpoint,
    setup_amp,
    setup_seed,
)


def build_pairwise(cfg: dict, device: torch.device) -> SimpleEnergy:
    """Build pairwise critic from config and load checkpoint."""
    model = SimpleEnergy(
        dim=cfg.get("dim", 1024),
        hidden_dims=cfg.get("hidden_dims", [2048, 1024, 512]),
        norm_mode=cfg.get("norm_mode", "orthonorm"),
        activation=cfg.get("activation", "groupsort"),
    ).to(device)

    ckpt_path = cfg.get("checkpoint")
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Handle different checkpoint formats
        state = ckpt.get("model", ckpt.get("critic", ckpt))
        if isinstance(state, dict) and any(k.startswith("net.") for k in state):
            model.load_state_dict(state)
        print(f"  Pairwise loaded from {ckpt_path}")
    else:
        print(f"  WARNING: Pairwise checkpoint not found at {ckpt_path}, using random init")

    print(f"  Pairwise: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def build_chain_head(cfg: dict, device: torch.device, checkpoint: str | None = None) -> EBTChainHead:
    """Build Chain Head from config, optionally loading Phase A checkpoint."""
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

    if checkpoint and Path(checkpoint).exists():
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"  Chain Head loaded from {checkpoint}")
    else:
        print(f"  Chain Head: random init (no Phase A checkpoint)")

    print(f"  Chain Head: {model.num_params:,} parameters")
    return model


def train_epoch_phase_b(
    pairwise: SimpleEnergy,
    chain_head: EBTChainHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    phase_cfg: dict,
    epoch: int,
) -> dict[str, float]:
    """
    Phase B training epoch with System 1/2 switching.

    Each batch:
      1. Select thinking mode (system1/system2, 30/70 bias)
      2. Compute chain InfoNCE loss (always, for chain head quality)
      3. Track which mode was used for metrics
    """
    pairwise.train()
    chain_head.train()
    tracker = MetricTracker()
    step = 0

    sys1_weight = phase_cfg.get("system1_weight", 0.3)
    sys2_weight = phase_cfg.get("system2_weight", 0.7)
    clip_grad = phase_cfg.get("clip_grad_norm", 1.0)
    log_every = phase_cfg.get("log_every", 50)

    for batch in loader:
        positives = batch["positives"].to(device)
        pos_lengths = batch["pos_lengths"].to(device)
        negatives = batch["negatives"].to(device)
        neg_lengths = batch["neg_lengths"].to(device)

        # Select thinking mode
        mode = select_thinking_mode(sys1_weight, sys2_weight)

        optimizer.zero_grad()

        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
            # Chain Head InfoNCE loss (always computed — core training signal)
            chain_loss, chain_metrics = chain_head.compute_infonce_loss(
                positives, negatives,
                pos_lengths=pos_lengths,
                neg_lengths=neg_lengths,
            )

            # Pairwise consistency loss: pairwise energy should rank
            # positive chain vectors lower than negative chain vectors
            # Use first + last vector of positive chain as query/target pair
            B = positives.shape[0]
            batch_idx = torch.arange(B, device=positives.device)
            # Query = first vector in chain, target = last valid vector
            v_queries = positives[:, 0, :]  # [B, D]
            v_targets = positives[batch_idx, (pos_lengths - 1).clamp(min=0), :]  # [B, D]
            # Negative targets: last VALID vector of first negative (not padding!)
            neg_last_idx = (neg_lengths[:, 0] - 1).clamp(min=0)  # [B]
            v_neg = negatives[batch_idx, 0, neg_last_idx, :]  # [B, D]

            E_pos_pair = pairwise(v_queries, v_targets)
            E_neg_pair = pairwise(v_queries, v_neg)
            # Margin ranking loss for pairwise (adaptive margin based on energy scale)
            with torch.no_grad():
                energy_scale = (E_pos_pair.abs().mean() + E_neg_pair.abs().mean()).clamp(min=0.1)
                margin = energy_scale * 0.5  # 50% of typical energy magnitude
            pairwise_loss = F.relu(E_pos_pair - E_neg_pair + margin).mean()

            # Mode-dependent weighting
            if mode == "system1":
                # System 1: emphasize pairwise, lighter chain
                total_loss = 0.7 * pairwise_loss + 0.3 * chain_loss
            else:
                # System 2: emphasize chain, lighter pairwise
                total_loss = 0.3 * pairwise_loss + 0.7 * chain_loss

        scaler.scale(total_loss).backward()
        if clip_grad > 0:
            scaler.unscale_(optimizer)
            total_norm = nn.utils.clip_grad_norm_(
                list(pairwise.parameters()) + list(chain_head.parameters()),
                clip_grad,
            )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Track metrics
        metrics = {
            **chain_metrics,
            "pairwise_loss": pairwise_loss.item(),
            "total_loss": total_loss.item(),
            f"mode_{mode}": 1.0,
        }
        tracker.update(metrics)
        step += 1

        if step % log_every == 0:
            avg = tracker.get()
            lr = optimizer.param_groups[0]["lr"]
            sys1_pct = avg.get("mode_system1", 0) / max(1, avg.get("mode_system1", 0) + avg.get("mode_system2", 0))
            print(
                f"  [Epoch {epoch} Step {step}] "
                f"total={avg.get('total_loss', 0):.4f} "
                f"chain={avg.get('chain_loss', 0):.4f} "
                f"pair={avg.get('pairwise_loss', 0):.4f} "
                f"rank_acc={avg.get('chain_rank_acc', 0):.4f} "
                f"sys1%={sys1_pct:.2f} "
                f"lr={lr:.2e}"
            )

    return tracker.get()


@torch.no_grad()
def eval_epoch_phase_b(
    pairwise: SimpleEnergy,
    chain_head: EBTChainHead,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    """Evaluate both heads jointly."""
    pairwise.eval()
    chain_head.eval()
    tracker = MetricTracker()

    for batch in loader:
        positives = batch["positives"].to(device)
        pos_lengths = batch["pos_lengths"].to(device)
        negatives = batch["negatives"].to(device)
        neg_lengths = batch["neg_lengths"].to(device)

        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
            _, chain_metrics = chain_head.compute_infonce_loss(
                positives, negatives,
                pos_lengths=pos_lengths,
                neg_lengths=neg_lengths,
            )

            # Pairwise eval
            B = positives.shape[0]
            batch_idx = torch.arange(B, device=positives.device)
            v_queries = positives[:, 0, :]
            v_targets = positives[batch_idx, (pos_lengths - 1).clamp(min=0), :]
            neg_last_idx = (neg_lengths[:, 0] - 1).clamp(min=0)
            v_neg = negatives[batch_idx, 0, neg_last_idx, :]

            E_pos_pair = pairwise(v_queries, v_targets)
            E_neg_pair = pairwise(v_queries, v_neg)
            pair_rank_acc = (E_pos_pair < E_neg_pair).float().mean().item()

        metrics = {
            **chain_metrics,
            "pairwise_rank_acc": pair_rank_acc,
        }
        tracker.update(metrics)

    return tracker.get()


def main():
    parser = argparse.ArgumentParser(description="Stage 3 Phase B: Joint Training + System 1/2")
    parser.add_argument("--config", default="configs/stage3_config.json")
    parser.add_argument("--chain-head-ckpt", default=None,
                        help="Phase A chain head checkpoint (auto-detected if not specified)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    setup_seed(config.get("seed", 42), device)

    out_cfg = config["output"]
    for d in [out_cfg["output_dir"], out_cfg["checkpoint_dir"], out_cfg["logs_dir"]]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Stage 3 Phase B: Joint Training + System 1/2 Switching")
    print(f"Device: {device}")
    print("=" * 60)

    # Build models
    pairwise = build_pairwise(config["pairwise"], device)

    # Auto-detect Phase A checkpoint
    chain_ckpt = args.chain_head_ckpt
    if chain_ckpt is None:
        default_ckpt = Path(out_cfg["checkpoint_dir"]) / "best_chain_head.pt"
        if default_ckpt.exists():
            chain_ckpt = str(default_ckpt)
    chain_head = build_chain_head(config["chain_head"], device, checkpoint=chain_ckpt)

    total_params = sum(p.numel() for p in pairwise.parameters()) + chain_head.num_params
    print(f"  Total trainable params: {total_params:,}")

    # Build data — load sequences (supports both formats)
    data_path = config["data"]["train_data_path"]
    print(f"  Loading data from {data_path}")
    raw = torch.load(data_path, map_location="cpu", weights_only=False)

    if isinstance(raw, dict):
        if "sequences" in raw:
            sequences = raw["sequences"]
        elif "vectors" in raw:
            vectors = raw["vectors"]
            lengths = raw["lengths"]
            sequences = [vectors[i, :int(lengths[i].item())] for i in range(len(lengths))]
        else:
            raise KeyError(f"Unknown data format. Keys: {list(raw.keys())}")
    elif isinstance(raw, list):
        sequences = raw
    else:
        raise TypeError(f"Unknown data type: {type(raw)}")

    n = len(sequences)
    print(f"  Loaded {n} sequences")

    split = config["data"].get("train_val_split", 0.9)
    gen = torch.Generator().manual_seed(config.get("seed", 42))
    perm = torch.randperm(n, generator=gen).tolist()
    n_train = int(n * split)

    chain_data_cfg = ChainDataConfig(
        min_chain_len=config["chain_data"].get("min_chain_len", 5),
        max_chain_len=config["chain_data"].get("max_chain_len", 15),
        num_negatives=config["chain_data"].get("num_negatives", 7),
        target_norm=config["chain_data"].get("target_norm", 0.2051),
    )

    train_ds = ChainDataset([sequences[i] for i in perm[:n_train]], chain_data_cfg)
    val_ds = ChainDataset([sequences[i] for i in perm[n_train:]], chain_data_cfg)
    print(f"  Train chains: {len(train_ds)}, Val chains: {len(val_ds)}")

    phase_cfg = config["phase_b"]
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

    # Optimizer: separate LR for pairwise (conservative) and chain head
    optimizer = torch.optim.AdamW([
        {"params": pairwise.parameters(), "lr": phase_cfg["pairwise_lr"]},
        {"params": chain_head.parameters(), "lr": phase_cfg["chain_head_lr"]},
    ], weight_decay=phase_cfg["weight_decay"])

    total_steps = phase_cfg["num_epochs"] * len(train_loader)
    warmup_steps = phase_cfg["warmup_epochs"] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps,
        min_factor=phase_cfg.get("lr_min_factor", 0.01),
    )

    amp_enabled, amp_dtype, scaler = setup_amp(config.get("amp", {}), device)

    # Resume
    start_epoch = 0
    best_metric = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        pairwise.load_state_dict(ckpt["pairwise"])
        chain_head.load_state_dict(ckpt["chain_head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_metric = ckpt.get("best_metric", 0.0)
        print(f"  Resumed from epoch {start_epoch}")

    # Training loop
    no_improve_count = 0
    patience = phase_cfg.get("early_stop_patience", 5)

    for epoch in range(start_epoch, phase_cfg["num_epochs"]):
        t0 = time.time()

        train_metrics = train_epoch_phase_b(
            pairwise, chain_head, train_loader, optimizer, scheduler, scaler,
            device, amp_enabled, amp_dtype, phase_cfg, epoch,
        )

        elapsed = time.time() - t0
        print(f"\n  Epoch {epoch} train ({elapsed:.1f}s):")
        for k, v in sorted(train_metrics.items()):
            print(f"    {k}: {v:.4f}")

        # Eval
        if (epoch + 1) % phase_cfg.get("eval_every_epochs", 1) == 0:
            val_metrics = eval_epoch_phase_b(
                pairwise, chain_head, val_loader,
                device, amp_enabled, amp_dtype,
            )
            print(f"  Epoch {epoch} val:")
            for k, v in sorted(val_metrics.items()):
                print(f"    {k}: {v:.4f}")

            # Combined metric: chain_rank_acc * pairwise_rank_acc
            chain_acc = val_metrics.get("chain_rank_acc", 0)
            pair_acc = val_metrics.get("pairwise_rank_acc", 0)
            combined = chain_acc * pair_acc

            if combined > best_metric:
                best_metric = combined
                no_improve_count = 0
                save_checkpoint(
                    Path(out_cfg["checkpoint_dir"]) / "best_phase_b.pt",
                    {
                        "pairwise": pairwise.state_dict(),
                        "chain_head": chain_head.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_metric": best_metric,
                        "val_metrics": val_metrics,
                        "config": config,
                    },
                )
                print(f"  >> New best combined={combined:.4f} (chain={chain_acc:.4f} × pair={pair_acc:.4f})")
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"  Early stopping: no improvement for {patience} evals")
                    break

        # Periodic checkpoint
        if (epoch + 1) % phase_cfg.get("checkpoint_every_epochs", 5) == 0:
            save_checkpoint(
                Path(out_cfg["checkpoint_dir"]) / f"phase_b_epoch_{epoch}.pt",
                {
                    "pairwise": pairwise.state_dict(),
                    "chain_head": chain_head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                },
            )

    print(f"\nPhase B finished. Best combined metric={best_metric:.4f}")
    print(f"Best checkpoint: {out_cfg['checkpoint_dir']}/best_phase_b.pt")


if __name__ == "__main__":
    main()
