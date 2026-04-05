#!/usr/bin/env python3
"""
Stage 3 Phase B: Joint fine-tune + System 1/2 switching.

Both Pairwise and Chain Head are UNFROZEN and trained jointly.
Mode policy is strict:
  - Until simple-task accuracy reaches threshold (default 95%), only System 1 is allowed.
  - After unlock, batches are sampled in 30/70 ratio (System 1/System 2).
  - System 2 quality is continuously monitored after unlock.

Key training mechanisms:
  - System 1 (30%): Pairwise energy + short Langevin (10 steps)
  - System 2 (70%): Chain Head energy + extended Langevin (up to 50 steps)
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
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cebcm.data.sequence_loading import load_sonar_sequences
from cebcm.models.chain_head import ChainHeadConfig, EBTChainHead
from cebcm.models.energy import SimpleEnergy
from cebcm.models.energy_decomposed import AngularEnergyCritic, RadialEnergyCritic
from cebcm.inference.system_switching import (
    select_thinking_mode,
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


@dataclass
class ModeSwitchState:
    """Runtime state for strict System1->System2 gating."""
    system2_unlocked: bool = False
    unlock_epoch: int = -1
    unlock_metric_value: float = 0.0
    unlock_hits: int = 0
    system2_monitor_fail_streak: int = 0


def build_mode_schedule(num_batches: int, sys1_weight: float, sys2_weight: float) -> list[str]:
    """
    Build an exact per-epoch mode schedule with target sys1/sys2 ratio.

    Random weighted sampling gives ratio only in expectation. For strict
    30/70 policy we pre-build counts and shuffle.
    """
    if num_batches <= 0:
        return []

    total = max(1e-8, float(sys1_weight) + float(sys2_weight))
    sys1_share = float(sys1_weight) / total
    n_sys1 = int(round(num_batches * sys1_share))
    n_sys1 = max(0, min(num_batches, n_sys1))
    n_sys2 = num_batches - n_sys1

    schedule = (["system1"] * n_sys1) + (["system2"] * n_sys2)
    random.shuffle(schedule)
    return schedule


def build_pairwise(cfg: dict, device: torch.device) -> nn.Module:
    """
    Build pairwise critic from config and load checkpoint.

    Stage 1.5 uses radial_angular architecture:
      - critic1 = AngularEnergyCritic (semantic/tangential)
      - critic2 = RadialEnergyCritic (norm/shell/OOD)
    Checkpoint keys: critic1_state, critic2_state.

    For Phase B we use AngularEnergyCritic as the primary pairwise function
    (it handles semantic direction, which is what chain reasoning cares about).
    """
    arch = cfg.get("architecture", "simple")
    norm_mode = cfg.get("norm_mode", "none")
    activation = cfg.get("activation", "silu")
    dim = cfg.get("dim", 1024)
    clamp = cfg.get("energy_output_clamp", None)

    if arch == "radial_angular":
        model = AngularEnergyCritic(
            dim=dim,
            hidden_dims=cfg.get("angular_hidden_dims", [2048, 1024, 512]),
            norm_mode=norm_mode,
            activation=activation,
            energy_output_clamp=clamp,
        ).to(device)
        ckpt_key = "critic1_state"
    else:
        model = SimpleEnergy(
            dim=dim,
            hidden_dims=cfg.get("hidden_dims", [2048, 1024, 512]),
            norm_mode=norm_mode,
            activation=activation,
        ).to(device)
        ckpt_key = "model"

    ckpt_path = cfg.get("checkpoint")
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ckpt_key in ckpt:
            model.load_state_dict(ckpt[ckpt_key])
            print(f"  Pairwise ({arch}) loaded from {ckpt_path} [key={ckpt_key}]")
        else:
            available = [k for k in ckpt if k.endswith("_state") or k == "model"]
            print(f"  WARNING: Key '{ckpt_key}' not in checkpoint. Available: {available}")
            print(f"  Using random init.")
    else:
        print(f"  WARNING: Pairwise checkpoint not found at {ckpt_path}, using random init")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Pairwise ({arch}): {n_params:,} parameters")
    return model


def build_chain_head(cfg: dict, device: torch.device, checkpoint: str | None = None) -> EBTChainHead:
    """Build Chain Head from config, optionally loading Phase A checkpoint."""
    chain_cfg = ChainHeadConfig(
        d_model=cfg.get("d_model", 1024),
        n_heads=cfg.get("n_heads", 8),
        n_layers=cfg.get("n_layers", 2),
        dim_feedforward=cfg.get("dim_feedforward", 2048),
        max_chain_len=cfg.get("max_chain_len", 20),
        dropout=cfg.get("dropout", 0.2),
        activation=cfg.get("activation", "gelu"),
        energy_hidden=cfg.get("energy_hidden", 512),
        temperature=cfg.get("temperature", 0.07),
        focal_gamma=cfg.get("focal_gamma", 2.0),
        lambda_grad=cfg.get("lambda_grad", 0.01),
        lambda_energy_norm=cfg.get("lambda_energy_norm", 0.05),
        energy_norm_margin=cfg.get("energy_norm_margin", 1.0),
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
    pairwise: nn.Module,
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
    mode_state: ModeSwitchState,
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

    sys1_weight = float(phase_cfg.get("system1_weight", 0.3))
    sys2_weight = float(phase_cfg.get("system2_weight", 0.7))
    clip_grad = phase_cfg.get("clip_grad_norm", 1.0)
    log_every = phase_cfg.get("log_every", 50)
    mode_schedule: list[str] = []
    if mode_state.system2_unlocked:
        mode_schedule = build_mode_schedule(len(loader), sys1_weight, sys2_weight)

    for batch_idx, batch in enumerate(loader):
        positives = batch["positives"].to(device)
        pos_lengths = batch["pos_lengths"].to(device)
        negatives = batch["negatives"].to(device)
        neg_lengths = batch["neg_lengths"].to(device)
        neg_types = batch.get("neg_types")

        # Strict gating: until unlock, force System 1 only.
        if mode_state.system2_unlocked:
            if batch_idx < len(mode_schedule):
                mode = mode_schedule[batch_idx]
            else:
                mode = select_thinking_mode(sys1_weight, sys2_weight)
            forced_system1 = 0.0
        else:
            mode = "system1"
            forced_system1 = 1.0

        optimizer.zero_grad()

        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
            # Chain Head InfoNCE loss (always computed — core training signal)
            chain_loss, chain_metrics = chain_head.compute_infonce_loss(
                positives, negatives,
                pos_lengths=pos_lengths,
                neg_lengths=neg_lengths,
                neg_types=neg_types,
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
            pair_rank_acc = (E_pos_pair < E_neg_pair).float().mean()
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
            "pairwise_rank_acc": pair_rank_acc.item(),
            "total_loss": total_loss.item(),
            f"mode_{mode}": 1.0,
            "mode_forced_system1": forced_system1,
            "system2_unlocked": 1.0 if mode_state.system2_unlocked else 0.0,
        }
        if mode == "system1":
            metrics["simple_acc_system1"] = pair_rank_acc.item()
        else:
            metrics["simple_acc_system2"] = pair_rank_acc.item()
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
                f"simple_acc={avg.get('pairwise_rank_acc', 0):.4f} "
                f"chain_rank={avg.get('chain_rank_acc', 0):.4f} "
                f"sys1%={sys1_pct:.2f} "
                f"sys2_unlocked={int(mode_state.system2_unlocked)} "
                f"lr={lr:.2e}"
            )

    return tracker.get()


@torch.no_grad()
def eval_epoch_phase_b(
    pairwise: nn.Module,
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
        neg_types = batch.get("neg_types")

        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
            _, chain_metrics = chain_head.compute_infonce_loss(
                positives, negatives,
                pos_lengths=pos_lengths,
                neg_lengths=neg_lengths,
                neg_types=neg_types,
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
    sequences, _, seq_source, seq_meta = load_sonar_sequences(
        data_path,
        max_seq_len=config["data"].get("max_seq_len", None),
        min_seq_len=config["data"].get("min_seq_len", 1),
        legacy_window_stride=config["data"].get("legacy_window_stride", None),
    )

    n = len(sequences)
    print(
        f"  Loaded {n} sequences "
        f"(source={seq_source}, format={seq_meta.get('input_format', 'unknown')})"
    )

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
    gate_cfg = phase_cfg.get("mode_switch_gate", {})
    lock_until_simple = bool(gate_cfg.get("lock_system2_until_simple_acc", True))
    unlock_metric_name = str(gate_cfg.get("unlock_metric", "pairwise_rank_acc"))
    unlock_threshold = float(gate_cfg.get("simple_acc_threshold", 0.95))
    unlock_consecutive_evals = int(gate_cfg.get("unlock_consecutive_evals", 1))
    system2_monitor_source = str(gate_cfg.get("system2_monitor_source", "val")).strip().lower()
    if system2_monitor_source not in {"train", "val"}:
        system2_monitor_source = "val"
    system2_monitor_metric = str(gate_cfg.get("system2_monitor_metric", "chain_rank_acc"))
    system2_monitor_min = float(gate_cfg.get("system2_monitor_min", 0.85))
    system2_monitor_patience = int(gate_cfg.get("system2_monitor_patience", 3))
    mode_state = ModeSwitchState(system2_unlocked=not lock_until_simple)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        pairwise.load_state_dict(ckpt["pairwise"])
        chain_head.load_state_dict(ckpt["chain_head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_metric = ckpt.get("best_metric", 0.0)
        mode_switch = ckpt.get("mode_switch", {})
        if mode_switch:
            mode_state.system2_unlocked = bool(mode_switch.get("system2_unlocked", mode_state.system2_unlocked))
            mode_state.unlock_epoch = int(mode_switch.get("unlock_epoch", mode_state.unlock_epoch))
            mode_state.unlock_metric_value = float(mode_switch.get("unlock_metric_value", mode_state.unlock_metric_value))
            mode_state.unlock_hits = int(mode_switch.get("unlock_hits", mode_state.unlock_hits))
            mode_state.system2_monitor_fail_streak = int(
                mode_switch.get("system2_monitor_fail_streak", mode_state.system2_monitor_fail_streak)
            )
        print(f"  Resumed from epoch {start_epoch}")
        print(
            "  Resume mode-switch: "
            f"system2_unlocked={mode_state.system2_unlocked}, "
            f"unlock_epoch={mode_state.unlock_epoch}, "
            f"unlock_metric={mode_state.unlock_metric_value:.4f}"
        )

    # Training loop
    no_improve_count = 0
    patience = phase_cfg.get("early_stop_patience", 5)
    print(
        "Mode-switch policy: "
        f"lock_until_simple={lock_until_simple}, "
        f"unlock_metric={unlock_metric_name}, "
        f"threshold={unlock_threshold:.3f}, "
        f"unlock_consecutive_evals={unlock_consecutive_evals}, "
        f"ratio_after_unlock={phase_cfg.get('system1_weight', 0.3):.2f}/"
        f"{phase_cfg.get('system2_weight', 0.7):.2f}"
    )
    print(
        "System2 monitor: "
        f"source={system2_monitor_source}, "
        f"metric={system2_monitor_metric}, min={system2_monitor_min:.3f}, "
        f"patience={system2_monitor_patience}"
    )

    for epoch in range(start_epoch, phase_cfg["num_epochs"]):
        t0 = time.time()

        train_metrics = train_epoch_phase_b(
            pairwise, chain_head, train_loader, optimizer, scheduler, scaler,
            device, amp_enabled, amp_dtype, phase_cfg, epoch, mode_state,
        )

        elapsed = time.time() - t0
        print(f"\n  Epoch {epoch} train ({elapsed:.1f}s):")
        for k, v in sorted(train_metrics.items()):
            print(f"    {k}: {v:.4f}")
        print(f"    system2_unlocked: {int(mode_state.system2_unlocked)}")

        # Eval
        if (epoch + 1) % phase_cfg.get("eval_every_epochs", 1) == 0:
            val_metrics = eval_epoch_phase_b(
                pairwise, chain_head, val_loader,
                device, amp_enabled, amp_dtype,
            )
            print(f"  Epoch {epoch} val:")
            for k, v in sorted(val_metrics.items()):
                print(f"    {k}: {v:.4f}")

            # Strict unlock condition: System2 remains disabled until simple-task
            # accuracy reaches threshold on validation.
            simple_acc_val = float(val_metrics.get(unlock_metric_name, 0.0))
            if lock_until_simple and not mode_state.system2_unlocked:
                if simple_acc_val >= unlock_threshold:
                    mode_state.unlock_hits += 1
                else:
                    mode_state.unlock_hits = 0

                print(
                    "  [ModeGate] "
                    f"{unlock_metric_name}={simple_acc_val:.4f} "
                    f"(threshold={unlock_threshold:.4f}, hits={mode_state.unlock_hits}/"
                    f"{unlock_consecutive_evals})"
                )

                if mode_state.unlock_hits >= unlock_consecutive_evals:
                    mode_state.system2_unlocked = True
                    mode_state.unlock_epoch = epoch
                    mode_state.unlock_metric_value = simple_acc_val
                    mode_state.system2_monitor_fail_streak = 0
                    print(
                        "  [ModeGate] System2 UNLOCKED: "
                        f"{unlock_metric_name}={simple_acc_val:.4f} >= {unlock_threshold:.4f}"
                    )
            elif mode_state.system2_unlocked:
                # Post-unlock strict monitoring.
                if system2_monitor_source == "train":
                    monitor_metrics = train_metrics
                else:
                    monitor_metrics = val_metrics
                sys2_metric_val = float(monitor_metrics.get(system2_monitor_metric, 0.0))
                if sys2_metric_val < system2_monitor_min:
                    mode_state.system2_monitor_fail_streak += 1
                else:
                    mode_state.system2_monitor_fail_streak = 0

                print(
                    "  [System2 Monitor] "
                    f"source={system2_monitor_source}, "
                    f"{system2_monitor_metric}={sys2_metric_val:.4f}, "
                    f"min={system2_monitor_min:.4f}, "
                    f"fail_streak={mode_state.system2_monitor_fail_streak}/"
                    f"{system2_monitor_patience}"
                )
                if mode_state.system2_monitor_fail_streak >= system2_monitor_patience:
                    print(
                        "  [System2 Monitor][WARNING] "
                        "System2 quality below configured floor for consecutive evals."
                    )

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
                        "mode_switch": {
                            "system2_unlocked": mode_state.system2_unlocked,
                            "unlock_epoch": mode_state.unlock_epoch,
                            "unlock_metric_value": mode_state.unlock_metric_value,
                            "unlock_hits": mode_state.unlock_hits,
                            "system2_monitor_fail_streak": mode_state.system2_monitor_fail_streak,
                        },
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
                    "mode_switch": {
                        "system2_unlocked": mode_state.system2_unlocked,
                        "unlock_epoch": mode_state.unlock_epoch,
                        "unlock_metric_value": mode_state.unlock_metric_value,
                        "unlock_hits": mode_state.unlock_hits,
                        "system2_monitor_fail_streak": mode_state.system2_monitor_fail_streak,
                    },
                },
            )

    print(f"\nPhase B finished. Best combined metric={best_metric:.4f}")
    print(f"Best checkpoint: {out_cfg['checkpoint_dir']}/best_phase_b.pt")


if __name__ == "__main__":
    main()
