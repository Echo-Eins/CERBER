#!/usr/bin/env python3
"""
Stage 2 Training: ContextEncoder + IPP + SurprisePredictor.

Three-phase training protocol:
  Phase A (epochs 0..surprise_pretrain_epochs):
    - Train SurprisePredictor alone (self-supervised next-vector prediction)
    - Freeze SP after this phase

  Phase B (epochs surprise_pretrain_epochs..num_epochs):
    - Train ContextEncoder + IPP jointly
    - SP provides surprise scores (frozen)
    - CE produces V_context → IPP generates V_init proposals
    - Loss: IPP flow matching loss (or MLP MSE+cos)

  Evaluation:
    - Surprise: mean surprise score, prediction cosine
    - IPP: generated V_init cosine similarity to ground truth answer
    - ContextEncoder: qualitative inspection of V_context

Usage:
    python experiments/08_autoregressor/train_stage2.py \
        --config configs/stage2_config.json

    python experiments/08_autoregressor/train_stage2.py \
        --config configs/stage2_config.json \
        --resume experiments/08_autoregressor/checkpoints/latest.pt

Spec reference: §9.5-§9.9, IMPLEMENTATION_PLAN.md Stage 2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cebcm.data.sequence_dataset import SONARSequenceDataset
from cebcm.models.context_encoder import ContextEncoder, ContextEncoderConfig
from cebcm.models.ipp import FlowIPP, MLPIPP, IPPConfig
from cebcm.models.surprise import SurprisePredictor, SurpriseConfig


# ============================================================
# Config loading
# ============================================================

def load_config(config_path: str) -> dict:
    """Load JSON config file."""
    with open(config_path) as f:
        return json.load(f)


def build_surprise_predictor(cfg: dict) -> SurprisePredictor:
    sp_cfg = cfg.get("surprise", {})
    return SurprisePredictor(SurpriseConfig(
        d_model=sp_cfg.get("d_model", 1024),
        ssm_d_state=sp_cfg.get("ssm_d_state", 64),
        ssm_d_conv=sp_cfg.get("ssm_d_conv", 4),
        ssm_expand=sp_cfg.get("ssm_expand", 2),
        ssm_n_layers=sp_cfg.get("ssm_n_layers", 2),
        ssm_dropout=sp_cfg.get("ssm_dropout", 0.0),
        pred_hidden=sp_cfg.get("pred_hidden", 1024),
        cos_loss_weight=sp_cfg.get("cos_loss_weight", 0.5),
        threshold_mode=sp_cfg.get("threshold_mode", "percentile"),
        threshold_percentile=sp_cfg.get("threshold_percentile", 95.0),
        threshold_fixed=sp_cfg.get("threshold_fixed", 0.3),
    ))


def build_context_encoder(cfg: dict) -> ContextEncoder:
    ce_cfg = cfg.get("context_encoder", {})
    return ContextEncoder(ContextEncoderConfig(
        d_model=ce_cfg.get("d_model", 1024),
        ssm_d_state=ce_cfg.get("ssm_d_state", 64),
        ssm_d_conv=ce_cfg.get("ssm_d_conv", 4),
        ssm_expand=ce_cfg.get("ssm_expand", 2),
        ssm_n_layers=ce_cfg.get("ssm_n_layers", 2),
        ssm_dropout=ce_cfg.get("ssm_dropout", 0.05),
        n_global_heads=ce_cfg.get("n_global_heads", 8),
        global_attn_dropout=ce_cfg.get("global_attn_dropout", 0.1),
        surprise_top_k_pct=ce_cfg.get("surprise_top_k_pct", 0.05),
        surprise_top_k_min_tokens=ce_cfg.get("surprise_top_k_min_tokens", 1),
        global_include_last_token=ce_cfg.get("global_include_last_token", True),
        n_types=ce_cfg.get("n_types", 3),
        output_dim=ce_cfg.get("output_dim", 1024),
        use_alibi=ce_cfg.get("use_alibi", False),
    ))


def build_ipp(cfg: dict) -> FlowIPP | MLPIPP:
    ipp_cfg = cfg.get("ipp", {})
    ipp_config = IPPConfig(
        d_model=ipp_cfg.get("d_model", 1024),
        d_context=ipp_cfg.get("d_context", 1024),
        hidden_dims=ipp_cfg.get("hidden_dims", [2048, 2048, 1024]),
        flow_activation=ipp_cfg.get("flow_activation", "silu"),
        flow_norm=ipp_cfg.get("flow_norm", "layernorm"),
        flow_dropout=ipp_cfg.get("flow_dropout", 0.0),
        flow_zero_init_last=ipp_cfg.get("flow_zero_init_last", True),
        flow_velocity_weight=ipp_cfg.get("flow_velocity_weight", 1.0),
        n_integration_steps=ipp_cfg.get("n_integration_steps", 50),
        d_time=ipp_cfg.get("d_time", 256),
        sigma_init=ipp_cfg.get("sigma_init", 0.05),
        solver=ipp_cfg.get("solver", "midpoint"),
        endpoint_loss_weight=ipp_cfg.get("endpoint_loss_weight", 0.0),
        endpoint_cos_weight=ipp_cfg.get("endpoint_cos_weight", 0.5),
        endpoint_steps=ipp_cfg.get("endpoint_steps", 20),
        endpoint_target_norm=ipp_cfg.get("endpoint_target_norm", 0.2051),
        mlp_hidden_dims=ipp_cfg.get("mlp_hidden_dims", [2048, 1024]),
        mlp_activation=ipp_cfg.get("mlp_activation", "silu"),
        mlp_norm=ipp_cfg.get("mlp_norm", "layernorm"),
        mlp_dropout=ipp_cfg.get("mlp_dropout", 0.0),
        mlp_mse_weight=ipp_cfg.get("mlp_mse_weight", 1.0),
        mlp_cos_weight=ipp_cfg.get("mlp_cos_weight", 0.5),
        mlp_contrastive_weight=ipp_cfg.get("mlp_contrastive_weight", 0.0),
        mlp_temperature=ipp_cfg.get("mlp_temperature", 0.07),
    )
    mode = ipp_cfg.get("mode", "flow")
    if mode == "flow":
        return FlowIPP(ipp_config)
    return MLPIPP(ipp_config)


# ============================================================
# Training utilities
# ============================================================

def get_cosine_schedule_with_warmup(
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_factor: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing with linear warmup."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(min_factor, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_factor, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_type_ids(lengths: Tensor, max_len: int, dataset_source: str) -> Tensor:
    """
    Build type_ids for the current dataset.

    SQuAD convention:
      last 2 elements are [question, answer].
      context -> type=1, question -> type=0, answer -> type=1.

    Non-SQuAD corpora (WikiText/CNN):
      no explicit QA markup in each sequence -> keep all type_ids=0.
    """
    B = lengths.shape[0]
    type_ids = torch.zeros(B, max_len, dtype=torch.long)

    if dataset_source.lower().startswith("squad"):
        for i in range(B):
            L = int(lengths[i].item())
            if L >= 3:
                # Context sentences: type=1 (answer-like context facts)
                type_ids[i, : L - 2] = 1
                # Question: type=0 (query)
                type_ids[i, L - 2] = 0
                # Answer: type=1 (answer)
                type_ids[i, L - 1] = 1

    return type_ids


class MetricTracker:
    """Rolling average metric tracker."""

    def __init__(self):
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def update(self, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            self._sums[k] = self._sums.get(k, 0.0) + v
            self._counts[k] = self._counts.get(k, 0) + 1

    def get(self) -> dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


# ============================================================
# Phase A: Surprise Predictor pretraining
# ============================================================

def train_surprise_epoch(
        model: SurprisePredictor,
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
    """Train SurprisePredictor for one epoch."""
    model.train()
    tracker = MetricTracker()
    step = 0

    for batch in loader:
        vectors = batch["vectors"].to(device)  # [B, L, D]
        lengths = batch["lengths"].to(device)  # [B]

        # Keep only sequences with at least 3 vectors (2 prediction steps).
        valid_rows = lengths >= 3
        if not valid_rows.any():
            continue
        vectors = vectors[valid_rows]
        lengths = lengths[valid_rows]
        _, L, _ = vectors.shape

        if L < 3:
            continue

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            loss, metrics = model.compute_loss(vectors, lengths=lengths)

        if not torch.isfinite(loss):
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        tracker.update(metrics)
        step += 1

        if step % log_every == 0:
            avg = tracker.get()
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [SP] epoch={epoch} step={step} "
                f"loss={avg.get('surprise_loss', 0):.4f} "
                f"mse={avg.get('surprise_mse', 0):.4f} "
                f"cos={avg.get('surprise_cos', 0):.4f} "
                f"surprise_mean={avg.get('surprise_mean', 0):.4f} "
                f"lr={lr:.2e}"
            )

    return tracker.get()


@torch.no_grad()
def eval_surprise(
        model: SurprisePredictor,
        loader: DataLoader,
        device: torch.device,
        amp_enabled: bool,
        amp_dtype: torch.dtype,
) -> dict[str, float]:
    """Evaluate SurprisePredictor on validation set."""
    model.eval()
    tracker = MetricTracker()

    for batch in loader:
        vectors = batch["vectors"].to(device)
        lengths = batch["lengths"].to(device)

        valid_rows = lengths >= 3
        if not valid_rows.any():
            continue
        vectors = vectors[valid_rows]
        lengths = lengths[valid_rows]

        if vectors.shape[1] < 3:
            continue

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            _, metrics = model.compute_loss(vectors, lengths=lengths)

        tracker.update(metrics)

    return tracker.get()


# ============================================================
# Phase B: Joint ContextEncoder + IPP training
# ============================================================

def train_joint_epoch(
        context_encoder: ContextEncoder,
        ipp: FlowIPP | MLPIPP,
        surprise_predictor: SurprisePredictor,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        scaler: torch.amp.GradScaler,
        device: torch.device,
        amp_enabled: bool,
        amp_dtype: torch.dtype,
        clip_grad: float,
        target_norm: float,
        log_every: int,
        epoch: int,
        dataset_source: str,
) -> dict[str, float]:
    """Train ContextEncoder + IPP jointly for one epoch."""
    context_encoder.train()
    ipp.train()
    surprise_predictor.eval()  # frozen

    tracker = MetricTracker()
    step = 0

    for batch in loader:
        vectors = batch["vectors"].to(device)  # [B, L, D]
        lengths = batch["lengths"].to(device)  # [B]
        valid_rows = lengths >= 4
        if not valid_rows.any():
            continue
        vectors = vectors[valid_rows]
        lengths = lengths[valid_rows]
        B, L, D = vectors.shape

        if L < 4:  # Need at least context + question + answer
            continue

        # For SQuAD-style: context is vectors[:, :-1], target is vectors[:, -1]
        # Context = everything except the last vector (the answer)
        context_vecs = vectors[:, :-1]  # [B, L-1, D]
        target_idx = (lengths - 1).long()
        target_vecs = vectors[torch.arange(B, device=device), target_idx]  # [B, D]
        context_lengths = lengths - 1  # [B]

        # Type IDs for the context portion
        type_ids = build_type_ids(lengths, L, dataset_source=dataset_source)[:, :-1].to(device)  # [B, L-1]

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            # Step 1: Get surprise scores (frozen SP)
            with torch.no_grad():
                surprise_scores = surprise_predictor.compute_surprise(context_vecs, lengths=context_lengths)
                # Pad surprise to match context length (first position has no surprise)
                surprise_padded = torch.zeros(B, L - 1, device=device)
                if surprise_scores.shape[1] > 0:
                    surprise_padded[:, 1:1 + surprise_scores.shape[1]] = surprise_scores

            # Step 2: ContextEncoder produces V_context
            v_context = context_encoder(
                context_vectors=context_vecs,
                type_ids=type_ids,
                surprise_scores=surprise_padded,
                lengths=context_lengths,
            )  # [B, output_dim]

            # Step 3: IPP loss — predict the target answer vector
            ipp_loss, ipp_metrics = ipp.compute_loss(v_context, target_vecs)

        if not torch.isfinite(ipp_loss):
            continue

        scaler.scale(ipp_loss).backward()
        scaler.unscale_(optimizer)

        # Clip grads for both models
        all_params = list(context_encoder.parameters()) + list(ipp.parameters())
        nn.utils.clip_grad_norm_(all_params, clip_grad)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        tracker.update(ipp_metrics)
        step += 1

        if step % log_every == 0:
            avg = tracker.get()
            lr = optimizer.param_groups[0]["lr"]
            loss_key = "flow_loss" if "flow_loss" in avg else "ipp_loss"
            cos_key = "flow_cos" if "flow_cos" in avg else "ipp_cos"
            print(
                f"  [CE+IPP] epoch={epoch} step={step} "
                f"loss={avg.get(loss_key, 0):.4f} "
                f"cos={avg.get(cos_key, 0):.4f} "
                f"lr={lr:.2e}"
            )

    return tracker.get()


@torch.no_grad()
def eval_joint(
        context_encoder: ContextEncoder,
        ipp: FlowIPP | MLPIPP,
        surprise_predictor: SurprisePredictor,
        loader: DataLoader,
        device: torch.device,
        amp_enabled: bool,
        amp_dtype: torch.dtype,
        target_norm: float,
        dataset_source: str,
) -> dict[str, float]:
    """Evaluate ContextEncoder + IPP on validation set."""
    context_encoder.eval()
    ipp.eval()
    surprise_predictor.eval()

    all_cos_sims: list[float] = []
    all_l2_dists: list[float] = []
    tracker = MetricTracker()

    for batch in loader:
        vectors = batch["vectors"].to(device)
        lengths = batch["lengths"].to(device)
        valid_rows = lengths >= 4
        if not valid_rows.any():
            continue
        vectors = vectors[valid_rows]
        lengths = lengths[valid_rows]
        B, L, D = vectors.shape
        if L < 4:
            continue

        context_vecs = vectors[:, :-1]
        target_idx = (lengths - 1).long()
        target_vecs = vectors[torch.arange(B, device=device), target_idx]  # [B, D]
        context_lengths = lengths - 1

        type_ids = build_type_ids(lengths, L, dataset_source=dataset_source)[:, :-1].to(device)  # [B, L-1]

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            # Surprise scores
            surprise_scores = surprise_predictor.compute_surprise(context_vecs, lengths=context_lengths)
            surprise_padded = torch.zeros(B, L - 1, device=device)
            if surprise_scores.shape[1] > 0:
                surprise_padded[:, 1:1 + surprise_scores.shape[1]] = surprise_scores

            # Context encoding
            v_context = context_encoder(
                context_vectors=context_vecs,
                type_ids=type_ids,
                surprise_scores=surprise_padded,
                lengths=context_lengths,
            )

            # Loss
            _, metrics = ipp.compute_loss(v_context, target_vecs)

            # Generate proposal and evaluate
            v_init = ipp.sample(v_context, target_norm=target_norm)

        # Cosine similarity between proposal and target
        cos_sim = F.cosine_similarity(v_init, target_vecs, dim=-1)
        l2_dist = (v_init - target_vecs).norm(dim=-1)

        all_cos_sims.extend(cos_sim.cpu().tolist())
        all_l2_dists.extend(l2_dist.cpu().tolist())
        tracker.update(metrics)

    avg = tracker.get()
    if not all_cos_sims:
        avg.setdefault("eval_cos_mean", 0.0)
        avg.setdefault("eval_cos_std", 0.0)
        avg.setdefault("eval_cos_gt05", 0.0)
        avg.setdefault("eval_cos_gt07", 0.0)
        avg.setdefault("eval_l2_mean", 0.0)
        return avg

    cos_tensor = torch.tensor(all_cos_sims)
    l2_tensor = torch.tensor(all_l2_dists)

    avg["eval_cos_mean"] = cos_tensor.mean().item()
    avg["eval_cos_std"] = cos_tensor.std().item()
    avg["eval_cos_gt05"] = (cos_tensor > 0.5).float().mean().item()
    avg["eval_cos_gt07"] = (cos_tensor > 0.7).float().mean().item()
    avg["eval_l2_mean"] = l2_tensor.mean().item()

    return avg


# ============================================================
# Checkpointing
# ============================================================

def save_checkpoint(
        path: Path,
        epoch: int,
        phase: str,
        surprise_predictor: SurprisePredictor,
        context_encoder: ContextEncoder,
        ipp: FlowIPP | MLPIPP,
        optimizers: dict[str, torch.optim.Optimizer],
        metrics: dict[str, float],
) -> None:
    """Save training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "phase": phase,
        "surprise_predictor": surprise_predictor.state_dict(),
        "context_encoder": context_encoder.state_dict(),
        "ipp": ipp.state_dict(),
        "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
        "metrics": metrics,
    }, path)


def load_checkpoint(
        path: Path,
        surprise_predictor: SurprisePredictor,
        context_encoder: ContextEncoder,
        ipp: FlowIPP | MLPIPP,
        optimizers: dict[str, torch.optim.Optimizer] | None = None,
        device: torch.device = torch.device("cpu"),
) -> tuple[int, str, dict]:
    """Load checkpoint, return (epoch, phase, metrics)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    surprise_predictor.load_state_dict(ckpt["surprise_predictor"])
    context_encoder.load_state_dict(ckpt["context_encoder"])
    ipp.load_state_dict(ckpt["ipp"])
    if optimizers and "optimizers" in ckpt:
        for k, opt in optimizers.items():
            if k in ckpt["optimizers"]:
                opt.load_state_dict(ckpt["optimizers"][k])
    return ckpt["epoch"], ckpt["phase"], ckpt.get("metrics", {})


# ============================================================
# Main training loop
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Autoregressor Training")
    parser.add_argument("--config", type=str, required=True, help="JSON config path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    amp_cfg = cfg.get("amp", {})
    out_cfg = cfg.get("output", {})

    # Device
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # Seed
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # AMP
    amp_enabled = amp_cfg.get("enabled", True) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_cfg.get("dtype", "bf16") == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

    # ---- Data ----
    print("Loading dataset...")
    dataset = SONARSequenceDataset(
        path=data_cfg.get("train_data_path", "data/squad_sequences.pt"),
        max_seq_len=data_cfg.get("max_seq_len", 64),
        min_seq_len=data_cfg.get("min_seq_len", 3),
    )

    val_path = data_cfg.get("val_data_path", "")
    if val_path:
        val_dataset = SONARSequenceDataset(
            path=val_path,
            max_seq_len=data_cfg.get("max_seq_len", 64),
            min_seq_len=data_cfg.get("min_seq_len", 3),
        )
        train_dataset = dataset
    else:
        train_dataset, val_dataset = dataset.split(
            train_ratio=data_cfg.get("train_val_split", 0.9),
            seed=seed,
        )

    stats = train_dataset.get_statistics()
    print(f"  Train: {stats['num_sequences']} sequences, "
          f"mean_len={stats['mean_length']:.1f}, mean_norm={stats['mean_norm']:.4f}")
    val_stats = val_dataset.get_statistics()
    print(f"  Val:   {val_stats['num_sequences']} sequences")
    train_source = str(getattr(train_dataset, "source", "unknown"))
    val_source = str(getattr(val_dataset, "source", train_source))

    num_workers = data_cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=True,
        collate_fn=SONARSequenceDataset.collate,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=False,
        collate_fn=SONARSequenceDataset.collate,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ---- Build models ----
    print("Building models...")
    surprise_predictor = build_surprise_predictor(cfg).to(device)
    context_encoder = build_context_encoder(cfg).to(device)
    ipp = build_ipp(cfg).to(device)
    ipp_mode = str(cfg.get("ipp", {}).get("mode", "flow")).lower()

    print(f"  SurprisePredictor: {surprise_predictor.num_params:,} params")
    print(f"  ContextEncoder:    {context_encoder.num_params:,} params")
    print(f"  IPP:               {ipp.num_params:,} params ({ipp_mode}, {ipp.__class__.__name__})")
    total = surprise_predictor.num_params + context_encoder.num_params + ipp.num_params
    print(f"  Total:             {total:,} params ({total / 1e6:.1f}M)")

    # ---- Optimizers ----
    sp_optimizer = torch.optim.AdamW(
        surprise_predictor.parameters(),
        lr=train_cfg.get("surprise_lr", 3e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    joint_params = [
        {"params": context_encoder.parameters(), "lr": train_cfg.get("context_encoder_lr", 1e-4)},
        {"params": ipp.parameters(), "lr": train_cfg.get("ipp_lr", 1e-4)},
    ]
    joint_optimizer = torch.optim.AdamW(
        joint_params,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    optimizers = {"surprise": sp_optimizer, "joint": joint_optimizer}

    # ---- LR Schedulers ----
    surprise_pretrain_epochs = train_cfg.get("surprise_pretrain_epochs", 10)
    num_epochs = train_cfg.get("num_epochs", 50)
    warmup_epochs = train_cfg.get("warmup_epochs", 3)
    lr_min_factor = train_cfg.get("lr_min_factor", 0.01)
    steps_per_epoch = len(train_loader)

    sp_scheduler = get_cosine_schedule_with_warmup(
        sp_optimizer,
        warmup_steps=warmup_epochs * steps_per_epoch,
        total_steps=surprise_pretrain_epochs * steps_per_epoch,
        min_factor=lr_min_factor,
    )

    joint_epochs = num_epochs - surprise_pretrain_epochs
    joint_scheduler = get_cosine_schedule_with_warmup(
        joint_optimizer,
        warmup_steps=warmup_epochs * steps_per_epoch,
        total_steps=joint_epochs * steps_per_epoch,
        min_factor=lr_min_factor,
    )

    # ---- Resume ----
    start_epoch = 0
    current_phase = "A"  # A=surprise pretrain, B=joint
    if args.resume:
        print(f"Resuming from {args.resume}...")
        start_epoch, current_phase, prev_metrics = load_checkpoint(
            Path(args.resume), surprise_predictor, context_encoder, ipp,
            optimizers=optimizers, device=device,
        )
        print(f"  Resumed at epoch {start_epoch}, phase {current_phase}")
        print(f"  Previous metrics: {prev_metrics}")
        start_epoch += 1

    # ---- Paths ----
    output_dir = Path(out_cfg.get("output_dir", "experiments/08_autoregressor"))
    checkpoint_dir = Path(out_cfg.get("checkpoint_dir", "experiments/08_autoregressor/checkpoints"))
    logs_dir = Path(out_cfg.get("logs_dir", "experiments/08_autoregressor/logs"))
    for d in [output_dir, checkpoint_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Training params
    clip_grad = train_cfg.get("clip_grad_norm", 1.0)
    target_norm = train_cfg.get("target_norm", 0.2051)
    log_every = train_cfg.get("log_every", 50)
    eval_every = train_cfg.get("eval_every_epochs", 2)
    ckpt_every = train_cfg.get("checkpoint_every_epochs", 5)

    # ---- Training log ----
    log_path = logs_dir / "training_log.jsonl"
    log_file = open(log_path, "a")

    def log_metrics(epoch: int, phase: str, metrics: dict, elapsed: float):
        entry = {"epoch": epoch, "phase": phase, "elapsed_s": elapsed, **metrics}
        log_file.write(json.dumps(entry) + "\n")
        log_file.flush()

    # ============================================================
    # PHASE A: Surprise Predictor Pretraining
    # ============================================================
    print(f"\n{'=' * 60}")
    print(f"PHASE A: SurprisePredictor Pretraining (epochs 0..{surprise_pretrain_epochs - 1})")
    print(f"{'=' * 60}")

    if current_phase == "A":
        for epoch in range(start_epoch, surprise_pretrain_epochs):
            t0 = time.time()
            train_metrics = train_surprise_epoch(
                model=surprise_predictor,
                loader=train_loader,
                optimizer=sp_optimizer,
                scheduler=sp_scheduler,
                scaler=scaler,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                clip_grad=clip_grad,
                log_every=log_every,
                epoch=epoch,
            )
            elapsed = time.time() - t0

            print(f"[SP] Epoch {epoch} done ({elapsed:.1f}s)")
            log_metrics(epoch, "A", train_metrics, elapsed)

            # Eval
            if (epoch + 1) % eval_every == 0 or epoch == surprise_pretrain_epochs - 1:
                val_metrics = eval_surprise(
                    surprise_predictor, val_loader, device, amp_enabled, amp_dtype
                )
                print(
                    f"  [SP VAL] "
                    f"loss={val_metrics.get('surprise_loss', 0):.4f} "
                    f"mse={val_metrics.get('surprise_mse', 0):.4f} "
                    f"cos={val_metrics.get('surprise_cos', 0):.4f} "
                    f"surprise_mean={val_metrics.get('surprise_mean', 0):.4f}"
                )
                log_metrics(epoch, "A_val", val_metrics, 0)

            # Checkpoint
            if (epoch + 1) % ckpt_every == 0 or epoch == surprise_pretrain_epochs - 1:
                save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch:03d}.pt",
                    epoch, "A", surprise_predictor, context_encoder, ipp,
                    optimizers, train_metrics,
                )
                save_checkpoint(
                    checkpoint_dir / "latest.pt",
                    epoch, "A", surprise_predictor, context_encoder, ipp,
                    optimizers, train_metrics,
                )

        # Freeze SurprisePredictor
        if train_cfg.get("freeze_surprise_after_pretrain", True):
            print("\n  Freezing SurprisePredictor (training complete)")
            for param in surprise_predictor.parameters():
                param.requires_grad_(False)
            surprise_predictor.eval()

        current_phase = "B"
        start_epoch = surprise_pretrain_epochs

    # ============================================================
    # PHASE B: Joint ContextEncoder + IPP Training
    # ============================================================
    print(f"\n{'=' * 60}")
    print(f"PHASE B: Joint CE + IPP Training (epochs {surprise_pretrain_epochs}..{num_epochs - 1})")
    print(f"{'=' * 60}")

    best_cos = -1.0

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        train_metrics = train_joint_epoch(
            context_encoder=context_encoder,
            ipp=ipp,
            surprise_predictor=surprise_predictor,
            loader=train_loader,
            optimizer=joint_optimizer,
            scheduler=joint_scheduler,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            clip_grad=clip_grad,
            target_norm=target_norm,
            log_every=log_every,
            epoch=epoch,
            dataset_source=train_source,
        )
        elapsed = time.time() - t0

        print(f"[CE+IPP] Epoch {epoch} done ({elapsed:.1f}s)")
        log_metrics(epoch, "B", train_metrics, elapsed)

        # Eval
        if (epoch + 1) % eval_every == 0 or epoch == num_epochs - 1:
            val_metrics = eval_joint(
                context_encoder, ipp, surprise_predictor,
                val_loader, device, amp_enabled, amp_dtype, target_norm, val_source,
            )
            loss_key = "flow_loss" if "flow_loss" in val_metrics else "ipp_loss"
            cos_key = "flow_cos" if "flow_cos" in val_metrics else "ipp_cos"

            print(
                f"  [VAL] "
                f"loss={val_metrics.get(loss_key, 0):.4f} "
                f"cos={val_metrics.get(cos_key, 0):.4f} "
                f"eval_cos_mean={val_metrics.get('eval_cos_mean', 0):.4f} "
                f"eval_cos>0.5={val_metrics.get('eval_cos_gt05', 0):.1%} "
                f"eval_cos>0.7={val_metrics.get('eval_cos_gt07', 0):.1%} "
                f"eval_l2={val_metrics.get('eval_l2_mean', 0):.4f}"
            )
            log_metrics(epoch, "B_val", val_metrics, 0)

            # Best model tracking
            cos_mean = val_metrics.get("eval_cos_mean", 0)
            if cos_mean > best_cos:
                best_cos = cos_mean
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    epoch, "B", surprise_predictor, context_encoder, ipp,
                    optimizers, val_metrics,
                )
                print(f"  ** New best: cos_mean={best_cos:.4f}")

        # Checkpoint
        if (epoch + 1) % ckpt_every == 0 or epoch == num_epochs - 1:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:03d}.pt",
                epoch, "B", surprise_predictor, context_encoder, ipp,
                optimizers, train_metrics,
            )
            save_checkpoint(
                checkpoint_dir / "latest.pt",
                epoch, "B", surprise_predictor, context_encoder, ipp,
                optimizers, train_metrics,
            )

    # ---- Final summary ----
    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"  Best eval cosine similarity: {best_cos:.4f}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Training log: {log_path}")
    print(f"{'=' * 60}")

    log_file.close()


if __name__ == "__main__":
    main()

