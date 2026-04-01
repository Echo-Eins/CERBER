#!/usr/bin/env python3
"""
Standalone Stage 2 CE pretraining.

Trains ContextEncoder with a lightweight projection head on top of V_context:
  V_context -> V_target (answer vector)

The projection head is training-only and is saved in checkpoints for resume,
but downstream modules consume the pretrained ContextEncoder weights.
"""

from __future__ import annotations

import argparse
import json
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

from cebcm.models.context_encoder import ContextEncoder
from cebcm.models.surprise import SurprisePredictor
from cebcm.training.stage2_utils import (
    MetricTracker,
    build_context_encoder,
    build_dataloaders,
    build_surprise_predictor,
    build_type_ids,
    get_cosine_schedule_with_warmup,
    load_checkpoint,
    load_config,
    load_train_val_datasets,
    resolve_device,
    save_checkpoint,
    setup_amp,
    setup_seed,
)


class CEPretrainHead(nn.Module):
    """Projection head used only for CE pretraining objective."""

    def __init__(self, d_in: int, d_out: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _build_surprise_padded(
    surprise_predictor: SurprisePredictor | None,
    context_vecs: Tensor,
    context_lengths: Tensor,
) -> Tensor | None:
    if surprise_predictor is None:
        return None
    B, L_ctx, _ = context_vecs.shape
    with torch.no_grad():
        surprise_scores = surprise_predictor.compute_surprise(context_vecs, lengths=context_lengths)
        surprise_padded = torch.zeros(B, L_ctx, device=context_vecs.device, dtype=context_vecs.dtype)
        if surprise_scores.shape[1] > 0:
            surprise_padded[:, 1 : 1 + surprise_scores.shape[1]] = surprise_scores
    return surprise_padded


def _ce_batch(
    context_encoder: ContextEncoder,
    pretrain_head: CEPretrainHead,
    surprise_predictor: SurprisePredictor | None,
    batch: dict,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    dataset_source: str,
    cos_loss_weight: float,
) -> tuple[Tensor | None, dict[str, float]]:
    vectors = batch["vectors"].to(device)
    lengths = batch["lengths"].to(device)
    valid_rows = lengths >= 4
    if not valid_rows.any():
        return None, {}
    vectors = vectors[valid_rows]
    lengths = lengths[valid_rows]
    B, L, _ = vectors.shape
    if L < 4:
        return None, {}

    context_vecs = vectors[:, :-1]
    context_lengths = lengths - 1
    target_idx = (lengths - 1).long()
    target_vecs = vectors[torch.arange(B, device=device), target_idx]
    type_ids = build_type_ids(lengths, L, dataset_source=dataset_source)[:, :-1].to(device)
    surprise_padded = _build_surprise_padded(surprise_predictor, context_vecs, context_lengths)

    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        v_context = context_encoder(
            context_vectors=context_vecs,
            type_ids=type_ids,
            surprise_scores=surprise_padded,
            lengths=context_lengths,
        )
        pred_vecs = pretrain_head(v_context)
        loss_mse = F.mse_loss(pred_vecs, target_vecs)
        loss_cos = (1.0 - F.cosine_similarity(pred_vecs, target_vecs, dim=-1)).mean()
        loss = loss_mse + cos_loss_weight * loss_cos

    with torch.no_grad():
        cos_mean = F.cosine_similarity(pred_vecs, target_vecs, dim=-1).mean().item()
        pred_norm = pred_vecs.norm(dim=-1).mean().item()

    metrics = {
        "ce_loss": loss.item(),
        "ce_mse": loss_mse.item(),
        "ce_cos_loss": loss_cos.item(),
        "ce_cos": cos_mean,
        "ce_pred_norm": pred_norm,
    }
    return loss, metrics


def train_epoch(
    context_encoder: ContextEncoder,
    pretrain_head: CEPretrainHead,
    surprise_predictor: SurprisePredictor | None,
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
    dataset_source: str,
    cos_loss_weight: float,
) -> dict[str, float]:
    context_encoder.train()
    pretrain_head.train()
    if surprise_predictor is not None:
        surprise_predictor.eval()

    tracker = MetricTracker()
    step = 0

    for batch in loader:
        loss, metrics = _ce_batch(
            context_encoder=context_encoder,
            pretrain_head=pretrain_head,
            surprise_predictor=surprise_predictor,
            batch=batch,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            dataset_source=dataset_source,
            cos_loss_weight=cos_loss_weight,
        )
        if loss is None or not torch.isfinite(loss):
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            list(context_encoder.parameters()) + list(pretrain_head.parameters()),
            clip_grad,
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        tracker.update(metrics)
        step += 1

        if step % log_every == 0:
            avg = tracker.get()
            print(
                f"  [CE] epoch={epoch} step={step} "
                f"loss={avg.get('ce_loss', 0.0):.4f} "
                f"cos={avg.get('ce_cos', 0.0):.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    return tracker.get()


@torch.no_grad()
def eval_epoch(
    context_encoder: ContextEncoder,
    pretrain_head: CEPretrainHead,
    surprise_predictor: SurprisePredictor | None,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    dataset_source: str,
    cos_loss_weight: float,
) -> dict[str, float]:
    context_encoder.eval()
    pretrain_head.eval()
    if surprise_predictor is not None:
        surprise_predictor.eval()

    tracker = MetricTracker()
    all_cos: list[float] = []
    all_l2: list[float] = []

    for batch in loader:
        vectors = batch["vectors"].to(device)
        lengths = batch["lengths"].to(device)
        valid_rows = lengths >= 4
        if not valid_rows.any():
            continue
        vectors = vectors[valid_rows]
        lengths = lengths[valid_rows]
        B, L, _ = vectors.shape
        if L < 4:
            continue

        context_vecs = vectors[:, :-1]
        context_lengths = lengths - 1
        target_idx = (lengths - 1).long()
        target_vecs = vectors[torch.arange(B, device=device), target_idx]
        type_ids = build_type_ids(lengths, L, dataset_source=dataset_source)[:, :-1].to(device)
        surprise_padded = _build_surprise_padded(surprise_predictor, context_vecs, context_lengths)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            v_context = context_encoder(
                context_vectors=context_vecs,
                type_ids=type_ids,
                surprise_scores=surprise_padded,
                lengths=context_lengths,
            )
            pred_vecs = pretrain_head(v_context)
            loss_mse = F.mse_loss(pred_vecs, target_vecs)
            loss_cos = (1.0 - F.cosine_similarity(pred_vecs, target_vecs, dim=-1)).mean()
            loss = loss_mse + cos_loss_weight * loss_cos

        cos = F.cosine_similarity(pred_vecs, target_vecs, dim=-1)
        l2 = (pred_vecs - target_vecs).norm(dim=-1)
        all_cos.extend(cos.cpu().tolist())
        all_l2.extend(l2.cpu().tolist())

        tracker.update(
            {
                "ce_loss": loss.item(),
                "ce_mse": loss_mse.item(),
                "ce_cos_loss": loss_cos.item(),
                "ce_cos": cos.mean().item(),
                "ce_pred_norm": pred_vecs.norm(dim=-1).mean().item(),
            }
        )

    avg = tracker.get()
    if all_cos:
        cos_t = torch.tensor(all_cos)
        l2_t = torch.tensor(all_l2)
        avg["eval_cos_mean"] = cos_t.mean().item()
        avg["eval_cos_std"] = cos_t.std().item()
        avg["eval_cos_gt05"] = (cos_t > 0.5).float().mean().item()
        avg["eval_cos_gt07"] = (cos_t > 0.7).float().mean().item()
        avg["eval_l2_mean"] = l2_t.mean().item()
    else:
        avg.setdefault("eval_cos_mean", 0.0)
        avg.setdefault("eval_cos_std", 0.0)
        avg.setdefault("eval_cos_gt05", 0.0)
        avg.setdefault("eval_cos_gt07", 0.0)
        avg.setdefault("eval_l2_mean", 0.0)
    return avg


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage2 CE-only pretraining")
    parser.add_argument("--config", type=str, required=True, help="JSON config path")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument("--sp-checkpoint", type=str, default=None, help="Override SP checkpoint path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    amp_cfg = cfg.get("amp", {})
    out_cfg = cfg.get("output", {})
    init_cfg = cfg.get("init", {})

    device = resolve_device(args.device)
    print(f"Device: {device}")
    seed = int(cfg.get("seed", 42))
    setup_seed(seed, device)
    amp_enabled, amp_dtype, scaler = setup_amp(amp_cfg, device)

    print("Loading dataset...")
    train_dataset, val_dataset, train_source, val_source = load_train_val_datasets(data_cfg, seed)
    train_stats = train_dataset.get_statistics()
    val_stats = val_dataset.get_statistics()
    print(
        f"  Train: {train_stats['num_sequences']} seq, "
        f"mean_len={train_stats['mean_length']:.1f}, source={train_source}"
    )
    print(f"  Val:   {val_stats['num_sequences']} seq, source={val_source}")

    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 4))
    train_loader, val_loader = build_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print("Building ContextEncoder...")
    context_encoder = build_context_encoder(cfg).to(device)
    head_hidden = int(train_cfg.get("pretrain_head_hidden", 1024))
    pretrain_head = CEPretrainHead(
        d_in=context_encoder.cfg.output_dim,
        d_out=context_encoder.cfg.d_model,
        hidden_dim=head_hidden,
    ).to(device)
    print(f"  ContextEncoder params: {context_encoder.num_params:,}")
    print(f"  Pretrain head params:  {sum(p.numel() for p in pretrain_head.parameters()):,}")

    use_surprise = bool(train_cfg.get("use_surprise_features", True))
    surprise_predictor: SurprisePredictor | None = None
    if use_surprise:
        sp_ckpt_path = args.sp_checkpoint or init_cfg.get("sp_checkpoint", "")
        if not sp_ckpt_path:
            raise ValueError("CE pretraining requires SP checkpoint when use_surprise_features=true")
        print(f"Loading frozen SP from {sp_ckpt_path}")
        surprise_predictor = build_surprise_predictor(cfg).to(device)
        sp_ckpt = load_checkpoint(sp_ckpt_path, device=device)
        surprise_predictor.load_state_dict(sp_ckpt["surprise_predictor"])
        surprise_predictor.eval()
        for p in surprise_predictor.parameters():
            p.requires_grad_(False)
    else:
        print("SP features disabled for CE pretraining.")

    lr = float(train_cfg.get("lr", train_cfg.get("context_encoder_lr", 1e-4)))
    optimizer = torch.optim.AdamW(
        [
            {"params": context_encoder.parameters(), "lr": lr},
            {"params": pretrain_head.parameters(), "lr": lr},
        ],
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    num_epochs = int(train_cfg.get("num_epochs", train_cfg.get("ce_num_epochs", 20)))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 3))
    lr_min_factor = float(train_cfg.get("lr_min_factor", 0.01))
    steps_per_epoch = max(1, len(train_loader))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=warmup_epochs * steps_per_epoch,
        total_steps=max(1, num_epochs * steps_per_epoch),
        min_factor=lr_min_factor,
    )

    checkpoint_dir = Path(out_cfg.get("checkpoint_dir", "experiments/08_autoregressor/ce/checkpoints"))
    logs_dir = Path(out_cfg.get("logs_dir", "experiments/08_autoregressor/ce/logs"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_cos = -1.0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = load_checkpoint(args.resume, device=device)
        context_encoder.load_state_dict(ckpt["context_encoder"])
        pretrain_head.load_state_dict(ckpt["ce_head"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_cos = float(ckpt.get("best_cos", best_cos))
        print(f"  Resume epoch={start_epoch}, best_cos={best_cos:.6f}")

    clip_grad = float(train_cfg.get("clip_grad_norm", 1.0))
    log_every = int(train_cfg.get("log_every", 50))
    eval_every = int(train_cfg.get("eval_every_epochs", 1))
    ckpt_every = int(train_cfg.get("checkpoint_every_epochs", 1))
    cos_loss_weight = float(train_cfg.get("cos_loss_weight", 0.5))

    log_path = logs_dir / "ce_training_log.jsonl"
    log_file = open(log_path, "a", encoding="utf-8")

    def write_log(epoch: int, split: str, metrics: dict[str, float], elapsed_s: float) -> None:
        entry = {"epoch": epoch, "split": split, "elapsed_s": elapsed_s, **metrics}
        log_file.write(json.dumps(entry) + "\n")
        log_file.flush()

    print(f"\n{'=' * 60}")
    print(f"CE-only pretraining ({num_epochs} epochs)")
    print(f"{'=' * 60}")

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        train_metrics = train_epoch(
            context_encoder=context_encoder,
            pretrain_head=pretrain_head,
            surprise_predictor=surprise_predictor,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            clip_grad=clip_grad,
            log_every=log_every,
            epoch=epoch,
            dataset_source=train_source,
            cos_loss_weight=cos_loss_weight,
        )
        elapsed = time.time() - t0
        print(f"[CE] Epoch {epoch} done ({elapsed:.1f}s)")
        write_log(epoch, "train", train_metrics, elapsed)

        do_eval = ((epoch + 1) % eval_every == 0) or (epoch == num_epochs - 1)
        val_metrics: dict[str, float] = {}
        if do_eval:
            val_metrics = eval_epoch(
                context_encoder=context_encoder,
                pretrain_head=pretrain_head,
                surprise_predictor=surprise_predictor,
                loader=val_loader,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                dataset_source=val_source,
                cos_loss_weight=cos_loss_weight,
            )
            print(
                f"  [CE VAL] loss={val_metrics.get('ce_loss', 0.0):.4f} "
                f"cos={val_metrics.get('ce_cos', 0.0):.4f} "
                f"eval_cos_mean={val_metrics.get('eval_cos_mean', 0.0):.4f} "
                f"eval_cos>0.5={val_metrics.get('eval_cos_gt05', 0.0):.1%} "
                f"eval_l2={val_metrics.get('eval_l2_mean', 0.0):.4f}"
            )
            write_log(epoch, "val", val_metrics, 0.0)

            cos_mean = float(val_metrics.get("eval_cos_mean", 0.0))
            if cos_mean > best_cos:
                best_cos = cos_mean
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    {
                        "epoch": epoch,
                        "phase": "CE",
                        "context_encoder": context_encoder.state_dict(),
                        "ce_head": pretrain_head.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "metrics": val_metrics,
                        "best_cos": best_cos,
                    },
                )
                print(f"  ** New best: eval_cos_mean={best_cos:.6f}")

        if ((epoch + 1) % ckpt_every == 0) or (epoch == num_epochs - 1):
            payload = {
                "epoch": epoch,
                "phase": "CE",
                "context_encoder": context_encoder.state_dict(),
                "ce_head": pretrain_head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "metrics": train_metrics,
                "best_cos": best_cos,
            }
            save_checkpoint(checkpoint_dir / f"epoch_{epoch:03d}.pt", payload)
            save_checkpoint(checkpoint_dir / "latest.pt", payload)

    log_file.close()
    print(f"\n{'=' * 60}")
    print("CE pretraining complete")
    print(f"  Best eval cos mean: {best_cos:.6f}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Logs: {log_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
