#!/usr/bin/env python3
"""
Standalone Stage 2A training: SurprisePredictor pretraining.

This script intentionally does not modify or split the existing
`train_stage2.py` monolithic pipeline.
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cebcm.models.surprise import SurprisePredictor
from cebcm.training.stage2_utils import (
    MetricTracker,
    build_dataloaders,
    build_surprise_predictor,
    get_cosine_schedule_with_warmup,
    load_checkpoint,
    load_config,
    load_train_val_datasets,
    resolve_device,
    save_checkpoint,
    setup_amp,
    setup_seed,
)


def train_epoch(
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
    model.train()
    tracker = MetricTracker()
    step = 0

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
            print(
                f"  [SP] epoch={epoch} step={step} "
                f"loss={avg.get('surprise_loss', 0.0):.4f} "
                f"mse={avg.get('surprise_mse', 0.0):.4f} "
                f"cos={avg.get('surprise_cos', 0.0):.4f} "
                f"surprise_mean={avg.get('surprise_mean', 0.0):.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    return tracker.get()


@torch.no_grad()
def eval_epoch(
    model: SurprisePredictor,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage2 SP-only pretraining")
    parser.add_argument("--config", type=str, required=True, help="JSON config path")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    amp_cfg = cfg.get("amp", {})
    out_cfg = cfg.get("output", {})

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

    print("Building SurprisePredictor...")
    model = build_surprise_predictor(cfg).to(device)
    print(f"  Params: {model.num_params:,}")

    lr = float(train_cfg.get("lr", train_cfg.get("surprise_lr", 3e-4)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    num_epochs = int(train_cfg.get("num_epochs", train_cfg.get("sp_num_epochs", 10)))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 3))
    lr_min_factor = float(train_cfg.get("lr_min_factor", 0.01))
    steps_per_epoch = max(1, len(train_loader))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=warmup_epochs * steps_per_epoch,
        total_steps=max(1, num_epochs * steps_per_epoch),
        min_factor=lr_min_factor,
    )

    checkpoint_dir = Path(out_cfg.get("checkpoint_dir", "experiments/08_autoregressor/sp/checkpoints"))
    logs_dir = Path(out_cfg.get("logs_dir", "experiments/08_autoregressor/sp/logs"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = load_checkpoint(args.resume, device=device)
        model.load_state_dict(ckpt["surprise_predictor"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        print(f"  Resume epoch={start_epoch}, best_val_loss={best_val_loss:.6f}")

    clip_grad = float(train_cfg.get("clip_grad_norm", 1.0))
    log_every = int(train_cfg.get("log_every", 50))
    eval_every = int(train_cfg.get("eval_every_epochs", 1))
    ckpt_every = int(train_cfg.get("checkpoint_every_epochs", 1))
    early_stop_patience = int(train_cfg.get("early_stop_patience", 5))
    no_improve_count = 0

    log_path = logs_dir / "sp_training_log.jsonl"
    log_file = open(log_path, "a", encoding="utf-8")

    def write_log(epoch: int, split: str, metrics: dict[str, float], elapsed_s: float) -> None:
        entry = {"epoch": epoch, "split": split, "elapsed_s": elapsed_s, **metrics}
        log_file.write(json.dumps(entry) + "\n")
        log_file.flush()

    print(f"\n{'=' * 60}")
    print(f"SP-only training ({num_epochs} epochs)")
    print(f"{'=' * 60}")

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        train_metrics = train_epoch(
            model=model,
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
        )
        elapsed = time.time() - t0
        print(f"[SP] Epoch {epoch} done ({elapsed:.1f}s)")
        write_log(epoch, "train", train_metrics, elapsed)

        do_eval = ((epoch + 1) % eval_every == 0) or (epoch == num_epochs - 1)
        val_metrics: dict[str, float] = {}
        if do_eval:
            val_metrics = eval_epoch(model, val_loader, device, amp_enabled, amp_dtype)
            val_loss = float(val_metrics.get("surprise_loss", float("inf")))
            print(
                f"  [SP VAL] loss={val_loss:.4f} "
                f"mse={val_metrics.get('surprise_mse', 0.0):.4f} "
                f"cos={val_metrics.get('surprise_cos', 0.0):.4f} "
                f"surprise_mean={val_metrics.get('surprise_mean', 0.0):.4f}"
            )
            write_log(epoch, "val", val_metrics, 0.0)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    {
                        "epoch": epoch,
                        "phase": "SP",
                        "surprise_predictor": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "metrics": val_metrics,
                        "best_val_loss": best_val_loss,
                    },
                )
                print(f"  ** New best: val_loss={best_val_loss:.6f}")
            else:
                no_improve_count += 1
                if no_improve_count >= early_stop_patience:
                    print(f"  Early stopping: no improvement for {early_stop_patience} evals")
                    break

        if ((epoch + 1) % ckpt_every == 0) or (epoch == num_epochs - 1):
            payload = {
                "epoch": epoch,
                "phase": "SP",
                "surprise_predictor": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "metrics": train_metrics,
                "best_val_loss": best_val_loss,
            }
            save_checkpoint(checkpoint_dir / f"epoch_{epoch:03d}.pt", payload)
            save_checkpoint(checkpoint_dir / "latest.pt", payload)

    log_file.close()
    print(f"\n{'=' * 60}")
    print("SP pretraining complete")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Logs: {log_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
