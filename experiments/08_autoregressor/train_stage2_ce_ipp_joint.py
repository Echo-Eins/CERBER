#!/usr/bin/env python3
"""
Standalone Stage 2 joint training: ContextEncoder + IPP with frozen SP.

This is equivalent to Phase B logic, but exposed as a dedicated script.
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
from cebcm.models.ipp import FlowIPP, MLPIPP
from cebcm.models.surprise import SurprisePredictor
from cebcm.training.stage2_utils import (
    MetricTracker,
    build_context_encoder,
    build_dataloaders,
    build_ipp,
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


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(enabled)


def _count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _grad_l2_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float(torch.sum(g * g).item())
    return total ** 0.5


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


def train_epoch(
    context_encoder: ContextEncoder,
    ipp: FlowIPP | MLPIPP,
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
) -> dict[str, float]:
    context_encoder.train()
    ipp.train()
    if surprise_predictor is not None:
        surprise_predictor.eval()

    tracker = MetricTracker()
    step = 0

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
            loss, metrics = ipp.compute_loss(v_context, target_vecs)

        if not torch.isfinite(loss):
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        ce_grad_norm = _grad_l2_norm(list(context_encoder.parameters()))
        ipp_grad_norm = _grad_l2_norm(list(ipp.parameters()))
        nn.utils.clip_grad_norm_(
            list(context_encoder.parameters()) + list(ipp.parameters()),
            clip_grad,
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        tracker.update({
            **metrics,
            "ce_grad_norm": ce_grad_norm,
            "ipp_grad_norm": ipp_grad_norm,
        })
        step += 1

        if step % log_every == 0:
            avg = tracker.get()
            loss_key = "flow_loss" if "flow_loss" in avg else "ipp_loss"
            cos_key = "flow_cos" if "flow_cos" in avg else "ipp_cos"
            print(
                f"  [CE+IPP] epoch={epoch} step={step} "
                f"loss={avg.get(loss_key, 0.0):.4f} "
                f"cos={avg.get(cos_key, 0.0):.4f} "
                f"ce_gn={avg.get('ce_grad_norm', 0.0):.3e} "
                f"ipp_gn={avg.get('ipp_grad_norm', 0.0):.3e} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    return tracker.get()


@torch.no_grad()
def eval_epoch(
    context_encoder: ContextEncoder,
    ipp: FlowIPP | MLPIPP,
    surprise_predictor: SurprisePredictor | None,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    target_norm: float,
    eval_num_samples: int,
    dataset_source: str,
) -> dict[str, float]:
    context_encoder.eval()
    ipp.eval()
    if surprise_predictor is not None:
        surprise_predictor.eval()

    tracker = MetricTracker()
    all_cos: list[float] = []
    all_l2: list[float] = []
    all_cos_best: list[float] = []
    all_l2_best: list[float] = []

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
            _, metrics = ipp.compute_loss(v_context, target_vecs)
            v_init = ipp.sample(v_context, target_norm=target_norm)

        cos = F.cosine_similarity(v_init, target_vecs, dim=-1)
        l2 = (v_init - target_vecs).norm(dim=-1)
        all_cos.extend(cos.cpu().tolist())
        all_l2.extend(l2.cpu().tolist())

        if eval_num_samples > 1:
            cos_samples = [cos]
            l2_samples = [l2]
            for _ in range(eval_num_samples - 1):
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    v_s = ipp.sample(v_context, target_norm=target_norm)
                cos_samples.append(F.cosine_similarity(v_s, target_vecs, dim=-1))
                l2_samples.append((v_s - target_vecs).norm(dim=-1))
            cos_stack = torch.stack(cos_samples, dim=0)  # [K, B]
            l2_stack = torch.stack(l2_samples, dim=0)  # [K, B]
            all_cos_best.extend(cos_stack.max(dim=0).values.cpu().tolist())
            all_l2_best.extend(l2_stack.min(dim=0).values.cpu().tolist())

        tracker.update(metrics)

    avg = tracker.get()
    if all_cos:
        cos_t = torch.tensor(all_cos)
        l2_t = torch.tensor(all_l2)
        avg["eval_cos_mean"] = cos_t.mean().item()
        avg["eval_cos_std"] = cos_t.std().item()
        avg["eval_cos_gt05"] = (cos_t > 0.5).float().mean().item()
        avg["eval_cos_gt07"] = (cos_t > 0.7).float().mean().item()
        avg["eval_l2_mean"] = l2_t.mean().item()
        if all_cos_best:
            cos_best_t = torch.tensor(all_cos_best)
            l2_best_t = torch.tensor(all_l2_best)
            avg["eval_cos_bestk_mean"] = cos_best_t.mean().item()
            avg["eval_cos_bestk_gt05"] = (cos_best_t > 0.5).float().mean().item()
            avg["eval_cos_bestk_gt07"] = (cos_best_t > 0.7).float().mean().item()
            avg["eval_l2_bestk_mean"] = l2_best_t.mean().item()
    else:
        avg.setdefault("eval_cos_mean", 0.0)
        avg.setdefault("eval_cos_std", 0.0)
        avg.setdefault("eval_cos_gt05", 0.0)
        avg.setdefault("eval_cos_gt07", 0.0)
        avg.setdefault("eval_l2_mean", 0.0)
        avg.setdefault("eval_cos_bestk_mean", 0.0)
        avg.setdefault("eval_cos_bestk_gt05", 0.0)
        avg.setdefault("eval_cos_bestk_gt07", 0.0)
        avg.setdefault("eval_l2_bestk_mean", 0.0)
    return avg


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage2 CE+IPP joint training")
    parser.add_argument("--config", type=str, required=True, help="JSON config path")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument("--sp-checkpoint", type=str, default=None, help="Override SP checkpoint path")
    parser.add_argument("--ce-checkpoint", type=str, default=None, help="Optional CE init checkpoint")
    parser.add_argument("--ipp-checkpoint", type=str, default=None, help="Optional IPP init checkpoint")
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

    print("Building models...")
    context_encoder = build_context_encoder(cfg).to(device)
    ipp = build_ipp(cfg).to(device)
    ipp_mode = str(cfg.get("ipp", {}).get("mode", "flow")).lower()
    print(f"  ContextEncoder params: {context_encoder.num_params:,}")
    print(f"  IPP mode:              {ipp_mode} ({ipp.__class__.__name__})")
    print(f"  IPP params:            {ipp.num_params:,}")

    # By default force both modules to remain trainable in joint stage.
    if bool(train_cfg.get("force_unfreeze_context_encoder", True)):
        _set_requires_grad(context_encoder, True)
    if bool(train_cfg.get("force_unfreeze_ipp", True)):
        _set_requires_grad(ipp, True)

    use_surprise = bool(train_cfg.get("use_surprise_features", True))
    surprise_predictor: SurprisePredictor | None = None
    if use_surprise:
        sp_ckpt_path = args.sp_checkpoint or init_cfg.get("sp_checkpoint", "")
        if not sp_ckpt_path:
            raise ValueError("Joint CE+IPP training requires SP checkpoint when use_surprise_features=true")
        print(f"Loading frozen SP from {sp_ckpt_path}")
        surprise_predictor = build_surprise_predictor(cfg).to(device)
        sp_ckpt = load_checkpoint(sp_ckpt_path, device=device)
        surprise_predictor.load_state_dict(sp_ckpt["surprise_predictor"])
        surprise_predictor.eval()
        for p in surprise_predictor.parameters():
            p.requires_grad_(False)
    else:
        print("SP features disabled for joint training.")

    ce_ckpt_path = args.ce_checkpoint or init_cfg.get("ce_checkpoint", "")
    if ce_ckpt_path:
        print(f"Loading CE init from {ce_ckpt_path}")
        ce_ckpt = load_checkpoint(ce_ckpt_path, device=device)
        context_encoder.load_state_dict(ce_ckpt["context_encoder"])

    ipp_ckpt_path = args.ipp_checkpoint or init_cfg.get("ipp_checkpoint", "")
    if ipp_ckpt_path:
        print(f"Loading IPP init from {ipp_ckpt_path}")
        ipp_ckpt = load_checkpoint(ipp_ckpt_path, device=device)
        ipp.load_state_dict(ipp_ckpt["ipp"])

    # Re-assert trainability after loading checkpoints.
    if bool(train_cfg.get("force_unfreeze_context_encoder", True)):
        _set_requires_grad(context_encoder, True)
    if bool(train_cfg.get("force_unfreeze_ipp", True)):
        _set_requires_grad(ipp, True)

    ce_trainable = _count_trainable_params(context_encoder)
    ipp_trainable = _count_trainable_params(ipp)
    print(f"  Trainable CE params:   {ce_trainable:,}")
    print(f"  Trainable IPP params:  {ipp_trainable:,}")

    optimizer = torch.optim.AdamW(
        [
            {"params": context_encoder.parameters(), "lr": float(train_cfg.get("context_encoder_lr", 1e-4))},
            {"params": ipp.parameters(), "lr": float(train_cfg.get("ipp_lr", 1e-4))},
        ],
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    num_epochs = int(train_cfg.get("num_epochs", train_cfg.get("joint_num_epochs", 40)))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 3))
    lr_min_factor = float(train_cfg.get("lr_min_factor", 0.01))
    steps_per_epoch = max(1, len(train_loader))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=warmup_epochs * steps_per_epoch,
        total_steps=max(1, num_epochs * steps_per_epoch),
        min_factor=lr_min_factor,
    )

    checkpoint_dir = Path(out_cfg.get("checkpoint_dir", "experiments/08_autoregressor/joint/checkpoints"))
    logs_dir = Path(out_cfg.get("logs_dir", "experiments/08_autoregressor/joint/logs"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_cos = -1.0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = load_checkpoint(args.resume, device=device)
        context_encoder.load_state_dict(ckpt["context_encoder"])
        ipp.load_state_dict(ckpt["ipp"])
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
    target_norm = float(train_cfg.get("target_norm", 0.2051))
    eval_num_samples = max(1, int(train_cfg.get("eval_num_samples", 1)))

    log_path = logs_dir / "ce_ipp_joint_training_log.jsonl"
    log_file = open(log_path, "a", encoding="utf-8")

    def write_log(epoch: int, split: str, metrics: dict[str, float], elapsed_s: float) -> None:
        entry = {"epoch": epoch, "split": split, "elapsed_s": elapsed_s, **metrics}
        log_file.write(json.dumps(entry) + "\n")
        log_file.flush()

    print(f"\n{'=' * 60}")
    print(f"Joint CE+IPP training ({num_epochs} epochs)")
    print(f"{'=' * 60}")

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        train_metrics = train_epoch(
            context_encoder=context_encoder,
            ipp=ipp,
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
        )
        elapsed = time.time() - t0
        print(f"[CE+IPP] Epoch {epoch} done ({elapsed:.1f}s)")
        write_log(epoch, "train", train_metrics, elapsed)

        do_eval = ((epoch + 1) % eval_every == 0) or (epoch == num_epochs - 1)
        val_metrics: dict[str, float] = {}
        if do_eval:
            val_metrics = eval_epoch(
                context_encoder=context_encoder,
                ipp=ipp,
                surprise_predictor=surprise_predictor,
                loader=val_loader,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                target_norm=target_norm,
                eval_num_samples=eval_num_samples,
                dataset_source=val_source,
            )
            loss_key = "flow_loss" if "flow_loss" in val_metrics else "ipp_loss"
            cos_key = "flow_cos" if "flow_cos" in val_metrics else "ipp_cos"
            print(
                f"  [VAL] loss={val_metrics.get(loss_key, 0.0):.4f} "
                f"cos={val_metrics.get(cos_key, 0.0):.4f} "
                f"eval_cos_mean={val_metrics.get('eval_cos_mean', 0.0):.4f} "
                f"eval_cos>0.5={val_metrics.get('eval_cos_gt05', 0.0):.1%} "
                f"eval_cos>0.7={val_metrics.get('eval_cos_gt07', 0.0):.1%} "
                f"eval_l2={val_metrics.get('eval_l2_mean', 0.0):.4f}"
            )
            if eval_num_samples > 1:
                print(
                    f"        eval_cos_best@{eval_num_samples}="
                    f"{val_metrics.get('eval_cos_bestk_mean', 0.0):.4f} "
                    f"eval_cos_best>0.7={val_metrics.get('eval_cos_bestk_gt07', 0.0):.1%} "
                    f"eval_l2_best={val_metrics.get('eval_l2_bestk_mean', 0.0):.4f}"
                )
            write_log(epoch, "val", val_metrics, 0.0)

            cos_mean = float(val_metrics.get("eval_cos_mean", 0.0))
            if cos_mean > best_cos:
                best_cos = cos_mean
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    {
                        "epoch": epoch,
                        "phase": "JOINT",
                        "context_encoder": context_encoder.state_dict(),
                        "ipp": ipp.state_dict(),
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
                "phase": "JOINT",
                "context_encoder": context_encoder.state_dict(),
                "ipp": ipp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "metrics": train_metrics,
                "best_cos": best_cos,
            }
            save_checkpoint(checkpoint_dir / f"epoch_{epoch:03d}.pt", payload)
            save_checkpoint(checkpoint_dir / "latest.pt", payload)

    log_file.close()
    print(f"\n{'=' * 60}")
    print("Joint CE+IPP training complete")
    print(f"  Best eval cos mean: {best_cos:.6f}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Logs: {log_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
