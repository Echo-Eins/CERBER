#!/usr/bin/env python3
"""
ChainGenerator training (autoregressive QA in SONAR space).

Objective:
    L = lambda_step * L_step_masked
      + lambda_ans  * L_final_answer
      + lambda_roll * L_free_run
      + lambda_rank * L_inbatch_contrastive
"""

from __future__ import annotations

import argparse
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


class ChainDataset(Dataset):
    """Dataset for chain generator training with context memory-bank."""

    def __init__(
        self,
        samples: list[dict],
        max_chain_len: int = 20,
        context_bank_size: int = 4,
        answer_repeat_pad: int = 2,
    ):
        self.samples = samples
        self.max_chain_len = int(max_chain_len)
        self.context_bank_size = max(1, int(context_bank_size))
        self.answer_repeat_pad = max(0, int(answer_repeat_pad))
        if self.max_chain_len < 1:
            raise ValueError("max_chain_len must be >= 1")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        v_q = s["v_question"]        # [D]
        v_a = s["v_answer"]          # [D]
        v_steps = s["v_steps"]       # [S, D]

        # Chain target: [steps..., answer, answer, answer, ...]
        # Answer-repeat padding teaches the model to CONVERGE: after reaching
        # the answer, keep outputting it. At inference, consecutive similarity
        # (cos > threshold) becomes a natural stopping signal — the SONAR-space
        # equivalent of EOS.
        if v_steps.shape[0] > 0:
            chain = torch.cat([v_steps, v_a.unsqueeze(0)], dim=0)
        else:
            chain = v_a.unsqueeze(0)

        # Pad with answer repeats (before truncation to max_chain_len).
        if self.answer_repeat_pad > 0:
            pad = v_a.unsqueeze(0).expand(self.answer_repeat_pad, -1)
            chain = torch.cat([chain, pad], dim=0)

        if chain.shape[0] > self.max_chain_len:
            keep_reasoning = max(self.max_chain_len - 1, 0)
            if keep_reasoning > 0:
                chain = torch.cat([chain[:keep_reasoning], chain[-1:]], dim=0)
            else:
                chain = chain[-1:]

        # Context memory-bank: [query, evidence_slots...]
        # Keep evidence from reasoning steps only (never answer token).
        max_evidence = max(0, self.context_bank_size - 1)
        if v_steps.shape[0] > 0 and max_evidence > 0:
            evidence = v_steps[:max_evidence]
            context_bank = torch.cat([v_q.unsqueeze(0), evidence], dim=0)
        else:
            context_bank = v_q.unsqueeze(0)

        return {
            "v_question": v_q,
            "chain": chain,
            "chain_len": chain.shape[0],
            "context_bank": context_bank,
            "context_len": context_bank.shape[0],
        }


def collate_chains(batch: list[dict]) -> dict:
    """Pad chains and context banks."""
    bsz = len(batch)
    d_model = batch[0]["chain"].shape[-1]

    v_questions = torch.stack([b["v_question"] for b in batch])

    chain_lens = torch.tensor([b["chain_len"] for b in batch], dtype=torch.long)
    max_chain = int(chain_lens.max().item())
    chains = torch.zeros(bsz, max_chain, d_model)
    for i, b in enumerate(batch):
        l = int(b["chain_len"])
        chains[i, :l] = b["chain"]

    context_lens = torch.tensor([b["context_len"] for b in batch], dtype=torch.long)
    max_ctx = int(context_lens.max().item())
    context_banks = torch.zeros(bsz, max_ctx, d_model)
    context_mask = torch.zeros(bsz, max_ctx, dtype=torch.bool)
    for i, b in enumerate(batch):
        l = int(b["context_len"])
        context_banks[i, :l] = b["context_bank"]
        context_mask[i, :l] = True

    return {
        "v_questions": v_questions,
        "chains": chains,
        "chain_lens": chain_lens,
        "context_banks": context_banks,
        "context_mask": context_mask,
    }


def get_chain_steps(epoch: int, cfg: dict) -> int:
    """Curriculum for chain horizon growth."""
    s1_epochs = int(cfg.get("system1_epochs", 10))
    ramp_epochs = int(cfg.get("system2_ramp_epochs", 10))
    max_steps = int(cfg.get("max_chain_steps", 20))

    if epoch < s1_epochs:
        return 1
    ramp_progress = min(1.0, (epoch - s1_epochs) / max(ramp_epochs, 1))
    return max(1, int(1 + ramp_progress * (max_steps - 1)))


def select_training_targets(
    chains: torch.Tensor,
    chain_lens: torch.Tensor,
    target_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select suffix-aligned targets ending at answer token.
    """
    bsz, full_len, d_model = chains.shape
    steps = max(1, min(int(target_steps), full_len))
    targets = torch.zeros(bsz, steps, d_model, device=chains.device, dtype=chains.dtype)
    mask = torch.zeros(bsz, steps, device=chains.device, dtype=torch.bool)

    for i in range(bsz):
        li = max(1, min(int(chain_lens[i].item()), full_len))
        ti = min(steps, li)
        start = li - ti
        targets[i, :ti] = chains[i, start:li]
        mask[i, :ti] = True

    return targets, mask


def _masked_step_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    d_model: int,
    cosine_weight: float,
    mse_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    maskf = mask.to(dtype=pred.dtype)
    mask_sum = maskf.sum().clamp(min=1.0)

    cos_sim = F.cosine_similarity(pred, target, dim=-1)
    cos_loss = ((1.0 - cos_sim) * maskf).sum() / mask_sum

    mse_per = (pred - target).pow(2).mean(dim=-1)  # already averaged over D
    mse_loss = (mse_per * maskf).sum() / mask_sum

    loss = cosine_weight * cos_loss + mse_weight * mse_loss
    return loss, {
        "cos_sim": cos_sim,
        "cos_loss": cos_loss,
        "mse_loss": mse_loss,
        "mask_sum": mask_sum,
    }


def _gather_last_valid(x: torch.Tensor, valid_lens: torch.Tensor) -> torch.Tensor:
    idx = (valid_lens - 1).clamp(min=0)
    return x[torch.arange(x.shape[0], device=x.device), idx]


def _inbatch_contrastive_loss(
    pred_final: torch.Tensor,
    tgt_final: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz = pred_final.shape[0]
    if bsz <= 1:
        zero = pred_final.new_zeros(())
        one = pred_final.new_ones(())
        return zero, one

    temp = max(float(temperature), 1e-4)
    p = F.normalize(pred_final, dim=-1)
    t = F.normalize(tgt_final, dim=-1)

    logits = (p @ t.t()) / temp
    labels = torch.arange(bsz, device=pred_final.device)

    l_pt = F.cross_entropy(logits, labels)
    l_tp = F.cross_entropy(logits.t(), labels)
    loss = 0.5 * (l_pt + l_tp)

    acc = (logits.argmax(dim=1) == labels).float().mean()
    return loss, acc


def compute_composite_objective(
    model: ChainGenerator,
    v_q: torch.Tensor,
    chains: torch.Tensor,
    chain_mask: torch.Tensor,
    context_banks: torch.Tensor,
    context_mask: torch.Tensor,
    cfg: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute full autoregressive objective for one batch."""
    bsz, steps, d_model = chains.shape
    valid_lens = chain_mask.sum(dim=1).long().clamp(min=1)

    # Lambda weights.
    lambda_step = float(cfg.get("loss_lambda_step", 1.0))
    lambda_ans = float(cfg.get("loss_lambda_answer", 1.0))
    lambda_roll = float(cfg.get("loss_lambda_roll", 1.0))
    lambda_rank = float(cfg.get("loss_lambda_rank", 0.1))

    # Base cosine/MSE weights from model config.
    w_cos = float(model.cfg.loss_cosine_weight)
    w_mse = float(model.cfg.loss_mse_weight)

    # 1) Teacher-forced masked step loss.
    v_tf = model.forward(
        v_q,
        chains,
        v_context_bank=context_banks,
        context_mask=context_mask,
    )
    l_step, tf_stats = _masked_step_losses(v_tf, chains, chain_mask, d_model, w_cos, w_mse)

    # 2) Final-answer supervised loss (last valid token per sample).
    tf_final = _gather_last_valid(v_tf, valid_lens)
    tgt_final = _gather_last_valid(chains, valid_lens)
    ans_cos = (1.0 - F.cosine_similarity(tf_final, tgt_final, dim=-1)).mean()
    ans_mse = (tf_final - tgt_final).pow(2).mean(dim=-1).mean()  # .mean(-1) already averages over D
    l_ans = w_cos * ans_cos + w_mse * ans_mse

    # 3) Free-run rollout loss (exposure-bias correction).
    v_roll = model.generate(
        v_q,
        num_steps=steps,
        v_context_bank=context_banks,
        context_mask=context_mask,
        temperature=float(cfg.get("free_run_temperature", 1.0)),
        latent_noise_std=float(cfg.get("free_run_noise_std", 0.0)),
        repeat_penalty=float(cfg.get("free_run_repeat_penalty", 0.0)),
        repeat_cos_threshold=float(cfg.get("free_run_repeat_cos_threshold", 0.98)),
        repeat_ban_threshold=float(cfg.get("free_run_repeat_ban_threshold", 0.995)),
        repeat_ban_max_retries=int(cfg.get("free_run_repeat_ban_retries", 2)),
    )
    l_roll, roll_stats = _masked_step_losses(v_roll, chains, chain_mask, d_model, w_cos, w_mse)

    # 4) In-batch contrastive ranking on final rollout answer.
    roll_final = _gather_last_valid(v_roll, valid_lens)
    l_rank, rank_acc = _inbatch_contrastive_loss(
        roll_final,
        tgt_final,
        temperature=float(cfg.get("contrastive_temperature", 0.07)),
    )

    loss = lambda_step * l_step + lambda_ans * l_ans + lambda_roll * l_roll + lambda_rank * l_rank

    with torch.no_grad():
        tf_cos = tf_stats["cos_sim"]
        roll_cos = F.cosine_similarity(v_roll, chains, dim=-1)
        maskf = chain_mask.to(dtype=chains.dtype)
        valid = maskf.sum().clamp(min=1.0)

        tf_cos_mean = ((tf_cos * maskf).sum() / valid).item()
        roll_cos_mean = ((roll_cos * maskf).sum() / valid).item()

        tf_cos_last = F.cosine_similarity(tf_final, tgt_final, dim=-1).mean().item()
        roll_cos_last = F.cosine_similarity(roll_final, tgt_final, dim=-1).mean().item()

        metrics = {
            "loss": float(loss.item()),
            "loss_step": float(l_step.item()),
            "loss_ans": float(l_ans.item()),
            "loss_roll": float(l_roll.item()),
            "loss_rank": float(l_rank.item()),
            "tf_cos_mean": float(tf_cos_mean),
            "tf_cos_last": float(tf_cos_last),
            "roll_cos_mean": float(roll_cos_mean),
            "roll_cos_last": float(roll_cos_last),
            "rank_acc": float(rank_acc.item()),
            "pred_norm_mean_tf": float((v_tf.norm(dim=-1) * maskf).sum().item() / valid.item()),
            "pred_norm_mean_roll": float((v_roll.norm(dim=-1) * maskf).sum().item() / valid.item()),
            "valid_tokens": float(valid.item()),
            "valid_tokens_per_sample": float(valid_lens.float().mean().item()),
            "lambda_step": lambda_step,
            "lambda_ans": lambda_ans,
            "lambda_roll": lambda_roll,
            "lambda_rank": lambda_rank,
        }

        # Additional raw components for debugging.
        metrics["tf_cos_loss_raw"] = float(tf_stats["cos_loss"].item())
        metrics["tf_mse_loss_raw"] = float(tf_stats["mse_loss"].item())
        metrics["roll_cos_loss_raw"] = float(roll_stats["cos_loss"].item())
        metrics["roll_mse_loss_raw"] = float(roll_stats["mse_loss"].item())

    return loss, metrics


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
    v_q = batch["v_questions"].to(device)
    chains = batch["chains"].to(device)
    chain_lens = batch["chain_lens"].to(device)
    context_banks = batch["context_banks"].to(device)
    context_mask = batch["context_mask"].to(device)

    chains_trunc, chain_mask = select_training_targets(chains, chain_lens, target_steps)

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        loss, metrics = compute_composite_objective(
            model,
            v_q,
            chains_trunc,
            chain_mask,
            context_banks,
            context_mask,
            cfg,
        )

    scaler.scale(loss).backward()

    clip_grad = float(cfg.get("clip_grad_norm", 1.0))
    if clip_grad > 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

    scaler.step(optimizer)
    scaler.update()

    metrics["target_steps"] = float(target_steps)
    metrics["target_is_answer"] = 1.0 if target_steps == 1 else 0.0
    return metrics


@torch.no_grad()
def eval_step(
    model: ChainGenerator,
    batch: dict,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    cfg: dict,
    gen_steps: int,
) -> dict[str, float]:
    v_q = batch["v_questions"].to(device)
    chains = batch["chains"].to(device)
    chain_lens = batch["chain_lens"].to(device)
    context_banks = batch["context_banks"].to(device)
    context_mask = batch["context_mask"].to(device)

    chains_trunc, chain_mask = select_training_targets(chains, chain_lens, gen_steps)

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        _, metrics = compute_composite_objective(
            model,
            v_q,
            chains_trunc,
            chain_mask,
            context_banks,
            context_mask,
            cfg,
        )

    return {
        "val_loss": metrics["loss"],
        "val_loss_step": metrics["loss_step"],
        "val_loss_ans": metrics["loss_ans"],
        "val_loss_roll": metrics["loss_roll"],
        "val_loss_rank": metrics["loss_rank"],
        "val_tf_cos": metrics["tf_cos_mean"],
        "val_tf_cos_last": metrics["tf_cos_last"],
        "val_roll_cos": metrics["roll_cos_mean"],
        "val_roll_cos_last": metrics["roll_cos_last"],
        "val_rank_acc": metrics["rank_acc"],
        "val_norm_tf": metrics["pred_norm_mean_tf"],
        "val_norm_roll": metrics["pred_norm_mean_roll"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ChainGenerator Training")
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

    print("=" * 70)
    print("ChainGenerator Training - Autoregressive QA in SONAR space")
    print(f"Device: {device}")
    print("=" * 70)

    gen_cfg = ChainGeneratorConfig(**config.get("generator", {}))
    model = ChainGenerator(gen_cfg).to(device)

    print(f"Parameters: {model.num_params:,}")
    print(f"Layers={gen_cfg.n_layers}, Heads={gen_cfg.n_heads}, FFN={gen_cfg.dim_feedforward}")
    print(f"target_norm={gen_cfg.target_norm}, max_chain_len={gen_cfg.max_chain_len}")

    data_cfg = config["data"]
    data_path = data_cfg["path"]
    data = torch.load(data_path, map_location="cpu", weights_only=False)

    train_cfg = config["training"]
    max_chain_steps = int(train_cfg.get("max_chain_steps", gen_cfg.max_chain_len))
    if max_chain_steps > gen_cfg.max_chain_len:
        print(
            f"[WARN] training.max_chain_steps={max_chain_steps} exceeds generator.max_chain_len={gen_cfg.max_chain_len}; clamping"
        )
        max_chain_steps = gen_cfg.max_chain_len
        train_cfg["max_chain_steps"] = max_chain_steps

    context_bank_size = int(train_cfg.get("context_bank_size", 4))
    answer_repeat_pad = int(train_cfg.get("answer_repeat_pad", 2))

    train_ds = ChainDataset(
        data["train"],
        max_chain_len=gen_cfg.max_chain_len,
        context_bank_size=context_bank_size,
        answer_repeat_pad=answer_repeat_pad,
    )
    val_ds = ChainDataset(
        data["val"],
        max_chain_len=gen_cfg.max_chain_len,
        context_bank_size=context_bank_size,
        answer_repeat_pad=answer_repeat_pad,
    )

    print(f"Data: train={len(train_ds)}, val={len(val_ds)}, "
          f"context_bank_size={context_bank_size}, answer_repeat_pad={answer_repeat_pad}")

    batch_size = int(train_cfg.get("batch_size", 16))
    num_workers = int(data_cfg.get("num_workers", 4))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_chains,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_chains,
        pin_memory=device.type == "cuda",
    )

    lr = float(train_cfg.get("lr", 1e-4))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    num_epochs = int(args.max_epochs or train_cfg.get("num_epochs", 50))
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(train_cfg.get("warmup_epochs", 3)) * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_steps,
        min_factor=float(train_cfg.get("lr_min_factor", 0.01)),
    )

    amp_enabled, amp_dtype, scaler = setup_amp(config.get("amp", {}), device)

    start_epoch = 0
    best_metric = float("-inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metric = float(ckpt.get("best_metric", best_metric))
        print(f"Resumed from {args.resume}: epoch={start_epoch}, best={best_metric:.4f}")

    log_every = int(train_cfg.get("log_every", 50))
    ckpt_every = int(train_cfg.get("checkpoint_every", 5))
    patience = int(train_cfg.get("early_stop_patience", 15))
    no_improve = 0

    tracker = MetricTracker()

    print("\nTraining settings")
    print(f"  epochs={num_epochs}, batch={batch_size}, lr={lr:.2e}")
    print(
        "  objective="
        f"step*{train_cfg.get('loss_lambda_step', 1.0)} + "
        f"ans*{train_cfg.get('loss_lambda_answer', 1.0)} + "
        f"roll*{train_cfg.get('loss_lambda_roll', 1.0)} + "
        f"rank*{train_cfg.get('loss_lambda_rank', 0.1)}"
    )
    print(
        f"  horizon: system1_epochs={train_cfg.get('system1_epochs', 10)}, "
        f"ramp={train_cfg.get('system2_ramp_epochs', 10)}, max_steps={max_chain_steps}"
    )
    print("=" * 70)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        target_steps = get_chain_steps(epoch, train_cfg)
        epoch_start = time.time()

        phase = "System1" if target_steps == 1 else f"System2({target_steps})"
        print(f"\n[E{epoch}] target_steps={target_steps} [{phase}]")

        for step, batch in enumerate(train_loader):
            metrics = train_step(
                model,
                batch,
                optimizer,
                scaler,
                device,
                amp_enabled,
                amp_dtype,
                train_cfg,
                target_steps=target_steps,
            )
            scheduler.step()
            tracker.update(metrics)

            if (step + 1) % log_every == 0:
                avg = tracker.get()
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"  [E{epoch} S{step+1}] "
                    f"loss={avg.get('loss', 0.0):.4f} "
                    f"step={avg.get('loss_step', 0.0):.4f} "
                    f"ans={avg.get('loss_ans', 0.0):.4f} "
                    f"roll={avg.get('loss_roll', 0.0):.4f} "
                    f"rank={avg.get('loss_rank', 0.0):.4f} "
                    f"tf_cos={avg.get('tf_cos_mean', 0.0):.4f} "
                    f"roll_cos={avg.get('roll_cos_mean', 0.0):.4f} "
                    f"rank_acc={avg.get('rank_acc', 0.0):.3f} "
                    f"lr={lr_now:.2e}"
                )
                tracker.reset()

        model.eval()
        val_tracker = MetricTracker()
        eval_steps = target_steps
        for batch in val_loader:
            vm = eval_step(
                model,
                batch,
                device,
                amp_enabled,
                amp_dtype,
                train_cfg,
                gen_steps=eval_steps,
            )
            val_tracker.update(vm)

        val = val_tracker.get()
        epoch_time = time.time() - epoch_start

        # Main selection metric: rollout final cosine.
        val_metric = float(val.get("val_roll_cos_last", val.get("val_roll_cos", 0.0)))

        print(
            f"  [E{epoch} VAL] "
            f"loss={val.get('val_loss', 0.0):.4f} "
            f"tf_cos={val.get('val_tf_cos', 0.0):.4f} "
            f"roll_cos={val.get('val_roll_cos', 0.0):.4f} "
            f"roll_cos_last={val.get('val_roll_cos_last', 0.0):.4f} "
            f"rank_acc={val.get('val_rank_acc', 0.0):.3f} "
            f"norm_roll={val.get('val_norm_roll', 0.0):.4f} "
            f"({epoch_time:.1f}s)"
        )

        improved = val_metric > best_metric
        if improved:
            best_metric = val_metric
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
            print(f"  ** New best: val_roll_cos_last={best_metric:.4f}")
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
            print(f"\nEarly stopping: no improvement for {patience} epochs")
            break

    print("\nTraining complete")
    print(f"Best val_roll_cos_last={best_metric:.4f}")
    print(f"Checkpoints: {out_cfg['checkpoint_dir']}")


if __name__ == "__main__":
    main()
