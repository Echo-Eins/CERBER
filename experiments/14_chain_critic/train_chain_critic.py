#!/usr/bin/env python3
"""
ChainCritic training — pure reranker for ChainGenerator pipeline.

Strictly follows lessons.md:
  - L1242: "NEVER activate all losses simultaneously — start with ranking only"
  - L1682: "NEVER use iterative navigation (Langevin, Flow, ODE)" → no Langevin even for monitoring
  - L1700: "Keep the trained critic as a RERANKER only"
  - L1225: "Unconstrained MLP + SiLU" for ranking
  - L62:   "Self-denoise critic CANNOT solve QA" → trained on (q, answer) pairs

Architecture: CompositeCritic = ConditionalAngularCritic + AnalyticalRadialGuard
Loss: Focal-InfoNCE ONLY — E(q, correct_answer) < E(q, distractor)
Negatives: random + in-batch hard (top-k cosine)
No path-contrastive, no direction_loss, no Langevin, no auxiliary losses.

The critic's SOLE purpose is best-of-N reranking of ChainGenerator candidates.

Usage:
    python experiments/14_chain_critic/train_chain_critic.py
    python experiments/14_chain_critic/train_chain_critic.py --config configs/chain_critic_config.json
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

from cebcm.models.composite_critic import CompositeCritic, CompositeCriticConfig
from cebcm.models.conditional_angular_critic import ConditionalAngularCriticConfig
from cebcm.models.radial_guard import RadialGuardConfig
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

class CriticDataset(Dataset):
    """
    Dataset for critic reranker training.

    Each sample has (v_question, v_answer, v_steps).
    Base negatives are random answers from other samples.
    Hard negatives are injected per-batch from nearest in-batch answers.
    Context = mean of reasoning steps (or v_question if no steps).
    """

    def __init__(self, samples: list[dict], num_negatives: int = 7, context_bank_size: int = 4):
        self.samples = samples
        self.num_negatives = num_negatives
        self.context_bank_size = max(1, int(context_bank_size))
        self._all_answers = torch.stack([s["v_answer"] for s in samples])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        v_q = s["v_question"]
        v_a = s["v_answer"]
        v_steps = s["v_steps"]

        # Context must match generator's context bank: [query, evidence_slots...]
        # Critic sees the mean of this bank (same semantic grounding as generator).
        max_evidence = max(0, self.context_bank_size - 1)
        if v_steps.shape[0] > 0 and max_evidence > 0:
            evidence = v_steps[:max_evidence]
            bank = torch.cat([v_q.unsqueeze(0), evidence], dim=0)  # [K, D]
        else:
            bank = v_q.unsqueeze(0)  # [1, D]
        v_context = bank.mean(dim=0)  # [D] — mean of full context bank

        # Sample negatives: random answers from other samples
        neg_indices = []
        while len(neg_indices) < self.num_negatives:
            j = random.randint(0, len(self.samples) - 1)
            if j != idx:
                neg_indices.append(j)
        v_negatives = self._all_answers[neg_indices]

        return {
            "v_question": v_q,
            "v_answer": v_a,
            "v_context": v_context,
            "v_negatives": v_negatives,
        }


def collate_critic(batch: list[dict]) -> dict:
    return {
        "v_questions": torch.stack([b["v_question"] for b in batch]),
        "v_answers": torch.stack([b["v_answer"] for b in batch]),
        "v_contexts": torch.stack([b["v_context"] for b in batch]),
        "v_negatives": torch.stack([b["v_negatives"] for b in batch]),
    }


def inject_inbatch_hard_negatives(
    v_positive: torch.Tensor,
    v_negatives: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, int]:
    """
    Replace first top_k negatives with hardest in-batch answers by cosine similarity.
    Converts random-only negatives into mixed random+hard negatives.
    """
    B, N, _ = v_negatives.shape
    k = min(int(top_k), N, max(B - 1, 0))
    if k <= 0:
        return v_negatives, 0

    pos_norm = F.normalize(v_positive, dim=-1)  # [B, D]
    sim = pos_norm @ pos_norm.T                 # [B, B]
    sim.fill_diagonal_(-1e9)
    hard_idx = sim.topk(k=k, dim=1).indices     # [B, k]
    hard_neg = v_positive[hard_idx]             # [B, k, D]

    mixed = v_negatives.clone()
    mixed[:, :k, :] = hard_neg
    return mixed, k


@torch.no_grad()
def inject_generator_hard_negatives(
    v_query: torch.Tensor,
    v_positive: torch.Tensor,
    v_negatives: torch.Tensor,
    generator: ChainGenerator | None,
    steps: int,
    slots: int,
    max_pos_cos: float,
) -> tuple[torch.Tensor, int]:
    """
    Replace up to `slots` negative slots with generator-produced candidates.

    Generator-hard negatives are filtered to avoid turning near-positives
    into false negatives (cos(gen, positive) too high).
    """
    if generator is None:
        return v_negatives, 0

    B, N, _ = v_negatives.shape
    k = min(int(slots), N)
    if k <= 0:
        return v_negatives, 0

    gen_steps = max(1, int(steps))
    gen_chain = generator.generate(v_query, num_steps=gen_steps)  # [B, S, D]
    v_gen = gen_chain[:, -1, :]  # [B, D]

    # Guard against false negatives when generator already matches target too closely.
    cos_gp = F.cosine_similarity(v_gen, v_positive, dim=-1)  # [B]
    use_mask = cos_gp < float(max_pos_cos)
    if not use_mask.any():
        return v_negatives, 0

    mixed = v_negatives.clone()
    # Fill the last slot with generator-produced negative.
    # Only 1 unique generator output per sample (single generate call),
    # so we place it in the last slot to preserve in-batch hard negatives in first slots.
    slot_idx = N - 1
    mixed[use_mask, slot_idx, :] = v_gen[use_mask]

    return mixed, int(use_mask.sum().item())


# ── Training step ────────────────────────────────────────────────

def train_step(
    critic: CompositeCritic,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    cfg: dict,
    generator_hard_model: ChainGenerator | None = None,
) -> dict[str, float]:
    """
    Single training step — Focal-InfoNCE only.

    No path-contrastive (lessons L1682: navigation is broken).
    No direction loss (lessons L20: 2nd-order dominance).
    No Langevin (lessons L1700: critic is reranker only).
    """
    v_q = batch["v_questions"].to(device)
    v_a = batch["v_answers"].to(device)
    v_ctx = batch["v_contexts"].to(device)
    v_neg = batch["v_negatives"].to(device)
    use_hard = bool(cfg.get("enable_hard_negatives", True))
    hard_top_k = int(cfg.get("hard_negatives_top_k", 2))
    hard_used = 0
    if use_hard and hard_top_k > 0:
        v_neg, hard_used = inject_inbatch_hard_negatives(v_a, v_neg, hard_top_k)

    use_gen_hard = bool(cfg.get("enable_generator_hard_negatives", False)) and (generator_hard_model is not None)
    gen_hard_used = 0
    if use_gen_hard:
        v_neg, gen_hard_used = inject_generator_hard_negatives(
            v_q,
            v_a,
            v_neg,
            generator=generator_hard_model,
            steps=int(cfg.get("generator_hard_steps", 1)),
            slots=int(cfg.get("generator_hard_slots", 1)),
            max_pos_cos=float(cfg.get("generator_hard_max_pos_cos", 0.98)),
        )

    clip_grad = cfg.get("clip_grad_norm", 1.0)

    optimizer.zero_grad()

    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
        loss, rank_metrics = critic.compute_contrastive_loss(
            v_q, v_a, v_neg, v_context=v_ctx,
        )

    scaler.scale(loss).backward()

    if clip_grad > 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(critic.parameters(), clip_grad)

    scaler.step(optimizer)
    scaler.update()

    return {
        **rank_metrics,
        "loss": loss.item(),
        "hard_neg_k": float(hard_used),
        "hard_neg_enabled": 1.0 if use_hard else 0.0,
        "gen_hard_neg_used": float(gen_hard_used),
        "gen_hard_enabled": 1.0 if use_gen_hard else 0.0,
    }


# ── Eval step ────────────────────────────────────────────────────

@torch.no_grad()
def eval_step(
    critic: CompositeCritic,
    batch: dict,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    hard_neg_enabled: bool = True,
    hard_neg_top_k: int = 2,
    generator_hard_model: ChainGenerator | None = None,
    generator_hard_enabled: bool = False,
    generator_hard_steps: int = 1,
    generator_hard_slots: int = 1,
    generator_hard_max_pos_cos: float = 0.98,
) -> dict[str, float]:
    """
    Evaluation: ranking accuracy + energy gap.

    No Langevin — critic is a reranker, not a navigator.
    """
    v_q = batch["v_questions"].to(device)
    v_a = batch["v_answers"].to(device)
    v_ctx = batch["v_contexts"].to(device)
    v_neg = batch["v_negatives"].to(device)
    hard_used = 0
    if hard_neg_enabled and hard_neg_top_k > 0:
        v_neg, hard_used = inject_inbatch_hard_negatives(v_a, v_neg, hard_neg_top_k)
    gen_hard_used = 0
    if generator_hard_enabled and (generator_hard_model is not None):
        v_neg, gen_hard_used = inject_generator_hard_negatives(
            v_q,
            v_a,
            v_neg,
            generator=generator_hard_model,
            steps=generator_hard_steps,
            slots=generator_hard_slots,
            max_pos_cos=generator_hard_max_pos_cos,
        )

    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
        # Energy for positive (correct answer)
        E_pos = critic(v_q, v_a, v_context=v_ctx)  # [B]

        # Energy for negatives
        B, N, D = v_neg.shape
        v_q_exp = v_q.unsqueeze(1).expand(B, N, D).reshape(B * N, D)
        v_ctx_exp = v_ctx.unsqueeze(1).expand(B, N, D).reshape(B * N, D)
        v_neg_flat = v_neg.reshape(B * N, D)
        E_neg = critic(v_q_exp, v_neg_flat, v_context=v_ctx_exp).reshape(B, N)

        # Ranking accuracy: E(positive) < E(negative) for all negatives
        rank_acc = (E_pos.unsqueeze(1) < E_neg).float().mean().item()
        # Per-sample rank accuracy (correct < ALL negatives)
        perfect_rank = (E_pos.unsqueeze(1) < E_neg).all(dim=1).float().mean().item()
        energy_gap = (E_neg.mean(dim=1) - E_pos).mean().item()

    return {
        "val_rank_acc": rank_acc,
        "val_perfect_rank": perfect_rank,
        "val_energy_gap": energy_gap,
        "val_E_pos_mean": E_pos.mean().item(),
        "val_E_neg_mean": E_neg.mean().item(),
        "val_E_pos_std": E_pos.std().item(),
        "val_E_neg_std": E_neg.std().item(),
        "val_hard_neg_k": float(hard_used),
        "val_gen_hard_neg_used": float(gen_hard_used),
    }


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ChainCritic Training — Pure Reranker for ChainGenerator"
    )
    parser.add_argument("--config", default="configs/chain_critic_config.json")
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
    print("ChainCritic Training — Pure Reranker for ChainGenerator")
    print(f"  Loss: Focal-InfoNCE ONLY (no path, no direction, no Langevin)")
    print(f"  Purpose: best-of-N reranking of generator candidates")
    print(f"  Device: {device}")
    print("=" * 60)

    # ── Build critic ──
    critic_raw = config.get("critic", {})
    ang_cfg = ConditionalAngularCriticConfig(**critic_raw.get("angular", {}))
    rad_cfg = RadialGuardConfig(**critic_raw.get("radial", {}))
    comp_cfg = CompositeCriticConfig(
        angular=ang_cfg,
        radial=rad_cfg,
        lambda_radial=critic_raw.get("lambda_radial", 5.0),
    )
    critic = CompositeCritic(comp_cfg).to(device)

    print(f"  Angular params: {critic.angular.num_params:,}")
    print(f"  Radial: analytical (0 params)")
    print(f"  lambda_radial: {comp_cfg.lambda_radial}")
    print(f"  target_norm: {rad_cfg.target_norm}")
    print(f"  temperature: {ang_cfg.temperature}")
    print(f"  focal_gamma: {ang_cfg.focal_gamma}")
    print(f"  num_negatives: {ang_cfg.num_negatives}")

    # ── Load data ──
    data_path = config["data"]["path"]
    print(f"\n  Loading data from {data_path}")
    data = torch.load(data_path, map_location="cpu", weights_only=False)

    num_neg = ang_cfg.num_negatives
    ctx_bank_size = config["data"].get("context_bank_size", 4)
    train_ds = CriticDataset(data["train"], num_negatives=num_neg, context_bank_size=ctx_bank_size)
    val_ds = CriticDataset(data["val"], num_negatives=num_neg, context_bank_size=ctx_bank_size)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_cfg = config["training"]
    batch_size = train_cfg["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=config["data"].get("num_workers", 4),
        collate_fn=collate_critic, pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        collate_fn=collate_critic, pin_memory=device.type == "cuda",
    )

    # ── Optimizer ──
    lr = train_cfg.get("lr", 3e-4)
    optimizer = torch.optim.AdamW(
        critic.parameters(), lr=lr,
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
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if "critic" in ckpt:
            critic.load_state_dict(ckpt["critic"])
        elif "model" in ckpt:
            critic.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metric = ckpt.get("best_metric", 0.0)
        print(f"  Resumed from epoch {start_epoch}, best rank_acc={best_metric:.4f}")

    # ── Training loop ──
    log_every = train_cfg.get("log_every", 50)
    ckpt_every = train_cfg.get("checkpoint_every", 5)
    patience = train_cfg.get("early_stop_patience", 15)
    no_improve = 0

    print(f"\n  Epochs: {num_epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"  Loss: Focal-InfoNCE (tau={ang_cfg.temperature}, gamma={ang_cfg.focal_gamma})")
    print(f"  Negatives: {num_neg} per sample")
    print(
        f"  Hard negatives: enabled={bool(train_cfg.get('enable_hard_negatives', True))}, "
        f"top_k={int(train_cfg.get('hard_negatives_top_k', 2))}"
    )
    print("=" * 60)

    # ── Optional generator-hard negatives ──
    generator_hard_model = None
    if bool(train_cfg.get("enable_generator_hard_negatives", False)):
        gen_ckpt_path = str(train_cfg.get("generator_hard_checkpoint", "")).strip()
        if gen_ckpt_path:
            gen_ckpt = Path(gen_ckpt_path)
            if gen_ckpt.exists():
                try:
                    g_ckpt = torch.load(str(gen_ckpt), map_location=device, weights_only=False)
                    if "config" in g_ckpt and "generator" in g_ckpt["config"]:
                        g_cfg = ChainGeneratorConfig(**g_ckpt["config"]["generator"])
                    else:
                        g_cfg = ChainGeneratorConfig()
                    generator_hard_model = ChainGenerator(g_cfg).to(device)
                    if "model" in g_ckpt:
                        generator_hard_model.load_state_dict(g_ckpt["model"])
                    elif "generator" in g_ckpt:
                        generator_hard_model.load_state_dict(g_ckpt["generator"])
                    else:
                        generator_hard_model.load_state_dict(g_ckpt)
                    generator_hard_model.eval()
                    for p in generator_hard_model.parameters():
                        p.requires_grad_(False)
                    print(
                        f"  Generator-hard negatives: enabled from {gen_ckpt_path} "
                        f"(steps={int(train_cfg.get('generator_hard_steps', 1))}, "
                        f"slots={int(train_cfg.get('generator_hard_slots', 1))})"
                    )
                except Exception as e:
                    print(f"  WARNING: failed to load generator-hard checkpoint '{gen_ckpt_path}': {e}")
            else:
                print(f"  WARNING: generator-hard checkpoint not found: {gen_ckpt_path}")
        else:
            print("  WARNING: enable_generator_hard_negatives=true but generator_hard_checkpoint is empty")

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        critic.train()
        tracker = MetricTracker()

        for step, batch in enumerate(train_loader):
            metrics = train_step(
                critic, batch, optimizer, scaler,
                device, amp_enabled, amp_dtype, train_cfg,
                generator_hard_model=generator_hard_model,
            )
            tracker.update(metrics)
            scheduler.step()

            if (step + 1) % log_every == 0:
                avg = tracker.get()
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"  [E{epoch} S{step+1}] "
                    f"loss={avg.get('loss', 0):.4f} "
                    f"rank_acc={avg.get('rank_acc', 0):.4f} "
                    f"E_gap={avg.get('energy_gap', 0):.4f} "
                    f"E_pos={avg.get('E_pos_mean', 0):.3f} "
                    f"E_neg={avg.get('E_neg_mean', 0):.3f} "
                    f"lr={lr_now:.2e}"
                )
                tracker.reset()

        elapsed = time.time() - t0

        # ── Validation ──
        critic.eval()
        val_tracker = MetricTracker()
        for batch in val_loader:
            vm = eval_step(
                critic,
                batch,
                device,
                amp_enabled,
                amp_dtype,
                hard_neg_enabled=bool(train_cfg.get("enable_hard_negatives", True)),
                hard_neg_top_k=int(train_cfg.get("hard_negatives_top_k", 2)),
                generator_hard_model=generator_hard_model,
                generator_hard_enabled=bool(train_cfg.get("enable_generator_hard_negatives", False)),
                generator_hard_steps=int(train_cfg.get("generator_hard_steps", 1)),
                generator_hard_slots=int(train_cfg.get("generator_hard_slots", 1)),
                generator_hard_max_pos_cos=float(train_cfg.get("generator_hard_max_pos_cos", 0.98)),
            )
            val_tracker.update(vm)

        val_avg = val_tracker.get()
        val_rank = val_avg.get("val_rank_acc", 0)
        val_perfect = val_avg.get("val_perfect_rank", 0)

        print(
            f"\n  [E{epoch} VAL] "
            f"rank_acc={val_rank:.4f} "
            f"perfect_rank={val_perfect:.4f} "
            f"E_gap={val_avg.get('val_energy_gap', 0):.4f} "
            f"E_pos={val_avg.get('val_E_pos_mean', 0):.3f}±{val_avg.get('val_E_pos_std', 0):.3f} "
            f"E_neg={val_avg.get('val_E_neg_mean', 0):.3f}±{val_avg.get('val_E_neg_std', 0):.3f} "
            f"({elapsed:.1f}s)"
        )

        # ── Checkpointing (metric = rank_acc) ──
        improved = val_rank > best_metric
        if improved:
            best_metric = val_rank
            no_improve = 0
            save_checkpoint(
                Path(out_cfg["checkpoint_dir"]) / "best_chain_critic.pt",
                {
                    "critic": critic.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "val_metrics": val_avg,
                    "config": config,
                },
            )
            print(f"  ** New best: rank_acc={best_metric:.4f}")
        else:
            no_improve += 1

        if (epoch + 1) % ckpt_every == 0:
            save_checkpoint(
                Path(out_cfg["checkpoint_dir"]) / f"chain_critic_epoch_{epoch}.pt",
                {
                    "critic": critic.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "config": config,
                },
            )

        if no_improve >= patience:
            print(f"\n  Early stopping after {patience} epochs without improvement")
            break

    print(f"\nTraining complete. Best rank_acc={best_metric:.4f}")
    print(f"Checkpoints in: {out_cfg['checkpoint_dir']}")
    print(f"\nUsage: load best_chain_critic.pt in ChainGenerator GUI for reranking")


if __name__ == "__main__":
    main()
