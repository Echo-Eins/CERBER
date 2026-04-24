#!/usr/bin/env python3
"""
ChainGenerator training (autoregressive QA in SONAR space).

Objective:
    L = lambda_step * L_step_masked
      + lambda_ans  * L_final_answer
      + lambda_roll * L_free_run
      + lambda_rank * L_inbatch_contrastive
      + lambda_aux  * L_auxiliary_heads
      + lambda_aux_df * L_auxiliary_df_x0
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cebcm.models.chain_generator import (
    ChainGenerator,
    ChainGeneratorConfig,
    SwiGLUFFN,
    _apply_rope,
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
            answer_pos = int(v_steps.shape[0])
        else:
            chain = v_a.unsqueeze(0)
            answer_pos = 0

        # Pad with answer repeats (before truncation to max_chain_len).
        if self.answer_repeat_pad > 0:
            pad = v_a.unsqueeze(0).expand(self.answer_repeat_pad, -1)
            chain = torch.cat([chain, pad], dim=0)

        if chain.shape[0] > self.max_chain_len:
            keep_reasoning = max(self.max_chain_len - 1, 0)
            if keep_reasoning > 0:
                chain = torch.cat([chain[:keep_reasoning], chain[-1:]], dim=0)
                answer_pos = keep_reasoning
            else:
                chain = chain[-1:]
                answer_pos = 0

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
            "answer_pos": answer_pos,
            "context_bank": context_bank,
            "context_len": context_bank.shape[0],
        }


def collate_chains(batch: list[dict]) -> dict:
    """Pad chains and context banks."""
    bsz = len(batch)
    d_model = batch[0]["chain"].shape[-1]

    v_questions = torch.stack([b["v_question"] for b in batch])

    chain_lens = torch.tensor([b["chain_len"] for b in batch], dtype=torch.long)
    answer_pos = torch.tensor([b["answer_pos"] for b in batch], dtype=torch.long)
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
        "answer_pos": answer_pos,
        "context_banks": context_banks,
        "context_mask": context_mask,
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * q))))
    return float(vals[idx])


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _collect_swiglu_stats(model: torch.nn.Module) -> dict:
    """Aggregate per-layer SwiGLU activation health stats.

    Returns mean across layers of:
      - gate_in_abs_mean: how far from zero the SiLU input lives.
        Tiny → SiLU near-linear → gradient ~0.5 → "dead zone" the third-
        party scale-collapse analysis warned about.
      - gate_in_std: spread of SiLU inputs.  Collapsing toward 0 means
        the gate is being squashed onto a single point.
      - silu_kurtosis: excess kurtosis of post-SiLU values.  ~0 is healthy
        (Gaussian-like); ≫0 is the "лес из нулей и редких пиков" failure
        mode where a few channels carry all the signal.
      - out_norm_mean: mean L2 norm of FFN output.  Collapsing → FFN is
        contributing nothing to the residual stream.
    """
    gate_means, gate_stds, kurts, out_norms = [], [], [], []
    for module in model.modules():
        if isinstance(module, SwiGLUFFN) and module.track_stats:
            gate_means.append(float(module._gate_in_abs_mean.item()))
            gate_stds.append(float(module._gate_in_std.item()))
            kurts.append(float(module._silu_kurtosis.item()))
            out_norms.append(float(module._out_norm_mean.item()))
    if not gate_means:
        return {}
    return {
        "ffn_gate_in_abs_mean": _mean(gate_means),
        "ffn_gate_in_std": _mean(gate_stds),
        "ffn_silu_kurtosis": _mean(kurts),
        "ffn_out_norm_mean": _mean(out_norms),
        "ffn_silu_kurtosis_max": max(kurts),
    }


def summarize_chain_dataset(
    samples: list[dict],
    label: str,
    max_chain_len: int,
    max_chain_steps: int,
    answer_repeat_pad: int,
    context_bank_size: int,
) -> dict[str, float]:
    """Print cheap dataset diagnostics before spending GPU time."""
    step_lens: list[float] = []
    chain_lens: list[float] = []
    answer_positions: list[float] = []
    ctx_lens: list[float] = []
    q_norms: list[float] = []
    a_norms: list[float] = []
    step_norms: list[float] = []
    answer_word_lens: list[float] = []
    truncated = 0

    for s in samples:
        v_steps = s["v_steps"]
        n_steps = int(v_steps.shape[0])
        raw_chain_len = n_steps + 1 + max(0, int(answer_repeat_pad))
        final_chain_len = min(raw_chain_len, int(max_chain_len))
        answer_pos = n_steps
        if raw_chain_len > max_chain_len:
            answer_pos = max(final_chain_len - 1, 0)

        step_lens.append(float(n_steps))
        chain_lens.append(float(final_chain_len))
        answer_positions.append(float(answer_pos))
        ctx_lens.append(float(min(1 + n_steps, max(1, int(context_bank_size)))))
        q_norms.append(float(s["v_question"].norm().item()))
        a_norms.append(float(s["v_answer"].norm().item()))
        if n_steps > 0:
            step_norms.append(float(v_steps.norm(dim=-1).mean().item()))
        answer_text = s.get("answer")
        if isinstance(answer_text, str):
            answer_word_lens.append(float(len(answer_text.split())))
        if raw_chain_len > max_chain_len:
            truncated += 1

    n = max(len(samples), 1)
    coverage = sum(1 for x in answer_positions if (x + 1) <= max_chain_steps) / n
    stats = {
        "n": float(len(samples)),
        "steps_mean": _mean(step_lens),
        "steps_p50": _percentile(step_lens, 0.50),
        "steps_p95": _percentile(step_lens, 0.95),
        "steps_max": max(step_lens) if step_lens else 0.0,
        "chain_mean": _mean(chain_lens),
        "chain_p50": _percentile(chain_lens, 0.50),
        "chain_p95": _percentile(chain_lens, 0.95),
        "chain_max": max(chain_lens) if chain_lens else 0.0,
        "truncated_pct": 100.0 * truncated / n,
        "answer_coverage_at_max_steps_pct": 100.0 * coverage,
        "context_len_mean": _mean(ctx_lens),
        "q_norm_mean": _mean(q_norms),
        "a_norm_mean": _mean(a_norms),
        "step_norm_mean": _mean(step_norms),
        "answer_words_mean": _mean(answer_word_lens),
        "answer_words_p95": _percentile(answer_word_lens, 0.95),
    }
    print(
        f"  [{label}] chain_len mean/p50/p95/max="
        f"{stats['chain_mean']:.1f}/{stats['chain_p50']:.0f}/{stats['chain_p95']:.0f}/{stats['chain_max']:.0f}; "
        f"steps mean/p95={stats['steps_mean']:.1f}/{stats['steps_p95']:.0f}; "
        f"truncated={stats['truncated_pct']:.1f}%; "
        f"answer_coverage@max_steps={stats['answer_coverage_at_max_steps_pct']:.1f}%; "
        f"answer_words mean/p95={stats['answer_words_mean']:.1f}/{stats['answer_words_p95']:.0f}; "
        f"norm q/a/step={stats['q_norm_mean']:.4f}/{stats['a_norm_mean']:.4f}/{stats['step_norm_mean']:.4f}; "
        f"context_len_mean={stats['context_len_mean']:.1f}"
    )
    return stats


def get_chain_steps(epoch: int, cfg: dict) -> int:
    """Curriculum for chain horizon growth.

    Supports two modes controlled by the ``horizon_schedule`` config key:

    1. **Fixed-horizon schedule** (``horizon_schedule`` is set):
       A list of ``[steps, duration_epochs]`` pairs.  Each pair trains at
       the given horizon for the specified number of epochs.  After the
       schedule is exhausted, the model stays at the last horizon.

       Example: ``[[4, 10], [5, 10]]`` — 10 epochs at 4 steps, then 10
       epochs at 5 steps.

    2. **Legacy linear ramp** (``horizon_schedule`` is absent):
       Each epoch after System-1 adds +1 step, starting from
       ``system2_start_steps``.  Original Fix #8 behaviour.
    """
    s1_epochs = int(cfg.get("system1_epochs", 10))
    max_steps = int(cfg.get("max_chain_steps", 20))

    if epoch < s1_epochs:
        return 1

    # ── Fixed-horizon schedule ──
    schedule = cfg.get("horizon_schedule")
    if schedule:
        elapsed = epoch - s1_epochs
        for entry in schedule:
            steps_val, duration = int(entry[0]), int(entry[1])
            if elapsed < duration:
                return min(max_steps, steps_val)
            elapsed -= duration
        # After schedule exhausted: stay at last scheduled horizon.
        return min(max_steps, int(schedule[-1][0]))

    # ── Legacy: linear ramp ──
    start_steps = int(cfg.get("system2_start_steps", 2))
    return min(max_steps, start_steps + (epoch - s1_epochs))


def select_training_targets(
    chains: torch.Tensor,
    chain_lens: torch.Tensor,
    answer_pos: torch.Tensor,
    target_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select training targets aligned to the generation process.

    Two regimes:
      - System1 (target_steps=1): answer-aligned; take the first answer token,
        not the first reasoning token and not an answer-repeat pad.
      - System2 (target_steps>1): prefix-aligned; take chains[0:target_steps],
        matching what generate() produces autoregressively from position 0.
    """
    bsz, full_len, d_model = chains.shape
    steps = max(1, min(int(target_steps), full_len))
    targets = torch.zeros(bsz, steps, d_model, device=chains.device, dtype=chains.dtype)
    mask = torch.zeros(bsz, steps, device=chains.device, dtype=torch.bool)
    target_answer_pos = torch.zeros(bsz, device=chains.device, dtype=torch.long)
    target_has_answer = torch.zeros(bsz, device=chains.device, dtype=torch.bool)

    for i in range(bsz):
        li = max(1, min(int(chain_lens[i].item()), full_len))
        ti = min(steps, li)
        ai = max(0, min(int(answer_pos[i].item()), li - 1))

        if steps == 1:
            # System1: direct answer.  Use the first answer vector, not an
            # arbitrary repeated answer pad at the end of the chain.
            targets[i, 0] = chains[i, ai]
            mask[i, 0] = True
            target_answer_pos[i] = 0
            target_has_answer[i] = True
        else:
            # System2: prefix-aligned — first ti tokens.
            targets[i, :ti] = chains[i, :ti]
            mask[i, :ti] = True
            if ai < ti:
                target_answer_pos[i] = ai
                target_has_answer[i] = True
            else:
                target_answer_pos[i] = max(ti - 1, 0)

    return targets, mask, target_answer_pos, target_has_answer


def _masked_step_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    d_model: int,
    cosine_weight: float,
    mse_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred = pred.float()
    target = target.float()
    # Replace non-finite pred with target (loss≈0) instead of 0 (cos_loss=1.0).
    _bad_pred = ~torch.isfinite(pred)
    if _bad_pred.any():
        pred = torch.where(_bad_pred, target, pred)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    mask_bool = mask.to(device=pred.device, dtype=torch.bool)
    maskf = mask_bool.to(dtype=pred.dtype)
    # NaN×0 trap: clamp(min=1.0) does NOT fix NaN; NaN passes through
    # clamp unchanged.  Must scrub BEFORE clamping (lesson 2026-04-14).
    mask_sum = torch.nan_to_num(maskf.sum(), nan=1.0, posinf=1.0, neginf=1.0).clamp(min=1.0)

    cos_sim = (_safe_normalize(pred, dim=-1) * _safe_normalize(target, dim=-1)).sum(dim=-1)
    # Clamp to valid cosine range and scrub NaN (fp-rounding or poisoned
    # rows can push values slightly outside [-1, 1] and break (1 - cos)).
    cos_sim = torch.nan_to_num(cos_sim.clamp(min=-1.0, max=1.0), nan=0.0)
    cos_term = torch.where(mask_bool, 1.0 - cos_sim, torch.zeros_like(cos_sim))
    cos_loss = cos_term.sum() / mask_sum

    mse_per = (pred - target).pow(2).sum(dim=-1)
    mse_per = torch.nan_to_num(mse_per, nan=0.0, posinf=1e6, neginf=0.0)
    mse_term = torch.where(mask_bool, mse_per, torch.zeros_like(mse_per))
    mse_loss = mse_term.sum() / mask_sum

    loss = cosine_weight * cos_loss + mse_weight * mse_loss
    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    return loss, {
        "cos_sim": cos_sim,
        "cos_loss": cos_loss,
        "mse_loss": mse_loss,
        "mask_sum": mask_sum,
    }


def _answer_loss(
    pred_answer_all: torch.Tensor,
    target_answer_all: torch.Tensor,
    has_answer: torch.Tensor,
    cosine_weight: float,
    mse_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Answer-vector loss on samples whose selected window contains answer."""
    if has_answer.any():
        pred = pred_answer_all[has_answer].float()
        target = target_answer_all[has_answer].float()
        cos_sim = (_safe_normalize(pred, dim=-1) * _safe_normalize(target, dim=-1)).sum(dim=-1)
        cos_sim = torch.nan_to_num(cos_sim.clamp(min=-1.0, max=1.0), nan=0.0)
        cos_loss = (1.0 - cos_sim).mean()
        mse_loss = (pred - target).pow(2).sum(dim=-1).mean()
        loss = cosine_weight * cos_loss + mse_weight * mse_loss
        cos_mean = cos_sim.mean()
    else:
        loss = pred_answer_all.new_zeros(())
        cos_loss = pred_answer_all.new_zeros(())
        mse_loss = pred_answer_all.new_zeros(())
        cos_mean = pred_answer_all.new_zeros(())
    return loss, {
        "cos_loss": cos_loss,
        "mse_loss": mse_loss,
        "cos_mean": cos_mean,
    }


def _auxiliary_heads_loss(
    aux_preds: dict[str, torch.Tensor] | None,
    target: torch.Tensor,
    step_mask: torch.Tensor,
    answer_pos: torch.Tensor,
    has_answer: torch.Tensor,
    *,
    d_model: int,
    cosine_weight: float,
    mse_weight: float,
    answer_weight: float = 1.0,
    prefix: str = "aux",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Deep-supervision loss for intermediate decoder heads.

    The loss is averaged across aux heads, so the effective lambda does not
    change when we add/remove diagnostic heads. Each head gets the same
    masked step objective plus the selected answer-vector objective.
    """
    if not aux_preds:
        return target.new_zeros(()), {
            f"loss_{prefix}": 0.0,
            f"{prefix}_cos_mean": 0.0,
            f"{prefix}_cos_answer": 0.0,
        }

    batch_idx = torch.arange(target.shape[0], device=target.device)
    target_answer_all = target[batch_idx, answer_pos]
    losses: list[torch.Tensor] = []
    cos_means: list[float] = []
    ans_cos_means: list[float] = []
    metrics: dict[str, float] = {}

    for layer_name, pred_raw in sorted(aux_preds.items(), key=lambda kv: int(kv[0])):
        pred = torch.nan_to_num(pred_raw, nan=0.0, posinf=0.0, neginf=0.0)
        l_step, step_stats = _masked_step_losses(
            pred,
            target,
            step_mask,
            d_model,
            cosine_weight,
            mse_weight,
        )
        pred_answer_all = pred[batch_idx, answer_pos]
        l_ans, ans_stats = _answer_loss(
            pred_answer_all,
            target_answer_all,
            has_answer,
            cosine_weight,
            mse_weight,
        )
        layer_loss = l_step + float(answer_weight) * l_ans
        losses.append(layer_loss)

        mask_bool = step_mask.to(device=pred.device, dtype=torch.bool)
        maskf = mask_bool.to(dtype=pred.dtype)
        layer_ans_cos = float(ans_stats["cos_mean"].detach().item())
        valid_count = float(maskf.sum().detach().item())
        if valid_count > 0.0:
            valid = maskf.sum().clamp(min=1.0)
            cos_masked = torch.where(mask_bool, step_stats["cos_sim"], torch.zeros_like(step_stats["cos_sim"]))
            layer_cos = float((cos_masked.sum() / valid).detach().item())
        else:
            # System1 direct-answer training masks the answer out of L_step,
            # leaving no non-answer step positions. In that phase the aux
            # answer cosine is the only meaningful aux quality metric.
            layer_cos = layer_ans_cos
        cos_means.append(layer_cos)
        ans_cos_means.append(layer_ans_cos)
        metrics[f"{prefix}_l{layer_name}_loss"] = float(layer_loss.detach().item())
        metrics[f"{prefix}_l{layer_name}_step_loss"] = float(l_step.detach().item())
        metrics[f"{prefix}_l{layer_name}_ans_loss"] = float(l_ans.detach().item())
        metrics[f"{prefix}_l{layer_name}_cos_mean"] = layer_cos
        metrics[f"{prefix}_l{layer_name}_cos_answer"] = layer_ans_cos

    loss = torch.stack(losses).mean() if losses else target.new_zeros(())
    metrics[f"loss_{prefix}"] = float(loss.detach().item())
    metrics[f"{prefix}_cos_mean"] = float(sum(cos_means) / max(len(cos_means), 1))
    metrics[f"{prefix}_cos_answer"] = float(sum(ans_cos_means) / max(len(ans_cos_means), 1))
    return loss, metrics


class ModelEMA:
    """Exponential Moving Average of model parameters (Arch-1).

    Maintains a shadow copy of the model weights updated after every
    optimizer step as ``ema = decay · ema + (1 − decay) · live``.
    At eval time we swap the live weights with the EMA shadow to get a
    smoother, less noisy prediction — standard in modern diffusion training.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        self.decay = float(decay)
        # Store shadow on the same device as the model.
        self.shadow: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()
        # Also track buffers so norm running_mean/var stay consistent.
        self.buffers: dict[str, torch.Tensor] = {
            n: b.detach().clone() for n, b in model.named_buffers()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            s = self.shadow.get(name)
            if s is None:
                self.shadow[name] = p.detach().clone()
                continue
            s.mul_(d).add_(p.detach(), alpha=1.0 - d)
        # Buffers: just copy (they're not part of SGD anyway).
        for name, b in model.named_buffers():
            self.buffers[name] = b.detach().clone()

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Swap model params with EMA shadow; return original weights for restore."""
        backup: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name])
        return backup

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        for name, p in model.named_parameters():
            if name in backup:
                p.data.copy_(backup[name])

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()},
            "buffers": {k: v.detach().cpu() for k, v in self.buffers.items()},
        }

    def load_state_dict(self, sd: dict) -> None:
        self.decay = float(sd.get("decay", self.decay))
        for k, v in sd.get("shadow", {}).items():
            self.shadow[k] = v
        for k, v in sd.get("buffers", {}).items():
            self.buffers[k] = v


def _build_wd_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict]:
    """Split parameters into decayed / non-decayed groups (Arch-6).

    Non-decayed: biases, norm layers (LayerNorm/AdaRMSNorm weights),
    start_token, null_context_token. Everything else gets weight decay.
    Standard recipe from transformer training (Loshchilov 2019).
    """
    decay, no_decay = [], []
    no_decay_names: set[str] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_special_token = name.endswith("start_token") or name.endswith("null_context_token")
        # LayerScale gates (ls_self, ls_cross, ls_ffn) are 1-D but are NOT
        # biases or norm gains — they are learnable residual-branch gates
        # that can grow unboundedly without weight decay, amplifying FFN
        # output norms toward NaN at horizon transitions.
        is_layerscale = ".ls_" in name
        is_bias_or_norm = p.ndim <= 1 and not is_layerscale
        if is_bias_or_norm or is_special_token:
            no_decay.append(p)
            no_decay_names.add(name)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _masked_weighted_step_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
    d_model: int,
    cosine_weight: float,
    mse_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred = pred.float()
    target = target.float()
    # Replace non-finite pred with target (loss≈0) instead of 0 (cos_loss=1.0).
    _bad_pred = ~torch.isfinite(pred)
    if _bad_pred.any():
        pred = torch.where(_bad_pred, target, pred)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    mask_bool = mask.to(device=pred.device, dtype=torch.bool)
    weights = weights.to(device=pred.device, dtype=pred.dtype)
    # Guard against non-finite weights from an upstream SNR overflow BEFORE
    # masking, so torch.where never has to choose between NaN and zero.
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = torch.where(mask_bool, weights, torch.zeros_like(weights))
    # NaN×0 trap: clamp(min=1.0) does NOT fix NaN, it passes through.
    # Must scrub BEFORE clamping (lesson 2026-04-14 Adam-zombie).
    weight_sum = torch.nan_to_num(
        weights.sum(), nan=1.0, posinf=1.0, neginf=1.0
    ).clamp(min=1.0)

    cos_sim = (_safe_normalize(pred, dim=-1) * _safe_normalize(target, dim=-1)).sum(dim=-1)
    cos_sim = torch.nan_to_num(cos_sim.clamp(min=-1.0, max=1.0), nan=0.0)
    # CRITICAL: avoid ``NaN * 0`` by zeroing inside ``torch.where`` BEFORE
    # multiplying by ``weights``.  Direct ``(1 - cos) * weights`` is the
    # exact footgun flagged in tasks/lessons.md (2026-04-11, rule #3):
    # a single NaN at a masked position pollutes the entire reduction.
    cos_term = torch.where(
        mask_bool, (1.0 - cos_sim) * weights, torch.zeros_like(cos_sim)
    )
    cos_loss = cos_term.sum() / weight_sum

    mse_per = (pred - target).pow(2).sum(dim=-1)
    mse_per = torch.nan_to_num(mse_per, nan=0.0, posinf=1e6, neginf=0.0)
    mse_term = torch.where(
        mask_bool, mse_per * weights, torch.zeros_like(mse_per)
    )
    mse_loss = mse_term.sum() / weight_sum

    loss = cosine_weight * cos_loss + mse_weight * mse_loss
    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    return loss, {
        "cos_sim": cos_sim,
        "cos_loss": cos_loss,
        "mse_loss": mse_loss,
        "weight_sum": weight_sum,
    }


def _gather_last_valid(x: torch.Tensor, valid_lens: torch.Tensor) -> torch.Tensor:
    idx = (valid_lens - 1).clamp(min=0)
    return x[torch.arange(x.shape[0], device=x.device), idx]


def _safe_normalize(v: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Numerically safe normalization that prevents NaN gradients.

    Mirrors ``ChainGenerator._safe_normalize``: (1) scrubs any non-finite
    values from the input so downstream division never sees NaN/Inf, and
    (2) clamps the norm BEFORE division.  Without the NaN scrub, a single
    corrupted element (e.g. from a bfloat16 overflow in AMP) would poison
    the entire masked-reduce — the classic ``NaN * 0 = NaN`` trap flagged
    in ``tasks/lessons.md`` (2026-04-11).
    """
    v_float = v.float()
    v_float = torch.nan_to_num(v_float, nan=0.0, posinf=0.0, neginf=0.0)
    norms = v_float.norm(dim=dim, keepdim=True).clamp(min=eps)
    return v_float / norms


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
    p = _safe_normalize(pred_final.float(), dim=-1)
    t = _safe_normalize(tgt_final.float(), dim=-1)

    logits = (p @ t.t()) / temp
    labels = torch.arange(bsz, device=pred_final.device)

    l_pt = F.cross_entropy(logits, labels)
    l_tp = F.cross_entropy(logits.t(), labels)
    loss = 0.5 * (l_pt + l_tp)

    acc = (logits.argmax(dim=1) == labels).float().mean()
    return loss, acc


def _get_scheduled_sampling_prob(epoch: int, cfg: dict) -> float:
    """Compute scheduled sampling probability for the current epoch.

    Returns 0.0 during System1 (target_steps=1), then ramps linearly
    from 0.0 to ss_max over ss_ramp_epochs after System2 begins.
    """
    s1_epochs = int(cfg.get("system1_epochs", 10))
    if epoch < s1_epochs:
        return 0.0
    ss_max = float(cfg.get("scheduled_sampling_max", 0.5))
    ss_ramp = int(cfg.get("scheduled_sampling_ramp_epochs", 10))
    if ss_ramp <= 0:
        return ss_max
    progress = min(1.0, (epoch - s1_epochs) / ss_ramp)
    return ss_max * progress


def _get_oracle_prob(epoch: int, cfg: dict) -> float:
    """Compute oracle guidance probability for the current epoch.

    Oracle guidance is disabled by default because it creates a train/eval
    distribution mismatch: during training it helps the model stay close to
    GT, but at eval the oracle is absent (self.training=False). That makes
    train rollouts look much better than real rollouts.

    If explicitly re-enabled for an ablation, decay oracle_prob over training.
    """
    if not bool(cfg.get("enable_oracle_dagger", False)):
        return 0.0

    oracle_max = float(cfg.get("oracle_prob_max", 0.0))
    oracle_min = float(cfg.get("oracle_prob_min", 0.0))
    oracle_ramp = int(cfg.get("oracle_decay_epochs", 20))
    if oracle_ramp <= 0:
        return oracle_min
    progress = min(1.0, epoch / oracle_ramp)
    return oracle_max - (oracle_max - oracle_min) * progress


def _get_scheduled_noise_std(epoch: int, cfg: dict) -> float:
    """Compute noise std for the current epoch.

    Returns base noise during System1, then ramps linearly
    to max_noise over ss_ramp_epochs after System2 begins.
    """
    s1_epochs = int(cfg.get("system1_epochs", 10))
    base_noise = float(cfg.get("free_run_noise_std", 0.0))
    if epoch < s1_epochs:
        return base_noise
    max_noise = float(cfg.get("free_run_noise_std_max", base_noise))
    ss_ramp = int(cfg.get("scheduled_sampling_ramp_epochs", 10))
    if ss_ramp <= 0:
        return max_noise
    progress = min(1.0, (epoch - s1_epochs) / ss_ramp)
    return base_noise + (max_noise - base_noise) * progress


def _get_tf_noise_std(epoch: int, cfg: dict) -> float:
    """Compute noisy teacher-forcing noise std for the current epoch.

    Diffusion-inspired: add noise to the teacher-forced prefix so the model
    learns to predict from imperfect contexts.  Noise is in SONAR space
    (pre-residual-scaling), so values are relative to target_norm≈0.2051.

    Returns 0.0 if tf_noise_std_max is 0 or absent (disabled by default).
    Ramps linearly from 0 to tf_noise_std_max over tf_noise_ramp_epochs,
    then stays at max.  Starting from 0 ensures early training focuses on
    learning the clean mapping before introducing perturbation.
    """
    max_noise = float(cfg.get("tf_noise_std_max", 0.0))
    if max_noise <= 0.0:
        return 0.0
    ramp_epochs = int(cfg.get("tf_noise_ramp_epochs", 5))
    if ramp_epochs <= 0:
        return max_noise
    progress = min(1.0, epoch / ramp_epochs)
    return max_noise * progress


def _sample_diffusion_noise_levels(
    mask: torch.Tensor,
    cfg: dict,
    model: ChainGenerator,
    *,
    eval_mode: bool = False,
) -> torch.Tensor:
    """Sample independent per-position diffusion levels for valid targets.

    Supports two training sampling strategies:
    - ``"independent"`` (default): Uniform random over [min_level, max_level].
    - ``"logsnr"``:  Log-SNR stratified sampling (Karras et al. 2022).
      Uniformly samples in log-SNR space, then maps to nearest timestep.
      This gives more samples at intermediate noise levels (where denoising
      is hardest) and fewer at extremes.
    """
    timesteps = max(2, int(model.cfg.diffusion_timesteps))
    min_level = max(0, int(cfg.get("df_noise_level_min", 0)))
    max_level = int(cfg.get("df_noise_level_max", timesteps - 1))
    max_level = max(min_level, min(max_level, timesteps - 1))

    if eval_mode:
        eval_level = int(cfg.get("df_eval_noise_level", (min_level + max_level) // 2))
        levels = torch.full(mask.shape, eval_level, device=mask.device, dtype=torch.long)
    else:
        sampling = str(cfg.get("df_noise_sampling", "logsnr")).lower()
        if sampling == "logsnr":
            # Log-SNR stratified sampling (Karras et al. 2022).
            snr_buf = model._df_snr.float()  # [K]
            log_snr = torch.log(snr_buf.clamp(min=1e-8))
            log_snr_min = log_snr[max_level].item()   # noisiest
            log_snr_max = log_snr[min_level].item()    # cleanest
            # Sample uniformly in log-SNR, map to nearest timestep.
            u = torch.rand(mask.shape, device=mask.device)
            target_log_snr = log_snr_min + u * (log_snr_max - log_snr_min)
            # Nearest-timestep lookup via argmin over the schedule.
            diffs = (log_snr[min_level:max_level + 1].unsqueeze(0).unsqueeze(0)
                     - target_log_snr.unsqueeze(-1)).abs()
            levels = diffs.argmin(dim=-1) + min_level
        elif sampling == "independent":
            levels = torch.randint(min_level, max_level + 1, mask.shape, device=mask.device)
        else:
            raise ValueError(
                f"Unsupported df_noise_sampling={sampling}; use 'independent' or 'logsnr'"
            )

    return torch.where(mask.to(dtype=torch.bool), levels, torch.zeros_like(levels))


def _diffusion_forcing_weights(
    model: ChainGenerator,
    noise_levels: torch.Tensor,
    cfg: dict,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Min-SNR-γ weights for DF loss (Hang et al. 2023).

        Weight formula depends on ``prediction_type`` (Table 1 of Min-SNR paper,
        confirmed by tasks/lessons.md 2026-04-13 rule #2):
          - x₀-prediction: ``min(SNR, γ)``           (lesson L2064 correction)
          - ε-prediction:  ``min(SNR, γ) / SNR``     (lesson L13 derivation)
          - v-prediction:  ``min(SNR, γ) / (SNR + 1)``

        The x0 / eps formulas were previously swapped.  v-prediction (the
        default in this config) was always correct, so existing runs with
        ``prediction_type="v"`` are unaffected — but we now fix the latent
        bugs behind the disabled branches per lesson L20.

        Default gamma=5.0 (recommended by Hang et al.).  Set to 0 to disable.
        """
    gamma = float(cfg.get("df_min_snr_gamma", 5.0))
    if gamma <= 0.0:
        return torch.ones_like(noise_levels, dtype=torch.float32)

    # SNR must be computed entirely in float32: bf16 has no denormals and
    # ``clamp(min=1e-8)`` is a no-op when the value underflows to 0, which
    # then produces div-by-zero → Inf → NaN downstream.  Lesson 2026-04-14
    # (frozen-model zombie): keep ENTIRE min-SNR pipeline in fp32, scrub
    # NaN/Inf at every stage, only cast to fp32 output at the end.
    with torch.autocast(device_type=noise_levels.device.type, enabled=False):
        snr_raw = model.diffusion_snr(noise_levels).to(
            device=noise_levels.device, dtype=torch.float32
        )
        snr_raw = torch.nan_to_num(snr_raw, nan=0.0, posinf=1e4, neginf=0.0)
        # Both-sides clamp: min to avoid div-by-0, max to avoid numerical
        # blow-up at the cleanest timesteps where SNR can reach 1e8+.
        snr = snr_raw.clamp(min=1e-6, max=1e4)
        gamma_t = torch.full_like(snr, gamma)
        clipped = torch.minimum(snr, gamma_t)

        pt = getattr(model.cfg, "prediction_type", "x0")
        if pt == "x0":
            weights = clipped  # Fixed: was clipped/snr
        elif pt == "eps":
            weights = clipped / snr  # Fixed: was clipped
        elif pt == "v":
            weights = clipped / (snr + 1.0)
        else:
            weights = clipped / (snr + 1.0)  # safe fallback matching v-pred
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.where(
        mask.to(dtype=torch.bool), weights, torch.zeros_like(weights)
    )


def _make_diffusion_eval_noise_like(
    chains: torch.Tensor,
    cfg: dict,
    model: ChainGenerator,
) -> torch.Tensor:
    """Deterministic Gaussian eval noise for stable DF validation metrics."""
    base_seed = int(cfg.get("df_eval_noise_seed", 12345))
    with torch.no_grad():
        # Data-dependent seed keeps the noise stable for the same validation
        # batch while avoiding identical noise for every batch with same shape.
        digest = torch.abs(chains[: min(chains.shape[0], 4), :1, :16].float()).sum()
        digest_int = int((digest * 1_000_000).detach().cpu().item()) % 2_147_483_647
    seed = (base_seed + digest_int) % 2_147_483_647
    scale = float(model.cfg.diffusion_noise_scale)
    if scale <= 0.0:
        scale = float(model.cfg.target_norm) / math.sqrt(float(model.cfg.d_model))

    try:
        gen = torch.Generator(device=chains.device)
        gen.manual_seed(seed)
        noise = torch.randn(
            chains.shape,
            device=chains.device,
            dtype=chains.dtype,
            generator=gen,
        )
    except (TypeError, RuntimeError):
        gen = torch.Generator()
        gen.manual_seed(seed)
        noise = torch.randn(chains.shape, dtype=chains.dtype, generator=gen).to(device=chains.device)
    return noise * scale


def _to_jsonable(value):
    """Convert common numeric/tensor values to JSON-safe scalars."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_to_jsonable(payload), ensure_ascii=False) + "\n")


def _positional_df_weights(
    chain_mask: torch.Tensor,
    answer_pos: torch.Tensor | None,
    cfg: dict,
) -> torch.Tensor:
    """Per-position multiplier emphasizing positions near the answer.

    DF-6: exponential ramp — later positions (closer to answer_pos) receive
    higher weight, because the autoregressive error compounds toward the
    answer and the answer position is the primary supervision target.

    Returns a [B, T] float tensor.  Identity (all-ones) when disabled.
    """
    weight_type = str(cfg.get("positional_weight_type", "none")).lower()
    if weight_type == "none":
        return torch.ones_like(chain_mask, dtype=torch.float32)

    bsz, steps = chain_mask.shape
    device = chain_mask.device
    pos = torch.arange(steps, device=device, dtype=torch.float32).unsqueeze(0).expand(bsz, -1)

    if answer_pos is None:
        # Fall back to last valid position per row.
        ap = chain_mask.sum(dim=1).clamp(min=1).long() - 1
    else:
        ap = answer_pos.to(device=device).long().clamp(min=0, max=max(steps - 1, 0))
    ap_f = ap.to(dtype=torch.float32).unsqueeze(-1)  # [B, 1]

    if weight_type == "exponential":
        # w(i) = base^(-(ap - i)), clipped at 0..ap; base > 1.
        base = float(cfg.get("positional_weight_base", 1.15))
        dist = (ap_f - pos).clamp(min=0.0)
        w = torch.pow(torch.tensor(base, device=device), -dist)
    elif weight_type == "linear":
        # w(i) = 1 + alpha * (i / ap); at i=ap weight is 1+alpha.
        alpha = float(cfg.get("positional_weight_alpha", 1.0))
        denom = ap_f.clamp(min=1.0)
        w = 1.0 + alpha * (pos / denom).clamp(min=0.0, max=1.0)
    else:
        raise ValueError(f"Unsupported positional_weight_type={weight_type}")

    # Zero-out padded positions so the mask still dominates.
    w = torch.where(chain_mask, w, torch.zeros_like(w))
    return w


def _diffusion_forcing_objective(
    model: ChainGenerator,
    v_q: torch.Tensor,
    chains: torch.Tensor,
    chain_mask: torch.Tensor,
    context_banks: torch.Tensor,
    context_mask: torch.Tensor,
    cfg: dict,
    *,
    answer_pos: torch.Tensor | None = None,
    has_answer: torch.Tensor | None = None,
    eval_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Masked Diffusion Forcing loss in SONAR space.

    Correctly handles ``prediction_type`` (x0 / eps / v) by computing the
    proper supervision target and DF-6 positional reweighting.
    """
    bsz, steps, d_model = chains.shape
    levels = _sample_diffusion_noise_levels(chain_mask, cfg, model, eval_mode=eval_mode)
    snr_weights = _diffusion_forcing_weights(model, levels, cfg, chain_mask)
    pos_weights = _positional_df_weights(chain_mask, answer_pos, cfg).to(
        device=snr_weights.device, dtype=snr_weights.dtype,
    )
    weights = snr_weights * pos_weights
    noise = None
    if eval_mode and bool(cfg.get("df_eval_deterministic_noise", True)):
        noise = _make_diffusion_eval_noise_like(chains, cfg, model)

    aux_df_enabled = len(getattr(model, "aux_heads", {})) > 0 and float(cfg.get("loss_lambda_aux_df", 0.0)) > 0.0
    df_out = model.forward_diffusion_forcing(
        v_q,
        chains,
        levels,
        v_context_bank=context_banks,
        context_mask=context_mask,
        noise=noise,
        return_noisy=True,
        return_aux=aux_df_enabled,
    )
    if aux_df_enabled:
        model_out, v_noisy, eps, aux_df_preds = df_out
    else:
        model_out, v_noisy, eps = df_out
        aux_df_preds = None
    # Compute prediction-type-aware target FIRST so we can use it as the
    # NaN-replacement for poisoned decoder outputs (lessons.md 2026-04-20).
    # Replacing NaN with 0 here was a "loss bomb": cos(0, target) = 0 →
    # cos_loss = 1.0 on every poisoned position, drowning the masked
    # reduction even though the downstream guard expects NaN, not zeros.
    target = model.diffusion_target(chains, eps, levels)
    # `target` itself should never be NaN; if it is, we have a bug in
    # diffusion_target (e.g. v-target with NaN eps).  Surface it loudly
    # rather than silently replacing — but still keep training alive.
    if not torch.isfinite(target).all():
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

    # Defense-in-depth: replace any non-finite decoder output with the
    # supervision target.  This makes the per-position loss on poisoned
    # positions exactly 0 (cos_sim=1, mse=0) — a true no-op — instead of
    # the cos_sim=0, mse>0 "loss bomb" that nan_to_num(0) produced.
    _bad_out = ~torch.isfinite(model_out)
    if _bad_out.any():
        model_out = torch.where(_bad_out, target, model_out)

    loss, stats = _masked_weighted_step_losses(
        model_out,
        target,
        chain_mask,
        weights,
        d_model,
        float(model.cfg.loss_cosine_weight),
        float(model.cfg.loss_mse_weight),
    )
    if aux_df_preds:
        # Aux DF heads predict clean x0 directly from noisy intermediate states.
        # This gives a layer-local denoising canary independent of pred_type.
        l_aux_df, aux_df_metrics = _auxiliary_heads_loss(
            aux_df_preds,
            chains,
            chain_mask,
            answer_pos if answer_pos is not None else (chain_mask.sum(dim=1).long().clamp(min=1) - 1),
            has_answer.to(device=chains.device, dtype=torch.bool)
            if has_answer is not None
            else torch.ones(chains.shape[0], device=chains.device, dtype=torch.bool),
            d_model=d_model,
            cosine_weight=float(model.cfg.loss_cosine_weight),
            mse_weight=float(model.cfg.loss_mse_weight),
            answer_weight=0.0,
            prefix="aux_df",
        )
    else:
        l_aux_df = chains.new_zeros(())
        aux_df_metrics = {
            "loss_aux_df": 0.0,
            "aux_df_cos_mean": 0.0,
            "aux_df_cos_answer": 0.0,
        }

    with torch.no_grad():
        maskf = chain_mask.to(dtype=torch.float32)
        valid = maskf.sum().clamp(min=1.0)

        # Prediction-type-space cosine (v vs v_target, or x0 vs x0, etc.).
        cos_sim_pred = stats["cos_sim"]
        cos_mean_pred = torch.where(chain_mask, cos_sim_pred, torch.zeros_like(cos_sim_pred)).sum() / valid

        # Diagnostic: decode model output to x0 space and report cos vs clean chain.
        x0_pred = model.predict_x0(v_noisy, model_out, levels).float()
        cos_x0 = (
            _safe_normalize(x0_pred, dim=-1) * _safe_normalize(chains.float(), dim=-1)
        ).sum(dim=-1)
        cos_x0_mean = torch.where(chain_mask, cos_x0, torch.zeros_like(cos_x0)).sum() / valid

        valid_levels = torch.where(chain_mask, levels.float(), torch.zeros_like(levels.float()))
        noisy_norm = torch.where(chain_mask, v_noisy.norm(dim=-1), torch.zeros_like(maskf)).sum() / valid
        eps_norm = torch.where(chain_mask, eps.norm(dim=-1), torch.zeros_like(maskf)).sum() / valid
        pred_norm = torch.where(chain_mask, model_out.norm(dim=-1), torch.zeros_like(maskf)).sum() / valid

        metrics = {
            "loss_df": float(loss.item()),
            "loss_aux_df": float(l_aux_df.detach().item()),
            "df_cos": float(cos_x0_mean.item()),  # x0-space (interpretable across pred_type)
            "df_cos_target": float(cos_mean_pred.item()),  # raw target-space cos (pred_type dep.)
            "df_cos_loss_raw": float(stats["cos_loss"].item()),
            "df_mse_loss_raw": float(stats["mse_loss"].item()),
            "df_noise_level_mean": float((valid_levels.sum() / valid).item()),
            "df_noise_level_max": float(levels[chain_mask].max().item()) if chain_mask.any() else 0.0,
            "df_noisy_norm": float(noisy_norm.item()),
            "df_eps_norm": float(eps_norm.item()),
            "df_pred_norm": float(pred_norm.item()),
            "df_weight_mean": float((weights * maskf).sum().item() / valid.item()),
            "df_pos_weight_mean": float((pos_weights * maskf).sum().item() / valid.item()),
        }
        metrics.update(aux_df_metrics)

    return loss, l_aux_df, metrics


def compute_composite_objective(
    model: ChainGenerator,
    v_q: torch.Tensor,
    chains: torch.Tensor,
    chain_mask: torch.Tensor,
    chain_lens: torch.Tensor,
    answer_pos: torch.Tensor,
    has_answer: torch.Tensor,
    context_banks: torch.Tensor,
    context_mask: torch.Tensor,
    cfg: dict,
    scheduled_sampling_prob: float = 0.0,
    free_run_noise_std: float = 0.0,
    oracle_prob: float = 0.0,
    tf_noise_std: float = 0.0,
    eval_mode: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute full autoregressive objective for one batch."""
    bsz, steps, d_model = chains.shape
    valid_lens = chain_mask.sum(dim=1).long().clamp(min=1)
    answer_pos = answer_pos.to(device=chains.device).long().clamp(min=0, max=max(steps - 1, 0))
    has_answer = has_answer.to(device=chains.device, dtype=torch.bool)

    # Lambda weights.
    lambda_step = float(cfg.get("loss_lambda_step", 1.0))
    lambda_ans = float(cfg.get("loss_lambda_answer", 1.0))
    lambda_roll = float(cfg.get("loss_lambda_roll", 1.0))
    lambda_rank = float(cfg.get("loss_lambda_rank", 0.1))
    lambda_df = float(cfg.get("loss_lambda_diffusion", 0.0))
    lambda_aux = float(cfg.get("loss_lambda_aux", 0.0))
    lambda_aux_df = float(cfg.get("loss_lambda_aux_df", 0.0))
    aux_answer_weight = float(cfg.get("aux_loss_answer_weight", 1.0))
    lambda_bptt = float(cfg.get("loss_lambda_bptt", 0.0))
    lambda_norm = float(cfg.get("loss_lambda_norm_penalty", 0.0))
    bptt_k = int(cfg.get("bptt_steps", -1))
    rank_enabled = lambda_rank > 0.0
    df_enabled = bool(cfg.get("enable_diffusion_forcing", False)) and lambda_df > 0.0
    bptt_enabled = bool(cfg.get("bptt_enabled", False)) and lambda_bptt > 0.0 and steps >= 1
    aux_enabled = len(getattr(model, "aux_heads", {})) > 0 and lambda_aux > 0.0
    oracle_enabled = bool(cfg.get("enable_oracle_dagger", False))
    effective_oracle_prob = float(oracle_prob) if oracle_enabled else 0.0

    # Base cosine/MSE weights from model config.
    w_cos = float(model.cfg.loss_cosine_weight)
    w_mse = float(model.cfg.loss_mse_weight)

    # 1) Teacher-forced masked step loss (with optional scheduled sampling).
    # Noisy TF: inject noise into the teacher-forced prefix so the model
    # learns to predict from imperfect contexts (diffusion-inspired).
    tf_out = model.forward(
        v_q,
        chains,
        v_context_bank=context_banks,
        context_mask=context_mask,
        scheduled_sampling_prob=scheduled_sampling_prob,
        tf_noise_std=tf_noise_std,
        return_aux=aux_enabled,
    )
    if aux_enabled:
        v_tf, aux_tf_preds = tf_out
    else:
        v_tf = tf_out
        aux_tf_preds = None
    # Defense-in-depth: replace non-finite predictions with GT so the
    # loss sees ≈0 (pred==target) instead of the nan_to_num(0) artifact
    # that gives cos_loss=1.0 — the "loss bomb" pattern (lessons 2026-04-19).
    _bad_tf = ~torch.isfinite(v_tf)
    if _bad_tf.any():
        v_tf = torch.where(_bad_tf, chains, v_tf)

    # Fix #7: Only exclude last position from L_step when L_ans is active
    # for that specific sample (i.e., when the window contains the answer).
    step_mask = chain_mask.clone()
    for i in range(bsz):
        if has_answer[i]:
            ans_idx = int(answer_pos[i].item())
            if ans_idx >= 0 and ans_idx < step_mask.shape[1]:
                step_mask[i, ans_idx] = False

    l_step, tf_stats = _masked_step_losses(v_tf, chains, step_mask, d_model, w_cos, w_mse)

    l_aux, aux_metrics = _auxiliary_heads_loss(
        aux_tf_preds,
        chains,
        step_mask,
        answer_pos,
        has_answer,
        d_model=d_model,
        cosine_weight=w_cos,
        mse_weight=w_mse,
        answer_weight=aux_answer_weight,
        prefix="aux",
    )

    # 2) Final-answer supervised loss — only on samples that reached the answer.
    tf_final = _gather_last_valid(v_tf, valid_lens)
    tgt_final = _gather_last_valid(chains, valid_lens)
    batch_idx = torch.arange(bsz, device=chains.device)
    tf_answer_all = v_tf[batch_idx, answer_pos]
    tgt_answer_all = chains[batch_idx, answer_pos]
    
    l_ans, _ans_stats = _answer_loss(
        tf_answer_all, tgt_answer_all, has_answer, w_cos, w_mse,
    )

    # 3) Free-run rollout loss (exposure-bias correction).
    v_roll, roll_info = model.generate(
        v_q,
        num_steps=steps,
        v_context_bank=context_banks,
        context_mask=context_mask,
        temperature=float(cfg.get("free_run_temperature", 1.0)),
        latent_noise_std=free_run_noise_std,
        repeat_penalty=float(cfg.get("free_run_repeat_penalty", 0.0)),
        repeat_cos_threshold=float(cfg.get("free_run_repeat_cos_threshold", 0.98)),
        repeat_ban_threshold=float(cfg.get("free_run_repeat_ban_threshold", 0.995)),
        repeat_ban_max_retries=int(cfg.get("free_run_repeat_ban_retries", 2)),
        oracle_guide=chains if oracle_enabled else None,
        oracle_max_retries=int(cfg.get("oracle_max_retries", 0)) if oracle_enabled else 0,
        oracle_prob=effective_oracle_prob,
        return_info=True,
    )
    # Replace non-finite rollout predictions with GT (same fix as v_tf above).
    # nan_to_num(0) was the "loss bomb": cos_sim(0,GT)=0 → cos_loss=1.0 per
    # NaN position, inflating rollout loss and accelerating NaN cascade.
    _bad_roll = ~torch.isfinite(v_roll)
    if _bad_roll.any():
        v_roll = torch.where(_bad_roll, chains, v_roll)
    l_roll, roll_stats = _masked_step_losses(v_roll, chains, chain_mask, d_model, w_cos, w_mse)

    # 4) In-batch contrastive ranking on final rollout answer.
    roll_final = _gather_last_valid(v_roll, valid_lens)
    roll_answer_all = v_roll[batch_idx, answer_pos]
    if rank_enabled and has_answer.any():
        l_rank, rank_acc = _inbatch_contrastive_loss(
            roll_answer_all[has_answer],
            tgt_answer_all[has_answer],
            temperature=float(cfg.get("contrastive_temperature", 0.07)),
        )
    else:
        l_rank = chains.new_zeros(())
        with torch.no_grad():
            if has_answer.any():
                _, rank_acc = _inbatch_contrastive_loss(
                    roll_answer_all[has_answer].detach(),
                    tgt_answer_all[has_answer].detach(),
                    temperature=float(cfg.get("contrastive_temperature", 0.07)),
                )
                if not torch.isfinite(rank_acc):
                    rank_acc = chains.new_zeros(())
            else:
                rank_acc = chains.new_zeros(())

    loss = lambda_step * l_step + lambda_ans * l_ans + lambda_roll * l_roll
    if rank_enabled:
        loss = loss + lambda_rank * l_rank
    if aux_enabled:
        loss = loss + lambda_aux * l_aux

    if df_enabled:
        l_df, l_aux_df, df_metrics = _diffusion_forcing_objective(
            model,
            v_q,
            chains,
            chain_mask,
            context_banks,
            context_mask,
            cfg,
            answer_pos=answer_pos,
            has_answer=has_answer,
            eval_mode=eval_mode,
        )
        loss = loss + lambda_df * l_df
        if lambda_aux_df > 0.0:
            loss = loss + lambda_aux_df * l_aux_df
    else:
        l_df = chains.new_zeros(())
        l_aux_df = chains.new_zeros(())
        df_metrics = {
            "loss_df": 0.0,
            "loss_aux_df": 0.0,
            "df_cos": 0.0,
            "df_cos_target": 0.0,
            "df_cos_loss_raw": 0.0,
            "df_mse_loss_raw": 0.0,
            "df_noise_level_mean": 0.0,
            "df_noise_level_max": 0.0,
            "df_noisy_norm": 0.0,
            "df_eps_norm": 0.0,
            "df_pred_norm": 0.0,
            "df_weight_mean": 0.0,
            "df_pos_weight_mean": 0.0,
            "aux_df_cos_mean": 0.0,
            "aux_df_cos_answer": 0.0,
        }

    # 5) Truncated BPTT through the generation chain (STE projection).
    if bptt_enabled:
        effective_bptt_k = steps if bptt_k < 0 else min(bptt_k, steps)
        all_bptt, bptt_preds, bptt_raw_norms = model.generate_with_bptt(
            v_q,
            num_steps=steps,
            bptt_steps=effective_bptt_k,
            v_context_bank=context_banks,
            context_mask=context_mask,
        )

        bptt_targets = chains[:, -effective_bptt_k:, :]
        bptt_mask = chain_mask[:, -effective_bptt_k:]
        l_bptt, bptt_stats = _masked_step_losses(
            bptt_preds, bptt_targets, bptt_mask, d_model, w_cos, w_mse,
        )
        l_bptt = torch.nan_to_num(l_bptt, nan=0.0, posinf=0.0, neginf=0.0)

        # Norm penalty on RAW output norms (pre-projection).
        # STE forward gives exact target_norm, but we penalize raw output
        # deviation to keep the STE approximation accurate (small gap between
        # forward sphere_project and backward identity).
        target_norm_val = float(model.cfg.target_norm)
        target_norm_safe = max(target_norm_val, 1e-6)
        rel_dev = (bptt_raw_norms - target_norm_val) / target_norm_safe
        l_norm_penalty = rel_dev.pow(2).mean()
        l_norm_penalty = torch.nan_to_num(l_norm_penalty, nan=0.0, posinf=0.0, neginf=0.0)

        loss = loss + lambda_bptt * l_bptt + lambda_norm * l_norm_penalty

        bptt_maskf = bptt_mask.to(dtype=chains.dtype)
        bptt_valid = bptt_maskf.sum().clamp(min=1.0)
        bptt_cos_masked = (bptt_stats["cos_sim"] * bptt_maskf).sum() / bptt_valid
        bptt_metrics = {
            "loss_bptt": float(l_bptt.item()),
            "loss_norm_penalty": float(l_norm_penalty.item()),
            "bptt_cos_mean": float(bptt_cos_masked.item()),
            "bptt_norm_mean": float(bptt_raw_norms.detach().mean().item()),
            "bptt_norm_std": float(bptt_raw_norms.detach().std().item()) if bptt_raw_norms.numel() > 1 else 0.0,
            "bptt_nan_count": float(getattr(model, "_bptt_nan_gate_count", 0)),
        }
    else:
        l_bptt = chains.new_zeros(())
        l_norm_penalty = chains.new_zeros(())
        bptt_metrics = {
            "loss_bptt": 0.0,
            "loss_norm_penalty": 0.0,
            "bptt_cos_mean": 0.0,
            "bptt_norm_mean": 0.0,
            "bptt_norm_std": 0.0,
            "bptt_nan_count": 0.0,
        }

    with torch.no_grad():
        # Use FULL mask (incl. answer) for reporting metrics.
        tf_cos = (_safe_normalize(v_tf.float(), dim=-1) * _safe_normalize(chains.float(), dim=-1)).sum(dim=-1)
        roll_cos = (_safe_normalize(v_roll.float(), dim=-1) * _safe_normalize(chains.float(), dim=-1)).sum(dim=-1)
        tf_cos_masked = torch.where(chain_mask, tf_cos, torch.zeros_like(tf_cos))
        roll_cos_masked = torch.where(chain_mask, roll_cos, torch.zeros_like(roll_cos))
        tf_norm = v_tf.norm(dim=-1)
        roll_norm = v_roll.norm(dim=-1)
        tf_norm_masked = torch.where(chain_mask, tf_norm, torch.zeros_like(tf_norm))
        roll_norm_masked = torch.where(chain_mask, roll_norm, torch.zeros_like(roll_norm))
        maskf = chain_mask.to(dtype=chains.dtype)
        valid = maskf.sum().clamp(min=1.0)

        tf_cos_mean = (tf_cos_masked.sum() / valid).item()
        roll_cos_mean = (roll_cos_masked.sum() / valid).item()

        tf_cos_last = (
            _safe_normalize(tf_final.float(), dim=-1)
            * _safe_normalize(tgt_final.float(), dim=-1)
        ).sum(dim=-1).mean().item()
        roll_cos_last = (
            _safe_normalize(roll_final.float(), dim=-1)
            * _safe_normalize(tgt_final.float(), dim=-1)
        ).sum(dim=-1).mean().item()
        if has_answer.any():
            roll_answer_cos = (
                _safe_normalize(roll_answer_all[has_answer].float(), dim=-1)
                * _safe_normalize(tgt_answer_all[has_answer].float(), dim=-1)
            ).sum(dim=-1).mean().item()
            tf_answer_cos = (
                _safe_normalize(tf_answer_all[has_answer].float(), dim=-1)
                * _safe_normalize(tgt_answer_all[has_answer].float(), dim=-1)
            ).sum(dim=-1).mean().item()
        else:
            roll_answer_cos = 0.0
            tf_answer_cos = 0.0

        # Fix #1: Per-step cosine diagnostics.
        tf_cos_first = float(tf_cos[:, 0].mean().item()) if steps >= 1 else 0.0
        roll_cos_first = float(roll_cos[:, 0].mean().item()) if steps >= 1 else 0.0

        metrics = {
            "loss": float(loss.item()),
            "loss_step": float(l_step.item()),
            "loss_ans": float(l_ans.item()),
            "loss_roll": float(l_roll.item()),
            "loss_rank": float(l_rank.item()),
            "loss_df": float(l_df.item()),
            "tf_cos_mean": float(tf_cos_mean),
            "tf_cos_last": float(tf_cos_last),
            "tf_cos_answer": float(tf_answer_cos),
            "tf_cos_first": float(tf_cos_first),
            "roll_cos_mean": float(roll_cos_mean),
            "roll_cos_last": float(roll_cos_last),
            "roll_cos_answer": float(roll_answer_cos),
            "roll_cos_first": float(roll_cos_first),
            "rank_acc": float(rank_acc.item()),
            "pred_norm_mean_tf": float(tf_norm_masked.sum().item() / valid.item()),
            "pred_norm_mean_roll": float(roll_norm_masked.sum().item() / valid.item()),
            "valid_tokens": float(valid.item()),
            "valid_tokens_per_sample": float(valid_lens.float().mean().item()),
            "lambda_step": lambda_step,
            "lambda_ans": lambda_ans,
            "lambda_roll": lambda_roll,
            "lambda_rank": lambda_rank,
            "lambda_df": lambda_df,
            "lambda_aux": lambda_aux,
            "lambda_aux_df": lambda_aux_df,
            "ss_prob": float(scheduled_sampling_prob),
            "free_run_noise_std": float(free_run_noise_std),
            "tf_noise_std": float(tf_noise_std),
            "df_enabled": 1.0 if df_enabled else 0.0,
            "oracle_enabled": 1.0 if oracle_enabled else 0.0,
            "oracle_prob": float(effective_oracle_prob),
            "answer_coverage": float(has_answer.float().mean().item()),
            "raw_norm_mean": float(roll_info.get("raw_norm_mean", 0.0)),
            "nan_gate_count": float(getattr(model, "_nan_gate_count", 0)),
        }

        # Additional raw components for debugging.
        metrics["tf_cos_loss_raw"] = float(tf_stats["cos_loss"].item())
        metrics["tf_mse_loss_raw"] = float(tf_stats["mse_loss"].item())
        metrics["roll_cos_loss_raw"] = float(roll_stats["cos_loss"].item())
        metrics["roll_mse_loss_raw"] = float(roll_stats["mse_loss"].item())
        metrics.update(aux_metrics)
        metrics.update(df_metrics)
        metrics.update(bptt_metrics)
        metrics["lambda_bptt"] = lambda_bptt
        metrics["lambda_norm_penalty"] = lambda_norm
        metrics["bptt_enabled"] = 1.0 if bptt_enabled else 0.0

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
    scheduled_sampling_prob: float = 0.0,
    free_run_noise_std: float = 0.0,
    oracle_prob: float = 0.0,
    tf_noise_std: float = 0.0,
    ema: "ModelEMA | None" = None,
) -> dict[str, float]:
    v_q = batch["v_questions"].to(device)
    chains = batch["chains"].to(device)
    chain_lens = batch["chain_lens"].to(device)
    answer_pos = batch["answer_pos"].to(device)
    context_banks = batch["context_banks"].to(device)
    context_mask = batch["context_mask"].to(device)

    chains_trunc, chain_mask, target_answer_pos, target_has_answer = select_training_targets(
        chains, chain_lens, answer_pos, target_steps
    )

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        loss, metrics = compute_composite_objective(
            model,
            v_q,
            chains_trunc,
            chain_mask,
            chain_lens,
            target_answer_pos,
            target_has_answer,
            context_banks,
            context_mask,
            cfg,
            scheduled_sampling_prob=scheduled_sampling_prob,
            free_run_noise_std=free_run_noise_std,
            oracle_prob=oracle_prob,
            tf_noise_std=tf_noise_std,
        )

    # Defense-in-depth: scrub any residual NaN/Inf in the final loss scalar
    # before backward.  Even if upstream guards catch most issues, a single
    # poisoned element elsewhere in the graph can still propagate; we must
    # never hand NaN to autograd (lesson 2026-04-14 Adam-zombie).
    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    # A completely scrubbed-to-zero loss has no signal — skip the backward
    # so we don't waste compute on a no-op step, but crucially keep the
    # optimizer / scaler state clean.
    if not torch.isfinite(loss) or float(loss.detach()) == 0.0:
        optimizer.zero_grad(set_to_none=True)
        metrics["nan_skipped"] = 1.0
        metrics["nan_loss_skipped"] = 1.0
        metrics["nan_grad_skipped"] = 0.0
        metrics["grad_sanitized"] = 0.0
        metrics["param_restored"] = 0.0
        metrics["optimizer_stepped"] = 0.0
        metrics["target_steps"] = float(target_steps)
        metrics["target_is_answer"] = 1.0 if target_steps == 1 else 0.0
        return metrics

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    # --- Gradient sanitation (preserve-momentum policy) --------------
    # Zero only the NaN/Inf entries in ``.grad``.  Adam sees a zero
    # gradient for those params → exp_avg decays by β₁, exp_avg_sq
    # decays by β₂ (natural "forget" signal).  We PRESERVE healthy
    # momentum buffers.  Lesson 2026-04-16: unconditionally zeroing
    # momentum created the Adam-zombie — after a reset the first
    # non-zero gradient produces an update of magnitude lr/√ε ≈ 1e4,
    # which instantly destabilizes the model and locks val metrics.
    # The only buffers we zero are those that are themselves non-finite.
    grad_had_nan = False
    sanitized_count = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad
        bad = ~torch.isfinite(g)
        if bad.any():
            grad_had_nan = True
            sanitized_count += 1
            g.masked_fill_(bad, 0.0)
            # DO NOT zero Adam momentum here.  With zero grad, Adam
            # naturally decays exp_avg by β₁ and exp_avg_sq by β₂ —
            # effectively "this step had no signal, slowly forget the
            # old direction".  Zeroing momentum instead creates the
            # Adam-zombie: after reset, Adam's update is dominated by
            # lr / sqrt(ε) ≈ 1e4, which produces a huge destabilizing
            # step on the next non-zero gradient (lesson 2026-04-16).
            # Only sanitize buffers that are themselves non-finite.
            state = optimizer.state.get(p)
            if state:
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    buf = state.get(key)
                    if buf is not None and not torch.isfinite(buf).all():
                        buf.zero_()

    # --- Zombie streak detector + hard EMA reset ---------------------
    # When grad sanitation happens on many consecutive steps, the
    # model is stuck in a corrupted basin from which per-step grad
    # scrubbing cannot escape (every forward still produces NaN because
    # a weight is in a numerically fragile region, even if it is still
    # technically `isfinite`).  We maintain a streak counter on the EMA
    # object and, once it crosses a threshold, force-restore ALL
    # parameters from the EMA shadow and zero ALL Adam state.  This is
    # the break-glass path that guarantees we can escape ANY basin.
    zombie_threshold = int(cfg.get("zombie_reset_threshold", 15))
    zombie_reset = 0
    zombie_resets_total = 0
    zombie_streak_now = 0
    if ema is not None:
        if not hasattr(ema, "_zombie_streak"):
            ema._zombie_streak = 0
            ema._zombie_resets = 0
        if grad_had_nan:
            ema._zombie_streak += 1
        else:
            ema._zombie_streak = 0

        if ema._zombie_streak >= zombie_threshold:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    shadow = ema.shadow.get(name)
                    if shadow is not None and torch.isfinite(shadow).all():
                        p.data.copy_(shadow)
                    else:
                        torch.nan_to_num(
                            p.data, nan=0.0, posinf=0.0, neginf=0.0,
                            out=p.data,
                        )
                    # Only sanitize non-finite Adam buffers.  Zeroing
                    # healthy momentum is what created the Adam-zombie:
                    # after full reset, update ≈ lr/√ε ≈ 1e4 on the
                    # first non-zero gradient, destabilizing the model.
                    # With preserved momentum, Adam continues from the
                    # restored shadow weights with working dynamics.
                    state = optimizer.state.get(p)
                    if state:
                        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                            buf = state.get(key)
                            if buf is not None and not torch.isfinite(buf).all():
                                buf.zero_()
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            ema._zombie_streak = 0
            ema._zombie_resets += 1
            zombie_reset = 1
            zombie_resets_total = ema._zombie_resets
            metrics["nan_skipped"] = 1.0
            metrics["nan_loss_skipped"] = 0.0
            metrics["nan_grad_skipped"] = 0.0
            metrics["grad_sanitized"] = float(sanitized_count)
            metrics["param_restored"] = 0.0
            metrics["zombie_reset"] = 1.0
            metrics["zombie_resets_total"] = float(zombie_resets_total)
            metrics["zombie_streak"] = 0.0
            metrics["optimizer_stepped"] = 0.0
            metrics["grad_norm"] = 0.0
            metrics["target_steps"] = float(target_steps)
            metrics["target_is_answer"] = 1.0 if target_steps == 1 else 0.0
            return metrics
        zombie_streak_now = ema._zombie_streak
        zombie_resets_total = ema._zombie_resets

    clip_grad = float(cfg.get("clip_grad_norm", 1.0))
    if clip_grad > 0:
        total_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        if not torch.isfinite(total_norm):
            # Defensive: extremely unlikely after the sanitation above,
            # but if it still happens, zero grads and skip the step.
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            metrics["nan_skipped"] = 1.0
            metrics["nan_loss_skipped"] = 0.0
            metrics["nan_grad_skipped"] = 1.0
            metrics["grad_sanitized"] = float(sanitized_count)
            metrics["param_restored"] = 0.0
            metrics["zombie_reset"] = 0.0
            metrics["zombie_resets_total"] = float(zombie_resets_total)
            metrics["zombie_streak"] = float(zombie_streak_now)
            metrics["optimizer_stepped"] = 0.0
            metrics["target_steps"] = float(target_steps)
            metrics["target_is_answer"] = 1.0 if target_steps == 1 else 0.0
            return metrics
    else:
        total_norm = torch.tensor(0.0, device=device)

    scaler.step(optimizer)
    scaler.update()

    # --- Post-step parameter sanity check (EMA break-glass) --------
    # If the optimizer step somehow produced NaN/Inf or HUGE-but-finite
    # drift in parameters, restore from EMA.  ``isfinite`` does NOT
    # catch values like 1e30 — we therefore also guard against norms
    # exceeding a generous threshold relative to initialization.
    params_restored = 0
    max_param_abs = float(cfg.get("param_abs_max", 1.0e4))
    if ema is not None:
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            non_finite = not torch.isfinite(p.data).all()
            too_large = False
            if not non_finite:
                with torch.no_grad():
                    pmax = float(p.data.abs().amax().item())
                too_large = pmax > max_param_abs
            if non_finite or too_large:
                shadow = ema.shadow.get(name)
                if shadow is not None and torch.isfinite(shadow).all():
                    p.data.copy_(shadow)
                    params_restored += 1
                else:
                    torch.nan_to_num(
                        p.data, nan=0.0, posinf=0.0, neginf=0.0, out=p.data
                    )
                    params_restored += 1
                # Only sanitize non-finite Adam buffers — preserve
                # healthy momentum so Adam can continue learning
                # after the parameter restore (lesson 2026-04-16).
                state = optimizer.state.get(p)
                if state:
                    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        buf = state.get(key)
                        if buf is not None and not torch.isfinite(buf).all():
                            buf.zero_()

    # EMA update ONLY on fully clean steps — NO grad sanitation AND
    # NO param restore.  Otherwise subtle drift (that passed isfinite
    # but is already polluted) leaks into the shadow, and the very
    # rescue source becomes the source of future corruption.
    if ema is not None and not grad_had_nan and params_restored == 0:
        ema.update(model)

    metrics["nan_skipped"] = 1.0 if (grad_had_nan or params_restored > 0) else 0.0
    metrics["nan_loss_skipped"] = 0.0
    metrics["nan_grad_skipped"] = 0.0
    metrics["grad_sanitized"] = float(sanitized_count)
    metrics["param_restored"] = float(params_restored)
    metrics["zombie_reset"] = 0.0
    metrics["zombie_resets_total"] = float(zombie_resets_total)
    metrics["zombie_streak"] = float(zombie_streak_now)
    metrics["optimizer_stepped"] = 1.0
    metrics["grad_norm"] = float(total_norm.item()) if clip_grad > 0 else 0.0
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
    answer_pos = batch["answer_pos"].to(device)
    context_banks = batch["context_banks"].to(device)
    context_mask = batch["context_mask"].to(device)

    chains_trunc, chain_mask, target_answer_pos, target_has_answer = select_training_targets(
        chains, chain_lens, answer_pos, gen_steps
    )

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        _, metrics = compute_composite_objective(
            model,
            v_q,
            chains_trunc,
            chain_mask,
            chain_lens,
            target_answer_pos,
            target_has_answer,
            context_banks,
            context_mask,
            cfg,
            free_run_noise_std=float(cfg.get("eval_free_run_noise_std", 0.0)),
            eval_mode=True,
        )

    return {
        "val_loss": metrics["loss"],
        "val_loss_step": metrics["loss_step"],
        "val_loss_ans": metrics["loss_ans"],
        "val_loss_roll": metrics["loss_roll"],
        "val_loss_rank": metrics["loss_rank"],
        "val_loss_df": metrics["loss_df"],
        "val_loss_aux": metrics.get("loss_aux", 0.0),
        "val_loss_aux_df": metrics.get("loss_aux_df", 0.0),
        "val_tf_cos": metrics["tf_cos_mean"],
        "val_tf_cos_last": metrics["tf_cos_last"],
        "val_tf_cos_answer": metrics["tf_cos_answer"],
        "val_roll_cos": metrics["roll_cos_mean"],
        "val_roll_cos_last": metrics["roll_cos_last"],
        "val_roll_cos_answer": metrics["roll_cos_answer"],
        "val_rank_acc": metrics["rank_acc"],
        "val_df_cos": metrics["df_cos"],
        "val_aux_cos": metrics.get("aux_cos_mean", 0.0),
        "val_aux_cos_answer": metrics.get("aux_cos_answer", 0.0),
        "val_aux_df_cos": metrics.get("aux_df_cos_mean", 0.0),
        "val_df_noise_level": metrics["df_noise_level_mean"],
        "val_df_noisy_norm": metrics["df_noisy_norm"],
        "val_df_eps_norm": metrics["df_eps_norm"],
        "val_df_pred_norm": metrics["df_pred_norm"],
        "val_norm_tf": metrics["pred_norm_mean_tf"],
        "val_norm_roll": metrics["pred_norm_mean_roll"],
        "val_answer_coverage": metrics["answer_coverage"],
    }


def _fit_probe_projection_basis(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a deterministic 3D PCA basis on CPU for stable probe visualization."""
    points = points.detach().float().cpu()
    if points.dim() != 2:
        points = points.reshape(-1, points.shape[-1])
    if points.shape[0] == 0:
        raise ValueError("Cannot fit projection basis on empty point set")

    mean = points.mean(dim=0)
    centered = points - mean
    if points.shape[0] < 2:
        basis = torch.eye(points.shape[1], dtype=torch.float32)[:3]
        if basis.shape[0] < 3:
            basis = F.pad(basis, (0, 0, 0, 3 - basis.shape[0]))
        explained = torch.zeros(3, dtype=torch.float32)
        return mean, basis, explained

    try:
        _, svals, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh[:3].contiguous()
        denom = centered.shape[0] - 1
        variance = (svals[:3].pow(2) / max(denom, 1)).float()
    except RuntimeError:
        cov = centered.t().matmul(centered) / max(centered.shape[0] - 1, 1)
        evals, evecs = torch.linalg.eigh(cov)
        idx = torch.argsort(evals, descending=True)[:3]
        basis = evecs[:, idx].t().contiguous()
        variance = evals[idx].float().clamp(min=0.0)

    if basis.shape[0] < 3:
        pad = torch.eye(points.shape[1], dtype=torch.float32)[: 3 - basis.shape[0]]
        basis = torch.cat([basis, pad], dim=0)
        variance = F.pad(variance, (0, 3 - variance.shape[0]))

    # Fix sign ambiguity so projections do not flip between independent fits.
    for i in range(basis.shape[0]):
        max_idx = int(torch.argmax(basis[i].abs()).item())
        if basis[i, max_idx] < 0:
            basis[i] = -basis[i]

    total_var = (centered.pow(2).sum() / max(centered.shape[0] - 1, 1)).clamp(min=1e-12)
    explained = (variance / total_var).float()
    return mean, basis[:3], explained[:3]


def _project_probe_tensor(x: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    x_cpu = x.detach().float().cpu()
    flat = x_cpu.reshape(-1, x_cpu.shape[-1])
    proj = (flat - mean).matmul(basis.t())
    return proj.reshape(*x_cpu.shape[:-1], 3).contiguous()


def _masked_probe_points(*vectors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_cpu = mask.detach().bool().cpu()
    chunks: list[torch.Tensor] = []
    for vec in vectors:
        vec_cpu = vec.detach().float().cpu()
        if vec_cpu.shape[:2] != mask_cpu.shape:
            continue
        chunks.append(vec_cpu[mask_cpu])
    if not chunks:
        raise ValueError("No valid probe vectors found for PCA basis")
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def _extract_probe_attention(
    model: ChainGenerator,
    x_input: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor | None,
) -> list[dict[str, torch.Tensor]]:
    """Extract self- and cross-attention maps from all decoder layers.

    Uses forward hooks on q/k projections to reconstruct attention scores
    without modifying the model's forward pass or using the slower
    non-fused attention path.  Returns one dict per layer with keys
    ``self_attn`` [H, L, L] and ``cross_attn`` [H, L, K].
    """
    from cebcm.models.chain_head import _apply_rope

    hooks: list[torch.utils.hooks.RemovableHook] = []
    captured: dict[int, dict[str, dict[str, torch.Tensor]]] = {}

    for layer_idx, layer in enumerate(model.layers):
        sa = layer.self_attn
        ca = layer.cross_attn
        self_cap: dict[str, torch.Tensor] = {}
        cross_cap: dict[str, torch.Tensor] = {}

        def _make_hook(storage: dict[str, torch.Tensor], key: str):
            def _hook(_mod, _inp, out):
                storage[key] = out.detach()
            return _hook

        hooks.append(sa.q_proj.register_forward_hook(_make_hook(self_cap, "q")))
        hooks.append(sa.k_proj.register_forward_hook(_make_hook(self_cap, "k")))
        hooks.append(ca.q_proj.register_forward_hook(_make_hook(cross_cap, "q")))
        hooks.append(ca.k_proj.register_forward_hook(_make_hook(cross_cap, "k")))
        captured[layer_idx] = {"self": self_cap, "cross": cross_cap}

    # Run the actual forward — hooks capture projections.
    x = x_input
    for layer in model.layers:
        x = layer(x, context, context_mask=context_mask)

    # Remove hooks immediately.
    for h in hooks:
        h.remove()

    attention_maps: list[dict[str, torch.Tensor]] = []
    for layer_idx in range(len(model.layers)):
        sa = model.layers[layer_idx].self_attn
        ca = model.layers[layer_idx].cross_attn
        sc = captured[layer_idx]["self"]
        cc = captured[layer_idx]["cross"]
        layer_map: dict[str, torch.Tensor] = {}

        if "q" in sc and "k" in sc:
            n_h, h_d = sa.n_heads, sa.head_dim
            q = sc["q"].view(-1, sc["q"].shape[1], n_h, h_d).transpose(1, 2)
            k = sc["k"].view(-1, sc["k"].shape[1], n_h, h_d).transpose(1, 2)
            rope = sa._get_rope(q.shape[2], q.device)
            q = _apply_rope(q, rope)
            k = _apply_rope(k, rope)
            scores = torch.matmul(q, k.transpose(-2, -1)) * (h_d ** -0.5)
            seq_len = scores.shape[-1]
            causal = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1,
            )
            scores = scores + causal
            layer_map["self_attn"] = F.softmax(scores, dim=-1).cpu()  # [B, H, L, L]

        if "q" in cc and "k" in cc:
            n_h, h_d = ca.n_heads, ca.head_dim
            q = cc["q"].view(-1, cc["q"].shape[1], n_h, h_d).transpose(1, 2)
            k = cc["k"].view(-1, cc["k"].shape[1], n_h, h_d).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) * (h_d ** -0.5)
            if context_mask is not None:
                invalid = ~context_mask.to(dtype=torch.bool, device=scores.device)
                scores = scores.masked_fill(invalid[:, None, None, :], float("-inf"))
            layer_map["cross_attn"] = F.softmax(scores, dim=-1).cpu()  # [B, H, L, K]

        attention_maps.append(layer_map)

    return attention_maps


@torch.no_grad()
def write_training_probe_snapshot(
    *,
    model: ChainGenerator,
    probe_batch: dict,
    cfg: dict,
    device: torch.device,
    probe_dir: Path,
    epoch: int,
    batch_idx: int,
    global_step: int,
    target_steps: int,
    projection_state: dict | None,
) -> dict:
    """Write a compact fixed-probe geometry snapshot for GUI inspection."""
    was_training = model.training
    model.eval()

    try:
        max_samples = max(1, int(cfg.get("probe_num_samples", 4)))
        probe_steps_cfg = int(cfg.get("probe_steps", 0))
        steps = target_steps if probe_steps_cfg <= 0 else min(
            probe_steps_cfg,
            int(cfg.get("max_chain_steps", target_steps)),
        )

        v_q = probe_batch["v_questions"][:max_samples].to(device)
        chains = probe_batch["chains"][:max_samples].to(device)
        chain_lens = probe_batch["chain_lens"][:max_samples].to(device)
        answer_pos = probe_batch["answer_pos"][:max_samples].to(device)
        context_banks = probe_batch["context_banks"][:max_samples].to(device)
        context_mask = probe_batch["context_mask"][:max_samples].to(device)

        chains_trunc, chain_mask, target_answer_pos, target_has_answer = select_training_targets(
            chains,
            chain_lens,
            answer_pos,
            steps,
        )
        context_mask = context_mask[:, : context_banks.shape[1]]

        levels = _sample_diffusion_noise_levels(chain_mask, cfg, model, eval_mode=True)
        noise = _make_diffusion_eval_noise_like(chains_trunc, cfg, model)
        v_df_raw, v_noisy, eps = model.forward_diffusion_forcing(
            v_q,
            chains_trunc,
            levels,
            v_context_bank=context_banks,
            context_mask=context_mask,
            noise=noise,
            return_noisy=True,
        )
        # ``forward_diffusion_forcing`` returns the raw model output which under
        # ``prediction_type="v"`` (or "eps") is NOT the clean x0 — comparing it
        # directly to the clean target yields an antipodal cosine (v ≈ −σ·x0
        # at mid noise, so cos(v, x0) → −√(1−α̅_t)). Decode to pred_x0 first so
        # every downstream cosine/L2 and the exported "pred_x0" field live in
        # the same space as the clean target.
        v_df = model.predict_x0(v_noisy, v_df_raw, levels).to(dtype=v_df_raw.dtype)
        v_tf = model.forward(
            v_q,
            chains_trunc,
            v_context_bank=context_banks,
            context_mask=context_mask,
            scheduled_sampling_prob=0.0,
            tf_noise_std=0.0,
        )
        v_roll, roll_info = model.generate(
            v_q,
            num_steps=chains_trunc.shape[1],
            v_context_bank=context_banks,
            context_mask=context_mask,
            temperature=float(cfg.get("free_run_temperature", 1.0)),
            latent_noise_std=float(cfg.get("eval_free_run_noise_std", 0.0)),
            repeat_penalty=float(cfg.get("free_run_repeat_penalty", 0.0)),
            repeat_cos_threshold=float(cfg.get("free_run_repeat_cos_threshold", 0.98)),
            repeat_ban_threshold=float(cfg.get("free_run_repeat_ban_threshold", 0.995)),
            repeat_ban_max_retries=int(cfg.get("free_run_repeat_ban_retries", 3)),
            return_info=True,
        )

        # Extract attention maps on the DF (noised) input for overlay visualization.
        df_context, df_ctx_mask = model._prepare_context(v_q, context_banks, context_mask)
        df_input = model._to_residual_space(v_noisy)
        t_emb = model._diffusion_timestep_embedding(levels).to(device=df_input.device, dtype=df_input.dtype)
        df_input_conditioned = df_input + t_emb
        attention_maps = _extract_probe_attention(
            model, df_input_conditioned, df_context, df_ctx_mask,
        )

        if projection_state is None:
            points = _masked_probe_points(chains_trunc, v_noisy, v_df, v_roll, v_tf, mask=chain_mask)
            mean, basis, explained = _fit_probe_projection_basis(points)
            projection_state = {
                "mean": mean,
                "basis": basis,
                "explained": explained,
            }
        else:
            mean = projection_state["mean"].detach().float().cpu()
            basis = projection_state["basis"].detach().float().cpu()
            explained = projection_state.get("explained", torch.zeros(3)).detach().float().cpu()

        clean = chains_trunc.detach().float()
        mask = chain_mask.detach().bool()

        def cos_to_clean(x: torch.Tensor) -> torch.Tensor:
            return (_safe_normalize(x.float(), dim=-1) * _safe_normalize(clean.float(), dim=-1)).sum(dim=-1)

        def l2_to_clean(x: torch.Tensor) -> torch.Tensor:
            return (x.float() - clean.float()).norm(dim=-1)

        snapshot = {
            "schema": "chain_generator_probe_v1",
            "global_step": int(global_step),
            "epoch": int(epoch),
            "batch_idx": int(batch_idx),
            "target_steps": int(steps),
            "mode": "system1" if int(steps) == 1 else "system2",
            "projection": {
                "mean": mean,
                "basis": basis,
                "explained": explained,
            },
            "projected": {
                "clean": _project_probe_tensor(clean, mean, basis),
                "noisy": _project_probe_tensor(v_noisy, mean, basis),
                "pred_x0": _project_probe_tensor(v_df, mean, basis),
                "teacher_forced": _project_probe_tensor(v_tf, mean, basis),
                "rollout": _project_probe_tensor(v_roll, mean, basis),
            },
            "arrays": {
                "mask": mask.cpu(),
                "noise_levels": levels.detach().cpu(),
                "answer_pos": target_answer_pos.detach().cpu(),
                "has_answer": target_has_answer.detach().cpu(),
                "df_cos": torch.where(mask, cos_to_clean(v_df), torch.zeros_like(levels, dtype=torch.float32)).cpu(),
                "noisy_cos": torch.where(mask, cos_to_clean(v_noisy), torch.zeros_like(levels, dtype=torch.float32)).cpu(),
                "tf_cos": torch.where(mask, cos_to_clean(v_tf), torch.zeros_like(levels, dtype=torch.float32)).cpu(),
                "roll_cos": torch.where(mask, cos_to_clean(v_roll), torch.zeros_like(levels, dtype=torch.float32)).cpu(),
                "df_l2": torch.where(mask, l2_to_clean(v_df), torch.zeros_like(levels, dtype=torch.float32)).cpu(),
                "noisy_l2": torch.where(mask, l2_to_clean(v_noisy), torch.zeros_like(levels, dtype=torch.float32)).cpu(),
                "roll_l2": torch.where(mask, l2_to_clean(v_roll), torch.zeros_like(levels, dtype=torch.float32)).cpu(),
                "clean_norm": torch.where(mask, clean.norm(dim=-1), torch.zeros_like(levels, dtype=torch.float32)).cpu(),
                "noisy_norm": torch.where(mask, v_noisy.norm(dim=-1), torch.zeros_like(levels, dtype=torch.float32)).cpu(),
                "pred_norm": torch.where(mask, v_df.norm(dim=-1), torch.zeros_like(levels, dtype=torch.float32)).cpu(),
                "roll_norm": torch.where(mask, v_roll.norm(dim=-1), torch.zeros_like(levels, dtype=torch.float32)).cpu(),
            },
            "metrics": {
                "df_cos_mean": float(cos_to_clean(v_df)[mask].mean().item()) if mask.any() else 0.0,
                "noisy_cos_mean": float(cos_to_clean(v_noisy)[mask].mean().item()) if mask.any() else 0.0,
                "roll_cos_mean": float(cos_to_clean(v_roll)[mask].mean().item()) if mask.any() else 0.0,
                "tf_cos_mean": float(cos_to_clean(v_tf)[mask].mean().item()) if mask.any() else 0.0,
                "noise_level_mean": float(levels[mask].float().mean().item()) if mask.any() else 0.0,
                "noise_level_max": float(levels[mask].float().max().item()) if mask.any() else 0.0,
                "raw_norm_mean": float(roll_info.get("raw_norm_mean", 0.0)),
            },
        }

        # Save attention maps: compact representation per layer.
        # Each layer has self_attn [B, H, L, L] and cross_attn [B, H, L, K].
        # We save only the first probe sample to limit file size.
        attn_data: list[dict[str, torch.Tensor]] = []
        for lm in attention_maps:
            layer_out: dict[str, torch.Tensor] = {}
            if "self_attn" in lm and lm["self_attn"].shape[0] > 0:
                layer_out["self_attn"] = lm["self_attn"][0].cpu()    # [H, L, L]
            if "cross_attn" in lm and lm["cross_attn"].shape[0] > 0:
                layer_out["cross_attn"] = lm["cross_attn"][0].cpu()  # [H, L, K]
            attn_data.append(layer_out)
        snapshot["attention"] = attn_data
        snapshot["attention_meta"] = {
            "n_layers": len(attn_data),
            "n_heads": int(model.cfg.n_heads),
            "context_len": int(context_mask.shape[1]) if context_mask is not None else 1,
        }

        if bool(cfg.get("probe_save_raw_vectors", False)):
            snapshot["raw"] = {
                "clean": clean.detach().cpu(),
                "noisy": v_noisy.detach().cpu(),
                "pred_x0": v_df.detach().cpu(),
                "teacher_forced": v_tf.detach().cpu(),
                "rollout": v_roll.detach().cpu(),
                "eps": eps.detach().cpu(),
            }

        probe_dir.mkdir(parents=True, exist_ok=True)
        out_path = probe_dir / f"probe_step_{int(global_step):010d}.pt"
        torch.save(snapshot, out_path)

        return {
            "path": str(out_path),
            "global_step": int(global_step),
            "df_cos_mean": snapshot["metrics"]["df_cos_mean"],
            "roll_cos_mean": snapshot["metrics"]["roll_cos_mean"],
            "target_steps": int(steps),
            "projection_state": projection_state,
        }
    finally:
        if was_training:
            model.train()


def main() -> None:
    parser = argparse.ArgumentParser(description="ChainGenerator Training")
    parser.add_argument("--config", default="configs/chain_generator_config.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--finetune", default=None,
        help="Path to checkpoint for fine-tuning (loads model weights only, "
             "fresh optimizer/scheduler/epoch).",
    )
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

    # Enable SwiGLU activation telemetry (dead-zone / kurtosis tracking).
    # Cost is ~4 reductions per FFN call when training; off when eval'ing.
    enable_ffn_stats = bool(config.get("training", {}).get("enable_ffn_stats", True))
    if enable_ffn_stats:
        for module in model.modules():
            if isinstance(module, SwiGLUFFN):
                module.track_stats = True
        print(f"  ffn_stats: enabled (gate-input mean/std, SiLU kurtosis, out-norm)")

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
    print("Dataset diagnostics:")
    summarize_chain_dataset(
        data["train"],
        "train",
        max_chain_len=gen_cfg.max_chain_len,
        max_chain_steps=max_chain_steps,
        answer_repeat_pad=answer_repeat_pad,
        context_bank_size=context_bank_size,
    )
    summarize_chain_dataset(
        data["val"],
        "val",
        max_chain_len=gen_cfg.max_chain_len,
        max_chain_steps=max_chain_steps,
        answer_repeat_pad=answer_repeat_pad,
        context_bank_size=context_bank_size,
    )

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
    wd = float(train_cfg.get("weight_decay", 1e-4))
    param_groups = _build_wd_param_groups(model, wd)
    optimizer = torch.optim.AdamW(param_groups, lr=lr)

    # EMA (Arch-1).  Decay=0 disables.
    ema_decay = float(train_cfg.get("model_ema_decay", 0.0))
    ema: ModelEMA | None = None
    if ema_decay > 0.0:
        ema = ModelEMA(model, decay=ema_decay)
        print(f"EMA enabled with decay={ema_decay}")

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
    global_step = 0
    if args.finetune:
        ckpt = torch.load(args.finetune, map_location=device, weights_only=False)
        load_result = model.load_state_dict(ckpt["model"], strict=False)
        src_epoch = ckpt.get("epoch", "?")
        src_metric = ckpt.get("best_metric", "?")
        print(f"Fine-tune from {args.finetune} (src epoch={src_epoch}, metric={src_metric})")
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                f"  Compat load: missing={len(load_result.missing_keys)}, "
                f"unexpected={len(load_result.unexpected_keys)}"
            )
        print("  Fresh optimizer, scheduler, epoch counter.")
    elif args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metric = float(ckpt.get("best_metric", best_metric))
        global_step = int(ckpt.get("global_step", 0))
        if ema is not None and ckpt.get("ema") is not None:
            ema.load_state_dict(ckpt["ema"])
        print(f"Resumed from {args.resume}: epoch={start_epoch}, best={best_metric:.4f}")

    log_every = int(train_cfg.get("log_every", 50))
    ckpt_every = int(train_cfg.get("checkpoint_every", 5))
    patience = int(train_cfg.get("early_stop_patience", 15))
    no_improve = 0
    logs_dir = Path(out_cfg["logs_dir"])
    metrics_log_path = logs_dir / str(train_cfg.get("metrics_log_name", "chain_generator_training.jsonl"))
    probe_enabled = bool(train_cfg.get("enable_training_geometry_probe", True))
    probe_every_steps = int(train_cfg.get("probe_every_steps", 200))
    probe_dir = logs_dir / str(train_cfg.get("probe_dir_name", "geometry_probes"))
    probe_state: dict | None = None
    probe_batch = None
    if probe_enabled and probe_every_steps > 0:
        try:
            probe_batch = next(iter(val_loader))
        except StopIteration:
            probe_enabled = False
    _append_jsonl(
        metrics_log_path,
        {
            "event": "run_start",
            "global_step": int(global_step),
            "start_epoch": int(start_epoch),
            "config_path": str(args.config),
            "probe_enabled": bool(probe_enabled),
            "probe_every_steps": int(probe_every_steps),
            "probe_dir": str(probe_dir),
            "timestamp": time.time(),
        },
    )

    tracker = MetricTracker()

    # Preserve base lambdas so warm-up schedules can scale without drift.
    base_df_lambda = float(train_cfg.get("loss_lambda_diffusion", 0.0))
    df_warmup_epochs = int(train_cfg.get("df_warmup_epochs", 0))
    base_ans_lambda = float(train_cfg.get("loss_lambda_answer", 1.0))
    base_bptt_lambda = float(train_cfg.get("loss_lambda_bptt", 0.0))
    bptt_warmup_epochs = int(train_cfg.get("bptt_warmup_epochs", 0))

    # Rolling ans_coverage history for collapse-warning heuristic.
    ans_cov_history: list[float] = []
    ans_cov_warn_threshold = float(train_cfg.get("ans_cov_warn_threshold", 0.10))
    ans_cov_warn_window = int(train_cfg.get("ans_cov_warn_window", 3))

    # Initialize SADT (Step-wise Adaptive Dynamic Throttling) state.
    # It changes LR only; oracle/DAger is disabled by default and not tied to SADT.
    sadt_tf_ema = None
    sadt_roll_ema = None
    sadt_events = {"throttle": 0, "turbo": 0}

    # Hard-fail guards: do not keep training after the model enters a
    # finite-metric / zero-gradient zombie state. Recovery may attempt EMA
    # restores inside train_step, but a repeated bad-step streak means the
    # run is no longer scientifically valid.
    #
    # enable_zombie_guard=false disables the hard-fail raise entirely, so
    # training can survive NaN cascades that the NaN gates are already
    # handling. NaN gates (model._nan_gate_count) stay active regardless —
    # they prevent NaN from entering the loss and gradients.  The zombie
    # guard only controls whether a persistent zero-grad or bad-step streak
    # terminates the run.  Disable when you observe genuine learning
    # continuing through a streak (e.g. GB10_1 tf_cos 0.46→0.63 while
    # zero_grad_streak was accumulating from NaN-gated batches).
    enable_zombie_guard = bool(train_cfg.get("enable_zombie_guard", True))
    bad_step_streak = 0
    zero_grad_streak = 0
    hard_fail_bad_step_streak = int(train_cfg.get("hard_fail_bad_step_streak", 25))
    hard_fail_zero_grad_streak = int(train_cfg.get("hard_fail_zero_grad_streak", 25))
    zero_grad_threshold = float(train_cfg.get("zero_grad_threshold", 1e-8))

    print("\nTraining settings")
    print(f"  epochs={num_epochs}, batch={batch_size}, lr={lr:.2e}")
    print(
        "  objective="
        f"step*{train_cfg.get('loss_lambda_step', 1.0)} + "
        f"ans*{train_cfg.get('loss_lambda_answer', 1.0)} + "
        f"roll*{train_cfg.get('loss_lambda_roll', 1.0)} + "
        f"rank*{train_cfg.get('loss_lambda_rank', 0.1)} + "
        f"df*{train_cfg.get('loss_lambda_diffusion', 0.0)} + "
        f"aux*{train_cfg.get('loss_lambda_aux', 0.0)} + "
        f"aux_df*{train_cfg.get('loss_lambda_aux_df', 0.0)}"
    )
    horizon_schedule = train_cfg.get("horizon_schedule")
    if horizon_schedule:
        sched_desc = " → ".join(f"{s}×{d}ep" for s, d in horizon_schedule)
        print(f"  horizon: system1={train_cfg.get('system1_epochs', 10)}ep, "
              f"schedule=[{sched_desc}]")
    else:
        print(
            f"  horizon: system1_epochs={train_cfg.get('system1_epochs', 10)}, "
            f"linear_ramp, max_steps={max_chain_steps}"
        )
    print(
        f"  scheduled_sampling: max={train_cfg.get('scheduled_sampling_max', 0.5)}, "
        f"ramp_epochs={train_cfg.get('scheduled_sampling_ramp_epochs', 10)}"
    )
    print(
        f"  oracle_dagger: enabled={bool(train_cfg.get('enable_oracle_dagger', False))}, "
        f"max_retries={train_cfg.get('oracle_max_retries', 0)}, "
        f"prob_max={train_cfg.get('oracle_prob_max', 0.0)}"
    )
    print(
        f"  diffusion_forcing: enabled={bool(train_cfg.get('enable_diffusion_forcing', False))}, "
        f"K={gen_cfg.diffusion_timesteps}, schedule={gen_cfg.diffusion_beta_schedule}, "
        f"levels=[{train_cfg.get('df_noise_level_min', 0)}, "
        f"{train_cfg.get('df_noise_level_max', gen_cfg.diffusion_timesteps - 1)}], "
        f"eval_level={train_cfg.get('df_eval_noise_level', (gen_cfg.diffusion_timesteps - 1) // 2)}"
    )
    print(
        f"  training_geometry: enabled={probe_enabled}, every={probe_every_steps}, "
        f"samples={train_cfg.get('probe_num_samples', 4)}, dir={probe_dir}"
    )
    print("=" * 70)

    prev_target_steps = 1  # Track for SADT cooldown
    # Horizon transition LR warmup state.
    horizon_warmup_remaining = 0
    horizon_warmup_total = 0
    horizon_warmup_factor = 0.1
    # Answer loss warmup state (ramps lambda_ans 0→base at horizon transitions).
    answer_warmup_remaining = 0
    answer_warmup_total = 0
    # NaN cascade detector state.
    nan_gate_window_size = int(train_cfg.get("nan_gate_window_size", 50))
    nan_gate_window: list[int] = []
    nan_gate_lr_halved = False

    for epoch in range(start_epoch, num_epochs):
        model.train()
        target_steps = get_chain_steps(epoch, train_cfg)
        ss_prob = _get_scheduled_sampling_prob(epoch, train_cfg)
        noise_std = _get_scheduled_noise_std(epoch, train_cfg)
        oracle_prob = _get_oracle_prob(epoch, train_cfg)
        tf_noise = _get_tf_noise_std(epoch, train_cfg)

        # Diffusion Forcing lambda warm-up: ramp from 0 to base over the first
        # df_warmup_epochs. Prevents the high-variance DF gradient from
        # dominating early training while the backbone is still warming up
        # and the geometry has not stabilised.
        if df_warmup_epochs > 0 and epoch < df_warmup_epochs:
            effective_df_lambda = base_df_lambda * (float(epoch + 1) / float(df_warmup_epochs))
        else:
            effective_df_lambda = base_df_lambda
        train_cfg["loss_lambda_diffusion"] = effective_df_lambda

        # BPTT warm-up: disable until bptt_warmup_epochs have elapsed.
        if bptt_warmup_epochs > 0 and epoch < bptt_warmup_epochs:
            train_cfg["loss_lambda_bptt"] = 0.0
        else:
            train_cfg["loss_lambda_bptt"] = base_bptt_lambda

        epoch_start = time.time()

        # Reset SADT EMA at System2 transition to prevent false throttling.
        # When target_steps increases, metrics naturally drop — this is NOT
        # degradation, it's a harder task. SADT must not punish it.
        if target_steps != prev_target_steps:
            sadt_tf_ema = None
            sadt_roll_ema = None
            sadt_cooldown = int(train_cfg.get("sadt_cooldown_steps", 200))
            # ── Horizon transition LR warmup ──────────────────────
            # The System1→System2 jump causes a ~50x gradient norm
            # spike (e.g. 0.39→19.67) which pushes parameters into
            # bfloat16-fragile zones and triggers NaN cascades.
            # Temporarily reduce LR and linearly ramp back up.
            hw_steps = int(train_cfg.get("horizon_warmup_steps", 0))
            hw_factor = float(train_cfg.get("horizon_warmup_factor", 0.1))
            # ── Answer loss warmup ─────────────────────────────
            # When target_steps increases, the answer embedding may
            # enter the training window for the first time.  l_ans
            # jumping from 0→full creates a gradient shock that
            # destabilises FFN output norms (lessons 2026-04-20 night).
            # Ramp lambda_ans from 0→base over answer_warmup_steps.
            aw_steps = int(train_cfg.get("answer_warmup_steps", hw_steps))
            if aw_steps > 0 and target_steps > prev_target_steps:
                answer_warmup_remaining = aw_steps
                answer_warmup_total = aw_steps
                train_cfg["loss_lambda_answer"] = 0.0
            else:
                train_cfg["loss_lambda_answer"] = base_ans_lambda

            if hw_steps > 0 and prev_target_steps > 0:
                horizon_warmup_remaining = hw_steps
                horizon_warmup_total = hw_steps
                horizon_warmup_factor = hw_factor
                print(f"  [SADT] Horizon changed {prev_target_steps}→{target_steps}, "
                      f"EMA reset, cooldown={sadt_cooldown} steps, "
                      f"LR warmup={hw_steps} steps (factor={hw_factor}), "
                      f"answer warmup={aw_steps} steps")
            else:
                print(f"  [SADT] Horizon changed {prev_target_steps}→{target_steps}, "
                      f"EMA reset, cooldown={sadt_cooldown} steps")

            # ── Soft Adam momentum reset (lessons.md 2026-04-20 evening) ──
            # Adam's exp_avg_sq accumulates the gradient-magnitude statistics
            # of System1.  When System2 starts, the gradient scale jumps
            # ~50x but Adam's denominator (sqrt(exp_avg_sq) + eps) is still
            # tiny, so a single new gradient drives a huge update — into
            # bfloat16-fragile zones — which the existing LR warmup only
            # partially compensates.  We *scale down* (don't zero) the
            # second moment so Adam becomes more sensitive to the new
            # gradient magnitude without inducing the lr/sqrt(eps) ≈ 1e4
            # update spike of a hard reset.  exp_avg (first moment, the
            # "momentum direction") is left untouched — direction is still
            # informative across the transition.
            sq_scale = float(train_cfg.get("horizon_adam_sq_scale", 0.0))
            if sq_scale > 0.0 and sq_scale < 1.0 and prev_target_steps > 0:
                with torch.no_grad():
                    n_scaled = 0
                    for group in optimizer.param_groups:
                        for p in group["params"]:
                            state = optimizer.state.get(p)
                            if state and "exp_avg_sq" in state:
                                state["exp_avg_sq"].mul_(sq_scale)
                                n_scaled += 1
                                if "max_exp_avg_sq" in state:
                                    state["max_exp_avg_sq"].mul_(sq_scale)
                    print(f"  [SADT] Soft Adam reset: exp_avg_sq *= {sq_scale} "
                          f"on {n_scaled} param tensors (exp_avg unchanged)")
        else:
            sadt_cooldown = 0
        prev_target_steps = target_steps

        phase = "System1" if target_steps == 1 else f"System2({target_steps})"
        tf_noise_info = f" tf_noise={tf_noise:.4f}" if tf_noise > 0 else ""
        bptt_info = (
            f" bptt=ON(K={int(train_cfg.get('bptt_steps', 2))}, "
            f"λ={float(train_cfg.get('loss_lambda_bptt', 0.0)):.3f})"
            if float(train_cfg.get("loss_lambda_bptt", 0.0)) > 0.0
            else ""
        )
        print(f"\n[E{epoch}] target_steps={target_steps} [{phase}] "
              f"ss_prob={ss_prob:.3f} noise_std={noise_std:.4f} oracle_prob={oracle_prob:.3f}"
              f"{tf_noise_info}{bptt_info}")

        nan_count = 0
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
                scheduled_sampling_prob=ss_prob,
                free_run_noise_std=noise_std,
                oracle_prob=oracle_prob,
                tf_noise_std=tf_noise,
                ema=ema,
            )
            if metrics.get("optimizer_stepped", 0.0) > 0.0:
                scheduler.step()
            # ── Horizon transition LR warmup ──────────────────────
            # After scheduler sets the base LR, apply a warmup
            # multiplier that linearly ramps from horizon_warmup_factor
            # to 1.0 over horizon_warmup_total steps.  This prevents
            # the ~50x gradient norm spike at System1→System2 transition
            # from pushing parameters into bfloat16-fragile zones.
            if horizon_warmup_remaining > 0:
                progress = 1.0 - (horizon_warmup_remaining / horizon_warmup_total)
                factor = horizon_warmup_factor + (1.0 - horizon_warmup_factor) * progress
                for pg in optimizer.param_groups:
                    pg["lr"] = pg["lr"] * factor
                horizon_warmup_remaining -= 1
            # ── Answer loss warmup (parallel to LR warmup) ────────
            if answer_warmup_remaining > 0:
                aw_progress = 1.0 - (answer_warmup_remaining / answer_warmup_total)
                train_cfg["loss_lambda_answer"] = base_ans_lambda * aw_progress
                answer_warmup_remaining -= 1
                if answer_warmup_remaining == 0:
                    train_cfg["loss_lambda_answer"] = base_ans_lambda
            global_step += 1
            tracker.update(metrics)

            if metrics.get("nan_skipped", 0.0) > 0:
                nan_count += 1

            # ── NaN cascade detector ──────────────────────────────
            # Track NaN gate activations (samples with NaN output
            # replaced by GT in the model's forward).  When the rate
            # accelerates, halve LR to prevent parameters from
            # drifting further into bfloat16-fragile zones.
            nan_gate_now = int(metrics.get("nan_gate_count", 0))
            if nan_gate_now > 0:
                nan_gate_window.append(nan_gate_now)
            else:
                nan_gate_window.append(0)
            if len(nan_gate_window) > nan_gate_window_size:
                nan_gate_window.pop(0)
            nan_gate_window_total = sum(nan_gate_window)
            nan_gate_max_rate = int(train_cfg.get("nan_gate_max_rate", 0))
            if (
                nan_gate_max_rate > 0
                and len(nan_gate_window) >= nan_gate_window_size
                and nan_gate_window_total > nan_gate_max_rate
                and not nan_gate_lr_halved
            ):
                old_lr = optimizer.param_groups[0]["lr"]
                new_lr = old_lr * 0.5
                for pg in optimizer.param_groups:
                    pg["lr"] = new_lr
                nan_gate_lr_halved = True
                print(f"  [NaN-CASCADE] {nan_gate_window_total} NaN gates "
                      f"in {nan_gate_window_size} steps → LR {old_lr:.2e}→{new_lr:.2e}")
            elif nan_gate_window_total == 0 and nan_gate_lr_halved:
                nan_gate_lr_halved = False  # reset when window is clean

            bad_step = (
                metrics.get("nan_skipped", 0.0) > 0.0
                or metrics.get("optimizer_stepped", 0.0) <= 0.0
                or metrics.get("zombie_reset", 0.0) > 0.0
            )
            if bad_step:
                bad_step_streak += 1
            else:
                bad_step_streak = 0

            grad_now = float(metrics.get("grad_norm", 0.0))
            zero_grad_step = (
                metrics.get("optimizer_stepped", 0.0) > 0.0
                and abs(grad_now) <= zero_grad_threshold
                and float(metrics.get("loss", 0.0)) > 0.0
            )
            if zero_grad_step:
                zero_grad_streak += 1
            else:
                zero_grad_streak = 0

            metrics["bad_step_streak"] = float(bad_step_streak)
            metrics["zero_grad_streak"] = float(zero_grad_streak)

            zombie_triggered = (
                (hard_fail_bad_step_streak > 0 and bad_step_streak >= hard_fail_bad_step_streak)
                or (
                    hard_fail_zero_grad_streak > 0
                    and zero_grad_streak >= hard_fail_zero_grad_streak
                )
            )
            if zombie_triggered:
                reason = (
                    f"bad_step_streak={bad_step_streak}, "
                    f"zero_grad_streak={zero_grad_streak}, "
                    f"grad_norm={grad_now:.3e}, nan_count_epoch={nan_count}"
                )
                _append_jsonl(
                    metrics_log_path,
                    {
                        "event": "hard_fail_zombie",
                        "epoch": int(epoch),
                        "batch_idx": int(step + 1),
                        "global_step": int(global_step),
                        "target_steps": int(target_steps),
                        "phase": phase,
                        "reason": reason,
                        "metrics": metrics,
                        "timestamp": time.time(),
                    },
                )
                if enable_zombie_guard:
                    raise RuntimeError(f"Hard-fail zombie guard triggered: {reason}")
                else:
                    print(f"  [ZOMBIE-WARN] Zombie guard suppressed (enable_zombie_guard=false): {reason}")
                    # Reset streaks so the warning doesn't repeat every step.
                    bad_step_streak = 0
                    zero_grad_streak = 0

            # --- SADT Dynamic Throttle Logic ---
            if train_cfg.get("dynamic_step_lr", False) and sadt_cooldown <= 0:
                alpha = float(train_cfg.get("sadt_ema_alpha", 0.1))
                tf_cur = metrics.get("tf_cos_mean", 0.0)
                roll_cur = metrics.get("roll_cos_mean", 0.0)

                # Skip NaN-skipped steps in SADT
                if metrics.get("nan_skipped", 0.0) > 0:
                    pass  # Don't update EMA or adjust LR on NaN steps
                elif sadt_tf_ema is None:
                    sadt_tf_ema = tf_cur
                    sadt_roll_ema = roll_cur
                else:
                    sadt_tf_ema = (1 - alpha) * sadt_tf_ema + alpha * tf_cur
                    sadt_roll_ema = (1 - alpha) * sadt_roll_ema + alpha * roll_cur

                    # Compare current to EMA to detect degradation
                    tolerance = float(train_cfg.get("sadt_tolerance", 0.05))
                    lr_now = optimizer.param_groups[0]["lr"]
                    min_lr = float(train_cfg.get("sadt_min_lr", 1e-6))
                    max_lr = float(train_cfg.get("sadt_max_lr", 5e-4))

                    # Degradation (Throttle) — use AND instead of OR to reduce
                    # false positives from single-metric noise
                    is_bad = (tf_cur < (sadt_tf_ema * (1 - tolerance))
                              and roll_cur < (sadt_roll_ema * (1 - tolerance)))
                    if is_bad:
                        new_lr = max(min_lr, lr_now * float(train_cfg.get("sadt_throttle_factor", 0.5)))
                        if new_lr < lr_now:
                            optimizer.param_groups[0]["lr"] = new_lr
                            sadt_events["throttle"] += 1
                    # Improvement (Turbo) - more conservative
                    elif (tf_cur > (sadt_tf_ema * (1 + tolerance/2))
                          and roll_cur > (sadt_roll_ema * (1 + tolerance/2))):
                        new_lr = min(max_lr, lr_now * float(train_cfg.get("sadt_turbo_factor", 1.02)))
                        if new_lr > lr_now:
                            optimizer.param_groups[0]["lr"] = new_lr
                            sadt_events["turbo"] += 1
            else:
                sadt_cooldown -= 1

            if (
                probe_enabled
                and probe_batch is not None
                and probe_every_steps > 0
                and global_step % probe_every_steps == 0
            ):
                try:
                    probe_info = write_training_probe_snapshot(
                        model=model,
                        probe_batch=probe_batch,
                        cfg=train_cfg,
                        device=device,
                        probe_dir=probe_dir,
                        epoch=epoch,
                        batch_idx=step + 1,
                        global_step=global_step,
                        target_steps=target_steps,
                        projection_state=probe_state,
                    )
                    probe_state = probe_info.pop("projection_state")
                    ffn_stats = _collect_swiglu_stats(model)
                    if ffn_stats:
                        probe_info["ffn_health"] = ffn_stats
                    _append_jsonl(
                        metrics_log_path,
                        {
                            "event": "probe_snapshot",
                            "epoch": int(epoch),
                            "batch_idx": int(step + 1),
                            "global_step": int(global_step),
                            "target_steps": int(target_steps),
                            "phase": phase,
                            "probe": probe_info,
                            "timestamp": time.time(),
                        },
                    )
                    ffn_log = ""
                    if ffn_stats:
                        ffn_log = (
                            f" ffn_gate_std={ffn_stats['ffn_gate_in_std']:.3f}"
                            f" silu_kurt={ffn_stats['ffn_silu_kurtosis']:+.2f}"
                            f" ffn_out={ffn_stats['ffn_out_norm_mean']:.2f}"
                        )
                    print(
                        f"  [PROBE] step={global_step} "
                        f"df_cos={probe_info.get('df_cos_mean', 0.0):.4f} "
                        f"roll_cos={probe_info.get('roll_cos_mean', 0.0):.4f}"
                        f"{ffn_log} "
                        f"path={probe_info.get('path')}"
                    )
                except Exception as exc:
                    _append_jsonl(
                        metrics_log_path,
                        {
                            "event": "probe_error",
                            "epoch": int(epoch),
                            "batch_idx": int(step + 1),
                            "global_step": int(global_step),
                            "error": str(exc),
                            "timestamp": time.time(),
                        },
                    )
                    print(f"  [PROBE ERROR] step={global_step}: {exc}")

            if (step + 1) % log_every == 0:
                avg = tracker.get()
                lr_now = optimizer.param_groups[0]["lr"]
                sadt_info = f" [SADT T:{sadt_events['throttle']} U:{sadt_events['turbo']}]" if train_cfg.get("dynamic_step_lr") else ""
                nan_gate_total = sum(nan_gate_window)
                nan_info = f" [NaN:{nan_count}]" if nan_count > 0 else ""
                nan_info += f" [gate:{nan_gate_total}]" if nan_gate_total > 0 else ""
                bptt_active = avg.get("bptt_enabled", 0.0) > 0.0
                bptt_info = (
                    f" bptt={avg.get('loss_bptt', 0.0):.4f}"
                    f" bptt_cos={avg.get('bptt_cos_mean', 0.0):.4f}"
                    f" bptt_norm={avg.get('bptt_norm_mean', 0.0):.3f}"
                    f" norm_pen={avg.get('loss_norm_penalty', 0.0):.4f}"
                    if bptt_active
                    else ""
                )
                print(
                    f"  [E{epoch} S{step+1}] "
                    f"loss={avg.get('loss', 0.0):.4f} "
                    f"step={avg.get('loss_step', 0.0):.4f} "
                    f"ans={avg.get('loss_ans', 0.0):.4f} "
                    f"roll={avg.get('loss_roll', 0.0):.4f} "
                    f"rank={avg.get('loss_rank', 0.0):.4f} "
                    f"df={avg.get('loss_df', 0.0):.4f} "
                    f"aux={avg.get('loss_aux', 0.0):.4f} "
                    f"df_lam={effective_df_lambda:.3f} "
                    f"tf_cos={avg.get('tf_cos_mean', 0.0):.4f} "
                    f"roll_cos={avg.get('roll_cos_mean', 0.0):.4f} "
                    f"roll_ans={avg.get('roll_cos_answer', 0.0):.4f} "
                    f"aux_cos={avg.get('aux_cos_mean', 0.0):.4f} "
                    f"df_cos={avg.get('df_cos', 0.0):.4f} "
                    f"df_t={avg.get('df_noise_level_mean', 0.0):.1f} "
                    f"rank_acc={avg.get('rank_acc', 0.0):.3f} "
                    f"ans_cov={avg.get('answer_coverage', 0.0):.2f} "
                    f"raw_norm={avg.get('raw_norm_mean', 0.0):.2f} "
                    f"grad={avg.get('grad_norm', 0.0):.4f} "
                    f"lr={lr_now:.2e}{bptt_info}{sadt_info}{nan_info}"
                )
                _append_jsonl(
                    metrics_log_path,
                    {
                        "event": "train_step",
                        "epoch": int(epoch),
                        "batch_idx": int(step + 1),
                        "global_step": int(global_step),
                        "target_steps": int(target_steps),
                        "phase": phase,
                        "lr": float(lr_now),
                        "scheduled_sampling_prob": float(ss_prob),
                        "free_run_noise_std": float(noise_std),
                        "tf_noise_std": float(tf_noise),
                        "oracle_prob": float(oracle_prob),
                        "nan_count_epoch": int(nan_count),
                        "sadt": dict(sadt_events),
                        "train_metrics": avg,
                        "timestamp": time.time(),
                    },
                )
                tracker.reset()
                sadt_events = {"throttle": 0, "turbo": 0}

        model.eval()
        # Swap in EMA weights for evaluation (Arch-1).
        ema_backup: dict[str, torch.Tensor] | None = None
        if ema is not None:
            ema_backup = ema.apply_to(model)
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
        if ema is not None and ema_backup is not None:
            ema.restore(model, ema_backup)

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
            f"roll_ans={val.get('val_roll_cos_answer', 0.0):.4f} "
            f"aux_cos={val.get('val_aux_cos', 0.0):.4f} "
            f"df_cos={val.get('val_df_cos', 0.0):.4f} "
            f"df_t={val.get('val_df_noise_level', 0.0):.1f} "
            f"rank_acc={val.get('val_rank_acc', 0.0):.3f} "
            f"ans_cov={val.get('val_answer_coverage', 0.0):.2f} "
            f"norm_roll={val.get('val_norm_roll', 0.0):.4f} "
            f"({epoch_time:.1f}s)"
        )
        _append_jsonl(
            metrics_log_path,
            {
                "event": "val_epoch",
                "epoch": int(epoch),
                "global_step": int(global_step),
                "target_steps": int(target_steps),
                "phase": phase,
                "epoch_time_sec": float(epoch_time),
                "val_metric": float(val_metric),
                "val_metrics": val,
                "timestamp": time.time(),
            },
        )

        # Rolling ans_coverage collapse warning: often the first visible
        # signal of the System1→System2 transition going wrong.
        cur_ans_cov = float(val.get("val_answer_coverage", 0.0))
        ans_cov_history.append(cur_ans_cov)
        if len(ans_cov_history) > ans_cov_warn_window:
            ans_cov_history = ans_cov_history[-ans_cov_warn_window:]
        if (
            len(ans_cov_history) >= ans_cov_warn_window
            and (sum(ans_cov_history) / len(ans_cov_history)) < ans_cov_warn_threshold
        ):
            rolling = sum(ans_cov_history) / len(ans_cov_history)
            print(
                f"  [WARN] val_answer_coverage rolling-mean={rolling:.3f} "
                f"< {ans_cov_warn_threshold:.2f} over last {ans_cov_warn_window} epochs "
                f"(target_steps={target_steps}). Possible System1→System2 collapse."
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
                    "global_step": global_step,
                    "ema": ema.state_dict() if ema is not None else None,
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
                    "global_step": global_step,
                    "ema": ema.state_dict() if ema is not None else None,
                },
            )

        if no_improve >= patience:
            print(f"\nEarly stopping: no improvement for {patience} epochs")
            break

    print("\nTraining complete")
    print(f"Best val_roll_cos_last={best_metric:.4f}")
    print(f"Checkpoints: {out_cfg['checkpoint_dir']}")
    _append_jsonl(
        metrics_log_path,
        {
            "event": "run_complete",
            "global_step": int(global_step),
            "best_metric": float(best_metric),
            "timestamp": time.time(),
        },
    )


if __name__ == "__main__":
    main()
