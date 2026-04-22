#!/usr/bin/env python3
"""
Forward-Forward training pipeline for ChainGenerator.

Layer-local training (Hinton 2022) with SymBa loss (Lee & Song 2023).
No global backprop through hidden layers — each decoder block is trained
by its own local goodness objective.  The output head (final_norm +
output_proj) is trained with standard BP on cosine/MSE (FFCL hybrid).

Key differences from the standard BP pipeline (train_chain_generator.py):
  - No compute_composite_objective — replaced by ff_compute_losses
  - No diffusion forcing, scheduled sampling, oracle/DAgger
  - Per-layer + head optimizer groups with separate backward passes
  - Negative chain curriculum: gaussian → shuffle → batch_swap
  - Activity normalization between blocks strips magnitude cue
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
)
from cebcm.training.forward_forward import (
    ff_forward,
    ff_compute_losses,
    generate_negatives,
    curriculum_strategy,
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


# ═══════════════════════════════════════════════════════════════════════
# Dataset (identical to BP pipeline)
# ═══════════════════════════════════════════════════════════════════════

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

        if v_steps.shape[0] > 0:
            chain = torch.cat([v_steps, v_a.unsqueeze(0)], dim=0)
            answer_pos = int(v_steps.shape[0])
        else:
            chain = v_a.unsqueeze(0)
            answer_pos = 0

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


# ═══════════════════════════════════════════════════════════════════════
# Small helpers (identical to BP pipeline)
# ═══════════════════════════════════════════════════════════════════════

def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * q))))
    return float(vals[idx])


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _collect_swiglu_stats(model: torch.nn.Module) -> dict:
    """Aggregate per-layer SwiGLU activation health stats."""
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


def _safe_normalize(v: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    v_float = v.float()
    v_float = torch.nan_to_num(v_float, nan=0.0, posinf=0.0, neginf=0.0)
    norms = v_float.norm(dim=dim, keepdim=True).clamp(min=eps)
    return v_float / norms


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
        f"norm q/a/step={stats['q_norm_mean']:.4f}/{stats['a_norm_mean']:.4f}/{stats['step_norm_mean']:.4f}; "
        f"context_len_mean={stats['context_len_mean']:.1f}"
    )
    return stats


# ═══════════════════════════════════════════════════════════════════════
# Horizon curriculum + target selection (identical to BP pipeline)
# ═══════════════════════════════════════════════════════════════════════

def get_chain_steps(epoch: int, cfg: dict) -> int:
    """Curriculum for chain horizon growth.

    Supports two modes controlled by the ``horizon_schedule`` config key:

    1. **Fixed-horizon schedule** (``horizon_schedule`` is set):
       A list of ``[steps, duration_epochs]`` pairs.

    2. **Legacy linear ramp** (``horizon_schedule`` is absent):
       Each epoch after System-1 adds +1 step, starting from
       ``system2_start_steps``.
    """
    s1_epochs = int(cfg.get("system1_epochs", 10))
    max_steps = int(cfg.get("max_chain_steps", 20))

    if epoch < s1_epochs:
        return 1

    schedule = cfg.get("horizon_schedule")
    if schedule:
        elapsed = epoch - s1_epochs
        for entry in schedule:
            steps_val, duration = int(entry[0]), int(entry[1])
            if elapsed < duration:
                return min(max_steps, steps_val)
            elapsed -= duration
        return min(max_steps, int(schedule[-1][0]))

    start_steps = int(cfg.get("system2_start_steps", 2))
    return min(max_steps, start_steps + (epoch - s1_epochs))


def select_training_targets(
    chains: torch.Tensor,
    chain_lens: torch.Tensor,
    answer_pos: torch.Tensor,
    target_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select training targets aligned to the generation process."""
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
            targets[i, 0] = chains[i, ai]
            mask[i, 0] = True
            target_answer_pos[i] = 0
            target_has_answer[i] = True
        else:
            targets[i, :ti] = chains[i, :ti]
            mask[i, :ti] = True
            if ai < ti:
                target_answer_pos[i] = ai
                target_has_answer[i] = True
            else:
                target_answer_pos[i] = max(ti - 1, 0)

    return targets, mask, target_answer_pos, target_has_answer


# ═══════════════════════════════════════════════════════════════════════
# ModelEMA (identical to BP pipeline)
# ═══════════════════════════════════════════════════════════════════════

class ModelEMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()
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


# ═══════════════════════════════════════════════════════════════════════
# Weight-decay param groups (identical to BP pipeline)
# ═══════════════════════════════════════════════════════════════════════

def _build_wd_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict]:
    """Split parameters into decayed / non-decayed groups."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_special_token = name.endswith("start_token") or name.endswith("null_context_token")
        is_layerscale = ".ls_" in name
        is_bias_or_norm = p.ndim <= 1 and not is_layerscale
        if is_bias_or_norm or is_special_token:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ═══════════════════════════════════════════════════════════════════════
# JSON logging (identical to BP pipeline)
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# FF train step — the core difference from the BP pipeline
# ═══════════════════════════════════════════════════════════════════════

def ff_train_step(
    model: ChainGenerator,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    cfg: dict,
    target_steps: int,
    neg_strategy: str = "gaussian",
    neg_sigma: float = 0.3,
    ema: ModelEMA | None = None,
) -> dict[str, float]:
    """One FF training step: per-layer SymBa + head cosine/MSE.

    Unlike BP train_step which does a single backward on the composite
    loss, FF requires separate backward passes:
      1. Each decoder layer's SymBa loss → backward → only that layer's
         params get gradient (guaranteed by .detach() in ff_forward).
      2. Head loss (cosine + MSE on output_proj) → backward → only
         final_norm + output_proj get gradient.

    All backwards accumulate into the same .grad tensors, then a single
    optimizer.step() applies them all.
    """
    v_q = batch["v_questions"].to(device)
    chains = batch["chains"].to(device)
    chain_lens = batch["chain_lens"].to(device)
    answer_pos_raw = batch["answer_pos"].to(device)
    context_banks = batch["context_banks"].to(device)
    context_mask = batch["context_mask"].to(device)

    chains_trunc, chain_mask, target_answer_pos, target_has_answer = select_training_targets(
        chains, chain_lens, answer_pos_raw, target_steps
    )

    optimizer.zero_grad(set_to_none=True)

    # Generate negatives from truncated positive chains.
    v_chain_neg = generate_negatives(
        chains_trunc, strategy=neg_strategy, mask=chain_mask, sigma=neg_sigma,
    )

    symba_alpha = float(cfg.get("ff_symba_alpha", 2.0))
    head_cos_w = float(cfg.get("ff_head_cosine_weight", 1.0))
    head_mse_w = float(cfg.get("ff_head_mse_weight", 0.1))

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        loss_dict = ff_compute_losses(
            model,
            v_q,
            chains_trunc,
            v_chain_neg,
            v_context_bank=context_banks,
            context_mask=context_mask,
            chain_mask=chain_mask,
            symba_alpha=symba_alpha,
            head_cosine_weight=head_cos_w,
            head_mse_weight=head_mse_w,
        )

    # ── Separate backwards for layer-local + head ──
    # Each ff_layer_i loss only has gradients for layer i's params
    # (due to .detach() at block boundaries in ff_forward).
    # head_total only has gradients for final_norm + output_proj.
    # We accumulate all grads, then do one optimizer step.
    n_layers = len(model.layers)
    grad_had_nan = False
    sanitized_count = 0

    for i in range(n_layers):
        layer_loss = loss_dict[f"ff_layer_{i}"]
        layer_loss = torch.nan_to_num(layer_loss, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isfinite(layer_loss) and float(layer_loss.detach()) > 0.0:
            scaler.scale(layer_loss).backward(retain_graph=True)

    head_loss = loss_dict["head_total"]
    head_loss = torch.nan_to_num(head_loss, nan=0.0, posinf=0.0, neginf=0.0)
    if torch.isfinite(head_loss) and float(head_loss.detach()) > 0.0:
        scaler.scale(head_loss).backward()

    scaler.unscale_(optimizer)

    # ── Gradient sanitation (preserve-momentum, same as BP pipeline) ──
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad
        bad = ~torch.isfinite(g)
        if bad.any():
            grad_had_nan = True
            sanitized_count += 1
            g.masked_fill_(bad, 0.0)
            state = optimizer.state.get(p)
            if state:
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    buf = state.get(key)
                    if buf is not None and not torch.isfinite(buf).all():
                        buf.zero_()

    # ── Gradient clipping ──
    clip_grad = float(cfg.get("clip_grad_norm", 1.0))
    if clip_grad > 0:
        total_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        if not torch.isfinite(total_norm):
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            return _ff_skip_metrics(loss_dict, target_steps, "grad_clip_nan")
    else:
        total_norm = torch.tensor(0.0, device=device)

    scaler.step(optimizer)
    scaler.update()

    # ── Post-step param sanity (EMA restore if needed) ──
    params_restored = 0
    if ema is not None:
        max_param_abs = float(cfg.get("param_abs_max", 1.0e4))
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
                else:
                    torch.nan_to_num(p.data, nan=0.0, posinf=0.0, neginf=0.0, out=p.data)
                params_restored += 1
                state = optimizer.state.get(p)
                if state:
                    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        buf = state.get(key)
                        if buf is not None and not torch.isfinite(buf).all():
                            buf.zero_()

    # EMA update only on fully clean steps.
    if ema is not None and not grad_had_nan and params_restored == 0:
        ema.update(model)

    # ── Metrics ──
    metrics: dict[str, float] = {
        "loss_total": float(loss_dict["loss_total"].detach().item()),
        "ff_total": float(loss_dict["ff_total"].detach().item()),
        "head_cosine": float(loss_dict["head_cosine"].detach().item()),
        "head_mse": float(loss_dict["head_mse"].detach().item()),
        "head_total": float(loss_dict["head_total"].detach().item()),
        "grad_norm": float(total_norm.item()) if clip_grad > 0 else 0.0,
        "grad_sanitized": float(sanitized_count),
        "param_restored": float(params_restored),
        "nan_skipped": 1.0 if (grad_had_nan or params_restored > 0) else 0.0,
        "optimizer_stepped": 1.0,
        "target_steps": float(target_steps),
        "neg_strategy": neg_strategy,
    }
    for i in range(n_layers):
        metrics[f"ff_layer_{i}"] = float(loss_dict[f"ff_layer_{i}"].detach().item())
        metrics[f"goodness_pos_{i}"] = float(loss_dict["goodness_pos"][i])
        metrics[f"goodness_neg_{i}"] = float(loss_dict["goodness_neg"][i])

    return metrics


def _ff_skip_metrics(
    loss_dict: dict, target_steps: int, reason: str,
) -> dict[str, float]:
    """Return metrics dict for a skipped FF step."""
    return {
        "loss_total": float(loss_dict.get("loss_total", torch.tensor(0.0)).detach().item()),
        "ff_total": 0.0,
        "head_cosine": 0.0,
        "head_mse": 0.0,
        "head_total": 0.0,
        "grad_norm": 0.0,
        "grad_sanitized": 0.0,
        "param_restored": 0.0,
        "nan_skipped": 1.0,
        "optimizer_stepped": 0.0,
        "target_steps": float(target_steps),
        "skip_reason": reason,
    }


# ═══════════════════════════════════════════════════════════════════════
# Eval step — uses standard model.generate() (not FF)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def ff_eval_step(
    model: ChainGenerator,
    batch: dict,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    cfg: dict,
    gen_steps: int,
) -> dict[str, float]:
    """Evaluate via standard autoregressive rollout (not FF).

    FF is a training method only — at inference we use the normal
    forward pass and generate(), so eval metrics are directly
    comparable to the BP pipeline.
    """
    v_q = batch["v_questions"].to(device)
    chains = batch["chains"].to(device)
    chain_lens = batch["chain_lens"].to(device)
    answer_pos = batch["answer_pos"].to(device)
    context_banks = batch["context_banks"].to(device)
    context_mask = batch["context_mask"].to(device)

    chains_trunc, chain_mask, target_answer_pos, target_has_answer = select_training_targets(
        chains, chain_lens, answer_pos, gen_steps
    )

    bsz, steps, d_model = chains_trunc.shape
    target_has_answer = target_has_answer.to(device=chains_trunc.device, dtype=torch.bool)
    target_answer_pos = target_answer_pos.to(device=chains_trunc.device).long()
    valid_lens = chain_mask.sum(dim=1).long().clamp(min=1)

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        # Teacher-forced prediction.
        v_tf = model.forward(
            v_q,
            chains_trunc,
            v_context_bank=context_banks,
            context_mask=context_mask,
            scheduled_sampling_prob=0.0,
            tf_noise_std=0.0,
        )

        # Free-run rollout.
        v_roll, roll_info = model.generate(
            v_q,
            num_steps=steps,
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

    # ── Compute eval cosine metrics ──
    tf_cos = (_safe_normalize(v_tf.float(), dim=-1) * _safe_normalize(chains_trunc.float(), dim=-1)).sum(dim=-1)
    roll_cos = (_safe_normalize(v_roll.float(), dim=-1) * _safe_normalize(chains_trunc.float(), dim=-1)).sum(dim=-1)
    maskf = chain_mask.to(dtype=torch.float32)
    valid = maskf.sum().clamp(min=1.0)

    tf_cos_masked = torch.where(chain_mask, tf_cos, torch.zeros_like(tf_cos))
    roll_cos_masked = torch.where(chain_mask, roll_cos, torch.zeros_like(roll_cos))
    tf_cos_mean = float((tf_cos_masked.sum() / valid).item())
    roll_cos_mean = float((roll_cos_masked.sum() / valid).item())

    # Last-valid and answer cosines.
    def _gather_last(x: torch.Tensor) -> torch.Tensor:
        idx = (valid_lens - 1).clamp(min=0)
        return x[torch.arange(x.shape[0], device=x.device), idx]

    tf_final = _gather_last(v_tf)
    roll_final = _gather_last(v_roll)
    tgt_final = _gather_last(chains_trunc)

    tf_cos_last = float(
        (_safe_normalize(tf_final.float()) * _safe_normalize(tgt_final.float())).sum(dim=-1).mean().item()
    )
    roll_cos_last = float(
        (_safe_normalize(roll_final.float()) * _safe_normalize(tgt_final.float())).sum(dim=-1).mean().item()
    )

    batch_idx = torch.arange(bsz, device=chains_trunc.device)
    roll_answer_cos = 0.0
    tf_answer_cos = 0.0
    if target_has_answer.any():
        roll_ans = v_roll[batch_idx, target_answer_pos]
        tf_ans = v_tf[batch_idx, target_answer_pos]
        tgt_ans = chains_trunc[batch_idx, target_answer_pos]
        roll_answer_cos = float(
            (_safe_normalize(roll_ans[target_has_answer].float())
             * _safe_normalize(tgt_ans[target_has_answer].float())).sum(dim=-1).mean().item()
        )
        tf_answer_cos = float(
            (_safe_normalize(tf_ans[target_has_answer].float())
             * _safe_normalize(tgt_ans[target_has_answer].float())).sum(dim=-1).mean().item()
        )

    return {
        "val_tf_cos": tf_cos_mean,
        "val_tf_cos_last": tf_cos_last,
        "val_tf_cos_answer": tf_answer_cos,
        "val_roll_cos": roll_cos_mean,
        "val_roll_cos_last": roll_cos_last,
        "val_roll_cos_answer": roll_answer_cos,
        "val_answer_coverage": float(target_has_answer.float().mean().item()),
        "val_norm_roll": float(v_roll.norm(dim=-1).mean().item()),
    }


# ═══════════════════════════════════════════════════════════════════════
# main() — FF training loop
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="ChainGenerator FF Training")
    parser.add_argument("--config", default="configs/chain_generator_ff_config.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--finetune", default=None,
        help="Path to checkpoint for fine-tuning (loads model weights only).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    setup_seed(config.get("seed", 42), device)

    out_cfg = config["output"]
    for d in [out_cfg["output_dir"], out_cfg["checkpoint_dir"], out_cfg["logs_dir"]]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ChainGenerator FF Training — Forward-Forward + FFCL head")
    print(f"Device: {device}")
    print("=" * 70)

    gen_cfg = ChainGeneratorConfig(**config.get("generator", {}))
    model = ChainGenerator(gen_cfg).to(device)

    print(f"Parameters: {model.num_params:,}")
    print(f"Layers={gen_cfg.n_layers}, Heads={gen_cfg.n_heads}, FFN={gen_cfg.dim_feedforward}")
    print(f"target_norm={gen_cfg.target_norm}, max_chain_len={gen_cfg.max_chain_len}")

    enable_ffn_stats = bool(config.get("training", {}).get("enable_ffn_stats", True))
    if enable_ffn_stats:
        for module in model.modules():
            if isinstance(module, SwiGLUFFN):
                module.track_stats = True
        print(f"  ffn_stats: enabled")

    # ── Data ──
    data_cfg = config["data"]
    data = torch.load(data_cfg["path"], map_location="cpu", weights_only=False)

    train_cfg = config["training"]
    max_chain_steps = int(train_cfg.get("max_chain_steps", gen_cfg.max_chain_len))
    if max_chain_steps > gen_cfg.max_chain_len:
        print(f"[WARN] max_chain_steps={max_chain_steps} > max_chain_len={gen_cfg.max_chain_len}; clamping")
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
        data["train"], "train",
        max_chain_len=gen_cfg.max_chain_len,
        max_chain_steps=max_chain_steps,
        answer_repeat_pad=answer_repeat_pad,
        context_bank_size=context_bank_size,
    )
    summarize_chain_dataset(
        data["val"], "val",
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

    # ── Optimizer ──
    lr = float(train_cfg.get("lr", 1e-4))
    wd = float(train_cfg.get("weight_decay", 1e-4))
    param_groups = _build_wd_param_groups(model, wd)
    optimizer = torch.optim.AdamW(param_groups, lr=lr)

    # ── EMA ──
    ema_decay = float(train_cfg.get("model_ema_decay", 0.0))
    ema: ModelEMA | None = None
    if ema_decay > 0.0:
        ema = ModelEMA(model, decay=ema_decay)
        print(f"EMA enabled with decay={ema_decay}")

    # ── Scheduler ──
    num_epochs = int(args.max_epochs or train_cfg.get("num_epochs", 50))
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(train_cfg.get("warmup_epochs", 3)) * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_steps,
        min_factor=float(train_cfg.get("lr_min_factor", 0.01)),
    )

    # ── AMP ──
    amp_enabled, amp_dtype, scaler = setup_amp(config.get("amp", {}), device)

    # ── Resume / Fine-tune ──
    start_epoch = 0
    best_metric = float("-inf")
    global_step = 0
    if args.finetune:
        ckpt = torch.load(args.finetune, map_location=device, weights_only=False)
        load_result = model.load_state_dict(ckpt["model"], strict=False)
        src_epoch = ckpt.get("epoch", "?")
        print(f"Fine-tune from {args.finetune} (src epoch={src_epoch})")
        if load_result.missing_keys or load_result.unexpected_keys:
            print(f"  Compat: missing={len(load_result.missing_keys)}, "
                  f"unexpected={len(load_result.unexpected_keys)}")
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

    # ── Logging & checkpoint config ──
    log_every = int(train_cfg.get("log_every", 50))
    ckpt_every = int(train_cfg.get("checkpoint_every", 5))
    patience = int(train_cfg.get("early_stop_patience", 15))
    no_improve = 0
    logs_dir = Path(out_cfg["logs_dir"])
    metrics_log_path = logs_dir / str(train_cfg.get("metrics_log_name", "chain_generator_ff_training.jsonl"))

    # FF-specific config.
    neg_sigma = float(train_cfg.get("ff_neg_sigma", 0.3))
    neg_schedule = train_cfg.get("ff_neg_schedule")

    _append_jsonl(
        metrics_log_path,
        {
            "event": "run_start",
            "pipeline": "forward_forward",
            "global_step": int(global_step),
            "start_epoch": int(start_epoch),
            "config_path": str(args.config),
            "timestamp": time.time(),
        },
    )

    tracker = MetricTracker()

    # ── Print settings ──
    print("\nFF Training settings")
    print(f"  epochs={num_epochs}, batch={batch_size}, lr={lr:.2e}")
    print(f"  symba_alpha={train_cfg.get('ff_symba_alpha', 2.0)}, "
          f"head_cos_w={train_cfg.get('ff_head_cosine_weight', 1.0)}, "
          f"head_mse_w={train_cfg.get('ff_head_mse_weight', 0.1)}")
    print(f"  neg_sigma={neg_sigma}, neg_schedule={neg_schedule}")
    horizon_schedule = train_cfg.get("horizon_schedule")
    if horizon_schedule:
        sched_desc = " → ".join(f"{s}×{d}ep" for s, d in horizon_schedule)
        print(f"  horizon: system1={train_cfg.get('system1_epochs', 10)}ep, "
              f"schedule=[{sched_desc}]")
    else:
        print(f"  horizon: system1_epochs={train_cfg.get('system1_epochs', 10)}, "
              f"linear_ramp, max_steps={max_chain_steps}")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════════
    # Epoch loop
    # ═══════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, num_epochs):
        model.train()
        target_steps = get_chain_steps(epoch, train_cfg)
        neg_strategy = curriculum_strategy(epoch, neg_schedule)

        epoch_start = time.time()
        phase = "System1" if target_steps == 1 else f"System2({target_steps})"
        print(f"\n[E{epoch}] target_steps={target_steps} [{phase}] neg={neg_strategy}")

        nan_count = 0
        for step, batch in enumerate(train_loader):
            metrics = ff_train_step(
                model,
                batch,
                optimizer,
                scaler,
                device,
                amp_enabled,
                amp_dtype,
                train_cfg,
                target_steps=target_steps,
                neg_strategy=neg_strategy,
                neg_sigma=neg_sigma,
                ema=ema,
            )
            if metrics.get("optimizer_stepped", 0.0) > 0.0:
                scheduler.step()
            global_step += 1
            tracker.update(metrics)

            if metrics.get("nan_skipped", 0.0) > 0:
                nan_count += 1

            if (step + 1) % log_every == 0:
                avg = tracker.get()
                lr_now = optimizer.param_groups[0]["lr"]
                n_layers = len(model.layers)
                nan_info = f" [NaN:{nan_count}]" if nan_count > 0 else ""

                # Per-layer goodness summary: show gap (pos - neg).
                g_gaps = []
                for li in range(n_layers):
                    gp = avg.get(f"goodness_pos_{li}", 0.0)
                    gn = avg.get(f"goodness_neg_{li}", 0.0)
                    g_gaps.append(f"{gp - gn:+.3f}")
                gap_str = ",".join(g_gaps)

                print(
                    f"  [E{epoch} S{step+1}] "
                    f"ff={avg.get('ff_total', 0.0):.4f} "
                    f"head={avg.get('head_total', 0.0):.4f} "
                    f"head_cos={avg.get('head_cosine', 0.0):.4f} "
                    f"head_mse={avg.get('head_mse', 0.0):.4f} "
                    f"g_gap=[{gap_str}] "
                    f"grad={avg.get('grad_norm', 0.0):.4f} "
                    f"lr={lr_now:.2e}{nan_info}"
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
                        "neg_strategy": neg_strategy,
                        "lr": float(lr_now),
                        "nan_count_epoch": int(nan_count),
                        "train_metrics": avg,
                        "timestamp": time.time(),
                    },
                )

                # Log SwiGLU stats at log boundaries.
                if enable_ffn_stats:
                    ffn_stats = _collect_swiglu_stats(model)
                    if ffn_stats:
                        _append_jsonl(
                            metrics_log_path,
                            {
                                "event": "ffn_health",
                                "global_step": int(global_step),
                                "stats": ffn_stats,
                                "timestamp": time.time(),
                            },
                        )

                tracker.reset()

        # ── Validation ──
        model.eval()
        ema_backup: dict[str, torch.Tensor] | None = None
        if ema is not None:
            ema_backup = ema.apply_to(model)
        val_tracker = MetricTracker()
        eval_steps = target_steps
        for vbatch in val_loader:
            vm = ff_eval_step(
                model, vbatch, device, amp_enabled, amp_dtype, train_cfg,
                gen_steps=eval_steps,
            )
            val_tracker.update(vm)
        if ema is not None and ema_backup is not None:
            ema.restore(model, ema_backup)

        val = val_tracker.get()
        epoch_time = time.time() - epoch_start

        val_metric = float(val.get("val_roll_cos_last", val.get("val_roll_cos", 0.0)))

        print(
            f"  [E{epoch} VAL] "
            f"tf_cos={val.get('val_tf_cos', 0.0):.4f} "
            f"roll_cos={val.get('val_roll_cos', 0.0):.4f} "
            f"roll_cos_last={val.get('val_roll_cos_last', 0.0):.4f} "
            f"roll_ans={val.get('val_roll_cos_answer', 0.0):.4f} "
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
                "neg_strategy": neg_strategy,
                "epoch_time_sec": float(epoch_time),
                "val_metric": float(val_metric),
                "val_metrics": val,
                "timestamp": time.time(),
            },
        )

        # ── Checkpointing + early stopping ──
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

    print("\nFF Training complete")
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
