"""
ContextEncoder diagnostics for GUI.

Provides:
  1) loading CE checkpoint (+ optional SP checkpoint),
  2) reproducible evaluation on sequence datasets,
  3) robustness sweeps over input noise,
  4) length-bucket metrics to diagnose bottlenecks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from cebcm.data.sequence_dataset import SONARSequenceDataset
from cebcm.models.context_encoder import ContextEncoder
from cebcm.models.surprise import SurprisePredictor
from cebcm.training.stage2_utils import (
    build_context_encoder,
    build_surprise_predictor,
    build_type_ids,
    load_config,
    load_checkpoint,
    resolve_device,
)


class CEPretrainHead(nn.Module):
    """Projection head used in CE pretraining."""

    def __init__(self, d_in: int, d_out: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@dataclass
class CEDiagnosticsState:
    config: dict[str, Any] | None = None
    config_path: str = ""
    ce_checkpoint_path: str = ""
    sp_checkpoint_path: str = ""
    device: str = "cpu"
    context_encoder: ContextEncoder | None = None
    ce_head: CEPretrainHead | None = None
    surprise_predictor: SurprisePredictor | None = None
    use_surprise: bool = False


_state = CEDiagnosticsState()


def _infer_ce_head_from_state(state_dict: dict[str, Tensor]) -> CEPretrainHead:
    w0 = state_dict.get("net.0.weight")
    w2 = state_dict.get("net.2.weight")
    if w0 is None or w2 is None:
        raise KeyError(
            "Unsupported CE head checkpoint format. "
            "Expected keys: 'net.0.weight' and 'net.2.weight'."
        )
    hidden_dim = int(w0.shape[0])
    d_in = int(w0.shape[1])
    d_out = int(w2.shape[0])
    head = CEPretrainHead(d_in=d_in, d_out=d_out, hidden_dim=hidden_dim)
    head.load_state_dict(state_dict)
    return head


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


def _bucket_name(ctx_len: int) -> str:
    if ctx_len <= 5:
        return "short (<=5)"
    if ctx_len <= 12:
        return "medium (6-12)"
    return "long (>=13)"


def _init_bucket() -> dict[str, float]:
    return {
        "n": 0.0,
        "cos_sum": 0.0,
        "l2_sum": 0.0,
        "mse_sum": 0.0,
        "pred_norm_sum": 0.0,
    }


def _update_bucket(bucket: dict[str, float], cos: Tensor, l2: Tensor, mse: Tensor, pred_norm: Tensor) -> None:
    n = float(cos.numel())
    bucket["n"] += n
    bucket["cos_sum"] += float(cos.sum().item())
    bucket["l2_sum"] += float(l2.sum().item())
    bucket["mse_sum"] += float(mse.sum().item())
    bucket["pred_norm_sum"] += float(pred_norm.sum().item())


def _finalize_bucket(bucket: dict[str, float]) -> dict[str, float]:
    n = max(1.0, bucket["n"])
    return {
        "n": int(bucket["n"]),
        "cos_mean": bucket["cos_sum"] / n,
        "l2_mean": bucket["l2_sum"] / n,
        "mse_mean": bucket["mse_sum"] / n,
        "pred_norm_mean": bucket["pred_norm_sum"] / n,
    }


def _parse_noise_levels(noise_levels_csv: str) -> list[float]:
    out: list[float] = []
    for token in str(noise_levels_csv).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(float(token))
        except ValueError:
            continue
    if not out:
        out = [0.0, 2.0, 5.0, 10.0]
    out = sorted(set(max(0.0, v) for v in out))
    return out


def load_ce_models(
    config_path: str,
    ce_checkpoint_path: str,
    sp_checkpoint_path: str,
    use_surprise: bool,
    device: str = "auto",
) -> str:
    """
    Load ContextEncoder (+ CE head) and optional SurprisePredictor for diagnostics.
    """
    cfg_path = Path(config_path)
    ce_path = Path(ce_checkpoint_path)
    if not cfg_path.exists():
        return f"Config not found: {cfg_path}"
    if not ce_path.exists():
        return f"CE checkpoint not found: {ce_path}"

    dev = resolve_device(None if device == "auto" else device)
    cfg = load_config(cfg_path)

    context_encoder = build_context_encoder(cfg).to(dev)
    ce_ckpt = load_checkpoint(ce_path, device=dev)
    if "context_encoder" not in ce_ckpt:
        return f"Checkpoint missing 'context_encoder': {ce_path}"
    context_encoder.load_state_dict(ce_ckpt["context_encoder"])
    context_encoder.eval()

    ce_head: CEPretrainHead | None = None
    if "ce_head" in ce_ckpt:
        ce_head = _infer_ce_head_from_state(ce_ckpt["ce_head"]).to(dev)
        ce_head.eval()

    surprise_predictor: SurprisePredictor | None = None
    effective_use_surprise = bool(use_surprise)
    if effective_use_surprise:
        sp_path = Path(sp_checkpoint_path) if sp_checkpoint_path else None
        if sp_path is None or (not sp_path.exists()):
            # Try default from config if explicit path missing.
            init_sp = str(cfg.get("init", {}).get("sp_checkpoint", "")).strip()
            if init_sp:
                sp_path = Path(init_sp)
        if sp_path is None or (not sp_path.exists()):
            return (
                "Surprise enabled but SP checkpoint not found. "
                "Provide SP path or disable surprise for this run."
            )
        sp_ckpt = load_checkpoint(sp_path, device=dev)
        if "surprise_predictor" not in sp_ckpt:
            return f"SP checkpoint missing 'surprise_predictor': {sp_path}"
        surprise_predictor = build_surprise_predictor(cfg).to(dev)
        surprise_predictor.load_state_dict(sp_ckpt["surprise_predictor"])
        surprise_predictor.eval()

    _state.config = cfg
    _state.config_path = str(cfg_path)
    _state.ce_checkpoint_path = str(ce_path)
    _state.sp_checkpoint_path = str(sp_checkpoint_path)
    _state.device = str(dev)
    _state.context_encoder = context_encoder
    _state.ce_head = ce_head
    _state.surprise_predictor = surprise_predictor
    _state.use_surprise = effective_use_surprise

    ce_params = context_encoder.num_params
    head_params = sum(p.numel() for p in ce_head.parameters()) if ce_head is not None else 0
    msg = [
        f"ContextEncoder loaded on `{dev}`",
        f"- CE params: `{ce_params:,}`",
        f"- CE head: {'loaded' if ce_head is not None else 'missing in checkpoint'}",
    ]
    if ce_head is not None:
        msg.append(f"- CE head params: `{head_params:,}`")
    msg.append(f"- Surprise features: `{effective_use_surprise}`")
    if effective_use_surprise and surprise_predictor is not None:
        msg.append(f"- SP checkpoint: `{sp_path}`")
    return "\n".join(msg)


def _evaluate_once(
    loader: DataLoader,
    dataset_source: str,
    noise_pct: float,
    target_norm: float | None,
    max_batches: int,
) -> dict[str, Any]:
    if _state.context_encoder is None:
        raise RuntimeError("ContextEncoder is not loaded. Load checkpoints first.")

    device = torch.device(_state.device)
    context_encoder = _state.context_encoder
    ce_head = _state.ce_head
    surprise_predictor = _state.surprise_predictor if _state.use_surprise else None

    context_encoder.eval()
    if ce_head is not None:
        ce_head.eval()
    if surprise_predictor is not None:
        surprise_predictor.eval()

    cos_values: list[float] = []
    l2_values: list[float] = []
    mse_values: list[float] = []
    pred_norm_values: list[float] = []
    tgt_norm_values: list[float] = []
    length_values: list[int] = []
    bucket_stats: dict[str, dict[str, float]] = {
        "short (<=5)": _init_bucket(),
        "medium (6-12)": _init_bucket(),
        "long (>=13)": _init_bucket(),
    }

    noise_scale = float(noise_pct) / 100.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break

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

            if noise_scale > 0.0:
                token_norm = context_vecs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                context_vecs = context_vecs + noise_scale * token_norm * torch.randn_like(context_vecs)

            type_ids = build_type_ids(lengths, L, dataset_source=dataset_source)[:, :-1].to(device)
            surprise_padded = _build_surprise_padded(
                surprise_predictor=surprise_predictor,
                context_vecs=context_vecs,
                context_lengths=context_lengths,
            )

            v_context = context_encoder(
                context_vectors=context_vecs,
                type_ids=type_ids,
                surprise_scores=surprise_padded,
                lengths=context_lengths,
            )

            if ce_head is not None:
                pred_vecs = ce_head(v_context)
            else:
                if v_context.shape[-1] != target_vecs.shape[-1]:
                    raise RuntimeError(
                        "CE head is missing and CE output dim != target dim. "
                        "Load a CE checkpoint with 'ce_head' for comparable metrics."
                    )
                pred_vecs = v_context

            if target_norm is not None and target_norm > 0.0:
                pred_vecs = F.normalize(pred_vecs, dim=-1) * float(target_norm)

            cos = F.cosine_similarity(pred_vecs, target_vecs, dim=-1)  # [B]
            l2 = (pred_vecs - target_vecs).norm(dim=-1)  # [B]
            mse = ((pred_vecs - target_vecs) ** 2).mean(dim=-1)  # [B]
            pred_norm = pred_vecs.norm(dim=-1)
            tgt_norm = target_vecs.norm(dim=-1)

            cos_values.extend(cos.cpu().tolist())
            l2_values.extend(l2.cpu().tolist())
            mse_values.extend(mse.cpu().tolist())
            pred_norm_values.extend(pred_norm.cpu().tolist())
            tgt_norm_values.extend(tgt_norm.cpu().tolist())

            ctx_lens = context_lengths.cpu().tolist()
            for i, ctx_len in enumerate(ctx_lens):
                bucket = bucket_stats[_bucket_name(int(ctx_len))]
                _update_bucket(
                    bucket,
                    cos=cos[i:i + 1],
                    l2=l2[i:i + 1],
                    mse=mse[i:i + 1],
                    pred_norm=pred_norm[i:i + 1],
                )
                length_values.append(int(ctx_len))

    if not cos_values:
        return {
            "noise_pct": noise_pct,
            "n": 0,
            "cos_mean": 0.0,
            "cos_std": 0.0,
            "cos_gt05": 0.0,
            "cos_gt07": 0.0,
            "l2_mean": 0.0,
            "mse_mean": 0.0,
            "pred_norm_mean": 0.0,
            "target_norm_mean": 0.0,
            "length_mean": 0.0,
            "buckets": {k: _finalize_bucket(v) for k, v in bucket_stats.items()},
        }

    cos_t = torch.tensor(cos_values)
    l2_t = torch.tensor(l2_values)
    mse_t = torch.tensor(mse_values)
    pred_norm_t = torch.tensor(pred_norm_values)
    tgt_norm_t = torch.tensor(tgt_norm_values)
    lengths_t = torch.tensor(length_values, dtype=torch.float32) if length_values else torch.tensor([0.0])

    return {
        "noise_pct": float(noise_pct),
        "n": int(cos_t.numel()),
        "cos_mean": float(cos_t.mean().item()),
        "cos_std": float(cos_t.std().item()),
        "cos_gt05": float((cos_t > 0.5).float().mean().item()),
        "cos_gt07": float((cos_t > 0.7).float().mean().item()),
        "l2_mean": float(l2_t.mean().item()),
        "mse_mean": float(mse_t.mean().item()),
        "pred_norm_mean": float(pred_norm_t.mean().item()),
        "target_norm_mean": float(tgt_norm_t.mean().item()),
        "length_mean": float(lengths_t.mean().item()),
        "buckets": {k: _finalize_bucket(v) for k, v in bucket_stats.items()},
    }


def _build_noise_plot(rows: list[dict[str, Any]]) -> go.Figure:
    noise = [r["noise_pct"] for r in rows]
    cos = [r["cos_mean"] for r in rows]
    l2 = [r["l2_mean"] for r in rows]
    gt07 = [100.0 * r["cos_gt07"] for r in rows]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=noise, y=cos, mode="lines+markers", name="cos_mean"))
    fig.add_trace(go.Scatter(x=noise, y=l2, mode="lines+markers", name="l2_mean", yaxis="y2"))
    fig.add_trace(go.Bar(x=noise, y=gt07, name="cos>0.7 (%)", opacity=0.35, yaxis="y3"))
    fig.update_layout(
        title="ContextEncoder Robustness vs Input Noise",
        xaxis=dict(title="Noise %"),
        yaxis=dict(title="Cosine Mean", range=[0.0, 1.0]),
        yaxis2=dict(title="L2 Mean", overlaying="y", side="right"),
        yaxis3=dict(
            title="cos>0.7 (%)",
            overlaying="y",
            side="left",
            anchor="free",
            position=0.08,
            range=[0.0, 100.0],
        ),
        barmode="overlay",
        legend=dict(orientation="h"),
    )
    return fig


def _build_bucket_plot(bucket_metrics: dict[str, dict[str, float]]) -> go.Figure:
    labels = list(bucket_metrics.keys())
    cos = [bucket_metrics[k]["cos_mean"] for k in labels]
    l2 = [bucket_metrics[k]["l2_mean"] for k in labels]
    counts = [bucket_metrics[k]["n"] for k in labels]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=cos, name="cos_mean"))
    fig.add_trace(go.Bar(x=labels, y=l2, name="l2_mean"))
    fig.add_trace(go.Scatter(x=labels, y=counts, mode="lines+markers", name="count", yaxis="y2"))
    fig.update_layout(
        title="Context Length Bucket Metrics (baseline noise)",
        xaxis=dict(title="Context Length Bucket"),
        yaxis=dict(title="Metric Value"),
        yaxis2=dict(title="Sample Count", overlaying="y", side="right"),
        barmode="group",
        legend=dict(orientation="h"),
    )
    return fig


def run_ce_diagnostics(
    data_path: str,
    split_mode: str,
    split_ratio: float,
    batch_size: int,
    max_batches: int,
    noise_levels_csv: str,
    seed: int,
    target_norm: float,
) -> tuple[str, go.Figure, go.Figure]:
    """
    Evaluate loaded ContextEncoder across requested noise conditions.
    """
    if _state.context_encoder is None or _state.config is None:
        raise RuntimeError("Load CE models first.")

    cfg = _state.config
    data_cfg = cfg.get("data", {})
    data_file = str(data_path).strip()
    if not data_file:
        data_file = str(data_cfg.get("train_data_path", "data/squad_sequences.pt"))

    dataset = SONARSequenceDataset(
        path=data_file,
        max_seq_len=int(data_cfg.get("max_seq_len", 64)),
        min_seq_len=int(data_cfg.get("min_seq_len", 3)),
        legacy_window_stride=data_cfg.get("legacy_window_stride", None),
    )
    source_name = str(getattr(dataset, "source", "unknown"))

    split_mode_norm = str(split_mode).strip().lower()
    eval_dataset = dataset
    if split_mode_norm in {"train", "val"}:
        train_ds, val_ds = dataset.split(train_ratio=float(split_ratio), seed=int(seed))
        eval_dataset = train_ds if split_mode_norm == "train" else val_ds

    loader = DataLoader(
        eval_dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        collate_fn=SONARSequenceDataset.collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    noise_levels = _parse_noise_levels(noise_levels_csv)
    rows: list[dict[str, Any]] = []
    for noise in noise_levels:
        rows.append(
            _evaluate_once(
                loader=loader,
                dataset_source=source_name,
                noise_pct=noise,
                target_norm=float(target_norm) if target_norm > 0 else None,
                max_batches=max(0, int(max_batches)),
            )
        )

    baseline = rows[0]
    bucket_fig = _build_bucket_plot(baseline["buckets"])
    noise_fig = _build_noise_plot(rows)

    ce_cfg = cfg.get("context_encoder", {})
    lines = [
        f"**ContextEncoder Diagnostics**  ",
        f"- dataset: `{data_file}` (source `{source_name}`), split `{split_mode_norm}`",
        f"- samples: `{baseline['n']}` (noise `{baseline['noise_pct']:.1f}%` baseline)",
        f"- cosine mean/std: `{baseline['cos_mean']:.4f}` / `{baseline['cos_std']:.4f}`",
        f"- cosine > 0.5: `{100.0 * baseline['cos_gt05']:.1f}%`",
        f"- cosine > 0.7: `{100.0 * baseline['cos_gt07']:.1f}%`",
        f"- l2 mean: `{baseline['l2_mean']:.4f}`",
        f"- mse mean: `{baseline['mse_mean']:.6f}`",
        f"- pred norm / target norm: `{baseline['pred_norm_mean']:.4f}` / `{baseline['target_norm_mean']:.4f}`",
        f"- context length mean: `{baseline['length_mean']:.2f}`",
        "",
        "**Global Token Policy (from config)**  ",
        f"- top_k_pct: `{float(ce_cfg.get('surprise_top_k_pct', 0.05)):.4f}`",
        f"- min_tokens: `{int(ce_cfg.get('surprise_top_k_min_tokens', 1))}`",
        f"- include_last_token: `{bool(ce_cfg.get('global_include_last_token', True))}`",
        f"- use_alibi: `{bool(ce_cfg.get('use_alibi', False))}`",
        f"- surprise_features_enabled: `{_state.use_surprise}`",
        "",
        "**Noise Sweep**",
    ]
    for r in rows:
        lines.append(
            f"- noise `{r['noise_pct']:.1f}%`: cos `{r['cos_mean']:.4f}`, "
            f"l2 `{r['l2_mean']:.4f}`, cos>0.7 `{100.0 * r['cos_gt07']:.1f}%`"
        )

    return "\n".join(lines), noise_fig, bucket_fig

