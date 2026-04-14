"""Training geometry visualization for ChainGenerator probes.

The training script writes compact fixed-probe snapshots. This module keeps the
GUI side read-only: it loads JSONL scalar events and per-step .pt artifacts, then
renders stable PCA-space diagnostics for Diffusion Forcing and rollout quality.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch


PLOT_TEMPLATE = "plotly_dark"
COLORS = {
    "clean": "#50fa7b",
    "noisy": "#ff6b6b",
    "pred": "#4dabf7",
    "tf": "#ffd43b",
    "roll": "#b197fc",
    "answer": "#ffffff",
    "muted": "rgba(210, 220, 255, 0.24)",
}


def _empty_fig(title: str, message: str | None = None) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=title,
        paper_bgcolor="#0b111d",
        plot_bgcolor="#0f1724",
        margin=dict(l=40, r=20, t=70, b=40),
    )
    fig.add_annotation(
        text=message or "No data available",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16, color="#d8e2ff"),
    )
    return fig


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    return np.asarray(value)


def _flatten_event(prefix: str, payload: dict[str, Any], out: dict[str, Any]) -> None:
    for key, value in payload.items():
        if isinstance(value, dict):
            _flatten_event(f"{prefix}{key}_", value, out)
        elif isinstance(value, (int, float, str, bool)) or value is None:
            out[f"{prefix}{key}"] = value


def _parse_step_from_probe(path: Path) -> int:
    match = re.search(r"probe_step_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def resolve_geometry_paths(path_like: str | Path) -> tuple[Path | None, Path | None, str]:
    path = Path(path_like).expanduser()
    if not path.exists():
        return None, None, f"Path not found: {path}"

    if path.is_file():
        if path.suffix.lower() == ".pt":
            probe_dir = path.parent
            metrics_path = probe_dir.parent / "chain_generator_training.jsonl"
            return (metrics_path if metrics_path.exists() else None), probe_dir, "Loaded from probe file"
        if path.suffix.lower() == ".jsonl":
            probe_dir = path.parent / "geometry_probes"
            return path, (probe_dir if probe_dir.exists() else None), "Loaded from JSONL metrics"
        return None, None, f"Unsupported file type: {path.suffix}"

    metrics_path = path / "chain_generator_training.jsonl"
    if not metrics_path.exists():
        jsonl_files = sorted(path.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        metrics_path = jsonl_files[0] if jsonl_files else None

    probe_dir = path / "geometry_probes"
    if not probe_dir.exists() and path.name == "geometry_probes":
        probe_dir = path
    if not probe_dir.exists():
        nested = sorted(path.glob("**/geometry_probes"), key=lambda p: len(p.parts))
        probe_dir = nested[0] if nested else None

    if metrics_path is None and probe_dir is None:
        return None, None, f"No JSONL metrics or geometry_probes directory found under {path}"
    return metrics_path, probe_dir, "Loaded from directory"


def _read_metrics_jsonl(metrics_path: Path | None) -> pd.DataFrame:
    if metrics_path is None or not metrics_path.exists():
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            row: dict[str, Any] = {}
            for key in ["event", "epoch", "batch_idx", "global_step", "target_steps", "phase", "lr", "timestamp"]:
                if key in rec:
                    row[key] = rec[key]
            if isinstance(rec.get("train_metrics"), dict):
                _flatten_event("train_", rec["train_metrics"], row)
            if isinstance(rec.get("val_metrics"), dict):
                for key, value in rec["val_metrics"].items():
                    out_key = key if str(key).startswith("val_") else f"val_{key}"
                    if isinstance(value, (int, float, str, bool)) or value is None:
                        row[out_key] = value
            if isinstance(rec.get("probe"), dict):
                _flatten_event("probe_", rec["probe"], row)
            if isinstance(rec.get("sadt"), dict):
                _flatten_event("sadt_", rec["sadt"], row)
            if "error" in rec:
                row["error"] = rec["error"]
            rows.append(row)

    df = pd.DataFrame(rows)
    if "global_step" in df.columns:
        df["global_step"] = pd.to_numeric(df["global_step"], errors="coerce")
        df = df.sort_values(["global_step", "event"], na_position="last")
    return df


def load_training_geometry_run(path_like: str | Path) -> dict[str, Any]:
    metrics_path, probe_dir, status = resolve_geometry_paths(path_like)
    df = _read_metrics_jsonl(metrics_path)
    probe_files = sorted((probe_dir.glob("probe_step_*.pt") if probe_dir else []), key=_parse_step_from_probe)
    choices = [(f"step {_parse_step_from_probe(p):,} | {p.name}", str(p)) for p in probe_files]
    if not probe_files:
        status = f"{status}; waiting for probe_step_*.pt artifacts"
    return {
        "metrics_path": str(metrics_path) if metrics_path else "",
        "probe_dir": str(probe_dir) if probe_dir else "",
        "metrics_df": df,
        "probe_files": probe_files,
        "probe_choices": choices,
        "status": status,
    }


def load_probe_snapshot(run: dict[str, Any] | None, selected: str | None = None) -> dict[str, Any] | None:
    if run is None:
        return None
    path: Path | None = None
    if selected:
        candidate = Path(str(selected))
        if candidate.exists():
            path = candidate
    if path is None:
        files = run.get("probe_files") or []
        if files:
            path = Path(files[-1])
    if path is None or not path.exists():
        return None
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(snapshot, dict):
        snapshot["_path"] = str(path)
    return snapshot


def _latest_value(df: pd.DataFrame, event: str, column: str) -> Any:
    if df.empty or column not in df.columns or "event" not in df.columns:
        return None
    part = df[df["event"] == event]
    if part.empty:
        return None
    value = part[column].dropna()
    return value.iloc[-1] if not value.empty else None


def format_geometry_summary(run: dict[str, Any] | None, snapshot: dict[str, Any] | None = None) -> str:
    if run is None:
        return "No training geometry run loaded."
    df = run.get("metrics_df", pd.DataFrame())
    probe_count = len(run.get("probe_files") or [])
    lines = [
        "### ChainGenerator Training Geometry",
        f"status: {run.get('status', 'unknown')}",
        f"metrics: `{run.get('metrics_path') or 'not found'}`",
        f"probe_dir: `{run.get('probe_dir') or 'not found'}`",
        f"events: {len(df)} | probes: {probe_count}",
    ]
    latest_train = _latest_value(df, "train_step", "train_roll_cos_answer")
    latest_roll = _latest_value(df, "train_step", "train_roll_cos_mean")
    latest_df = _latest_value(df, "train_step", "train_df_cos")
    latest_val = _latest_value(df, "val_epoch", "val_roll_cos_answer")
    latest_val_last = _latest_value(df, "val_epoch", "val_roll_cos_last")
    if latest_train is not None or latest_val is not None:
        lines.append("")
        lines.append("Latest scalar metrics:")
        if latest_roll is not None:
            lines.append(f"- train roll_cos_mean: {float(latest_roll):.4f}")
        if latest_train is not None:
            lines.append(f"- train roll_cos_answer: {float(latest_train):.4f}")
        if latest_df is not None:
            lines.append(f"- train df_cos: {float(latest_df):.4f}")
        if latest_val is not None:
            lines.append(f"- val roll_cos_answer: {float(latest_val):.4f}")
        if latest_val_last is not None:
            lines.append(f"- val roll_cos_last: {float(latest_val_last):.4f}")
    if snapshot is not None:
        metrics = snapshot.get("metrics", {})
        proj = snapshot.get("projection", {})
        explained = _to_numpy(proj.get("explained"))
        exp_text = ", ".join(f"{float(v):.3f}" for v in explained[:3]) if explained.size else "n/a"
        lines.extend([
            "",
            "Loaded probe snapshot:",
            f"- file: `{snapshot.get('_path', 'in-memory')}`",
            f"- global_step: {snapshot.get('global_step')} | epoch: {snapshot.get('epoch')} | target_steps: {snapshot.get('target_steps')}",
            f"- df_cos_mean: {float(metrics.get('df_cos_mean', 0.0)):.4f}",
            f"- noisy_cos_mean: {float(metrics.get('noisy_cos_mean', 0.0)):.4f}",
            f"- teacher_forced_cos_mean: {float(metrics.get('tf_cos_mean', 0.0)):.4f}",
            f"- rollout_cos_mean: {float(metrics.get('roll_cos_mean', 0.0)):.4f}",
            f"- PCA explained variance ratio: {exp_text}",
        ])
    return "\n".join(lines)


def _add_line_if_present(fig: go.Figure, df: pd.DataFrame, event: str, y: str, name: str, color: str, secondary_y: bool = False) -> None:
    if df.empty or "event" not in df.columns or "global_step" not in df.columns or y not in df.columns:
        return
    part = df[df["event"] == event]
    if part.empty:
        return
    valid = part[["global_step", y]].dropna()
    if valid.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=valid["global_step"],
            y=valid[y],
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=2.4),
            marker=dict(size=5),
        ),
        secondary_y=secondary_y,
    )


def create_step_metrics_figure(run: dict[str, Any] | None) -> go.Figure:
    if run is None:
        return _empty_fig("Step-Level Training Metrics")
    df = run.get("metrics_df", pd.DataFrame())
    if df.empty:
        return _empty_fig("Step-Level Training Metrics", "No scalar JSONL events found yet")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    _add_line_if_present(fig, df, "train_step", "train_loss", "train loss", "#ff8787", secondary_y=False)
    _add_line_if_present(fig, df, "train_step", "train_roll_cos_mean", "train rollout cos", "#b197fc", secondary_y=True)
    _add_line_if_present(fig, df, "train_step", "train_roll_cos_answer", "train answer cos", "#50fa7b", secondary_y=True)
    _add_line_if_present(fig, df, "train_step", "train_df_cos", "train DF cos", "#4dabf7", secondary_y=True)
    _add_line_if_present(fig, df, "val_epoch", "val_roll_cos", "val rollout cos", "#ffd43b", secondary_y=True)
    _add_line_if_present(fig, df, "val_epoch", "val_roll_cos_answer", "val answer cos", "#69db7c", secondary_y=True)
    _add_line_if_present(fig, df, "probe_snapshot", "probe_roll_cos_mean", "probe rollout cos", "#da77f2", secondary_y=True)
    _add_line_if_present(fig, df, "probe_snapshot", "probe_df_cos_mean", "probe DF cos", "#74c0fc", secondary_y=True)
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title="Training Scalars by Global Step",
        paper_bgcolor="#0b111d",
        plot_bgcolor="#0f1724",
        hovermode="x unified",
        margin=dict(l=50, r=30, t=70, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    fig.update_xaxes(title_text="global_step", gridcolor="rgba(180,195,255,0.14)")
    fig.update_yaxes(title_text="loss", secondary_y=False, gridcolor="rgba(180,195,255,0.14)")
    fig.update_yaxes(title_text="cosine", secondary_y=True, range=[-0.05, 1.02], gridcolor="rgba(180,195,255,0.05)")
    return fig


def _sample_data(snapshot: dict[str, Any], key: str, sample_idx: int) -> np.ndarray:
    data = snapshot.get("projected", {}).get(key)
    arr = _to_numpy(data)
    if arr.ndim < 3:
        return np.asarray([])
    sample_idx = int(np.clip(sample_idx, 0, arr.shape[0] - 1))
    return arr[sample_idx]


def _array_data(snapshot: dict[str, Any], key: str, sample_idx: int) -> np.ndarray:
    data = snapshot.get("arrays", {}).get(key)
    arr = _to_numpy(data)
    if arr.ndim < 2:
        return np.asarray([])
    sample_idx = int(np.clip(sample_idx, 0, arr.shape[0] - 1))
    return arr[sample_idx]


def _valid_positions(snapshot: dict[str, Any], sample_idx: int) -> np.ndarray:
    mask = _array_data(snapshot, "mask", sample_idx)
    if mask.size == 0:
        return np.asarray([], dtype=int)
    return np.where(mask.astype(bool))[0]


def _answer_pos(snapshot: dict[str, Any], sample_idx: int) -> int | None:
    arr = _to_numpy(snapshot.get("arrays", {}).get("answer_pos"))
    has = _to_numpy(snapshot.get("arrays", {}).get("has_answer"))
    if arr.ndim == 0 or arr.size == 0:
        return None
    sample_idx = int(np.clip(sample_idx, 0, arr.shape[0] - 1))
    if has.size and not bool(has[sample_idx]):
        return None
    return int(arr[sample_idx])


def _add_path3d(fig: go.Figure, pts: np.ndarray, idx: np.ndarray, name: str, color: str, width: float = 5.0) -> None:
    if pts.size == 0 or idx.size == 0:
        return
    p = pts[idx]
    fig.add_trace(go.Scatter3d(
        x=p[:, 0], y=p[:, 1], z=p[:, 2],
        mode="lines+markers",
        name=name,
        line=dict(color=color, width=width),
        marker=dict(size=4, color=color),
    ))


def create_diffusion_geometry_3d(snapshot: dict[str, Any] | None, sample_idx: int = 0) -> go.Figure:
    if snapshot is None:
        return _empty_fig("Diffusion Forcing 3D Geometry")
    clean = _sample_data(snapshot, "clean", sample_idx)
    noisy = _sample_data(snapshot, "noisy", sample_idx)
    pred = _sample_data(snapshot, "pred_x0", sample_idx)
    idx = _valid_positions(snapshot, sample_idx)
    if clean.size == 0 or noisy.size == 0 or pred.size == 0 or idx.size == 0:
        return _empty_fig("Diffusion Forcing 3D Geometry", "Snapshot has no valid projected vectors")

    noise = _array_data(snapshot, "noise_levels", sample_idx)
    noise_valid = noise[idx] if noise.size else np.zeros_like(idx, dtype=float)
    answer_pos = _answer_pos(snapshot, sample_idx)

    fig = go.Figure()
    _add_path3d(fig, clean, idx, "clean x0 chain", COLORS["clean"], width=6.0)

    for pos in idx:
        pts = np.vstack([clean[pos], noisy[pos], pred[pos]])
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="lines",
            showlegend=False,
            line=dict(color=COLORS["muted"], width=2.0),
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter3d(
        x=noisy[idx, 0], y=noisy[idx, 1], z=noisy[idx, 2],
        mode="markers",
        name="noisy x_t (colored by t)",
        marker=dict(size=5, color=noise_valid, colorscale="Turbo", colorbar=dict(title="noise t"), opacity=0.92),
        text=[f"pos={int(p)} noise={float(n):.0f}" for p, n in zip(idx, noise_valid)],
        hovertemplate="%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=pred[idx, 0], y=pred[idx, 1], z=pred[idx, 2],
        mode="markers+lines",
        name="pred x0",
        marker=dict(size=5, color=COLORS["pred"], symbol="diamond"),
        line=dict(color=COLORS["pred"], width=3.0),
    ))
    if answer_pos is not None and answer_pos in set(idx.tolist()):
        for pts, name, color, symbol in [
            (clean, "answer clean", COLORS["answer"], "circle"),
            (noisy, "answer noisy", COLORS["noisy"], "x"),
            (pred, "answer pred", COLORS["pred"], "diamond"),
        ]:
            p = pts[answer_pos]
            fig.add_trace(go.Scatter3d(
                x=[p[0]], y=[p[1]], z=[p[2]],
                mode="markers",
                name=name,
                marker=dict(size=9, color=color, symbol=symbol, line=dict(color="#ffffff", width=1.5)),
            ))
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=f"Diffusion Forcing Geometry | sample {sample_idx} | step {snapshot.get('global_step')}",
        paper_bgcolor="#0b111d",
        scene=dict(
            xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3",
            bgcolor="#0f1724",
            xaxis=dict(gridcolor="rgba(180,195,255,0.16)"),
            yaxis=dict(gridcolor="rgba(180,195,255,0.16)"),
            zaxis=dict(gridcolor="rgba(180,195,255,0.16)"),
        ),
        margin=dict(l=0, r=0, t=70, b=0),
        legend=dict(x=0.02, y=0.98),
    )
    return fig


def create_rollout_geometry_3d(snapshot: dict[str, Any] | None, sample_idx: int = 0) -> go.Figure:
    if snapshot is None:
        return _empty_fig("Rollout vs Target 3D Geometry")
    clean = _sample_data(snapshot, "clean", sample_idx)
    tf = _sample_data(snapshot, "teacher_forced", sample_idx)
    roll = _sample_data(snapshot, "rollout", sample_idx)
    pred = _sample_data(snapshot, "pred_x0", sample_idx)
    idx = _valid_positions(snapshot, sample_idx)
    if clean.size == 0 or roll.size == 0 or idx.size == 0:
        return _empty_fig("Rollout vs Target 3D Geometry", "Snapshot has no rollout vectors")

    answer_pos = _answer_pos(snapshot, sample_idx)
    fig = go.Figure()
    _add_path3d(fig, clean, idx, "target chain", COLORS["clean"], width=7.0)
    _add_path3d(fig, tf, idx, "teacher-forced prediction", COLORS["tf"], width=4.0)
    _add_path3d(fig, pred, idx, "DF pred x0", COLORS["pred"], width=3.5)
    _add_path3d(fig, roll, idx, "free rollout", COLORS["roll"], width=6.0)
    for pos in idx:
        pts = np.vstack([clean[pos], roll[pos]])
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="lines",
            showlegend=False,
            line=dict(color="rgba(255,255,255,0.18)", width=1.8),
            hoverinfo="skip",
        ))
    if answer_pos is not None and answer_pos in set(idx.tolist()):
        for pts, name, color in [(clean, "answer target", COLORS["answer"]), (roll, "answer rollout", COLORS["roll"]), (tf, "answer TF", COLORS["tf"] )]:
            if pts.size == 0:
                continue
            p = pts[answer_pos]
            fig.add_trace(go.Scatter3d(
                x=[p[0]], y=[p[1]], z=[p[2]],
                mode="markers",
                name=name,
                marker=dict(size=10, color=color, symbol="diamond", line=dict(color="#ffffff", width=1.5)),
            ))
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=f"Autoregressive Rollout Geometry | sample {sample_idx} | step {snapshot.get('global_step')}",
        paper_bgcolor="#0b111d",
        scene=dict(
            xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3",
            bgcolor="#0f1724",
            xaxis=dict(gridcolor="rgba(180,195,255,0.16)"),
            yaxis=dict(gridcolor="rgba(180,195,255,0.16)"),
            zaxis=dict(gridcolor="rgba(180,195,255,0.16)"),
        ),
        margin=dict(l=0, r=0, t=70, b=0),
        legend=dict(x=0.02, y=0.98),
    )
    return fig


def create_noise_heatmap(snapshot: dict[str, Any] | None, sample_idx: int = 0) -> go.Figure:
    if snapshot is None:
        return _empty_fig("Noise and Denoising Heatmap")
    idx = _valid_positions(snapshot, sample_idx)
    if idx.size == 0:
        return _empty_fig("Noise and Denoising Heatmap", "Snapshot has no valid positions")

    noise = _array_data(snapshot, "noise_levels", sample_idx)[idx]
    noise_norm = noise / max(float(np.max(noise)), 1.0)
    rows_cos = []
    labels_cos = []
    for key, label in [
        ("noisy_cos", "cos(noisy, clean)"),
        ("df_cos", "cos(DF pred, clean)"),
        ("tf_cos", "cos(TF pred, clean)"),
        ("roll_cos", "cos(rollout, clean)"),
    ]:
        arr = _array_data(snapshot, key, sample_idx)
        if arr.size:
            rows_cos.append(arr[idx])
            labels_cos.append(label)
    rows_cos.append(noise_norm)
    labels_cos.append("noise level / max")

    rows_l2 = []
    labels_l2 = []
    for key, label in [
        ("noisy_l2", "L2 noisy"),
        ("df_l2", "L2 DF pred"),
        ("roll_l2", "L2 rollout"),
    ]:
        arr = _array_data(snapshot, key, sample_idx)
        if arr.size:
            rows_l2.append(arr[idx])
            labels_l2.append(label)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.08,
        subplot_titles=("Cosine quality and normalized noise", "L2 reconstruction error"),
    )
    fig.add_trace(go.Heatmap(
        z=np.vstack(rows_cos),
        x=idx,
        y=labels_cos,
        zmin=0.0,
        zmax=1.0,
        colorscale="Viridis",
        colorbar=dict(title="quality", len=0.62, y=0.72),
        hovertemplate="pos=%{x}<br>%{y}: %{z:.4f}<extra></extra>",
    ), row=1, col=1)
    if rows_l2:
        fig.add_trace(go.Heatmap(
            z=np.vstack(rows_l2),
            x=idx,
            y=labels_l2,
            colorscale="Magma",
            colorbar=dict(title="L2", len=0.25, y=0.17),
            hovertemplate="pos=%{x}<br>%{y}: %{z:.6f}<extra></extra>",
        ), row=2, col=1)
    answer_pos = _answer_pos(snapshot, sample_idx)
    if answer_pos is not None:
        fig.add_vline(x=answer_pos, line_color="#ffffff", line_dash="dash", opacity=0.85)
        fig.add_annotation(x=answer_pos, y=1.08, xref="x", yref="paper", text="answer", showarrow=False, font=dict(color="#ffffff"))
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=f"Noise / Denoising by Chain Position | sample {sample_idx}",
        paper_bgcolor="#0b111d",
        plot_bgcolor="#0f1724",
        margin=dict(l=110, r=70, t=80, b=45),
    )
    fig.update_xaxes(title_text="chain position", row=2, col=1)
    return fig


def create_token_metrics_figure(snapshot: dict[str, Any] | None, sample_idx: int = 0) -> go.Figure:
    if snapshot is None:
        return _empty_fig("Per-Token Probe Metrics")
    idx = _valid_positions(snapshot, sample_idx)
    if idx.size == 0:
        return _empty_fig("Per-Token Probe Metrics", "Snapshot has no valid positions")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for key, label, color in [
        ("noisy_cos", "noisy cos", COLORS["noisy"]),
        ("df_cos", "DF pred cos", COLORS["pred"]),
        ("tf_cos", "TF pred cos", COLORS["tf"]),
        ("roll_cos", "rollout cos", COLORS["roll"]),
    ]:
        arr = _array_data(snapshot, key, sample_idx)
        if arr.size:
            fig.add_trace(go.Scatter(x=idx, y=arr[idx], mode="lines+markers", name=label, line=dict(color=color, width=2.6)), secondary_y=False)
    noise = _array_data(snapshot, "noise_levels", sample_idx)
    if noise.size:
        fig.add_trace(go.Bar(x=idx, y=noise[idx], name="noise level", marker_color="rgba(116,192,252,0.32)"), secondary_y=True)
    answer_pos = _answer_pos(snapshot, sample_idx)
    if answer_pos is not None:
        fig.add_vline(x=answer_pos, line_color="#ffffff", line_dash="dash", opacity=0.85)
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=f"Per-Token Metrics | sample {sample_idx} | step {snapshot.get('global_step')}",
        paper_bgcolor="#0b111d",
        plot_bgcolor="#0f1724",
        hovermode="x unified",
        margin=dict(l=50, r=55, t=70, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    fig.update_xaxes(title_text="chain position", gridcolor="rgba(180,195,255,0.14)")
    fig.update_yaxes(title_text="cosine", secondary_y=False, range=[-0.05, 1.02], gridcolor="rgba(180,195,255,0.14)")
    fig.update_yaxes(title_text="noise level", secondary_y=True, gridcolor="rgba(180,195,255,0.05)")
    return fig
