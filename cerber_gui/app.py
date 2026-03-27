"""
CERBER Model Monitor - main Gradio application.

Usage:
    python cerber_gui/app.py

    # Opens http://localhost:7860
"""

import sys
import re
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

# Add project root to import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import gradio as gr
import plotly.graph_objects as go
import numpy as np

from cerber_gui.checkpoint_analyzer import (
    load_checkpoint,
    extract_metrics,
    compare_checkpoints,
    export_comparison,
    get_checkpoint_summary,
    batch_load_checkpoints,
)
from cerber_gui.metrics_viewer import (
    load_training_metrics,
    metrics_to_dataframe,
    create_loss_plot,
    create_cosine_similarity_plot,
    create_metrics_dashboard,
    compute_summary_statistics,
)
from cerber_gui.landscape_3d import (
    scan_energy_landscape_3d,
    create_surface_plot,
    create_surface_plot_matplotlib,
    create_contour_plot,
    create_comparison_plot,
    export_figure_to_html,
)
from cerber_gui.live_monitor import (
    TrainingMetricsWatcher,
    create_live_metrics_plot,
)
from cerber_gui.sota_eval import (
    SOTAEvalConfig,
    compute_distribution_suite,
    compute_manifold_knn_metrics,
)
from configs.base import Stage1Config
from cebcm.data.dataset import SONARVectorDataset
from cebcm.inference.langevin import run_langevin


# ÃƒÆ’Ã‚ÂÃƒÂ¢Ã¢â€šÂ¬Ã…â€œÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â»ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â±ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â»ÃƒÆ’Ã¢â‚¬ËœÃƒâ€¦Ã¢â‚¬â„¢ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âµ ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âµ ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸
session_state = {
    "checkpoints": {},  # path -> checkpoint data
    "current_checkpoint": None,
    "metrics_file": None,
    "watcher": None,
    "landscape_cache": {},  # (checkpoint, grid, range, noise, steps, seed) -> landscape data
    "dataset_cache": {},  # dataset_path -> SONARVectorDataset
    "runtime_metrics": {},  # checkpoint_path -> latest live inference metrics
    "sota_eval_cache": {},  # (checkpoint, noise, steps, lr, batch, bank) -> dict
    "live_plot_last_update": None,
    "live_landscape_last_epoch": None,
    "live_landscape_fig": None,
    "live_landscape_traj_fig": None,
    "live_landscape_status": "No live landscape yet",
}
SOTA_EVAL_CACHE_VERSION = 3


def update_dataclass(target, updates: dict) -> None:
    """Recursively apply dict updates to a dataclass instance."""
    for key, value in updates.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            update_dataclass(current, value)
        else:
            setattr(target, key, value)


def build_stage1_config(checkpoint: dict) -> Stage1Config:
    """Build Stage1 config from checkpoint payload with safe defaults."""
    cfg = Stage1Config()
    stage1_cfg = checkpoint.get("stage1_config")
    if isinstance(stage1_cfg, dict):
        update_dataclass(cfg, stage1_cfg)
    return cfg


def _get_default_dataset_path() -> Path | None:
    """Resolve a default local dataset for visualization parity with CLI."""
    candidates = [
        Path("data/wikitext_sonar_10k.pt"),
        Path("data/wikitext_sonar_100k.pt"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _get_dataset(path: Path) -> SONARVectorDataset:
    key = str(path.resolve())
    if key not in session_state["dataset_cache"]:
        session_state["dataset_cache"][key] = SONARVectorDataset(path)
    return session_state["dataset_cache"][key]


def _sample_reference_clean_vector(
    device: torch.device,
    target_norm: float | None,
    seed: int = 42,
) -> tuple[torch.Tensor, str]:
    """
    Sample a deterministic reference clean vector.

    Priority:
    1) real SONAR vector from local dataset (CLI parity),
    2) deterministic synthetic vector with stage-config norm.
    """
    dataset_path = _get_default_dataset_path()
    if dataset_path is not None:
        try:
            dataset = _get_dataset(dataset_path)
            # Deterministic sample index for reproducible landscape previews
            idx = seed % len(dataset)
            v_clean = dataset[idx].unsqueeze(0).to(device)
            return v_clean, "dataset"
        except Exception:
            # Fallback to synthetic vector if dataset cannot be loaded.
            pass

    dim = 1024
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    v_clean = torch.randn((1, dim), generator=generator, device=device)
    if target_norm is not None:
        v_clean = torch.nn.functional.normalize(v_clean, dim=-1) * target_norm
    return v_clean, "synthetic"


def _add_relative_noise(v: torch.Tensor, scale: float, seed: int | None = None) -> torch.Tensor:
    """Match CLI noise injection semantics for fair GUI-vs-CLI comparison."""
    norms = v.norm(dim=-1, keepdim=True)
    if seed is None:
        noise = torch.randn_like(v)
    else:
        generator = torch.Generator(device=v.device)
        generator.manual_seed(int(seed))
        noise = torch.randn(v.shape, generator=generator, device=v.device, dtype=v.dtype)
    return v + noise * scale * norms


def _resolve_scan_params(
    grid_size: float,
    range_factor: float,
    absolute_half_range: float = 0.0,
) -> tuple[int, float, float | None]:
    """Validate and normalize landscape scan controls."""
    grid = int(round(grid_size))
    grid = max(15, min(200, grid))
    rf = max(0.1, float(range_factor))
    abs_range = float(absolute_half_range)
    if abs_range <= 0.0:
        abs_range = None
    else:
        abs_range = max(0.05, min(100.0, abs_range))
    return grid, rf, abs_range


def _make_landscape_cache_key(
    checkpoint_path: str,
    grid_size: int,
    range_factor: float,
    absolute_half_range: float | None,
    noise_scale: float,
    steps: int,
    seed: int,
    eval_batch_size: int,
    eval_bank_size: int,
) -> tuple:
    return (
        checkpoint_path,
        int(grid_size),
        round(float(range_factor), 5),
        None if absolute_half_range is None else round(float(absolute_half_range), 5),
        round(float(noise_scale), 5),
        int(steps),
        int(seed),
        int(eval_batch_size),
        int(eval_bank_size),
    )


def _make_sota_eval_cache_key(
    checkpoint_path: str,
    noise_scale: float,
    steps: int,
    learning_rate: float,
    eval_batch_size: int,
    eval_bank_size: int,
) -> tuple:
    return (
        int(SOTA_EVAL_CACHE_VERSION),
        checkpoint_path,
        round(float(noise_scale), 5),
        int(steps),
        round(float(learning_rate), 8),
        int(eval_batch_size),
        int(eval_bank_size),
    )


def _invalidate_checkpoint_cache(checkpoint_path: str) -> None:
    """Drop all cached landscapes that belong to a checkpoint."""
    to_remove = []
    for key in session_state["landscape_cache"].keys():
        if isinstance(key, tuple):
            if key and key[0] == checkpoint_path:
                to_remove.append(key)
        elif key == checkpoint_path:
            to_remove.append(key)
    for key in to_remove:
        session_state["landscape_cache"].pop(key, None)
    to_remove_sota = []
    for key in session_state["sota_eval_cache"].keys():
        if not isinstance(key, tuple) or not key:
            continue
        # v2 key format: (cache_version, checkpoint_path, ...)
        if len(key) >= 2 and isinstance(key[0], int):
            if key[1] == checkpoint_path:
                to_remove_sota.append(key)
            continue
        # backward compatibility for legacy key format: (checkpoint_path, ...)
        if key[0] == checkpoint_path:
            to_remove_sota.append(key)
    for key in to_remove_sota:
        session_state["sota_eval_cache"].pop(key, None)


class _UnconditionalEnergyAdapter:
    """
    Adapter for run_langevin() to support unconditional E(x) models.

    run_langevin expects energy_fn(v_query, v_candidate) and
    energy_and_grad(v_query, v_candidate). For unconditional models we ignore
    v_query and route all computations through x=v_candidate.
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, _v_query: torch.Tensor, v_candidate: torch.Tensor) -> torch.Tensor:
        return self.model(v_candidate)

    def energy_and_grad(
        self,
        _v_query: torch.Tensor,
        v_candidate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.energy_and_grad(v_candidate)


def load_checkpoints_fn(files):
    """Load uploaded checkpoint files and refresh dropdown choices."""
    if not files:
        return "No files uploaded", gr.Dropdown(choices=[]), gr.Dropdown(choices=[])

    results = []
    errors = []

    for file in files:
        try:
            checkpoint = load_checkpoint(file.name)
            metadata = checkpoint["metadata"]

            session_state["checkpoints"][file.name] = checkpoint
            _invalidate_checkpoint_cache(file.name)

            results.append({
                "name": _safe_display_name(Path(file.name).name),
                "epoch": metadata.epoch,
                "model_type": metadata.model_type,
                "hidden_dims": metadata.energy_hidden_dims,
            })
        except Exception as e:
            errors.append(f"{_safe_display_name(Path(file.name).name)}: {e}")

    if results:
        summary = "\n".join([
            f"- {r['name']}: epoch={r['epoch']}, {r['model_type']}, hidden={r['hidden_dims']}"
            for r in results
        ])
    else:
        summary = "No checkpoints loaded."

    if errors:
        summary += "\n\nErrors:\n" + "\n".join(errors)

    dropdown_choices = list(session_state["checkpoints"].keys())

    return summary, gr.update(choices=dropdown_choices), gr.update(choices=dropdown_choices)



def _fmt_float(value: float | None, ndigits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{ndigits}f}"


def _fmt_percent(value: float | None, ndigits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{100.0 * float(value):.{ndigits}f}%"


def _fmt_signed(value: float | None, ndigits: int = 6) -> str:
    if value is None:
        return "N/A"
    v = float(value)
    if abs(v) < 10 ** (-(ndigits + 1)):
        return f"{v:+.3e}"
    return f"{v:+.{ndigits}f}"


def _fmt_hidden_chain(input_dim: int, hidden_dims: list[int]) -> str:
    dims = [str(input_dim)] + [str(int(h)) for h in hidden_dims] + ["1"]
    return " -> ".join(dims)


def _safe_display_name(raw_name: str) -> str:
    """
    Best-effort mojibake recovery for filenames/titles shown in GUI.

    Some upload paths can arrive with UTF-8 bytes decoded as latin1/cp1252.
    We keep this conservative and only replace when recovery clearly improves text.
    """
    text = str(raw_name)

    def _looks_better(candidate: str, original: str) -> bool:
        if candidate == original:
            return False
        # Typical mojibake markers become fewer after recovery.
        bad_tokens = ("\u00c3", "\u00d0", "\u00d1", "\ufffd")
        old_bad = sum(original.count(t) for t in bad_tokens)
        new_bad = sum(candidate.count(t) for t in bad_tokens)
        return new_bad < old_bad

    for src_enc, dst_enc in (("latin1", "utf-8"), ("cp1252", "utf-8"), ("cp1251", "utf-8")):
        try:
            candidate = text.encode(src_enc).decode(dst_enc)
        except Exception:
            continue
        if _looks_better(candidate, text):
            text = candidate

    return text


def _build_checkpoint_summary(checkpoint_path: str, checkpoint: dict) -> str:
    metadata = checkpoint["metadata"]
    metrics = extract_metrics(checkpoint)
    resolved_dim, resolved_hidden, resolved_norm, resolved_activation = _resolve_model_hparams(checkpoint)
    architecture = _fmt_hidden_chain(resolved_dim, resolved_hidden)
    checkpoint_name = _safe_display_name(Path(checkpoint_path).name)
    runtime = session_state["runtime_metrics"].get(checkpoint_path)

    train_loss = metrics.train_loss
    eval_loss = metrics.eval_loss
    cos_before = metrics.cos_sim_before
    cos_after = metrics.cos_sim_after
    cos_improvement = metrics.cos_improvement
    success_rate = metrics.success_rate
    used_runtime_fallback = False

    if runtime is not None:
        if cos_before is None and runtime.get("cos_before") is not None:
            cos_before = float(runtime["cos_before"])
            used_runtime_fallback = True
        if cos_after is None and runtime.get("cos_after") is not None:
            cos_after = float(runtime["cos_after"])
            used_runtime_fallback = True
        if cos_improvement is None and runtime.get("cos_improvement") is not None:
            cos_improvement = float(runtime["cos_improvement"])
            used_runtime_fallback = True
        if success_rate is None:
            sota_runtime = runtime.get("sota", {})
            if isinstance(sota_runtime, dict) and sota_runtime.get("available", False):
                if sota_runtime.get("cos_success_rate") is not None:
                    success_rate = float(sota_runtime["cos_success_rate"])
                    used_runtime_fallback = True
            elif runtime.get("cosine_success") is not None:
                success_rate = 1.0 if bool(runtime["cosine_success"]) else 0.0
                used_runtime_fallback = True

    lines = [
        f"**Checkpoint:** {checkpoint_name}",
        f"**Epoch:** {metadata.epoch}",
        f"**Model Type:** {metadata.model_type}",
        f"**Architecture:** `{architecture}`",
        f"**Normalization:** `{resolved_norm}`",
        f"**Activation:** `{resolved_activation}`",
    ]

    if runtime is not None:
        lines.extend(
            [
                "",
                "**Quick Inference Snapshot (latest run):**",
                f"- Cosine: `{runtime['cos_before']:.6f}` -> `{runtime['cos_after']:.6f}` ({_fmt_signed(runtime['cos_improvement'])})",
                f"- Energy: `{runtime['energy_noisy']:.6f}` -> `{runtime['energy_final']:.6f}` ({_fmt_signed(runtime['energy_improvement'])})",
                f"- Steps executed: `{runtime['steps_executed']}` / `{runtime['steps_requested']}`",
            ]
        )
        if runtime.get("plane_dist_before") is not None and runtime.get("plane_dist_after") is not None:
            lines.append(
                f"- 2D slice distance to reference: `{runtime['plane_dist_before']:.6f}` -> `{runtime['plane_dist_after']:.6f}` "
                f"({ _fmt_signed(runtime.get('plane_dist_improvement')) })"
            )
        if runtime.get("offplane_noisy") is not None and runtime.get("offplane_denoised") is not None:
            lines.append(
                f"- Off-plane residual (1024D): `{float(runtime['offplane_noisy']):.6f}` -> `{float(runtime['offplane_denoised']):.6f}`"
            )
        lines.append("- Note: plotted coordinates are 2D projections; primary metrics remain 1024D.")

    lines.extend(
        [
            "",
            "**Checkpoint Metrics (saved during training):**",
            f"- Train Loss: `{_fmt_float(train_loss, 6)}`",
            f"- Eval Loss: `{_fmt_float(eval_loss, 6)}`",
            f"- Cosine Before: `{_fmt_float(cos_before, 6)}`",
            f"- Cosine After: `{_fmt_float(cos_after, 6)}`",
            f"- Improvement: `{_fmt_signed(cos_improvement, 6)}`",
            f"- Success Rate: `{_fmt_percent(success_rate, 2)}`",
        ]
    )
    if used_runtime_fallback:
        lines.extend(
            [
                "- Info: unavailable checkpoint metrics were backfilled from latest runtime inference.",
            ]
        )

    if (
        metadata.energy_dim != resolved_dim
        or list(metadata.energy_hidden_dims) != list(resolved_hidden)
        or metadata.norm_mode != resolved_norm
        or metadata.activation != resolved_activation
    ):
        lines.extend(
            [
                "",
                "- Warning: metadata/config mismatch detected; architecture shown from loaded `state_dict`.",
            ]
        )

    if runtime is not None:
        lines.extend(
            [
                "",
                "**Live Inference Metrics (latest run):**",
                f"- Reference source: `{runtime['reference_source']}`",
                f"- Noise scale: `{runtime['noise_scale']:.4f}`",
                f"- Steps requested/executed: `{runtime['steps_requested']}` / `{runtime['steps_executed']}`",
                f"- Early stop: `{runtime['stopped_early']}`",
                f"- Cosine before: `{runtime['cos_before']:.6f}`",
                f"- Cosine after: `{runtime['cos_after']:.6f}`",
                f"- Cosine improvement: `{_fmt_signed(runtime['cos_improvement'])}`",
                f"- Energy(clean ref): `{runtime['energy_clean']:.6f}`",
                f"- Energy(start noisy): `{runtime['energy_noisy']:.6f}`",
                f"- Energy(final): `{runtime['energy_final']:.6f}`",
                f"- Energy improvement (start-final): `{_fmt_signed(runtime['energy_improvement'])}`",
                f"- Success (energy descent): `{runtime['energy_success']}`",
                f"- Success (cosine gain): `{runtime['cosine_success']}`",
                f"- Displacement ||x_T - x_0||: `{runtime['displacement']:.6f}`",
            ]
        )
        if runtime.get("plane_dist_before") is not None and runtime.get("plane_dist_after") is not None:
            lines.extend(
                [
                    f"- 2D slice dist(reference): `{runtime['plane_dist_before']:.6f}` -> `{runtime['plane_dist_after']:.6f}` "
                    f"({ _fmt_signed(runtime.get('plane_dist_improvement')) })",
                ]
            )
        if runtime.get("offplane_noisy") is not None and runtime.get("offplane_denoised") is not None:
            lines.extend(
                [
                    f"- Off-plane residual: `{float(runtime['offplane_noisy']):.6f}` -> `{float(runtime['offplane_denoised']):.6f}`",
                ]
            )
        sota = runtime.get("sota")
        if isinstance(sota, dict):
            if bool(sota.get("available", False)):
                lines.extend(
                    [
                        "",
                        "**SOTA Batch Eval (latest run):**",
                        f"- Eval batch / bank: `{int(sota['eval_batch_size'])}` / `{int(sota['eval_bank_size'])}`",
                        f"- Cosine before/after: `{float(sota['cos_before_mean']):.6f}` -> `{float(sota['cos_after_mean']):.6f}`",
                        f"- Cosine improvement mean: `{_fmt_signed(float(sota['cos_improvement_mean']), 8)}`",
                        f"- Cosine success rate: `{float(sota['cos_success_rate']):.2%}`",
                        f"- L2(clean,x) mean: `{float(sota.get('l2_before_mean', float('nan'))):.6f}` -> `{float(sota.get('l2_after_mean', float('nan'))):.6f}`",
                        f"- L2 improvement mean: `{_fmt_signed(float(sota.get('l2_improvement_mean', float('nan'))), 8)}`",
                        f"- L2 success rate: `{float(sota.get('l2_success_rate', float('nan'))):.2%}`",
                        f"- Energy improvement mean: `{_fmt_signed(float(sota['energy_improvement_mean']), 8)}`",
                        f"- Energy success rate: `{float(sota['energy_success_rate']):.2%}`",
                        f"- MMD (RBF): `{float(sota['mmd_rbf']):.6f}`",
                        f"- C2ST accuracy: `{float(sota['c2st_acc']):.2%}`",
                        f"- C2ST raw accuracy: `{float(sota.get('c2st_raw_acc', float('nan'))):.2%}`",
                        f"- PRDC precision/recall: `{float(sota['prdc_precision']):.4f}` / `{float(sota['prdc_recall']):.4f}`",
                        f"- PRDC density/coverage: `{float(sota['prdc_density']):.4f}` / `{float(sota['prdc_coverage']):.4f}`",
                        f"- kNN cosine top1 improvement: `{_fmt_signed(float(sota.get('knn_cos_improvement', float('nan'))), 8)}`",
                        f"- kNN L2 improvement: `{_fmt_signed(float(sota.get('knn_l2_improvement', float('nan'))), 8)}`",
                    ]
                )
            else:
                lines.extend(
                    [
                        "",
                        "**SOTA Batch Eval (latest run):**",
                        f"- Unavailable: `{sota.get('reason', 'unknown')}`",
                    ]
                )
    else:
        lines.extend(
            [
                "",
                "**Live Inference Metrics (latest run):**",
                "- Not available yet. Select checkpoint/run inference to populate.",
            ]
        )
    return "\n".join(lines)


def _render_landscape_figure(checkpoint_path: str, landscape_data: dict, vis_backend: str):
    title = f"Energy Landscape - {_safe_display_name(Path(checkpoint_path).name)}"
    if vis_backend == "matplotlib":
        return create_surface_plot_matplotlib(landscape_data, title=title)
    return create_surface_plot(landscape_data, title=title)


def select_checkpoint_fn(
    checkpoint_path,
    vis_backend="plotly",
    grid_size=40,
    range_factor=1.5,
    absolute_half_range=0.0,
    sota_eval_batch_size=32,
    sota_eval_bank_size=512,
):
    """Render checkpoint summary and default landscape preview."""
    if not checkpoint_path or checkpoint_path not in session_state["checkpoints"]:
        return "No checkpoint selected", None, "No metrics available", None

    checkpoint = session_state["checkpoints"][checkpoint_path]
    session_state["current_checkpoint"] = checkpoint_path

    grid, rf, abs_range = _resolve_scan_params(grid_size, range_factor, absolute_half_range)

    summary = _build_checkpoint_summary(checkpoint_path, checkpoint)
    landscape_fig = None
    trajectory_plot = None
    info = "Run inference to update denoising trajectory and post-inference landscape."

    try:
        landscape_data = generate_landscape_for_checkpoint(
            checkpoint_path=checkpoint_path,
            grid_size=grid,
            range_factor=rf,
            absolute_half_range=0.0 if abs_range is None else abs_range,
            sota_eval_batch_size=int(sota_eval_batch_size),
            sota_eval_bank_size=int(sota_eval_bank_size),
        )
        runtime = landscape_data.get("runtime_metrics")
        if runtime is not None:
            session_state["runtime_metrics"][checkpoint_path] = runtime
            summary = _build_checkpoint_summary(checkpoint_path, checkpoint)
            info = "Preview inference and SOTA batch eval completed."
        landscape_fig = _render_landscape_figure(checkpoint_path, landscape_data, vis_backend)
        trajectory_plot = create_trajectory_plot(landscape_data)
    except Exception as e:
        info = f"Landscape generation error: {e}"

    return summary, landscape_fig, info, trajectory_plot


def _extract_hidden_dims_from_state_dict(model_state: dict) -> list[int]:
    """
    Infer hidden dims from state_dict for both plain and spectral-norm layers.
    """
    linear_layers: list[tuple[int, int]] = []
    for key, value in model_state.items():
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            continue

        m = re.match(r"^net\.(\d+)\.weight$", key)
        if m is None:
            m = re.match(r"^net\.(\d+)\.parametrizations\.weight\.original$", key)
        if m is None:
            continue

        layer_idx = int(m.group(1))
        out_dim = int(value.shape[0])
        linear_layers.append((layer_idx, out_dim))

    if not linear_layers:
        return []

    hidden_dims: list[int] = []
    for _, out_dim in sorted(linear_layers, key=lambda x: x[0]):
        if out_dim == 1:
            break
        hidden_dims.append(out_dim)
    return hidden_dims


def _resolve_model_hparams(checkpoint: dict) -> tuple[int, list[int], str, str]:
    model_state = checkpoint["model_state"]
    cfg = checkpoint.get("config", {}) or {}
    metadata = checkpoint.get("metadata")

    dim = int(cfg.get("energy_dim", getattr(metadata, "energy_dim", 1024)))

    hidden_dims = cfg.get("energy_hidden_dims")
    if not isinstance(hidden_dims, list) or not hidden_dims:
        hidden_dims = getattr(metadata, "energy_hidden_dims", None)
    if not isinstance(hidden_dims, list) or not hidden_dims:
        hidden_dims = _extract_hidden_dims_from_state_dict(model_state)
    if not hidden_dims:
        hidden_dims = [2048, 1024, 512]
    hidden_dims = [int(h) for h in hidden_dims if int(h) > 1]

    norm_mode = str(cfg.get("norm_mode", getattr(metadata, "norm_mode", "orthonorm")))
    activation = str(cfg.get("activation", getattr(metadata, "activation", "groupsort")))

    # Infer norm mode only if checkpoint config is missing/invalid.
    if norm_mode not in {"orthonorm", "spectral_norm", "none"}:
        has_spectral = any(".parametrizations.weight.original" in k for k in model_state)
        norm_mode = "spectral_norm" if has_spectral else "orthonorm"

    return dim, hidden_dims, norm_mode, activation


def create_trajectory_plot(landscape_data: dict) -> go.Figure:
    """
    Create 2D Langevin trajectory plot using the same contour semantics as CLI.
    """
    return create_contour_plot(
        landscape_data,
        title="Langevin Dynamics Trajectory",
        colorscale="Inferno",
        show_trajectory=True,
    )

def _load_energy_model_from_checkpoint(checkpoint: dict, device: torch.device):
    """Instantiate and load an energy model from checkpoint payload."""
    from cebcm.models.energy import SimpleEnergy
    from cebcm.models.energy_unconditional import UnconditionalEnergy

    model_type = checkpoint["model_type"]
    model_state = checkpoint["model_state"]
    dim, hidden_dims, norm_mode, activation = _resolve_model_hparams(checkpoint)

    if model_type == "simple":
        model = SimpleEnergy(
            dim=dim,
            hidden_dims=hidden_dims,
            norm_mode=norm_mode,
            activation=activation,
            energy_output_clamp=None,  # keep true energy scale for visualization/debugging
        ).to(device)
    else:
        model = UnconditionalEnergy(
            dim=dim,
            hidden_dims=hidden_dims,
            norm_mode=norm_mode,
            activation=activation,
        ).to(device)

    load_result = model.load_state_dict(model_state, strict=False)
    missing = [k for k in load_result.missing_keys if k != "_sigma_freqs"]
    unexpected = [k for k in load_result.unexpected_keys]
    if missing or unexpected:
        missing_preview = ", ".join(missing[:8])
        unexpected_preview = ", ".join(unexpected[:8])
        raise RuntimeError(
            "Checkpoint/model mismatch while loading GUI model. "
            f"model_type={model_type}, dim={dim}, hidden_dims={hidden_dims}, "
            f"norm_mode={norm_mode}, activation={activation}. "
            f"Missing({len(missing)}): {missing_preview}. "
            f"Unexpected({len(unexpected)}): {unexpected_preview}."
        )

    model.eval()
    return model, model_type


def run_langevin_denoise(
    model,
    v_clean: torch.Tensor,
    v_noisy: torch.Tensor,
    model_type: str,
    stage1_cfg: Stage1Config,
    max_steps: int,
    lr_override: float | None = None,
    force_full_steps: bool = True,
    track_vectors: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor], object]:
    """
    Run Langevin denoising with Stage1-consistent math and trajectory capture.
    """
    method = stage1_cfg.langevin.method
    method_kwargs = {}
    if method == "pid":
        method_kwargs = dict(
            kp=stage1_cfg.langevin.pid_kp,
            ki=stage1_cfg.langevin.pid_ki,
            kd=stage1_cfg.langevin.pid_kd,
            integral_decay=stage1_cfg.langevin.pid_integral_decay,
        )
    elif method == "underdamped":
        method_kwargs = dict(
            friction=stage1_cfg.langevin.underdamped_friction,
            mass=stage1_cfg.langevin.underdamped_mass,
        )
    elif method == "overdamped":
        method_kwargs = dict(momentum_beta=stage1_cfg.langevin.momentum_beta)

    lr = float(lr_override) if lr_override is not None else float(stage1_cfg.langevin.lr)
    max_steps = int(max_steps)
    energy_threshold = None if force_full_steps else stage1_cfg.langevin.energy_threshold
    plateau_patience = (max_steps + 1) if force_full_steps else stage1_cfg.langevin.plateau_patience

    if model_type == "unconditional":
        energy_fn = _UnconditionalEnergyAdapter(model)
        v_query = torch.zeros_like(v_clean)
    else:
        energy_fn = model
        v_query = v_clean

    result = run_langevin(
        method=method,
        energy_fn=energy_fn,
        v_query=v_query,
        v_init=v_noisy,
        lr=lr,
        noise_scale=stage1_cfg.langevin.noise_scale,
        max_steps=max_steps,
        target_norm=stage1_cfg.langevin.target_norm,
        energy_threshold=energy_threshold,
        plateau_patience=plateau_patience,
        plateau_delta=stage1_cfg.langevin.plateau_delta,
        v_target=v_clean,
        track_vectors=track_vectors,
        **method_kwargs,
    )
    if result.v_trajectory:
        trajectory = result.v_trajectory
        # GUI should report and render the actually reached final state (last executed step),
        # not the internal "best energy" fallback, otherwise metrics and trajectory disagree.
        v_out = trajectory[-1].to(v_noisy.device)
        return v_out, trajectory, result

    # Fallback for non-tracking runs (for example batch SOTA eval).
    v_last = result.v_last if result.v_last is not None else result.v_final
    trajectory = [v_noisy.detach().cpu().clone(), v_last.detach().cpu().clone()]
    return v_last, trajectory, result



def _sample_clean_noisy_pair(
    stage1_cfg: Stage1Config,
    device: torch.device,
    noise_scale: float,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    v_clean, reference_source = _sample_reference_clean_vector(
        device=device,
        target_norm=stage1_cfg.langevin.target_norm,
        seed=seed,
    )
    v_noisy = _add_relative_noise(v_clean, float(noise_scale), seed=seed + 1)
    return v_clean, v_noisy, reference_source


@torch.no_grad()
def _compute_plane_diagnostics(
    landscape_data: dict,
    v_clean: torch.Tensor,
    v_noisy: torch.Tensor,
    v_denoised: torch.Tensor,
) -> dict[str, float]:
    clean_xy = np.asarray(landscape_data.get("clean_point", (0.0, 0.0)), dtype=np.float64)
    noisy_xy = np.asarray(landscape_data.get("noisy_point", (0.0, 0.0)), dtype=np.float64)
    denoised_xy = np.asarray(landscape_data.get("denoised_point", (0.0, 0.0)), dtype=np.float64)

    plane_dist_before = float(np.linalg.norm(noisy_xy - clean_xy))
    plane_dist_after = float(np.linalg.norm(denoised_xy - clean_xy))
    plane_dist_improvement = float(plane_dist_before - plane_dist_after)

    basis = landscape_data.get("basis")
    noisy_offplane = float("nan")
    denoised_offplane = float("nan")
    if isinstance(basis, tuple) and len(basis) == 2:
        axis1, axis2 = basis
        axis1 = axis1.to(v_clean.device, dtype=v_clean.dtype)
        axis2 = axis2.to(v_clean.device, dtype=v_clean.dtype)
        v0 = v_clean.squeeze(0)

        def _offplane_norm(v: torch.Tensor) -> float:
            delta = v.squeeze(0) - v0
            proj = (delta @ axis1) * axis1 + (delta @ axis2) * axis2
            return float((delta - proj).norm().item())

        noisy_offplane = _offplane_norm(v_noisy)
        denoised_offplane = _offplane_norm(v_denoised)

    return {
        "plane_dist_before": plane_dist_before,
        "plane_dist_after": plane_dist_after,
        "plane_dist_improvement": plane_dist_improvement,
        "offplane_noisy": noisy_offplane,
        "offplane_denoised": denoised_offplane,
    }


def _sample_dataset_vectors(
    device: torch.device,
    batch_size: int,
    seed: int,
) -> tuple[torch.Tensor | None, str]:
    dataset_path = _get_default_dataset_path()
    if dataset_path is None:
        return None, "unavailable"

    try:
        dataset = _get_dataset(dataset_path)
    except Exception:
        return None, "unavailable"

    n = len(dataset)
    if n == 0:
        return None, "unavailable"

    bs = max(1, min(int(batch_size), n))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    indices = torch.randperm(n, generator=gen)[:bs]
    vectors = dataset.embeddings[indices].to(device)
    return vectors, "dataset"


@torch.no_grad()
def _compute_energy_batch(
    model,
    model_type: str,
    v_clean: torch.Tensor,
    v_candidate: torch.Tensor,
) -> torch.Tensor:
    if model_type == "simple":
        return model(v_clean, v_candidate).detach()
    return model(v_candidate).detach()


def _compute_sota_eval_metrics(
    checkpoint_path: str,
    model,
    model_type: str,
    stage1_cfg: Stage1Config,
    noise_scale: float,
    num_steps: int,
    learning_rate: float,
    eval_batch_size: int,
    eval_bank_size: int,
) -> dict:
    key = _make_sota_eval_cache_key(
        checkpoint_path=checkpoint_path,
        noise_scale=float(noise_scale),
        steps=int(num_steps),
        learning_rate=float(learning_rate),
        eval_batch_size=int(eval_batch_size),
        eval_bank_size=int(eval_bank_size),
    )
    if key in session_state["sota_eval_cache"]:
        return session_state["sota_eval_cache"][key]

    device = next(model.parameters()).device
    v_clean_batch, source = _sample_dataset_vectors(device, int(eval_batch_size), seed=31415)
    if v_clean_batch is None:
        out = {"available": False, "reason": "dataset_unavailable"}
        session_state["sota_eval_cache"][key] = out
        return out

    v_noisy_batch = _add_relative_noise(v_clean_batch, float(noise_scale), seed=31416)
    v_denoised_batch, _, batch_result = run_langevin_denoise(
        model=model,
        v_clean=v_clean_batch,
        v_noisy=v_noisy_batch,
        model_type=model_type,
        stage1_cfg=stage1_cfg,
        max_steps=int(num_steps),
        lr_override=float(learning_rate),
        force_full_steps=True,
        track_vectors=False,
    )

    cos_before = F.cosine_similarity(v_clean_batch, v_noisy_batch, dim=-1)
    cos_after = F.cosine_similarity(v_clean_batch, v_denoised_batch, dim=-1)

    e_noisy = _compute_energy_batch(
        model=model,
        model_type=model_type,
        v_clean=v_clean_batch,
        v_candidate=v_noisy_batch,
    )
    e_final = _compute_energy_batch(
        model=model,
        model_type=model_type,
        v_clean=v_clean_batch,
        v_candidate=v_denoised_batch,
    )

    ref_bank, _ = _sample_dataset_vectors(device, int(eval_bank_size), seed=27182)
    cfg = SOTAEvalConfig(
        prdc_k=max(3, min(10, int(eval_batch_size) // 4)),
        manifold_k=max(5, min(20, int(eval_bank_size) // 16)),
        c2st_steps=100,
        c2st_lr=0.05,
        c2st_train_frac=0.8,
        mmd_subsample=min(256, int(eval_batch_size)),
    )
    suite = compute_distribution_suite(v_clean_batch, v_denoised_batch, cfg=cfg)
    if ref_bank is not None:
        suite.update(
            compute_manifold_knn_metrics(
                ref_bank=ref_bank,
                noisy=v_noisy_batch,
                denoised=v_denoised_batch,
                k=cfg.manifold_k,
            )
        )

    out = {
        "available": True,
        "reference_source": source,
        "eval_batch_size": int(v_clean_batch.shape[0]),
        "eval_bank_size": int(0 if ref_bank is None else ref_bank.shape[0]),
        "steps_executed_mean": float(batch_result.num_steps),
        "cos_before_mean": float(cos_before.mean().item()),
        "cos_after_mean": float(cos_after.mean().item()),
        "cos_improvement_mean": float((cos_after - cos_before).mean().item()),
        "cos_success_rate": float((cos_after > cos_before).float().mean().item()),
        "l2_before_mean": float((v_clean_batch - v_noisy_batch).norm(dim=-1).mean().item()),
        "l2_after_mean": float((v_clean_batch - v_denoised_batch).norm(dim=-1).mean().item()),
        "l2_improvement_mean": float(
            ((v_clean_batch - v_noisy_batch).norm(dim=-1) - (v_clean_batch - v_denoised_batch).norm(dim=-1))
            .mean()
            .item()
        ),
        "l2_success_rate": float(
            ((v_clean_batch - v_denoised_batch).norm(dim=-1) < (v_clean_batch - v_noisy_batch).norm(dim=-1))
            .float()
            .mean()
            .item()
        ),
        "denoise_step_norm_mean": float((v_denoised_batch - v_noisy_batch).norm(dim=-1).mean().item()),
        "energy_before_mean": float(e_noisy.mean().item()),
        "energy_after_mean": float(e_final.mean().item()),
        "energy_improvement_mean": float((e_noisy - e_final).mean().item()),
        "energy_success_rate": float((e_final < e_noisy).float().mean().item()),
    }
    out.update(suite)

    session_state["sota_eval_cache"][key] = out
    return out


@torch.no_grad()
def _compute_energy_triplet(
    model,
    model_type: str,
    v_clean: torch.Tensor,
    v_noisy: torch.Tensor,
    v_denoised: torch.Tensor,
) -> dict[str, float]:
    if model_type == "simple":
        e_clean = float(model(v_clean, v_clean).mean().item())
        e_noisy = float(model(v_clean, v_noisy).mean().item())
        e_denoised = float(model(v_clean, v_denoised).mean().item())
    else:
        e_clean = float(model(v_clean).mean().item())
        e_noisy = float(model(v_noisy).mean().item())
        e_denoised = float(model(v_denoised).mean().item())

    return {
        "clean": e_clean,
        "noisy": e_noisy,
        "denoised": e_denoised,
        "delta_noisy_to_denoised": e_denoised - e_noisy,
        "delta_clean_to_denoised": e_denoised - e_clean,
    }


def _compute_alignment_diagnostic(
    model,
    model_type: str,
    v_clean: torch.Tensor,
    v_noisy: torch.Tensor,
) -> float | None:
    if model_type != "unconditional":
        return None

    # Must run under enabled autograd because energy_and_grad differentiates wrt input.
    with torch.enable_grad():
        _, grad = model.energy_and_grad(v_noisy)
    target = v_clean - v_noisy
    return float(F.cosine_similarity(-grad, target, dim=-1).mean().item())


def _build_runtime_metrics(
    model,
    model_type: str,
    v_clean: torch.Tensor,
    v_noisy: torch.Tensor,
    v_denoised: torch.Tensor,
    noise_scale: float,
    steps_requested: int,
    steps_executed: int,
    stopped_early: bool,
    reference_source: str,
) -> dict:
    v_clean_np = v_clean.squeeze(0).detach().cpu().numpy()
    v_noisy_np = v_noisy.squeeze(0).detach().cpu().numpy()
    v_denoised_np = v_denoised.squeeze(0).detach().cpu().numpy()

    cos_before = float(np.dot(v_clean_np, v_noisy_np) / (np.linalg.norm(v_clean_np) * np.linalg.norm(v_noisy_np)))
    cos_after = float(np.dot(v_clean_np, v_denoised_np) / (np.linalg.norm(v_clean_np) * np.linalg.norm(v_denoised_np)))
    cos_improvement = cos_after - cos_before

    energies = _compute_energy_triplet(
        model=model,
        model_type=model_type,
        v_clean=v_clean,
        v_noisy=v_noisy,
        v_denoised=v_denoised,
    )
    displacement = float((v_denoised - v_noisy).norm().item())

    return {
        "reference_source": reference_source,
        "noise_scale": float(noise_scale),
        "steps_requested": int(steps_requested),
        "steps_executed": int(steps_executed),
        "stopped_early": bool(stopped_early),
        "cos_before": cos_before,
        "cos_after": cos_after,
        "cos_improvement": cos_improvement,
        "energy_clean": float(energies["clean"]),
        "energy_noisy": float(energies["noisy"]),
        "energy_final": float(energies["denoised"]),
        "energy_improvement": float(energies["noisy"] - energies["denoised"]),
        "energy_success": bool(energies["denoised"] < energies["noisy"]),
        "cosine_success": bool(cos_after > cos_before),
        "displacement": displacement,
    }


def _format_inference_info(
    model_type: str,
    stage1_cfg: Stage1Config,
    noise_scale: float,
    num_steps: int,
    learning_rate: float,
    trajectory_len: int,
    landscape_data: dict,
    energies: dict[str, float],
    cos_before: float,
    cos_after: float,
    alignment: float | None,
    displacement: float,
    stopped_early: bool,
    force_full_steps: bool,
    reference_source: str,
    sota_metrics: dict | None = None,
) -> str:
    energy_min = float(landscape_data.get("energy_min", np.nan))
    energy_max = float(landscape_data.get("energy_max", np.nan))

    lines = [
        "**Inference Results:**",
        f"- Model type: `{model_type}`",
        f"- Initial noise scale (relative): `{noise_scale:.4f}`",
        f"- Langevin method: `{stage1_cfg.langevin.method}`",
        f"- Steps requested: `{int(num_steps)}`",
        f"- Learning rate: `{float(learning_rate):.6f}`",
        f"- Trajectory steps executed: `{trajectory_len}`",
        f"- Early stop triggered: `{stopped_early}`",
        f"- Forced full steps (GUI debug mode): `{force_full_steps}`",
        f"- Reference source: `{reference_source}`",
        f"- Energy range on scanned plane: `[{energy_min:.4f}, {energy_max:.4f}]`",
        f"- Energy(clean ref): `{energies['clean']:.6f}`",
        f"- Energy(noisy start): `{energies['noisy']:.6f}`",
        f"- Energy(denoised/final): `{energies['denoised']:.6f}`",
        f"- Delta energy (final - start): `{energies['delta_noisy_to_denoised']:+.6f}`",
        f"- Primary improvement (start - final energy): `{(-energies['delta_noisy_to_denoised']):+.6f}`",
        f"- Delta energy (final - clean): `{energies['delta_clean_to_denoised']:+.6f}`",
        f"- Cosine(clean, noisy): `{cos_before:.6f}`",
        f"- Cosine(clean, final): `{cos_after:.6f}`",
        f"- Cosine improvement: `{_fmt_signed(cos_after - cos_before)}`",
        f"- Final displacement ||x_T - x_0||: `{displacement:.6f}`",
    ]

    if alignment is not None:
        lines.append(
            f"- Alignment diagnostic cos(-gradE(noisy), clean-noisy): `{alignment:.6f}`"
        )

    if model_type == "unconditional":
        lines.append("- Note: for unconditional models, cosine-to-clean is only a local diagnostic.")
        lines.append("- Primary criterion is energy descent and manifold-level sampling metrics.")
        lines.append("- The reference clean vector is not a mandatory target minimum for this model.")

    if reference_source != "dataset":
        lines.append("- Warning: dataset reference was unavailable; using synthetic clean vector.")
        lines.append("- This weakens any cosine-to-clean interpretation for the current run.")

    if displacement < 1e-8:
        lines.append("- Warning: final state equals start state (no-op). Check LR/noise/steps or model gradients.")
    if (
        isinstance(landscape_data, dict)
        and landscape_data.get("clean_point") is not None
        and landscape_data.get("noisy_point") is not None
        and landscape_data.get("denoised_point") is not None
    ):
        clean_xy = np.asarray(landscape_data["clean_point"], dtype=np.float64)
        noisy_xy = np.asarray(landscape_data["noisy_point"], dtype=np.float64)
        denoised_xy = np.asarray(landscape_data["denoised_point"], dtype=np.float64)
        d2_before = float(np.linalg.norm(noisy_xy - clean_xy))
        d2_after = float(np.linalg.norm(denoised_xy - clean_xy))
        lines.append(
            f"- 2D slice distance to reference: `{d2_before:.6f}` -> `{d2_after:.6f}` ({_fmt_signed(d2_before - d2_after)})"
        )
        lines.append("- Note: this is 2D projection only; 1024D cosine/energy can differ.")

    if isinstance(sota_metrics, dict):
        lines.append("")
        lines.append("**SOTA Batch Eval:**")
        if bool(sota_metrics.get("available", False)):
            lines.append(
                f"- Eval batch / bank: `{int(sota_metrics['eval_batch_size'])}` / `{int(sota_metrics['eval_bank_size'])}`"
            )
            lines.append(
                f"- Cosine before/after: `{float(sota_metrics['cos_before_mean']):.6f}` -> `{float(sota_metrics['cos_after_mean']):.6f}`"
            )
            lines.append(
                f"- Cosine improvement mean: `{_fmt_signed(float(sota_metrics['cos_improvement_mean']), 8)}`"
            )
            lines.append(
                f"- Cosine success rate: `{float(sota_metrics['cos_success_rate']):.2%}`"
            )
            lines.append(
                f"- L2(clean,x) mean: `{float(sota_metrics.get('l2_before_mean', float('nan'))):.6f}` -> "
                f"`{float(sota_metrics.get('l2_after_mean', float('nan'))):.6f}` "
                f"({ _fmt_signed(float(sota_metrics.get('l2_improvement_mean', float('nan'))), 8) })"
            )
            lines.append(
                f"- L2 success rate: `{float(sota_metrics.get('l2_success_rate', float('nan'))):.2%}`"
            )
            lines.append(
                f"- Mean denoise step ||x_T-x_0||: `{float(sota_metrics.get('denoise_step_norm_mean', float('nan'))):.6f}`"
            )
            lines.append(
                f"- Energy improvement mean: `{_fmt_signed(float(sota_metrics['energy_improvement_mean']), 8)}`"
            )
            lines.append(
                f"- Energy success rate: `{float(sota_metrics['energy_success_rate']):.2%}`"
            )
            lines.append(f"- MMD (RBF): `{float(sota_metrics['mmd_rbf']):.6f}`")
            lines.append(f"- C2ST accuracy: `{float(sota_metrics['c2st_acc']):.2%}`")
            lines.append(f"- C2ST raw accuracy: `{float(sota_metrics.get('c2st_raw_acc', float('nan'))):.2%}`")
            lines.append(
                f"- PRDC (P/R/D/C): `{float(sota_metrics['prdc_precision']):.4f}` / `{float(sota_metrics['prdc_recall']):.4f}` / "
                f"`{float(sota_metrics['prdc_density']):.4f}` / `{float(sota_metrics['prdc_coverage']):.4f}`"
            )
            if "knn_cos_improvement" in sota_metrics:
                lines.append(
                    f"- kNN cosine top1 improvement: `{_fmt_signed(float(sota_metrics['knn_cos_improvement']), 8)}`"
                )
            if "knn_l2_improvement" in sota_metrics:
                lines.append(
                    f"- kNN L2 improvement: `{_fmt_signed(float(sota_metrics['knn_l2_improvement']), 8)}`"
                )
        else:
            lines.append(f"- Unavailable: `{sota_metrics.get('reason', 'unknown')}`")

    return "\n".join(lines)


def run_inference_fn(
    checkpoint_path,
    noise_scale,
    num_steps,
    learning_rate,
    vis_backend,
    grid_size,
    range_factor,
    absolute_half_range,
    sota_eval_batch_size,
    sota_eval_bank_size,
):
    """Run denoising/refinement inference and refresh both surface + trajectory plots."""
    if not checkpoint_path or checkpoint_path not in session_state["checkpoints"]:
        return "No checkpoint selected", None, None, "No checkpoint selected"

    checkpoint = session_state["checkpoints"][checkpoint_path]
    stage1_cfg = build_stage1_config(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type = _load_energy_model_from_checkpoint(checkpoint, device)

    v_clean, v_noisy, reference_source = _sample_clean_noisy_pair(
        stage1_cfg=stage1_cfg,
        device=device,
        noise_scale=float(noise_scale),
        seed=42,
    )

    v_denoised, trajectory, langevin_result = run_langevin_denoise(
        model=model,
        v_clean=v_clean,
        v_noisy=v_noisy,
        model_type=model_type,
        stage1_cfg=stage1_cfg,
        max_steps=int(num_steps),
        lr_override=float(learning_rate),
        force_full_steps=True,
    )

    grid, rf, abs_range = _resolve_scan_params(grid_size, range_factor, absolute_half_range)

    landscape_data = scan_energy_landscape_3d(
        energy_fn=model,
        v_clean=v_clean,
        v_noisy=v_noisy,
        grid_size=grid,
        range_factor=rf,
        absolute_half_range=abs_range,
        v_denoised=v_denoised,
        trajectory=trajectory,
        model_type=model_type,
    )

    landscape_fig = _render_landscape_figure(checkpoint_path, landscape_data, vis_backend)
    trajectory_fig = create_trajectory_plot(landscape_data)

    runtime_metrics = _build_runtime_metrics(
        model=model,
        model_type=model_type,
        v_clean=v_clean,
        v_noisy=v_noisy,
        v_denoised=v_denoised,
        noise_scale=float(noise_scale),
        steps_requested=int(num_steps),
        steps_executed=int(langevin_result.num_steps),
        stopped_early=bool(langevin_result.stopped_early),
        reference_source=reference_source,
    )
    runtime_metrics.update(
        _compute_plane_diagnostics(
            landscape_data=landscape_data,
            v_clean=v_clean,
            v_noisy=v_noisy,
            v_denoised=v_denoised,
        )
    )
    runtime_metrics["sota"] = _compute_sota_eval_metrics(
        checkpoint_path=checkpoint_path,
        model=model,
        model_type=model_type,
        stage1_cfg=stage1_cfg,
        noise_scale=float(noise_scale),
        num_steps=int(num_steps),
        learning_rate=float(learning_rate),
        eval_batch_size=int(sota_eval_batch_size),
        eval_bank_size=int(sota_eval_bank_size),
    )
    session_state["runtime_metrics"][checkpoint_path] = runtime_metrics

    alignment = _compute_alignment_diagnostic(
        model=model,
        model_type=model_type,
        v_clean=v_clean,
        v_noisy=v_noisy,
    )

    info = _format_inference_info(
        model_type=model_type,
        stage1_cfg=stage1_cfg,
        noise_scale=float(noise_scale),
        num_steps=int(num_steps),
        learning_rate=float(learning_rate),
        trajectory_len=langevin_result.num_steps,
        landscape_data=landscape_data,
        energies={
            "clean": runtime_metrics["energy_clean"],
            "noisy": runtime_metrics["energy_noisy"],
            "denoised": runtime_metrics["energy_final"],
            "delta_noisy_to_denoised": runtime_metrics["energy_final"] - runtime_metrics["energy_noisy"],
            "delta_clean_to_denoised": runtime_metrics["energy_final"] - runtime_metrics["energy_clean"],
        },
        cos_before=runtime_metrics["cos_before"],
        cos_after=runtime_metrics["cos_after"],
        alignment=alignment,
        displacement=runtime_metrics["displacement"],
        stopped_early=bool(langevin_result.stopped_early),
        force_full_steps=True,
        reference_source=reference_source,
        sota_metrics=runtime_metrics.get("sota"),
    )

    summary = _build_checkpoint_summary(checkpoint_path, checkpoint)
    return info, landscape_fig, trajectory_fig, summary


def generate_landscape_for_checkpoint(
    checkpoint_path,
    grid_size=40,
    range_factor=1.5,
    absolute_half_range=0.0,
    preview_noise=0.15,
    preview_steps=50,
    sota_eval_batch_size=32,
    sota_eval_bank_size=512,
):
    """Generate deterministic landscape preview for the selected checkpoint."""
    grid, rf, abs_range = _resolve_scan_params(grid_size, range_factor, absolute_half_range)
    cache_key = _make_landscape_cache_key(
        checkpoint_path=checkpoint_path,
        grid_size=grid,
        range_factor=rf,
        absolute_half_range=abs_range,
        noise_scale=float(preview_noise),
        steps=int(preview_steps),
        seed=42,
        eval_batch_size=int(sota_eval_batch_size),
        eval_bank_size=int(sota_eval_bank_size),
    )
    if cache_key in session_state["landscape_cache"]:
        return session_state["landscape_cache"][cache_key]

    checkpoint = session_state["checkpoints"].get(checkpoint_path)
    if not checkpoint:
        raise ValueError("Checkpoint not loaded")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type = _load_energy_model_from_checkpoint(checkpoint, device)
    stage1_cfg = build_stage1_config(checkpoint)

    v_clean, v_noisy, reference_source = _sample_clean_noisy_pair(
        stage1_cfg=stage1_cfg,
        device=device,
        noise_scale=float(preview_noise),
        seed=42,
    )

    v_denoised, trajectory, result = run_langevin_denoise(
        model=model,
        v_clean=v_clean,
        v_noisy=v_noisy,
        model_type=model_type,
        stage1_cfg=stage1_cfg,
        max_steps=int(preview_steps),
        lr_override=None,
        force_full_steps=True,
    )

    landscape_data = scan_energy_landscape_3d(
        energy_fn=model,
        v_clean=v_clean,
        v_noisy=v_noisy,
        grid_size=grid,
        range_factor=rf,
        absolute_half_range=abs_range,
        v_denoised=v_denoised,
        trajectory=trajectory,
        model_type=model_type,
    )
    landscape_data["runtime_metrics"] = _build_runtime_metrics(
        model=model,
        model_type=model_type,
        v_clean=v_clean,
        v_noisy=v_noisy,
        v_denoised=v_denoised,
        noise_scale=float(preview_noise),
        steps_requested=int(preview_steps),
        steps_executed=int(result.num_steps),
        stopped_early=bool(result.stopped_early),
        reference_source=reference_source,
    )
    landscape_data["runtime_metrics"].update(
        _compute_plane_diagnostics(
            landscape_data=landscape_data,
            v_clean=v_clean,
            v_noisy=v_noisy,
            v_denoised=v_denoised,
        )
    )
    landscape_data["runtime_metrics"]["sota"] = _compute_sota_eval_metrics(
        checkpoint_path=checkpoint_path,
        model=model,
        model_type=model_type,
        stage1_cfg=stage1_cfg,
        noise_scale=float(preview_noise),
        num_steps=int(preview_steps),
        learning_rate=float(stage1_cfg.langevin.lr),
        eval_batch_size=int(sota_eval_batch_size),
        eval_bank_size=int(sota_eval_bank_size),
    )

    session_state["landscape_cache"][cache_key] = landscape_data
    return landscape_data


def compare_selected_fn(checkpoint_paths):
    """ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¡ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âµ ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¹ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â±ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¹ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂºÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¿ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²."""
    if not checkpoint_paths or len(checkpoint_paths) < 2:
        return "Select at least 2 checkpoints", None, None

    # ÃƒÆ’Ã‚ÂÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â³ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã¢â‚¬ËœÃƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¶ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¼ ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂºÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¿ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¹ ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â»ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âµ ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âµ ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â·ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â³ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã¢â‚¬ËœÃƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¶ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¹
    for path in checkpoint_paths:
        if path not in session_state["checkpoints"]:
            try:
                checkpoint = load_checkpoint(path)
                session_state["checkpoints"][path] = checkpoint
            except Exception as e:
                return f"Error loading {Path(path).name}: {e}", None, None

    # ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¡ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¼
    df = compare_checkpoints(checkpoint_paths)

    # ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¢ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â±ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â»ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°
    table_md = df.to_markdown(index=False)

    # ÃƒÆ’Ã‚ÂÃƒÂ¢Ã¢â€šÂ¬Ã…â€œÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âº ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚Â
    fig = None
    if "cos_after" in df.columns and "epoch" in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["epoch"],
            y=df["cos_after"],
            mode="lines+markers",
            name="Cosine After",
            line=dict(color="green", width=2),
        ))
        if "cos_before" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["epoch"],
                y=df["cos_before"],
                mode="lines+markers",
                name="Cosine Before",
                line=dict(color="red", width=2),
            ))
        fig.update_layout(
            title="Checkpoint Comparison",
            xaxis_title="Epoch",
            yaxis_title="Cosine Similarity",
        )

    return table_md, fig, None


def load_metrics_file_fn(file):
    """ÃƒÆ’Ã‚ÂÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â³ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã¢â‚¬ËœÃƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â·ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂºÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â° ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¹ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â»ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â° training_metrics.json."""
    if not file:
        return "No file selected", None

    try:
        session_state["metrics_file"] = file.name
        df = metrics_to_dataframe(file.name)

        # ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â·ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â´ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¼ dashboard
        if "event" in df.columns and df["event"].astype(str).isin(["batch", "epoch", "final"]).any():
            fig = create_live_metrics_plot(df, plot_type="all")
        else:
            fig = create_metrics_dashboard(df, title="Training Metrics")

        # Statistics
        stats = compute_summary_statistics(df)
        stats_text = "\n".join([f"**{k}:** {v:.4f}" if isinstance(v, float) else f"**{k}:** {v}" for k, v in stats.items()])

        return stats_text, fig
    except Exception as e:
        return f"Error: {e}", None


def _extract_latest_epoch_from_metrics_file(metrics_path: Path) -> int | None:
    if not metrics_path.exists():
        return None

    if metrics_path.suffix.lower() == ".jsonl":
        latest_epoch: int | None = None
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                epoch_val = rec.get("epoch")
                if isinstance(epoch_val, (int, float)):
                    epoch_int = int(epoch_val)
                    if latest_epoch is None or epoch_int > latest_epoch:
                        latest_epoch = epoch_int
        return latest_epoch

    try:
        payload = load_training_metrics(metrics_path)
    except Exception:
        return None

    if isinstance(payload, dict):
        epochs = payload.get("epochs")
        if isinstance(epochs, list) and epochs:
            last = epochs[-1]
            if isinstance(last, (int, float)):
                return int(last)
        epoch_val = payload.get("epoch")
        if isinstance(epoch_val, (int, float)):
            return int(epoch_val)
    return None


def _find_latest_checkpoint_path(
    checkpoint_dir: Path,
    rolling_name: str = "latest_epoch.pt",
) -> tuple[Path | None, str]:
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        return None, "Checkpoint directory not found"

    rolling_path = checkpoint_dir / rolling_name
    if rolling_path.exists():
        return rolling_path, "rolling"

    epoch_files = list(checkpoint_dir.glob("epoch_*.pt"))
    best_file: Path | None = None
    best_epoch = -1
    for file_path in epoch_files:
        m = re.match(r"^epoch_(\d+)\.pt$", file_path.name)
        if not m:
            continue
        epoch_val = int(m.group(1))
        if epoch_val > best_epoch:
            best_epoch = epoch_val
            best_file = file_path
    if best_file is not None:
        return best_file, "milestone"

    final_path = checkpoint_dir / "final.pt"
    if final_path.exists():
        return final_path, "final"
    best_path = checkpoint_dir / "best.pt"
    if best_path.exists():
        return best_path, "best"
    return None, "No checkpoint files found"


def _generate_live_landscape_from_checkpoint(
    checkpoint_path: str,
    vis_backend: str,
    grid_size: float,
    range_factor: float,
    absolute_half_range: float,
    preview_noise: float,
    preview_steps: int,
    sota_eval_batch_size: int,
    sota_eval_bank_size: int,
) -> tuple[str, Any, Any]:
    checkpoint = load_checkpoint(checkpoint_path)
    session_state["checkpoints"][checkpoint_path] = checkpoint
    _invalidate_checkpoint_cache(checkpoint_path)

    landscape_data = generate_landscape_for_checkpoint(
        checkpoint_path,
        grid_size=grid_size,
        range_factor=range_factor,
        absolute_half_range=absolute_half_range,
        preview_noise=preview_noise,
        preview_steps=preview_steps,
        sota_eval_batch_size=sota_eval_batch_size,
        sota_eval_bank_size=sota_eval_bank_size,
    )
    landscape_fig = _render_landscape_figure(checkpoint_path, landscape_data, vis_backend)
    trajectory_fig = create_trajectory_plot(landscape_data)

    metadata = checkpoint.get("metadata")
    epoch_display = getattr(metadata, "epoch", "?") if metadata is not None else checkpoint.get("epoch", "?")
    status = (
        f"Landscape from: `{_safe_display_name(Path(checkpoint_path).name)}` | "
        f"epoch: `{epoch_display}`"
    )
    return status, landscape_fig, trajectory_fig


def check_live_landscape_fn(
    metrics_path,
    checkpoint_dir,
    vis_backend,
    grid_size,
    range_factor,
    absolute_half_range,
    preview_noise,
    preview_steps,
    auto_every_epochs,
    sota_eval_batch_size,
    sota_eval_bank_size,
):
    if not checkpoint_dir:
        return "Checkpoint directory is required", None, None

    ckpt_dir = Path(checkpoint_dir)
    ckpt_path, reason = _find_latest_checkpoint_path(ckpt_dir, rolling_name="latest_epoch.pt")
    if ckpt_path is None:
        return f"Cannot find checkpoint: {reason}", None, None

    try:
        status, landscape_fig, trajectory_fig = _generate_live_landscape_from_checkpoint(
            checkpoint_path=str(ckpt_path),
            vis_backend=vis_backend,
            grid_size=float(grid_size),
            range_factor=float(range_factor),
            absolute_half_range=float(absolute_half_range),
            preview_noise=float(preview_noise),
            preview_steps=int(preview_steps),
            sota_eval_batch_size=int(sota_eval_batch_size),
            sota_eval_bank_size=int(sota_eval_bank_size),
        )
        latest_epoch = _extract_latest_epoch_from_metrics_file(Path(metrics_path)) if metrics_path else None
        session_state["live_landscape_last_epoch"] = latest_epoch
        session_state["live_landscape_fig"] = landscape_fig
        session_state["live_landscape_traj_fig"] = trajectory_fig
        session_state["live_landscape_status"] = (
            f"{status} | source={reason} | auto every {int(auto_every_epochs)} epochs"
        )
        return session_state["live_landscape_status"], landscape_fig, trajectory_fig
    except Exception as e:
        return f"Landscape update failed: {e}", session_state.get("live_landscape_fig"), session_state.get("live_landscape_traj_fig")


def auto_update_live_landscape_fn(
    metrics_path,
    checkpoint_dir,
    vis_backend,
    grid_size,
    range_factor,
    absolute_half_range,
    preview_noise,
    preview_steps,
    auto_every_epochs,
    auto_enabled,
    sota_eval_batch_size,
    sota_eval_bank_size,
):
    if not auto_enabled:
        return (
            session_state.get("live_landscape_status", "Auto-update disabled"),
            session_state.get("live_landscape_fig"),
            session_state.get("live_landscape_traj_fig"),
        )

    if not metrics_path or not checkpoint_dir:
        return (
            session_state.get("live_landscape_status", "Waiting for metrics/checkpoints path"),
            session_state.get("live_landscape_fig"),
            session_state.get("live_landscape_traj_fig"),
        )

    latest_epoch = _extract_latest_epoch_from_metrics_file(Path(metrics_path))
    if latest_epoch is None:
        return (
            session_state.get("live_landscape_status", "Waiting for first epoch in metrics"),
            session_state.get("live_landscape_fig"),
            session_state.get("live_landscape_traj_fig"),
        )

    every = max(1, int(auto_every_epochs))
    if latest_epoch % every != 0:
        return (
            session_state.get("live_landscape_status", f"Latest epoch {latest_epoch}: waiting for multiple of {every}"),
            session_state.get("live_landscape_fig"),
            session_state.get("live_landscape_traj_fig"),
        )

    if session_state.get("live_landscape_last_epoch") == latest_epoch:
        return (
            session_state.get("live_landscape_status", f"Landscape already updated for epoch {latest_epoch}"),
            session_state.get("live_landscape_fig"),
            session_state.get("live_landscape_traj_fig"),
        )

    return check_live_landscape_fn(
        metrics_path=metrics_path,
        checkpoint_dir=checkpoint_dir,
        vis_backend=vis_backend,
        grid_size=grid_size,
        range_factor=range_factor,
        absolute_half_range=absolute_half_range,
        preview_noise=preview_noise,
        preview_steps=preview_steps,
        auto_every_epochs=auto_every_epochs,
        sota_eval_batch_size=sota_eval_batch_size,
        sota_eval_bank_size=sota_eval_bank_size,
    )


def start_live_monitor_fn(metrics_path):
    """ÃƒÆ’Ã‚ÂÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¿ÃƒÆ’Ã¢â‚¬ËœÃƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âº live ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¼ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â³ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°."""
    if not metrics_path:
        return "No path specified", None

    path = Path(metrics_path)
    if not path.exists():
        return f"File not found: {path}", None

    try:
        # Stop previous watcher to avoid duplicate observers on file changes.
        old_watcher = session_state.get("watcher")
        if old_watcher is not None:
            try:
                old_watcher.stop()
            except Exception:
                pass

        watcher = TrainingMetricsWatcher(path)
        watcher.start()
        session_state["watcher"] = watcher
        session_state["live_plot_last_update"] = None
        df = watcher.history.to_dataframe()
        fig = create_live_metrics_plot(df, plot_type="all") if not df.empty else None
        return f"Monitoring started: {path}", fig
    except Exception as e:
        return f"Error: {e}", None


def update_live_plot_fn():
    """ÃƒÆ’Ã‚ÂÃƒâ€¦Ã‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â±ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â»ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âµ live ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â³ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂºÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â° (ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¹ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â·ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¹ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚Â ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¿ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¹ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¼ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã¢â‚¬ËœÃƒâ€ Ã¢â‚¬â„¢)."""
    watcher = session_state.get("watcher")
    if not watcher:
        return None

    status = watcher.get_status()
    update_key = status.last_update.isoformat() if status.last_update is not None else None
    if update_key is not None and update_key == session_state.get("live_plot_last_update"):
        return gr.update()

    df = watcher.history.to_dataframe()
    if df.empty:
        return None

    fig = create_live_metrics_plot(df, plot_type="all")
    session_state["live_plot_last_update"] = update_key
    return fig


def export_comparison_fn(checkpoint_paths, output_format):
    """ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â­ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂºÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¿ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚Â ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂºÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¿ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²."""
    if not checkpoint_paths:
        return "No checkpoints selected"

    try:
        df = compare_checkpoints(checkpoint_paths)
        output_path = Path("cerber_gui/exports") / f"comparison_{len(checkpoint_paths)}_checkpoints.{output_format}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_comparison(df, output_path)
        return f"Exported to {output_path}"
    except Exception as e:
        return f"Error: {e}"


# === Gradio UI ===

with gr.Blocks(title="CERBER Model Monitor") as demo:
    gr.Markdown(
        """
    # CERBER Model Monitor

    Unified tool for checkpoint analysis, training metrics, and 3D energy landscape inspection.
    """
    )

    with gr.Tabs():
        # === Tab 1: Checkpoint Analysis ===
        with gr.TabItem("Checkpoint Analysis"):
            gr.Markdown("### Upload Checkpoints")

            with gr.Row():
                with gr.Column(scale=1):
                    file_upload = gr.File(
                        label="Upload Checkpoints (.pt)",
                        file_count="multiple",
                        file_types=[".pt", ".pth"],
                    )
                    load_btn = gr.Button("Load Checkpoints", variant="primary")

                with gr.Column(scale=2):
                    load_output = gr.Textbox(label="Load Results", lines=5)

            gr.Markdown("### Select Checkpoint")

            with gr.Row():
                checkpoint_dropdown = gr.Dropdown(
                    label="Select Checkpoint",
                    choices=[],
                    interactive=True,
                )

            with gr.Row():
                vis_backend_radio = gr.Radio(
                    choices=["plotly", "matplotlib"],
                    value="plotly",
                    label="Visualization Backend",
                    info="Plotly: interactive 3D. Matplotlib: static fallback renderer.",
                )
                grid_size_slider = gr.Slider(
                    minimum=20,
                    maximum=120,
                    value=40,
                    step=2,
                    label="Landscape Grid Size",
                    info="Higher value gives more detail but slower scans.",
                )
                range_factor_slider = gr.Slider(
                    minimum=0.5,
                    maximum=12.0,
                    value=1.5,
                    step=0.1,
                    label="Landscape Range Factor",
                    info="Controls how wide the explored 2D slice is.",
                )
                landscape_abs_range_slider = gr.Slider(
                    minimum=0.0,
                    maximum=100.0,
                    value=0.0,
                    step=0.1,
                    label="Landscape Half-Range (Absolute)",
                    info="0 = auto from Range Factor; >0 forces explicit axis span for Direction 1/2.",
                )

            with gr.Row():
                checkpoint_summary = gr.Markdown()

            with gr.Row():
                landscape_plot = gr.Plot(label="3D Energy Landscape", scale=2)
                trajectory_plot = gr.Plot(label="Langevin Trajectory", scale=1)

            gr.Markdown("### Inference Settings")

            with gr.Row():
                noise_scale_slider = gr.Slider(
                    minimum=0.05,
                    maximum=0.5,
                    value=0.15,
                    step=0.01,
                    label="Noise Scale",
                )
                num_steps_slider = gr.Slider(
                    minimum=10,
                    maximum=200,
                    value=50,
                    step=10,
                    label="Langevin Steps",
                )
                lr_slider = gr.Slider(
                    minimum=0.001,
                    maximum=0.1,
                    value=0.01,
                    step=0.001,
                    label="Learning Rate",
                )
                run_inference_btn = gr.Button("Run Inference", variant="primary")

            with gr.Row():
                sota_eval_batch_size_slider = gr.Slider(
                    minimum=8,
                    maximum=256,
                    value=32,
                    step=8,
                    label="SOTA Eval Batch Size",
                    info="Number of clean/noisy pairs for batched distribution evaluation.",
                )
                sota_eval_bank_size_slider = gr.Slider(
                    minimum=64,
                    maximum=4096,
                    value=512,
                    step=64,
                    label="SOTA Eval Reference Bank Size",
                    info="Reference manifold bank size for kNN-based distribution diagnostics.",
                )

            with gr.Row():
                inference_output = gr.Markdown()

        # === Tab 2: Comparison ===
        with gr.TabItem("Comparison"):
            gr.Markdown("### Checkpoint Comparison")

            with gr.Row():
                compare_dropdown = gr.Dropdown(
                    label="Select Checkpoints (min 2)",
                    choices=[],
                    multiselect=True,
                    interactive=True,
                )

            with gr.Row():
                compare_btn = gr.Button("Compare", variant="primary")
                export_format = gr.Radio(choices=["json", "csv"], value="json", label="Export Format")
                export_btn = gr.Button("Export Comparison")

            with gr.Row():
                compare_output = gr.Textbox(label="Comparison Results", lines=10)
                compare_plot = gr.Plot(label="Comparison Chart")

            export_output = gr.Textbox(label="Export Result")
            compare_status = gr.Textbox(label="Status", visible=False)

        # === Tab 3: Metrics ===
        with gr.TabItem("Training Metrics"):
            gr.Markdown("### Upload training metrics (`.json` or `.jsonl`)")

            with gr.Row():
                metrics_upload = gr.File(
                    label="Upload Metrics File",
                    file_types=[".json", ".jsonl"],
                )

            with gr.Row():
                metrics_stats = gr.Markdown()
                metrics_plot = gr.Plot(label="Metrics Dashboard")

        # === Tab 4: Live Monitor ===
        with gr.TabItem("Live Monitor"):
            gr.Markdown(
                """
            ### Real-time Training Monitor

            Provide a path to a metrics file being updated during training.
            Stage 1.5 writes `training_metrics.jsonl` (stream of batch/epoch events).
            """
            )

            with gr.Row():
                live_path_input = gr.Textbox(
                    label="Path to training metrics (`.json` or `.jsonl`)",
                    placeholder="experiments/03_Stage_1.5/logs/training_metrics.jsonl",
                )
                start_live_btn = gr.Button("Start Monitoring", variant="primary")

            live_status = gr.Textbox(label="Status")
            live_plot = gr.Plot(label="Live Metrics")

            gr.Markdown("### Live 3D Landscape")
            with gr.Row():
                live_ckpt_dir_input = gr.Textbox(
                    label="Checkpoint Directory",
                    value="experiments/03_Stage_1.5/checkpoints",
                    placeholder="experiments/03_Stage_1.5/checkpoints",
                )
                check_landscape_btn = gr.Button("Check Landscape", variant="secondary")

            with gr.Row():
                live_landscape_backend = gr.Radio(
                    choices=["plotly", "matplotlib"],
                    value="plotly",
                    label="Landscape Backend",
                )
                live_auto_landscape = gr.Checkbox(
                    value=True,
                    label="Auto-update landscape",
                    info="Refresh when new epoch is multiple of N",
                )
                live_auto_every_epochs = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Auto Every N Epochs",
                )

            with gr.Row():
                live_landscape_grid = gr.Slider(
                    minimum=20,
                    maximum=120,
                    value=40,
                    step=2,
                    label="Landscape Grid Size",
                )
                live_landscape_range = gr.Slider(
                    minimum=0.5,
                    maximum=12.0,
                    value=1.5,
                    step=0.1,
                    label="Landscape Range Factor",
                )
                live_landscape_abs_range = gr.Slider(
                    minimum=0.0,
                    maximum=100.0,
                    value=0.0,
                    step=0.1,
                    label="Landscape Half-Range (Absolute)",
                )
                live_preview_noise = gr.Slider(
                    minimum=0.05,
                    maximum=0.5,
                    value=0.15,
                    step=0.01,
                    label="Preview Noise",
                )
                live_preview_steps = gr.Slider(
                    minimum=10,
                    maximum=200,
                    value=50,
                    step=10,
                    label="Preview Langevin Steps",
                )

            with gr.Row():
                live_sota_eval_batch_size = gr.Slider(
                    minimum=8,
                    maximum=256,
                    value=32,
                    step=8,
                    label="Landscape Eval Batch Size",
                )
                live_sota_eval_bank_size = gr.Slider(
                    minimum=64,
                    maximum=4096,
                    value=512,
                    step=64,
                    label="Landscape Eval Bank Size",
                )

            live_landscape_status = gr.Textbox(label="Landscape Status")
            with gr.Row():
                live_landscape_plot = gr.Plot(label="Live 3D Landscape", scale=2)
                live_landscape_traj_plot = gr.Plot(label="Live Langevin Trajectory", scale=1)

            # Auto-refresh every 5 seconds
            live_timer = gr.Timer(value=5, active=True)
    # === Event Handlers ===

    # ÃƒÆ’Ã‚ÂÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â³ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã¢â‚¬ËœÃƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â·ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂºÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â° ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂºÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¿ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²
    load_btn.click(
        load_checkpoints_fn,
        inputs=[file_upload],
        outputs=[load_output, checkpoint_dropdown, compare_dropdown],
    )

    # ÃƒÆ’Ã‚ÂÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¹ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â±ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂºÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¿ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°
    checkpoint_dropdown.change(
        select_checkpoint_fn,
        inputs=[
            checkpoint_dropdown,
            vis_backend_radio,
            grid_size_slider,
            range_factor_slider,
            landscape_abs_range_slider,
            sota_eval_batch_size_slider,
            sota_eval_bank_size_slider,
        ],
        outputs=[checkpoint_summary, landscape_plot, inference_output, trajectory_plot],
    )

    # ÃƒÆ’Ã‚ÂÃƒâ€¦Ã‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â±ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â»ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âµ ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¿ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¼ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âµ backend
    vis_backend_radio.change(
        select_checkpoint_fn,
        inputs=[
            checkpoint_dropdown,
            vis_backend_radio,
            grid_size_slider,
            range_factor_slider,
            landscape_abs_range_slider,
            sota_eval_batch_size_slider,
            sota_eval_bank_size_slider,
        ],
        outputs=[checkpoint_summary, landscape_plot, inference_output, trajectory_plot],
    )

    grid_size_slider.change(
        select_checkpoint_fn,
        inputs=[
            checkpoint_dropdown,
            vis_backend_radio,
            grid_size_slider,
            range_factor_slider,
            landscape_abs_range_slider,
            sota_eval_batch_size_slider,
            sota_eval_bank_size_slider,
        ],
        outputs=[checkpoint_summary, landscape_plot, inference_output, trajectory_plot],
    )

    range_factor_slider.change(
        select_checkpoint_fn,
        inputs=[
            checkpoint_dropdown,
            vis_backend_radio,
            grid_size_slider,
            range_factor_slider,
            landscape_abs_range_slider,
            sota_eval_batch_size_slider,
            sota_eval_bank_size_slider,
        ],
        outputs=[checkpoint_summary, landscape_plot, inference_output, trajectory_plot],
    )

    landscape_abs_range_slider.change(
        select_checkpoint_fn,
        inputs=[
            checkpoint_dropdown,
            vis_backend_radio,
            grid_size_slider,
            range_factor_slider,
            landscape_abs_range_slider,
            sota_eval_batch_size_slider,
            sota_eval_bank_size_slider,
        ],
        outputs=[checkpoint_summary, landscape_plot, inference_output, trajectory_plot],
    )

    sota_eval_batch_size_slider.change(
        select_checkpoint_fn,
        inputs=[
            checkpoint_dropdown,
            vis_backend_radio,
            grid_size_slider,
            range_factor_slider,
            landscape_abs_range_slider,
            sota_eval_batch_size_slider,
            sota_eval_bank_size_slider,
        ],
        outputs=[checkpoint_summary, landscape_plot, inference_output, trajectory_plot],
    )

    sota_eval_bank_size_slider.change(
        select_checkpoint_fn,
        inputs=[
            checkpoint_dropdown,
            vis_backend_radio,
            grid_size_slider,
            range_factor_slider,
            landscape_abs_range_slider,
            sota_eval_batch_size_slider,
            sota_eval_bank_size_slider,
        ],
        outputs=[checkpoint_summary, landscape_plot, inference_output, trajectory_plot],
    )

    # ÃƒÆ’Ã‚ÂÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¿ÃƒÆ’Ã¢â‚¬ËœÃƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âº ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°
    run_inference_btn.click(
        run_inference_fn,
        inputs=[
            checkpoint_dropdown,
            noise_scale_slider,
            num_steps_slider,
            lr_slider,
            vis_backend_radio,
            grid_size_slider,
            range_factor_slider,
            landscape_abs_range_slider,
            sota_eval_batch_size_slider,
            sota_eval_bank_size_slider,
        ],
        outputs=[inference_output, landscape_plot, trajectory_plot, checkpoint_summary],
    )

    # ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¡ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âµ
    compare_btn.click(
        compare_selected_fn,
        inputs=[compare_dropdown],
        outputs=[compare_output, compare_plot, compare_status],
    )

    # ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â­ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂºÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¿ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡
    export_btn.click(
        export_comparison_fn,
        inputs=[compare_dropdown, export_format],
        outputs=[export_output],
    )

    # ÃƒÆ’Ã‚ÂÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â³ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã¢â‚¬ËœÃƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â·ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂºÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â° ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¼ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âº
    metrics_upload.change(
        load_metrics_file_fn,
        inputs=[metrics_upload],
        outputs=[metrics_stats, metrics_plot],
    )

    # Live ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¼ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â³
    start_live_btn.click(
        start_live_monitor_fn,
        inputs=[live_path_input],
        outputs=[live_status, live_plot],
    )

    check_landscape_btn.click(
        check_live_landscape_fn,
        inputs=[
            live_path_input,
            live_ckpt_dir_input,
            live_landscape_backend,
            live_landscape_grid,
            live_landscape_range,
            live_landscape_abs_range,
            live_preview_noise,
            live_preview_steps,
            live_auto_every_epochs,
            live_sota_eval_batch_size,
            live_sota_eval_bank_size,
        ],
        outputs=[live_landscape_status, live_landscape_plot, live_landscape_traj_plot],
    )

    live_timer.tick(
        update_live_plot_fn,
        outputs=[live_plot],
    )

    live_timer.tick(
        auto_update_live_landscape_fn,
        inputs=[
            live_path_input,
            live_ckpt_dir_input,
            live_landscape_backend,
            live_landscape_grid,
            live_landscape_range,
            live_landscape_abs_range,
            live_preview_noise,
            live_preview_steps,
            live_auto_every_epochs,
            live_auto_landscape,
            live_sota_eval_batch_size,
            live_sota_eval_bank_size,
        ],
        outputs=[live_landscape_status, live_landscape_plot, live_landscape_traj_plot],
    )

    demo.load(
        fn=lambda: None,
        inputs=None,
        outputs=None,
        js="""
        () => {
          let lastY = window.scrollY || 0;
          let rafPending = false;
          window.addEventListener("scroll", () => { lastY = window.scrollY || 0; }, { passive: true });
          const restoreScroll = () => {
            if (rafPending) return;
            rafPending = true;
            requestAnimationFrame(() => {
              window.scrollTo(0, lastY);
              rafPending = false;
            });
          };
          const attachObserver = () => {
            const app = document.querySelector("gradio-app");
            const root = app && app.shadowRoot ? app.shadowRoot : document.body;
            if (!root) return;
            const mo = new MutationObserver(() => restoreScroll());
            mo.observe(root, { childList: true, subtree: true });
          };
          attachObserver();
          setTimeout(attachObserver, 1500);
        }
        """,
    )


if __name__ == "__main__":
    # ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â·ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â´ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¼ ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â´ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂºÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã¢â‚¬ËœÃƒâ€¦Ã‚Â½ ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â´ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â»ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚Â ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂºÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¿ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°
    (Path(__file__).parent / "exports").mkdir(exist_ok=True)

    demo.queue(max_size=10)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
    )







