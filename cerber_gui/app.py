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

# Add project root to import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
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
from configs.base import Stage1Config
from cebcm.data.dataset import SONARVectorDataset
from cebcm.inference.langevin import run_langevin


# ÃƒÆ’Ã‚ÂÃƒÂ¢Ã¢â€šÂ¬Ã…â€œÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â»ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â±ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â»ÃƒÆ’Ã¢â‚¬ËœÃƒâ€¦Ã¢â‚¬â„¢ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âµ ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âµ ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸
session_state = {
    "checkpoints": {},  # path -> checkpoint data
    "current_checkpoint": None,
    "metrics_file": None,
    "watcher": None,
    "landscape_cache": {},  # checkpoint_path -> landscape data
    "dataset_cache": {},  # dataset_path -> SONARVectorDataset
}


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
) -> torch.Tensor:
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
            return v_clean
        except Exception:
            # Fallback to synthetic vector if dataset cannot be loaded.
            pass

    dim = 1024
    v_clean = torch.randn(1, dim, device=device)
    if target_norm is not None:
        v_clean = torch.nn.functional.normalize(v_clean, dim=-1) * target_norm
    return v_clean


def _add_relative_noise(v: torch.Tensor, scale: float) -> torch.Tensor:
    """Match CLI noise injection semantics for fair GUI-vs-CLI comparison."""
    norms = v.norm(dim=-1, keepdim=True)
    return v + torch.randn_like(v) * scale * norms


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
    """ÃƒÆ’Ã‚ÂÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â³ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã¢â‚¬ËœÃƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â·ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂºÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â° ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂºÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¿ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â² ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â· uploaded ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¹ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â»ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²."""
    if not files:
        return "No files uploaded", gr.Dropdown(choices=[]), gr.Dropdown(choices=[])

    results = []
    errors = []

    for file in files:
        try:
            # Gradio ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¿ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â´ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¹ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â»ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¹ ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂºÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âº tempfile
            checkpoint = load_checkpoint(file.name)
            metadata = checkpoint["metadata"]

            session_state["checkpoints"][file.name] = checkpoint
            session_state["landscape_cache"].pop(file.name, None)

            results.append({
                "name": Path(file.name).name,
                "epoch": metadata.epoch,
                "model_type": metadata.model_type,
                "hidden_dims": metadata.energy_hidden_dims,
            })
        except Exception as e:
            errors.append(f"{Path(file.name).name}: {e}")

    # ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¤ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¼ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã¢â‚¬ËœÃƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¼ summary
    if results:
        summary = "\n".join([
            f"ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ {r['name']} ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â epoch {r['epoch']}, {r['model_type']}, hidden={r['hidden_dims']}"
            for r in results
        ])
    else:
        summary = ""

    if errors:
        summary += "\n\nErrors:\n" + "\n".join(errors)

    # ÃƒÆ’Ã‚ÂÃƒâ€¦Ã‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â±ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â»ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¼ dropdown
    dropdown_choices = list(session_state["checkpoints"].keys())

    return summary, gr.update(choices=dropdown_choices), gr.update(choices=dropdown_choices)


def select_checkpoint_fn(checkpoint_path, vis_backend="plotly"):
    """ÃƒÆ’Ã‚ÂÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¹ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â±ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂºÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¿ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â° ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â´ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â»ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚Â ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â»ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â·ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°."""
    if not checkpoint_path or checkpoint_path not in session_state["checkpoints"]:
        return "No checkpoint selected", None, "No metrics available", None

    checkpoint = session_state["checkpoints"][checkpoint_path]
    metadata = checkpoint["metadata"]
    metrics = extract_metrics(checkpoint)

    session_state["current_checkpoint"] = checkpoint_path

    # Summary
    cos_before_str = f"{metrics.cos_sim_before:.4f}" if metrics.cos_sim_before is not None else "N/A"
    cos_after_str = f"{metrics.cos_sim_after:.4f}" if metrics.cos_sim_after is not None else "N/A"
    improvement_str = f"{metrics.cos_improvement:.4f}" if metrics.cos_improvement is not None else "N/A"
    success_rate_str = f"{metrics.success_rate}" if metrics.success_rate is not None else "N/A"

    summary = f"""
**Checkpoint:** {Path(checkpoint_path).name}
**Epoch:** {metadata.epoch}
**Model Type:** {metadata.model_type}
**Architecture:** {metadata.energy_dim} ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ {metadata.energy_hidden_dims} ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ 1
**Normalization:** {metadata.norm_mode}
**Activation:** {metadata.activation}

**Metrics:**
- Train Loss: {metrics.train_loss}
- Eval Loss: {metrics.eval_loss}
- Cosine Before: {cos_before_str}
- Cosine After: {cos_after_str}
- Improvement: {improvement_str}
- Success Rate: {success_rate_str}
""".strip()

    # 3D ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â»ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â´ÃƒÆ’Ã¢â‚¬ËœÃƒâ€¹Ã¢â‚¬Â ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡
    landscape_fig = None
    inference_info = ""
    trajectory_plot = None

    try:
        landscape_data = generate_landscape_for_checkpoint(checkpoint_path)
        session_state["landscape_cache"][checkpoint_path] = landscape_data

        # ÃƒÆ’Ã‚ÂÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¹ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â±ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¼ backend ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â´ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â»ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚Â ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â·ÃƒÆ’Ã¢â‚¬ËœÃƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â»ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â·ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸
        if vis_backend == "matplotlib":
            # Matplotlib backend ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â·ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¼ ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â·ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â±ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¶ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âµ
            landscape_fig = create_surface_plot_matplotlib(
                landscape_data,
                title=f"Energy Landscape ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â {Path(checkpoint_path).name}",
            )
        else:
            # Plotly backend ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂºÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¹ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¹ ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â³ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âº
            try:
                landscape_fig = create_surface_plot(
                    landscape_data,
                    title=f"Energy Landscape ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â {Path(checkpoint_path).name}",
                )
            except Exception as plotly_err:
                # Fallback so the UI stays usable even with strict Plotly schema/version mismatch.
                landscape_fig = create_surface_plot_matplotlib(
                    landscape_data,
                    title=f"Energy Landscape ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â {Path(checkpoint_path).name} (matplotlib fallback)",
                )
                summary += f"\n\n**Plotly Error:** {plotly_err}\nFell back to matplotlib rendering."

        # ÃƒÆ’Ã‚ÂÃƒâ€¹Ã…â€œÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¼ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚Â ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â± ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âµ
        if landscape_data.get("v_denoised") is not None:
            v_clean = landscape_data["v_clean"]
            v_noisy = landscape_data["v_noisy"]
            v_denoised = landscape_data["v_denoised"]

            cos_before = float(np.dot(v_clean, v_noisy) / (np.linalg.norm(v_clean) * np.linalg.norm(v_noisy)))
            cos_after = float(np.dot(v_clean, v_denoised) / (np.linalg.norm(v_clean) * np.linalg.norm(v_denoised)))
            improvement = cos_after - cos_before

            inference_info = f"""
**Inference Results:**
- Cosine (clean, noisy): {cos_before:.4f}
- Cosine (clean, denoised): {cos_after:.4f}
- Improvement: {improvement:+.4f}
- Trajectory steps: {len(landscape_data.get('trajectory_2d', [])) or 0}
"""
            # ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â·ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â´ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¼ ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â³ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âº ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂºÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ (ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â³ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â´ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â° Plotly ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â´ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â»ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚Â ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂºÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸)
            trajectory_plot = create_trajectory_plot(landscape_data)

    except Exception as e:
        summary += f"\n\n**Landscape Error:** {e}"

    return summary, landscape_fig, inference_info, trajectory_plot


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
) -> tuple[torch.Tensor, list[torch.Tensor]]:
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
        energy_threshold=stage1_cfg.langevin.energy_threshold,
        plateau_patience=stage1_cfg.langevin.plateau_patience,
        plateau_delta=stage1_cfg.langevin.plateau_delta,
        v_target=v_clean,
        track_vectors=True,
        **method_kwargs,
    )
    return result.v_final, result.v_trajectory


def run_inference_fn(checkpoint_path, noise_scale, num_steps, learning_rate):
    """
    Run denoising inference and return trajectory plot.
    """
    if not checkpoint_path or checkpoint_path not in session_state["checkpoints"]:
        return "No checkpoint selected", None

    checkpoint = session_state["checkpoints"][checkpoint_path]
    stage1_cfg = build_stage1_config(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type = _load_energy_model_from_checkpoint(checkpoint, device)

    # Match CLI semantics: real SONAR sample when available, else deterministic fallback.
    v_clean = _sample_reference_clean_vector(
        device=device,
        target_norm=stage1_cfg.langevin.target_norm,
        seed=42,
    )
    v_noisy = _add_relative_noise(v_clean, float(noise_scale))

    v_denoised, trajectory = run_langevin_denoise(
        model=model,
        v_clean=v_clean,
        v_noisy=v_noisy,
        model_type=model_type,
        stage1_cfg=stage1_cfg,
        max_steps=int(num_steps),
        lr_override=float(learning_rate),
    )

    landscape_data = scan_energy_landscape_3d(
        energy_fn=model,
        v_clean=v_clean,
        v_noisy=v_noisy,
        grid_size=40,
        range_factor=1.5,
        v_denoised=v_denoised,
        trajectory=trajectory,
        model_type=model_type,
    )

    v_clean_np = v_clean.squeeze(0).cpu().numpy()
    v_noisy_np = v_noisy.squeeze(0).cpu().numpy()
    v_denoised_np = v_denoised.squeeze(0).cpu().numpy()

    cos_before = float(np.dot(v_clean_np, v_noisy_np) / (np.linalg.norm(v_clean_np) * np.linalg.norm(v_noisy_np)))
    cos_after = float(np.dot(v_clean_np, v_denoised_np) / (np.linalg.norm(v_clean_np) * np.linalg.norm(v_denoised_np)))
    improvement = cos_after - cos_before

    energy_min = float(landscape_data.get("energy_min", np.nan))
    energy_max = float(landscape_data.get("energy_max", np.nan))

    info = f"""
**Inference Results:**
- Initial noise scale (relative): {noise_scale}
- Langevin method: {stage1_cfg.langevin.method}
- Steps: {num_steps}
- Learning rate: {learning_rate}
- Energy range: [{energy_min:.4f}, {energy_max:.4f}]

- Cosine (clean, noisy): {cos_before:.4f}
- Cosine (clean, denoised): {cos_after:.4f}
- Improvement: {improvement:+.4f}
- Trajectory steps: {len(trajectory)}
"""

    trajectory_fig = create_trajectory_plot(landscape_data)
    return info, trajectory_fig


def generate_landscape_for_checkpoint(checkpoint_path):
    """Generate deterministic landscape preview for the selected checkpoint."""
    if checkpoint_path in session_state["landscape_cache"]:
        return session_state["landscape_cache"][checkpoint_path]

    checkpoint = session_state["checkpoints"].get(checkpoint_path)
    if not checkpoint:
        raise ValueError("Checkpoint not loaded")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type = _load_energy_model_from_checkpoint(checkpoint, device)
    stage1_cfg = build_stage1_config(checkpoint)

    v_clean = _sample_reference_clean_vector(
        device=device,
        target_norm=stage1_cfg.langevin.target_norm,
        seed=42,
    )
    v_noisy = _add_relative_noise(v_clean, 0.15)

    v_denoised, trajectory = run_langevin_denoise(
        model=model,
        v_clean=v_clean,
        v_noisy=v_noisy,
        model_type=model_type,
        stage1_cfg=stage1_cfg,
        max_steps=50,
        lr_override=None,
    )

    landscape_data = scan_energy_landscape_3d(
        energy_fn=model,
        v_clean=v_clean,
        v_noisy=v_noisy,
        grid_size=40,
        range_factor=1.5,
        v_denoised=v_denoised,
        trajectory=trajectory,
        model_type=model_type,
    )

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
        fig = create_metrics_dashboard(df, title="Training Metrics")

        # Statistics
        stats = compute_summary_statistics(df)
        stats_text = "\n".join([f"**{k}:** {v:.4f}" if isinstance(v, float) else f"**{k}:** {v}" for k, v in stats.items()])

        return stats_text, fig
    except Exception as e:
        return f"Error: {e}", None


def start_live_monitor_fn(metrics_path):
    """ÃƒÆ’Ã‚ÂÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¿ÃƒÆ’Ã¢â‚¬ËœÃƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âº live ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¼ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â³ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°."""
    if not metrics_path:
        return "No path specified", None

    path = Path(metrics_path)
    if not path.exists():
        return f"File not found: {path}", None

    try:
        # ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â·ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â´ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¼ watcher
        watcher = TrainingMetricsWatcher(path)
        watcher.start()
        session_state["watcher"] = watcher

        return f"Monitoring started: {path}", gr.update(value=watcher.history.to_dataframe())
    except Exception as e:
        return f"Error: {e}", None


def update_live_plot_fn():
    """ÃƒÆ’Ã‚ÂÃƒâ€¦Ã‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â±ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â»ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âµ live ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â³ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂºÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â° (ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¹ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â·ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¹ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚Â ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¿ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¹ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¼ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã¢â‚¬ËœÃƒâ€ Ã¢â‚¬â„¢)."""
    watcher = session_state.get("watcher")
    if not watcher:
        return None

    df = watcher.history.to_dataframe()
    if df.empty:
        return None

    return create_live_metrics_plot(df, plot_type="all")


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
                    info="Plotly: interactive 3D; Matplotlib: static fallback renderer.",
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
            gr.Markdown("### Upload training_metrics.json")

            with gr.Row():
                metrics_upload = gr.File(
                    label="Upload Metrics JSON",
                    file_types=[".json"],
                )

            with gr.Row():
                metrics_stats = gr.Markdown()
                metrics_plot = gr.Plot(label="Metrics Dashboard")

        # === Tab 4: Live Monitor ===
        with gr.TabItem("Live Monitor"):
            gr.Markdown(
                """
            ### Real-time Training Monitor

            Provide a path to a `training_metrics.json` file that is being updated during training.
            """
            )

            with gr.Row():
                live_path_input = gr.Textbox(
                    label="Path to training_metrics.json",
                    placeholder="experiments/02_energy_matching/training_metrics.json",
                )
                start_live_btn = gr.Button("Start Monitoring", variant="primary")

            live_status = gr.Textbox(label="Status")
            live_plot = gr.Plot(label="Live Metrics")

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
        inputs=[checkpoint_dropdown, vis_backend_radio],
        outputs=[checkpoint_summary, landscape_plot, inference_output, trajectory_plot],
    )

    # ÃƒÆ’Ã‚ÂÃƒâ€¦Ã‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â±ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â»ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âµ ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¿ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¼ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âµ backend
    vis_backend_radio.change(
        select_checkpoint_fn,
        inputs=[checkpoint_dropdown, vis_backend_radio],
        outputs=[checkpoint_summary, landscape_plot, inference_output, trajectory_plot],
    )

    # ÃƒÆ’Ã‚ÂÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¿ÃƒÆ’Ã¢â‚¬ËœÃƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Âº ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã¢â‚¬ËœÃƒâ€šÃ‚ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°
    run_inference_btn.click(
        run_inference_fn,
        inputs=[checkpoint_dropdown, noise_scale_slider, num_steps_slider, lr_slider],
        outputs=[inference_output, trajectory_plot],
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

    live_timer.tick(
        update_live_plot_fn,
        outputs=[live_plot],
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



