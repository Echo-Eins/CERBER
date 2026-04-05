"""
CERBER Model Monitor - main Gradio application.

Usage:
    python cerber_gui/app.py

    # Opens http://localhost:7860
"""

import sys
import re
import json
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
from cerber_gui.chain_diagnostics import (
    load_chain_head_from_checkpoint,
    make_sample_chain,
    extract_attention_weights,
    create_attention_heatmap,
    create_all_heads_heatmap,
    compute_swap_sensitivity,
    create_swap_sensitivity_plot,
    compute_chain_growth_energy,
    create_chain_growth_plot,
    compute_pos_neg_comparison,
    create_pos_neg_violin,
    parse_training_log,
    create_overfitting_plot,
    create_energy_scale_plot,
)
from cerber_gui.inference_diagnostics import (
    PRESET_QUESTIONS,
    DiagnosticsState,
    InferenceResult,
    load_pairwise_model as diag_load_pairwise,
    load_chain_head_model as diag_load_chain,
    load_sonar as diag_load_sonar,
    run_inference as diag_run_inference,
    run_text_inference as diag_run_text_inference,
    create_energy_trajectory_plot,
    create_cosine_trajectory_plot,
    create_attention_animation,
    create_attention_grid,
    create_landscape_with_trajectory,
    create_contour_with_trajectory,
    format_metrics_markdown,
    export_metrics_json,
    export_metrics_csv,
    get_state as get_diag_state,
)
from cerber_gui.context_encoder_diagnostics import (
    load_ce_models as ce_diag_load_models,
    run_ce_diagnostics as ce_diag_run,
)
from configs.base import Stage1Config
from cebcm.data.dataset import SONARVectorDataset
from cebcm.inference.langevin import run_langevin
from cebcm.inference.sigma_schedule import AdaptiveSigmaEnergyWrapper, SigmaScheduleConfig


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
SOTA_EVAL_CACHE_VERSION = 4


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
    if not isinstance(stage1_cfg, dict):
        # Stage1.5 checkpoints persist runtime settings under "config".
        stage1_cfg = checkpoint.get("config")
    if isinstance(stage1_cfg, dict):
        update_dataclass(cfg, stage1_cfg)
    return cfg


def _extract_stage15_runtime_options(checkpoint_payload: dict) -> dict:
    """
    Extract Stage1.5 runtime knobs used by GUI inference/sampling parity.
    """
    opts = {
        "retrieval_topk_pos": 6,
        "retrieval_hard_start": 6,
        "retrieval_hard_end": 24,
        "retrieval_min_pos_similarity": 0.15,
        "actor_seed_mix_query": 0.5,
        "actor_seed_noise_scale": 1.0,
        "langevin_tangent_noise": False,
    }
    if not isinstance(checkpoint_payload, dict):
        return opts
    cfg_payload = checkpoint_payload.get("config", {})
    if not isinstance(cfg_payload, dict):
        return opts

    def _safe_float(name: str, default: float) -> float:
        try:
            return float(cfg_payload.get(name, default))
        except (TypeError, ValueError):
            return default

    def _safe_int(name: str, default: int) -> int:
        try:
            return int(cfg_payload.get(name, default))
        except (TypeError, ValueError):
            return default

    opts["retrieval_topk_pos"] = max(2, _safe_int("retrieval_topk_pos", opts["retrieval_topk_pos"]))
    opts["retrieval_hard_start"] = max(0, _safe_int("retrieval_hard_start", opts["retrieval_hard_start"]))
    opts["retrieval_hard_end"] = max(
        opts["retrieval_hard_start"] + 1,
        _safe_int("retrieval_hard_end", opts["retrieval_hard_end"]),
    )
    opts["retrieval_min_pos_similarity"] = max(
        -1.0,
        min(1.0, _safe_float("retrieval_min_pos_similarity", opts["retrieval_min_pos_similarity"])),
    )
    opts["actor_seed_mix_query"] = max(
        0.0, min(1.0, _safe_float("actor_seed_mix_query", opts["actor_seed_mix_query"]))
    )
    opts["actor_seed_noise_scale"] = max(
        0.0, _safe_float("actor_seed_noise_scale", opts["actor_seed_noise_scale"])
    )
    opts["langevin_tangent_noise"] = bool(
        cfg_payload.get("langevin_tangent_noise", opts["langevin_tangent_noise"])
    )
    return opts


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


class _SigmaBoundPairEnergyAdapter:
    """
    Adapter that binds fixed sigma for sigma-conditioned pairwise energy models.
    """

    def __init__(self, model, sigma: torch.Tensor):
        self.model = model
        self.sigma = sigma

    def __call__(self, v_query: torch.Tensor, v_candidate: torch.Tensor) -> torch.Tensor:
        return self.model(v_query, v_candidate, sigma=self.sigma)

    def energy_and_grad(
        self,
        v_query: torch.Tensor,
        v_candidate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.energy_and_grad(v_query, v_candidate, sigma=self.sigma)


class _TwinConditionalEnergyAdapter:
    """
    Runtime adapter for Stage1.5 twin-critic conditional energy in GUI paths.

    Mirrors training-time aggregation:
      - cond = agg(E1(q,v), E2(q,v))
      - total = cond + lambda_prior * E_prior(v)  (optional)
    """

    def __init__(
        self,
        critic1,
        critic2,
        aggregate: str = "softmax",
        softmax_temperature: float = 0.1,
        prior=None,
        lambda_prior: float = 0.0,
        critic_architecture: str = "homogeneous",
        sigma_min: float = 0.01,
        sigma_max: float = 0.3,
        sigma_head_weighting_enabled: bool = False,
        angular_weight_low_sigma: float = 0.5,
        angular_weight_high_sigma: float = 0.5,
        head_weight_power: float = 1.0,
    ):
        self.critic1 = critic1
        self.critic2 = critic2
        self.aggregate = str(aggregate)
        self.softmax_temperature = float(softmax_temperature)
        self.prior = prior
        self.lambda_prior = float(lambda_prior)
        self.critic_architecture = str(critic_architecture)
        self.sigma_min = float(max(sigma_min, 1e-8))
        self.sigma_max = float(max(sigma_max, self.sigma_min + 1e-8))
        self.sigma_head_weighting_enabled = bool(sigma_head_weighting_enabled)
        self.angular_weight_low_sigma = float(angular_weight_low_sigma)
        self.angular_weight_high_sigma = float(angular_weight_high_sigma)
        self.head_weight_power = float(max(head_weight_power, 1e-6))

    def eval(self):
        self.critic1.eval()
        self.critic2.eval()
        if self.prior is not None:
            self.prior.eval()
        return self

    def _conditional_energy(
        self,
        v_query: torch.Tensor,
        v_candidate: torch.Tensor,
        sigma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        e1 = self.critic1(v_query, v_candidate, sigma=sigma)
        e2 = self.critic2(v_query, v_candidate, sigma=sigma)
        if self.critic_architecture == "radial_angular":
            sigma_use = sigma
            if sigma_use is None:
                with torch.no_grad():
                    d = (v_candidate - v_query).norm(dim=-1, keepdim=True)
                    qn = v_query.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    sigma_use = (d / qn).clamp(min=self.sigma_min, max=self.sigma_max)
            if self.sigma_head_weighting_enabled:
                log_s = sigma_use.clamp(min=self.sigma_min, max=self.sigma_max).log()
                denom = max(np.log(self.sigma_max) - np.log(self.sigma_min), 1e-8)
                t = ((log_s - np.log(self.sigma_min)) / denom).clamp(min=0.0, max=1.0).pow(self.head_weight_power)
                w_ang = self.angular_weight_low_sigma + (
                    self.angular_weight_high_sigma - self.angular_weight_low_sigma
                ) * t
            else:
                w_ang = torch.full_like(sigma_use, 0.5 * (self.angular_weight_low_sigma + self.angular_weight_high_sigma))
            w_ang = w_ang.clamp(min=0.0, max=1.0).squeeze(-1)
            return w_ang * e1 + (1.0 - w_ang) * e2
        if self.aggregate == "mean":
            return 0.5 * (e1 + e2)
        if self.aggregate == "max":
            return torch.maximum(e1, e2)
        if self.aggregate == "softmax":
            tau = max(1e-6, float(self.softmax_temperature))
            stacked = torch.stack([e1, e2], dim=0)
            return tau * torch.logsumexp(stacked / tau, dim=0)
        raise ValueError(f"Unknown twin aggregate mode: {self.aggregate}")

    def __call__(
        self,
        v_query: torch.Tensor,
        v_candidate: torch.Tensor,
        sigma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        e = self._conditional_energy(v_query, v_candidate, sigma=sigma)
        if self.prior is not None and self.lambda_prior > 0.0:
            e = e + self.lambda_prior * self.prior(v_candidate)
        return e

    def energy_and_grad(
        self,
        v_query: torch.Tensor,
        v_candidate: torch.Tensor,
        sigma: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v_req = v_candidate.detach().requires_grad_(True)
        energy = self(v_query, v_req, sigma=sigma)
        grad = torch.autograd.grad(energy.sum(), v_req, create_graph=False)[0]
        return energy.detach(), grad.detach()


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
    cfg = checkpoint.get("config", {}) or {}
    if str(cfg.get("critic_architecture", "homogeneous")) == "radial_angular":
        angular_hidden = [int(x) for x in cfg.get("angular_hidden_dims", resolved_hidden)]
        radial_hidden = [int(x) for x in cfg.get("radial_hidden_dims", [512, 256, 128])]
        architecture = (
            f"Angular[{_fmt_hidden_chain(resolved_dim, angular_hidden)}] + "
            f"Radial[{_fmt_hidden_chain(resolved_dim, radial_hidden)}]"
        )
    else:
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
        (
            f"**Normalization:** `angular={cfg.get('angular_norm_mode', resolved_norm)}, "
            f"radial={cfg.get('radial_norm_mode', resolved_norm)}`"
            if str(cfg.get("critic_architecture", "homogeneous")) == "radial_angular"
            else f"**Normalization:** `{resolved_norm}`"
        ),
        (
            f"**Activation:** `angular={cfg.get('angular_activation', resolved_activation)}, "
            f"radial={cfg.get('radial_activation', resolved_activation)}`"
            if str(cfg.get("critic_architecture", "homogeneous")) == "radial_angular"
            else f"**Activation:** `{resolved_activation}`"
        ),
    ]

    if runtime is not None:
        runtime_target_label = str(runtime.get("target_label", "clean"))
        runtime_objective = str(runtime.get("eval_objective", "self_denoise"))
        lines.extend(
            [
                "",
                "**Quick Inference Snapshot (latest run):**",
                f"- Objective/target: `{runtime_objective}` / `{runtime_target_label}`",
                f"- Cosine(target): `{runtime['cos_before']:.6f}` -> `{runtime['cos_after']:.6f}` ({_fmt_signed(runtime['cos_improvement'])})",
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
                f"- Eval objective: `{runtime_objective}` (target=`{runtime_target_label}`)",
                f"- Noise scale: `{runtime['noise_scale']:.4f}`",
                f"- Steps requested/executed: `{runtime['steps_requested']}` / `{runtime['steps_executed']}`",
                f"- Early stop: `{runtime['stopped_early']}`",
                f"- Cosine(target, before): `{runtime['cos_before']:.6f}`",
                f"- Cosine(target, after): `{runtime['cos_after']:.6f}`",
                f"- Cosine improvement: `{_fmt_signed(runtime['cos_improvement'])}`",
                f"- Energy(target ref): `{runtime['energy_clean']:.6f}`",
                f"- Energy(start noisy): `{runtime['energy_noisy']:.6f}`",
                f"- Energy(final): `{runtime['energy_final']:.6f}`",
                f"- Energy improvement (start-final): `{_fmt_signed(runtime['energy_improvement'])}`",
                f"- Success (energy descent): `{runtime.get('energy_descent_success', runtime['energy_success'])}`",
                f"- Success (target-energy proximity): `{runtime.get('energy_target_closer_success', float('nan'))}`",
                f"- Success (cosine gain): `{runtime['cosine_success']}`",
                f"- Displacement ||x_T - x_0||: `{runtime['displacement']:.6f}`",
            ]
        )
        if runtime.get("query_target_cos") is not None and np.isfinite(float(runtime.get("query_target_cos"))):
            lines.extend([f"- Query-target cosine: `{float(runtime['query_target_cos']):.6f}`"])
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
                target_label = str(sota.get("target_label", "clean"))
                objective = str(sota.get("eval_objective", "self_denoise"))
                lines.extend(
                    [
                        "",
                        "**SOTA Batch Eval (latest run):**",
                        f"- Eval batch / bank: `{int(sota['eval_batch_size'])}` / `{int(sota['eval_bank_size'])}`",
                        f"- Eval objective: `{objective}` (target=`{target_label}`)",
                        f"- Cosine before/after: `{float(sota['cos_before_mean']):.6f}` -> `{float(sota['cos_after_mean']):.6f}`",
                        f"- Cosine improvement mean: `{_fmt_signed(float(sota['cos_improvement_mean']), 8)}`",
                        f"- Cosine success rate: `{float(sota['cos_success_rate']):.2%}`",
                        f"- L2(target,x) mean: `{float(sota.get('l2_before_mean', float('nan'))):.6f}` -> `{float(sota.get('l2_after_mean', float('nan'))):.6f}`",
                        f"- L2 improvement mean: `{_fmt_signed(float(sota.get('l2_improvement_mean', float('nan'))), 8)}`",
                        f"- L2 success rate: `{float(sota.get('l2_success_rate', float('nan'))):.2%}`",
                        f"- Energy improvement mean: `{_fmt_signed(float(sota['energy_improvement_mean']), 8)}`",
                        f"- Energy descent rate: `{float(sota.get('energy_descent_rate', float('nan'))):.2%}`",
                        f"- Target-energy proximity rate: `{float(sota.get('energy_target_closer_rate', sota['energy_success_rate'])):.2%}`",
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
    """Render checkpoint summary and default landscape preview.

    Also returns updated slider values (noise_scale, steps, lr) extracted
    from the checkpoint's persisted config so the GUI always reflects the
    parameters the model was trained with.
    """
    empty_sliders = (
        gr.update(),  # noise_scale
        gr.update(),  # num_steps
        gr.update(),  # lr
    )
    if not checkpoint_path or checkpoint_path not in session_state["checkpoints"]:
        return "No checkpoint selected", None, "No metrics available", None, *empty_sliders

    checkpoint = session_state["checkpoints"][checkpoint_path]
    session_state["current_checkpoint"] = checkpoint_path

    # --- Extract Langevin config from checkpoint to update sliders ---
    cfg = build_stage1_config(checkpoint)
    langevin_cfg = getattr(cfg, "langevin", None)
    if langevin_cfg is not None:
        slider_noise = gr.update(value=getattr(langevin_cfg, "noise_scale", 0.0002))
        slider_steps = gr.update(value=getattr(langevin_cfg, "max_steps", 100))
        slider_lr = gr.update(value=getattr(langevin_cfg, "lr", 0.01))
    else:
        slider_noise, slider_steps, slider_lr = empty_sliders

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

    return summary, landscape_fig, info, trajectory_plot, slider_noise, slider_steps, slider_lr


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
    from cebcm.models.energy_decomposed import AngularEnergyCritic, RadialEnergyCritic
    from cebcm.models.energy_unconditional import UnconditionalEnergy

    def _load_state_or_raise(model_obj, state_dict: dict, label: str) -> None:
        load_result = model_obj.load_state_dict(state_dict, strict=False)
        missing = [k for k in load_result.missing_keys if k != "_sigma_freqs"]
        unexpected = [k for k in load_result.unexpected_keys]
        if missing or unexpected:
            missing_preview = ", ".join(missing[:8])
            unexpected_preview = ", ".join(unexpected[:8])
            raise RuntimeError(
                "Checkpoint/model mismatch while loading GUI model. "
                f"{label} | Missing({len(missing)}): {missing_preview}. "
                f"Unexpected({len(unexpected)}): {unexpected_preview}."
            )

    model_type = checkpoint["model_type"]
    model_state = checkpoint["model_state"]
    dim, hidden_dims, norm_mode, activation = _resolve_model_hparams(checkpoint)

    if model_type == "simple":
        # Stage1.5 checkpoints contain twin critics. GUI must load the same
        # hybrid energy used in training/eval, not a single critic fallback.
        has_twin = ("critic1_state" in checkpoint) or ("critic2_state" in checkpoint)
        if has_twin:
            cfg = checkpoint.get("config", {}) or {}
            critic_arch = str(cfg.get("critic_architecture", "homogeneous"))
            if critic_arch == "radial_angular":
                c1 = AngularEnergyCritic(
                    dim=dim,
                    hidden_dims=[int(x) for x in cfg.get("angular_hidden_dims", hidden_dims)],
                    norm_mode=str(cfg.get("angular_norm_mode", norm_mode)),
                    activation=str(cfg.get("angular_activation", activation)),
                    energy_output_clamp=None,
                ).to(device)
                c2 = RadialEnergyCritic(
                    dim=dim,
                    hidden_dims=[int(x) for x in cfg.get("radial_hidden_dims", [512, 256, 128])],
                    norm_mode=str(cfg.get("radial_norm_mode", norm_mode)),
                    activation=str(cfg.get("radial_activation", activation)),
                    target_norm=float(cfg.get("langevin", {}).get("target_norm", 0.0)) or None,
                    energy_output_clamp=None,
                ).to(device)
            else:
                c1 = SimpleEnergy(
                    dim=dim,
                    hidden_dims=hidden_dims,
                    norm_mode=norm_mode,
                    activation=activation,
                    energy_output_clamp=None,
                ).to(device)
                c2 = SimpleEnergy(
                    dim=dim,
                    hidden_dims=hidden_dims,
                    norm_mode=norm_mode,
                    activation=activation,
                    energy_output_clamp=None,
                ).to(device)
            c1_state = checkpoint.get("critic1_state", model_state)
            c2_state = checkpoint.get("critic2_state", c1_state)
            _load_state_or_raise(c1, c1_state, "critic1")
            _load_state_or_raise(c2, c2_state, "critic2")

            aggregate = str(cfg.get("twin_aggregate", "max"))
            softmax_temperature = float(cfg.get("twin_softmax_temperature", 0.1))

            prior = None
            lambda_prior = 0.0
            prior_state = checkpoint.get("prior_state")
            if prior_state is not None:
                prior = UnconditionalEnergy(
                    dim=dim,
                    hidden_dims=hidden_dims,
                    norm_mode=norm_mode,
                    activation=activation,
                ).to(device)
                _load_state_or_raise(prior, prior_state, "prior")
                lambda_prior = float(cfg.get("lambda_prior", 0.0))

            model = _TwinConditionalEnergyAdapter(
                critic1=c1,
                critic2=c2,
                aggregate=aggregate,
                softmax_temperature=softmax_temperature,
                prior=prior,
                lambda_prior=lambda_prior,
                critic_architecture=critic_arch,
                sigma_min=float(cfg.get("sigma_min", 0.01)),
                sigma_max=float(cfg.get("sigma_max", 0.3)),
                sigma_head_weighting_enabled=bool(cfg.get("sigma_head_weighting_enabled", False)),
                angular_weight_low_sigma=float(cfg.get("angular_weight_low_sigma", 0.5)),
                angular_weight_high_sigma=float(cfg.get("angular_weight_high_sigma", 0.5)),
                head_weight_power=float(cfg.get("head_weight_power", 1.0)),
            )
            model.eval()
            return model, model_type

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

    _load_state_or_raise(
        model,
        model_state,
        f"model_type={model_type}, dim={dim}, hidden_dims={hidden_dims}, norm_mode={norm_mode}, activation={activation}",
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
    v_query_override: torch.Tensor | None = None,
    v_target_override: torch.Tensor | None = None,
    tangent_noise_override: bool | None = None,
    sigma_override: torch.Tensor | None = None,
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
    tangent_noise = bool(tangent_noise_override) if tangent_noise_override is not None else bool(
        getattr(stage1_cfg, "langevin_tangent_noise", False)
    )

    v_target = v_target_override if v_target_override is not None else v_clean

    if model_type == "unconditional":
        energy_fn = _UnconditionalEnergyAdapter(model)
        v_query = torch.zeros_like(v_clean)
    else:
        # Check if adaptive sigma is enabled in the config
        _langevin_cfg = getattr(stage1_cfg, 'langevin', None)
        _sigma_anneal = getattr(_langevin_cfg, 'sigma_anneal', False) if _langevin_cfg else False
        if _sigma_anneal and sigma_override is not None:
            _sched = SigmaScheduleConfig(
                enabled=True,
                mode=getattr(_langevin_cfg, 'sigma_anneal_mode', 'hybrid'),
                sigma_max=getattr(_langevin_cfg, 'sigma_anneal_max', 0.3),
                sigma_min=getattr(_langevin_cfg, 'sigma_anneal_min', 0.01),
                adaptive_blend=getattr(_langevin_cfg, 'sigma_anneal_blend', 0.5),
                noise_anneal=getattr(_langevin_cfg, 'noise_anneal', True),
                noise_mode=getattr(_langevin_cfg, 'noise_anneal_mode', 'hybrid'),
                noise_max=getattr(_langevin_cfg, 'noise_anneal_max', 0.15),
                noise_min=getattr(_langevin_cfg, 'noise_anneal_min', stage1_cfg.langevin.noise_scale),
                noise_sync_with_sigma=getattr(_langevin_cfg, 'noise_anneal_sync_with_sigma', True),
            )
            energy_fn = AdaptiveSigmaEnergyWrapper(model, _sched, max_steps=max_steps)
        elif sigma_override is not None:
            energy_fn = _SigmaBoundPairEnergyAdapter(model, sigma=sigma_override)
        else:
            energy_fn = model
        v_query = v_query_override if v_query_override is not None else v_clean

    result = run_langevin(
        method=method,
        energy_fn=energy_fn,
        v_query=v_query,
        v_init=v_noisy,
        lr=lr,
        noise_scale=stage1_cfg.langevin.noise_scale,
        max_steps=max_steps,
        target_norm=stage1_cfg.langevin.target_norm,
        tangent_noise=tangent_noise,
        energy_threshold=energy_threshold,
        plateau_patience=plateau_patience,
        plateau_delta=stage1_cfg.langevin.plateau_delta,
        v_target=v_target,
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


def _build_target_plane_basis(
    v_query: torch.Tensor,
    v_target: torch.Tensor,
    v_noisy_default: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    target = v_target.squeeze(0)
    primary = v_noisy_default.squeeze(0) - target
    if float(primary.norm().item()) < 1e-8:
        primary = v_query.squeeze(0) - target
    if float(primary.norm().item()) < 1e-8:
        gen = torch.Generator(device=target.device)
        gen.manual_seed(int(seed) + 17)
        primary = torch.randn(target.shape, generator=gen, device=target.device, dtype=target.dtype)

    u1 = F.normalize(primary.unsqueeze(0), dim=-1).squeeze(0)

    secondary = v_query.squeeze(0) - target
    secondary = secondary - (secondary * u1).sum() * u1
    if float(secondary.norm().item()) < 1e-8:
        gen = torch.Generator(device=target.device)
        gen.manual_seed(int(seed) + 29)
        secondary = torch.randn(target.shape, generator=gen, device=target.device, dtype=target.dtype)
        secondary = secondary - (secondary * u1).sum() * u1
    if float(secondary.norm().item()) < 1e-8:
        # Last-resort deterministic orthogonal direction.
        secondary = torch.roll(u1, shifts=1)
        secondary = secondary - (secondary * u1).sum() * u1

    u2 = F.normalize(secondary.unsqueeze(0), dim=-1).squeeze(0)
    base_dist = max(float((v_noisy_default.squeeze(0) - target).norm().item()), 1e-6)
    return u1, u2, base_dist


def _apply_noisy_start_strategy(
    v_query: torch.Tensor,
    v_target: torch.Tensor,
    v_noisy_default: torch.Tensor,
    stage1_cfg: Stage1Config,
    start_mode: str,
    far_scale: float,
    manual_x: float,
    manual_y: float,
    project_to_target_norm: bool,
    seed: int,
) -> tuple[torch.Tensor, str]:
    mode = str(start_mode or "objective_seed").strip().lower()
    if mode not in {"objective_seed", "far_auto", "manual_plane_xy"}:
        mode = "objective_seed"

    if mode == "objective_seed":
        return v_noisy_default, mode

    u1, u2, base_dist = _build_target_plane_basis(
        v_query=v_query,
        v_target=v_target,
        v_noisy_default=v_noisy_default,
        seed=seed,
    )
    target = v_target.squeeze(0)

    if mode == "far_auto":
        scale = max(1.0, float(far_scale))
        delta = scale * base_dist * u1
    else:  # manual_plane_xy
        delta = (float(manual_x) * base_dist) * u1 + (float(manual_y) * base_dist) * u2

    v_noisy = (target + delta).unsqueeze(0).to(device=v_target.device, dtype=v_target.dtype)
    if bool(project_to_target_norm) and stage1_cfg.langevin.target_norm is not None:
        v_noisy = F.normalize(v_noisy, dim=-1) * float(stage1_cfg.langevin.target_norm)
    return v_noisy, mode


def _is_stage15_conditional_checkpoint(checkpoint_payload: dict, model_type: str) -> bool:
    if model_type != "simple" or not isinstance(checkpoint_payload, dict):
        return False
    return (
        ("critic1_state" in checkpoint_payload)
        or ("critic2_state" in checkpoint_payload)
        or ("actor_state" in checkpoint_payload)
    )


def _sample_inference_triplet(
    checkpoint_payload: dict,
    model_type: str,
    stage1_cfg: Stage1Config,
    device: torch.device,
    noise_scale: float,
    seed: int,
    retrieval_bank_size: int,
    start_mode: str = "objective_seed",
    far_start_scale: float = 8.0,
    manual_start_x: float = 8.0,
    manual_start_y: float = 0.0,
    project_start_to_target_norm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str, str, str]:
    """
    Unified sampler for single-run inference:
    returns (v_query, v_target, v_noisy, reference_source, target_label, objective).
    """
    if _is_stage15_conditional_checkpoint(checkpoint_payload, model_type):
        rt = _extract_stage15_runtime_options(checkpoint_payload)
        q, pos, hard, src = _sample_conditional_retrieval_batch(
            device=device,
            batch_size=1,
            bank_size=max(64, int(retrieval_bank_size)),
            seed=int(seed),
            topk_pos=int(rt["retrieval_topk_pos"]),
            hard_start=int(rt["retrieval_hard_start"]),
            hard_end=int(rt["retrieval_hard_end"]),
            min_pos_similarity=float(rt["retrieval_min_pos_similarity"]),
        )
        if q is not None and pos is not None and hard is not None:
            seed_mix = float(rt["actor_seed_mix_query"])
            seed_scale = float(rt["actor_seed_noise_scale"])
            seed_vec = seed_mix * q + (1.0 - seed_mix) * hard
            noisy_default = _add_relative_noise(seed_vec, float(noise_scale) * seed_scale, seed=seed + 1)
            noisy, mode_used = _apply_noisy_start_strategy(
                v_query=q,
                v_target=pos,
                v_noisy_default=noisy_default,
                stage1_cfg=stage1_cfg,
                start_mode=start_mode,
                far_scale=far_start_scale,
                manual_x=manual_start_x,
                manual_y=manual_start_y,
                project_to_target_norm=project_start_to_target_norm,
                seed=seed,
            )
            return q, pos, noisy, src, "retrieved_pos", "conditional_retrieval", mode_used

    v_clean, v_noisy_default, src = _sample_clean_noisy_pair(
        stage1_cfg=stage1_cfg,
        device=device,
        noise_scale=float(noise_scale),
        seed=int(seed),
    )
    v_noisy, mode_used = _apply_noisy_start_strategy(
        v_query=v_clean,
        v_target=v_clean,
        v_noisy_default=v_noisy_default,
        stage1_cfg=stage1_cfg,
        start_mode=start_mode,
        far_scale=far_start_scale,
        manual_x=manual_start_x,
        manual_y=manual_start_y,
        project_to_target_norm=project_start_to_target_norm,
        seed=seed,
    )
    return v_clean, v_clean, v_noisy, src, "clean", "self_denoise", mode_used


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
def _sample_conditional_retrieval_batch(
    device: torch.device,
    batch_size: int,
    bank_size: int,
    seed: int,
    topk_pos: int = 6,
    hard_start: int = 6,
    hard_end: int = 24,
    min_pos_similarity: float = 0.15,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, str]:
    """
    Sample (query, positive, hard) triplets from dataset using cosine retrieval.
    This mirrors Stage 1.5 conditional objective better than self-target eval.
    """
    dataset_path = _get_default_dataset_path()
    if dataset_path is None:
        return None, None, None, "unavailable"
    try:
        dataset = _get_dataset(dataset_path)
    except Exception:
        return None, None, None, "unavailable"

    n = len(dataset)
    if n < 4:
        return None, None, None, "unavailable"

    bs = max(1, min(int(batch_size), n))
    bank_bs = max(bs, min(int(bank_size), n))

    gen_q = torch.Generator(device="cpu")
    gen_q.manual_seed(int(seed))
    q_idx_cpu = torch.randperm(n, generator=gen_q)[:bs]

    gen_b = torch.Generator(device="cpu")
    gen_b.manual_seed(int(seed) + 9973)
    bank_idx_cpu = torch.randperm(n, generator=gen_b)[:bank_bs]

    q_idx = q_idx_cpu.to(device=device, dtype=torch.long)
    bank_idx = bank_idx_cpu.to(device=device, dtype=torch.long)
    q = dataset.embeddings[q_idx_cpu].to(device)
    bank = dataset.embeddings[bank_idx_cpu].to(device)

    qn = F.normalize(q, dim=-1)
    bn = F.normalize(bank, dim=-1)
    sims = qn @ bn.T

    # Exclude exact self index in retrieval bank.
    same_index = q_idx.unsqueeze(1).eq(bank_idx.unsqueeze(0))
    sims = sims.masked_fill(same_index, -1e9)

    k = min(bank.shape[0], max(2, int(topk_pos), int(hard_end)))
    top_vals, top_idx = torch.topk(sims, k=k, dim=-1, largest=True)

    # Positive: first neighbor that passes min similarity threshold.
    valid_pos = top_vals >= float(min_pos_similarity)
    has_valid = valid_pos.any(dim=1)
    first_valid_col = valid_pos.to(torch.int64).argmax(dim=1)
    pos_rel = top_idx[torch.arange(bs, device=device), first_valid_col]
    # If no valid retrieval positive, provisional fallback uses top-1.
    pos_rel = torch.where(has_valid, pos_rel, top_idx[:, 0])

    hs = min(max(0, int(hard_start)), k - 1)
    he = min(max(hs + 1, int(hard_end)), k)
    width = max(1, he - hs)
    hard_rng = torch.Generator(device=device)
    hard_rng.manual_seed(int(seed) + 2027)
    hard_offset = torch.randint(0, width, (bs,), device=device, generator=hard_rng)
    hard_rel = top_idx[torch.arange(bs, device=device), hs + hard_offset]

    # Avoid hard==positive.
    hard_eq_pos = hard_rel.eq(pos_rel)
    if hard_eq_pos.any():
        fallback = top_idx[hard_eq_pos, -1]
        hard_rel = hard_rel.clone()
        hard_rel[hard_eq_pos] = fallback
        hard_still_eq = hard_rel.eq(pos_rel)
        if hard_still_eq.any() and k > 1:
            hard_rel[hard_still_eq] = top_idx[hard_still_eq, k - 2]

    pos = bank[pos_rel]
    if (~has_valid).any():
        # Keep objective well-posed: if retrieval neighborhood is too weak,
        # fall back to identity target for those rows.
        pos = pos.clone()
        pos[~has_valid] = q[~has_valid]
    hard = bank[hard_rel]
    return q, pos, hard, "dataset"


@torch.no_grad()
def _compute_energy_batch(
    model,
    model_type: str,
    v_clean: torch.Tensor,
    v_candidate: torch.Tensor,
    sigma_override: torch.Tensor | None = None,
) -> torch.Tensor:
    if model_type == "simple":
        if sigma_override is not None:
            return model(v_clean, v_candidate, sigma=sigma_override).detach()
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

    # Support both nn.Module and adapter wrappers (e.g. _TwinConditionalEnergyAdapter)
    if hasattr(model, 'parameters'):
        device = next(model.parameters()).device
    elif hasattr(model, 'critic1'):
        device = next(model.critic1.parameters()).device
    else:
        device = torch.device("cpu")
    checkpoint_payload = session_state["checkpoints"].get(checkpoint_path, {})
    is_stage15_conditional = _is_stage15_conditional_checkpoint(checkpoint_payload, model_type)
    stage15_rt = _extract_stage15_runtime_options(checkpoint_payload)
    tangent_noise = bool(stage15_rt["langevin_tangent_noise"]) if is_stage15_conditional else False

    if is_stage15_conditional:
        q_batch, v_target_batch, v_hard_batch, source = _sample_conditional_retrieval_batch(
            device=device,
            batch_size=int(eval_batch_size),
            bank_size=int(eval_bank_size),
            seed=31415,
            topk_pos=int(stage15_rt["retrieval_topk_pos"]),
            hard_start=int(stage15_rt["retrieval_hard_start"]),
            hard_end=int(stage15_rt["retrieval_hard_end"]),
            min_pos_similarity=float(stage15_rt["retrieval_min_pos_similarity"]),
        )
        if q_batch is None or v_target_batch is None or v_hard_batch is None:
            out = {"available": False, "reason": "dataset_unavailable"}
            session_state["sota_eval_cache"][key] = out
            return out
        seed_mix = float(stage15_rt["actor_seed_mix_query"])
        seed_scale = float(stage15_rt["actor_seed_noise_scale"])
        seed_vec = seed_mix * q_batch + (1.0 - seed_mix) * v_hard_batch
        v_noisy_batch = _add_relative_noise(seed_vec, float(noise_scale) * seed_scale, seed=31416)
        sigma_eval = torch.full(
            (v_noisy_batch.shape[0], 1),
            float(noise_scale),
            device=device,
            dtype=v_noisy_batch.dtype,
        )
        v_denoised_batch, _, batch_result = run_langevin_denoise(
            model=model,
            v_clean=v_target_batch,
            v_noisy=v_noisy_batch,
            model_type=model_type,
            stage1_cfg=stage1_cfg,
            max_steps=int(num_steps),
            lr_override=float(learning_rate),
            force_full_steps=True,
            track_vectors=False,
            v_query_override=q_batch,
            v_target_override=v_target_batch,
            tangent_noise_override=tangent_noise,
            sigma_override=sigma_eval,
        )
        eval_query_batch = q_batch
    else:
        v_clean_batch, source = _sample_dataset_vectors(device, int(eval_batch_size), seed=31415)
        if v_clean_batch is None:
            out = {"available": False, "reason": "dataset_unavailable"}
            session_state["sota_eval_cache"][key] = out
            return out
        v_target_batch = v_clean_batch
        v_noisy_batch = _add_relative_noise(v_clean_batch, float(noise_scale), seed=31416)
        sigma_eval = (
            torch.full(
                (v_noisy_batch.shape[0], 1),
                float(noise_scale),
                device=device,
                dtype=v_noisy_batch.dtype,
            )
            if model_type == "simple"
            else None
        )
        v_denoised_batch, _, batch_result = run_langevin_denoise(
            model=model,
            v_clean=v_target_batch,
            v_noisy=v_noisy_batch,
            model_type=model_type,
            stage1_cfg=stage1_cfg,
            max_steps=int(num_steps),
            lr_override=float(learning_rate),
            force_full_steps=True,
            track_vectors=False,
            sigma_override=sigma_eval,
        )
        eval_query_batch = v_clean_batch

    if v_target_batch is None:
        out = {"available": False, "reason": "dataset_unavailable"}
        session_state["sota_eval_cache"][key] = out
        return out

    cos_before = F.cosine_similarity(v_target_batch, v_noisy_batch, dim=-1)
    cos_after = F.cosine_similarity(v_target_batch, v_denoised_batch, dim=-1)

    e_target = _compute_energy_batch(
        model=model,
        model_type=model_type,
        v_clean=eval_query_batch,
        v_candidate=v_target_batch,
        sigma_override=sigma_eval,
    )
    e_noisy = _compute_energy_batch(
        model=model,
        model_type=model_type,
        v_clean=eval_query_batch,
        v_candidate=v_noisy_batch,
        sigma_override=sigma_eval,
    )
    e_final = _compute_energy_batch(
        model=model,
        model_type=model_type,
        v_clean=eval_query_batch,
        v_candidate=v_denoised_batch,
        sigma_override=sigma_eval,
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
    suite = compute_distribution_suite(v_target_batch, v_denoised_batch, cfg=cfg)
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
        "eval_objective": (
            "conditional_retrieval" if is_stage15_conditional else "self_denoise"
        ),
        "target_label": ("retrieved_pos" if is_stage15_conditional else "clean"),
        "eval_batch_size": int(v_target_batch.shape[0]),
        "eval_bank_size": int(0 if ref_bank is None else ref_bank.shape[0]),
        "steps_executed_mean": float(batch_result.num_steps),
        "cos_before_mean": float(cos_before.mean().item()),
        "cos_after_mean": float(cos_after.mean().item()),
        "cos_improvement_mean": float((cos_after - cos_before).mean().item()),
        "cos_success_rate": float((cos_after > cos_before).float().mean().item()),
        "l2_before_mean": float((v_target_batch - v_noisy_batch).norm(dim=-1).mean().item()),
        "l2_after_mean": float((v_target_batch - v_denoised_batch).norm(dim=-1).mean().item()),
        "l2_improvement_mean": float(
            ((v_target_batch - v_noisy_batch).norm(dim=-1) - (v_target_batch - v_denoised_batch).norm(dim=-1))
            .mean()
            .item()
        ),
        "l2_success_rate": float(
            ((v_target_batch - v_denoised_batch).norm(dim=-1) < (v_target_batch - v_noisy_batch).norm(dim=-1))
            .float()
            .mean()
            .item()
        ),
        "denoise_step_norm_mean": float((v_denoised_batch - v_noisy_batch).norm(dim=-1).mean().item()),
        "energy_before_mean": float(e_noisy.mean().item()),
        "energy_after_mean": float(e_final.mean().item()),
        "energy_improvement_mean": float((e_noisy - e_final).mean().item()),
        # Canonical success for dynamics: descent from start noisy state.
        "energy_descent_rate": float((e_final < e_noisy).float().mean().item()),
        # Additional diagnostic: final energy is closer to target energy than start was.
        "energy_target_closer_rate": float(
            ((e_final - e_target).abs() < (e_noisy - e_target).abs()).float().mean().item()
        ),
        # Backward-compatible alias used by older UI/report paths.
        "energy_success_rate": float(
            ((e_final - e_target).abs() < (e_noisy - e_target).abs()).float().mean().item()
        ),
    }
    out.update(suite)

    session_state["sota_eval_cache"][key] = out
    return out


@torch.no_grad()
def _compute_energy_triplet(
    model,
    model_type: str,
    v_query: torch.Tensor,
    v_target: torch.Tensor,
    v_noisy: torch.Tensor,
    v_denoised: torch.Tensor,
    sigma_override: torch.Tensor | None = None,
) -> dict[str, float]:
    if model_type == "simple":
        if sigma_override is not None:
            e_target = float(model(v_query, v_target, sigma=sigma_override).mean().item())
            e_noisy = float(model(v_query, v_noisy, sigma=sigma_override).mean().item())
            e_denoised = float(model(v_query, v_denoised, sigma=sigma_override).mean().item())
        else:
            e_target = float(model(v_query, v_target).mean().item())
            e_noisy = float(model(v_query, v_noisy).mean().item())
            e_denoised = float(model(v_query, v_denoised).mean().item())
    else:
        e_target = float(model(v_target).mean().item())
        e_noisy = float(model(v_noisy).mean().item())
        e_denoised = float(model(v_denoised).mean().item())

    return {
        "target": e_target,
        # Backward-compatible alias for existing UI fields:
        "clean": e_target,
        "noisy": e_noisy,
        "denoised": e_denoised,
        "delta_noisy_to_denoised": e_denoised - e_noisy,
        "delta_clean_to_denoised": e_denoised - e_target,
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
    v_query: torch.Tensor,
    v_target: torch.Tensor,
    v_noisy: torch.Tensor,
    v_denoised: torch.Tensor,
    noise_scale: float,
    steps_requested: int,
    steps_executed: int,
    stopped_early: bool,
    reference_source: str,
    target_label: str = "clean",
    eval_objective: str = "self_denoise",
    sigma_override: torch.Tensor | None = None,
) -> dict:
    v_target_np = v_target.squeeze(0).detach().cpu().numpy()
    v_noisy_np = v_noisy.squeeze(0).detach().cpu().numpy()
    v_denoised_np = v_denoised.squeeze(0).detach().cpu().numpy()

    cos_before = float(np.dot(v_target_np, v_noisy_np) / (np.linalg.norm(v_target_np) * np.linalg.norm(v_noisy_np)))
    cos_after = float(np.dot(v_target_np, v_denoised_np) / (np.linalg.norm(v_target_np) * np.linalg.norm(v_denoised_np)))
    cos_improvement = cos_after - cos_before

    energies = _compute_energy_triplet(
        model=model,
        model_type=model_type,
        v_query=v_query,
        v_target=v_target,
        v_noisy=v_noisy,
        v_denoised=v_denoised,
        sigma_override=sigma_override,
    )
    displacement = float((v_denoised - v_noisy).norm().item())

    query_target_cos = float(
        F.cosine_similarity(v_query, v_target, dim=-1).mean().item()
    ) if model_type == "simple" else float("nan")

    energy_descent_success = bool(energies["denoised"] < energies["noisy"])
    energy_target_closer_success = bool(
        abs(energies["denoised"] - energies["target"]) < abs(energies["noisy"] - energies["target"])
    )

    return {
        "reference_source": reference_source,
        "target_label": str(target_label),
        "eval_objective": str(eval_objective),
        "noise_scale": float(noise_scale),
        "steps_requested": int(steps_requested),
        "steps_executed": int(steps_executed),
        "stopped_early": bool(stopped_early),
        "cos_before": cos_before,
        "cos_after": cos_after,
        "cos_improvement": cos_improvement,
        "energy_target": float(energies["target"]),
        "energy_clean": float(energies["target"]),
        "energy_noisy": float(energies["noisy"]),
        "energy_final": float(energies["denoised"]),
        "energy_improvement": float(energies["noisy"] - energies["denoised"]),
        # Backward-compatible canonical field: true energy descent from start to final.
        "energy_success": energy_descent_success,
        # Explicit diagnostic for target-energy proximity (different criterion).
        "energy_target_closer_success": energy_target_closer_success,
        "energy_descent_success": energy_descent_success,
        "cosine_success": bool(cos_after > cos_before),
        "displacement": displacement,
        "query_target_cos": query_target_cos,
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
    target_label: str = "clean",
    eval_objective: str = "self_denoise",
    query_target_cos: float | None = None,
    start_mode: str | None = None,
    start_distance_to_target: float | None = None,
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
        f"- Eval objective: `{eval_objective}` (target=`{target_label}`)",
        f"- Noisy start mode: `{start_mode or 'objective_seed'}`",
        (
            f"- Initial distance ||x0-target||: `{float(start_distance_to_target):.6f}`"
            if start_distance_to_target is not None
            else "- Initial distance ||x0-target||: `N/A`"
        ),
        f"- Energy range on scanned plane: `[{energy_min:.4f}, {energy_max:.4f}]`",
        f"- Energy(target ref): `{energies['clean']:.6f}`",
        f"- Energy(noisy start): `{energies['noisy']:.6f}`",
        f"- Energy(denoised/final): `{energies['denoised']:.6f}`",
        f"- Delta energy (final - start): `{energies['delta_noisy_to_denoised']:+.6f}`",
        f"- Primary improvement (start - final energy): `{(-energies['delta_noisy_to_denoised']):+.6f}`",
        f"- Delta energy (final - target): `{energies['delta_clean_to_denoised']:+.6f}`",
        f"- Cosine(target, noisy): `{cos_before:.6f}`",
        f"- Cosine(target, final): `{cos_after:.6f}`",
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
    elif eval_objective == "conditional_retrieval":
        lines.append("- Note: conditional Stage1.5 evaluates against retrieved positive target (not self-clean reconstruction).")
        if query_target_cos is not None and np.isfinite(query_target_cos):
            lines.append(f"- Query-target cosine (retrieval quality): `{query_target_cos:.6f}`")

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
            target_label = str(sota_metrics.get("target_label", "clean"))
            objective = str(sota_metrics.get("eval_objective", "self_denoise"))
            lines.append(
                f"- Eval batch / bank: `{int(sota_metrics['eval_batch_size'])}` / `{int(sota_metrics['eval_bank_size'])}`"
            )
            lines.append(f"- Eval objective: `{objective}` (target=`{target_label}`)")
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
                f"- L2(target,x) mean: `{float(sota_metrics.get('l2_before_mean', float('nan'))):.6f}` -> "
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
                f"- Energy descent rate: `{float(sota_metrics.get('energy_descent_rate', float('nan'))):.2%}`"
            )
            lines.append(
                f"- Target-energy proximity rate: `{float(sota_metrics.get('energy_target_closer_rate', sota_metrics.get('energy_success_rate', float('nan')))):.2%}`"
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
    start_mode,
    far_start_scale,
    manual_start_x,
    manual_start_y,
    project_start_to_target_norm,
):
    """Run denoising/refinement inference and refresh both surface + trajectory plots."""
    if not checkpoint_path or checkpoint_path not in session_state["checkpoints"]:
        return "No checkpoint selected", None, None, "No checkpoint selected"

    checkpoint = session_state["checkpoints"][checkpoint_path]
    stage1_cfg = build_stage1_config(checkpoint)
    stage15_rt = _extract_stage15_runtime_options(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type = _load_energy_model_from_checkpoint(checkpoint, device)
    v_query, v_target, v_noisy, reference_source, target_label, eval_objective, start_mode_used = _sample_inference_triplet(
        checkpoint_payload=checkpoint,
        model_type=model_type,
        stage1_cfg=stage1_cfg,
        device=device,
        noise_scale=float(noise_scale),
        seed=42,
        retrieval_bank_size=int(sota_eval_bank_size),
        start_mode=str(start_mode),
        far_start_scale=float(far_start_scale),
        manual_start_x=float(manual_start_x),
        manual_start_y=float(manual_start_y),
        project_start_to_target_norm=bool(project_start_to_target_norm),
    )
    sigma_infer = (
        torch.full((v_noisy.shape[0], 1), float(noise_scale), device=device, dtype=v_noisy.dtype)
        if model_type == "simple"
        else None
    )
    tangent_noise = bool(stage15_rt["langevin_tangent_noise"]) if eval_objective == "conditional_retrieval" else False

    v_denoised, trajectory, langevin_result = run_langevin_denoise(
        model=model,
        v_clean=v_target,
        v_noisy=v_noisy,
        model_type=model_type,
        stage1_cfg=stage1_cfg,
        max_steps=int(num_steps),
        lr_override=float(learning_rate),
        force_full_steps=True,
        v_query_override=v_query,
        v_target_override=v_target,
        tangent_noise_override=tangent_noise,
        sigma_override=sigma_infer,
    )

    grid, rf, abs_range = _resolve_scan_params(grid_size, range_factor, absolute_half_range)

    landscape_data = scan_energy_landscape_3d(
        energy_fn=model,
        v_clean=v_target,
        v_noisy=v_noisy,
        grid_size=grid,
        range_factor=rf,
        absolute_half_range=abs_range,
        v_denoised=v_denoised,
        trajectory=trajectory,
        model_type=model_type,
    )
    if eval_objective == "conditional_retrieval":
        if isinstance(landscape_data.get("point_labels"), dict):
            landscape_data["point_labels"]["clean"] = "Retrieved Target"
            landscape_data["point_labels"]["noisy"] = "Noisy Seed"
            landscape_data["point_labels"]["denoised"] = "Refined"
        landscape_data["axis1_label"] = "Direction 1 (target -> noisy)"

    landscape_fig = _render_landscape_figure(checkpoint_path, landscape_data, vis_backend)
    trajectory_fig = create_trajectory_plot(landscape_data)

    runtime_metrics = _build_runtime_metrics(
        model=model,
        model_type=model_type,
        v_query=v_query,
        v_target=v_target,
        v_noisy=v_noisy,
        v_denoised=v_denoised,
        noise_scale=float(noise_scale),
        steps_requested=int(num_steps),
        steps_executed=int(langevin_result.num_steps),
        stopped_early=bool(langevin_result.stopped_early),
        reference_source=reference_source,
        target_label=target_label,
        eval_objective=eval_objective,
        sigma_override=sigma_infer,
    )
    runtime_metrics.update(
        _compute_plane_diagnostics(
            landscape_data=landscape_data,
            v_clean=v_target,
            v_noisy=v_noisy,
            v_denoised=v_denoised,
        )
    )
    runtime_metrics["start_mode"] = str(start_mode_used)
    runtime_metrics["start_distance_to_target"] = float((v_target - v_noisy).norm().item())
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
        v_clean=v_target,
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
        target_label=str(runtime_metrics.get("target_label", target_label)),
        eval_objective=str(runtime_metrics.get("eval_objective", eval_objective)),
        query_target_cos=float(runtime_metrics.get("query_target_cos"))
        if runtime_metrics.get("query_target_cos") is not None
        else None,
        start_mode=str(runtime_metrics.get("start_mode", start_mode_used)),
        start_distance_to_target=float(runtime_metrics.get("start_distance_to_target", float("nan"))),
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
    stage15_rt = _extract_stage15_runtime_options(checkpoint)

    v_query, v_target, v_noisy, reference_source, target_label, eval_objective, _ = _sample_inference_triplet(
        checkpoint_payload=checkpoint,
        model_type=model_type,
        stage1_cfg=stage1_cfg,
        device=device,
        noise_scale=float(preview_noise),
        seed=42,
        retrieval_bank_size=int(sota_eval_bank_size),
    )
    sigma_preview = (
        torch.full((v_noisy.shape[0], 1), float(preview_noise), device=device, dtype=v_noisy.dtype)
        if model_type == "simple"
        else None
    )
    tangent_noise = bool(stage15_rt["langevin_tangent_noise"]) if eval_objective == "conditional_retrieval" else False

    v_denoised, trajectory, result = run_langevin_denoise(
        model=model,
        v_clean=v_target,
        v_noisy=v_noisy,
        model_type=model_type,
        stage1_cfg=stage1_cfg,
        max_steps=int(preview_steps),
        lr_override=None,
        force_full_steps=True,
        v_query_override=v_query,
        v_target_override=v_target,
        tangent_noise_override=tangent_noise,
        sigma_override=sigma_preview,
    )

    landscape_data = scan_energy_landscape_3d(
        energy_fn=model,
        v_clean=v_target,
        v_noisy=v_noisy,
        grid_size=grid,
        range_factor=rf,
        absolute_half_range=abs_range,
        v_denoised=v_denoised,
        trajectory=trajectory,
        model_type=model_type,
    )
    if eval_objective == "conditional_retrieval":
        if isinstance(landscape_data.get("point_labels"), dict):
            landscape_data["point_labels"]["clean"] = "Retrieved Target"
            landscape_data["point_labels"]["noisy"] = "Noisy Seed"
            landscape_data["point_labels"]["denoised"] = "Refined"
        landscape_data["axis1_label"] = "Direction 1 (target -> noisy)"
    landscape_data["runtime_metrics"] = _build_runtime_metrics(
        model=model,
        model_type=model_type,
        v_query=v_query,
        v_target=v_target,
        v_noisy=v_noisy,
        v_denoised=v_denoised,
        noise_scale=float(preview_noise),
        steps_requested=int(preview_steps),
        steps_executed=int(result.num_steps),
        stopped_early=bool(result.stopped_early),
        reference_source=reference_source,
        target_label=target_label,
        eval_objective=eval_objective,
        sigma_override=sigma_preview,
    )
    landscape_data["runtime_metrics"].update(
        _compute_plane_diagnostics(
            landscape_data=landscape_data,
            v_clean=v_target,
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
                    minimum=0.0,
                    maximum=0.5,
                    value=0.0002,
                    step=0.0001,
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
                start_mode_dropdown = gr.Dropdown(
                    choices=[
                        ("Objective Seed (Default)", "objective_seed"),
                        ("Far Auto From Target", "far_auto"),
                        ("Manual XY (Target Plane)", "manual_plane_xy"),
                    ],
                    value="objective_seed",
                    label="Noisy Start Mode",
                    info="Use far/manual start to stress-test long-range navigation to target.",
                )
                far_start_scale_slider = gr.Slider(
                    minimum=1.0,
                    maximum=40.0,
                    value=8.0,
                    step=0.5,
                    label="Far Start Scale",
                    info="Multiplier for baseline ||seed-target|| distance in `far_auto` mode.",
                )
                manual_start_x_slider = gr.Slider(
                    minimum=-40.0,
                    maximum=40.0,
                    value=8.0,
                    step=0.5,
                    label="Manual Start X (plane)",
                    info="X coefficient in target-centric 2D plane (units: baseline distance).",
                )
                manual_start_y_slider = gr.Slider(
                    minimum=-40.0,
                    maximum=40.0,
                    value=0.0,
                    step=0.5,
                    label="Manual Start Y (plane)",
                    info="Y coefficient in target-centric 2D plane (units: baseline distance).",
                )
                project_start_norm_checkbox = gr.Checkbox(
                    value=False,
                    label="Project Start To Target Norm",
                    info="If enabled, start is normalized to target norm after far/manual placement.",
                )

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
                    minimum=0.0,
                    maximum=0.5,
                    value=0.0002,
                    step=0.0001,
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

        # === Tab 5: Chain Head Diagnostics ===
        with gr.TabItem("Chain Head Diagnostics"):
            gr.Markdown(
                """
            ### Chain Head Analysis

            Diagnose Chain Head behavior: attention patterns, energy sensitivity,
            positive vs negative discrimination, and overfitting detection.
            """
            )

            with gr.Row():
                chain_ckpt_input = gr.Textbox(
                    label="Chain Head Checkpoint",
                    value="experiments/09_stage3_chain/checkpoints/best_chain_head.pt",
                )
                chain_data_input = gr.Textbox(
                    label="SONAR Data Path",
                    value="data/squad_sequences.pt",
                )

            with gr.Row():
                chain_seq_idx = gr.Slider(
                    minimum=0, maximum=999, value=0, step=1,
                    label="Sequence Index",
                )
                chain_length = gr.Slider(
                    minimum=5, maximum=20, value=10, step=1,
                    label="Chain Length",
                )
                chain_analyze_btn = gr.Button("Analyze Chain Head", variant="primary")

            chain_info_md = gr.Markdown("Load a checkpoint to begin.")

            gr.Markdown("#### Attention Patterns")
            with gr.Row():
                chain_layer_select = gr.Radio(
                    choices=["0", "1"], value="0", label="Layer",
                )
            chain_attn_all_heads = gr.Plot(label="All Heads Attention")
            chain_attn_single = gr.Plot(label="Average Attention")

            gr.Markdown("#### Energy Sensitivity")
            with gr.Row():
                chain_swap_plot = gr.Plot(label="Adjacent Swap Sensitivity")
                chain_growth_plot = gr.Plot(label="Energy vs Chain Length")

            gr.Markdown("#### Positive vs Negative Discrimination")
            chain_violin_plot = gr.Plot(label="Energy Distributions by Perturbation Type")

            gr.Markdown("#### Training Log Analysis")
            chain_log_input = gr.Textbox(
                label="Paste training log here (or path to log file)",
                lines=5,
                placeholder="Paste Phase A training output...",
            )
            chain_log_btn = gr.Button("Analyze Training Log")
            with gr.Row():
                chain_overfit_plot = gr.Plot(label="Train vs Val (Overfitting Detection)")
                chain_energy_scale_plot = gr.Plot(label="Energy Scale Monitoring")

        # === Tab 6: Inference Diagnostics ===
        with gr.TabItem("Inference Diagnostics"):
            gr.Markdown(
                """
            ### Comprehensive Inference Diagnostics

            Load pairwise + chain head checkpoints, run System 1/2 inference with full visibility:
            attention heatmaps, energy landscape, trajectory tracking, and numerical metrics.
            """
            )

            # --- Model Loading ---
            gr.Markdown("#### Load Models")
            with gr.Row():
                diag_pairwise_path = gr.Textbox(
                    label="Pairwise Checkpoint",
                    value="experiments/05_stage15_twin/checkpoints/best_model.pt",
                )
                diag_pairwise_load_btn = gr.Button("Load Pairwise", variant="secondary")
                diag_pairwise_status = gr.Textbox(label="Pairwise Status", lines=3, interactive=False)

            with gr.Row():
                diag_chain_path = gr.Textbox(
                    label="Chain Head Checkpoint",
                    value="experiments/09_stage3_chain/checkpoints/best_phase_b.pt",
                )
                diag_chain_load_btn = gr.Button("Load Chain Head", variant="secondary")
                diag_chain_status = gr.Textbox(label="Chain Head Status", lines=3, interactive=False)

            with gr.Row():
                diag_sonar_btn = gr.Button("Load SONAR (for text)", variant="secondary")
                diag_sonar_status = gr.Textbox(label="SONAR Status", lines=1, interactive=False)

            # --- Inference Parameters ---
            gr.Markdown("#### Inference Parameters")
            with gr.Row():
                diag_mode = gr.Radio(
                    choices=["system1", "system2", "both"], value="system2",
                    label="Mode",
                )
                diag_max_steps = gr.Slider(
                    minimum=10, maximum=50, value=50, step=1,
                    label="Max Steps (S1<=10, S2<=50)",
                )
                diag_lr = gr.Number(value=0.01, label="Learning Rate")
                diag_noise_scale = gr.Number(value=0.005, label="Noise Scale")

            with gr.Row():
                diag_target_norm = gr.Number(value=0.2051, label="Target Norm")
                diag_chain_eval_every = gr.Slider(
                    minimum=1, maximum=20, value=5, step=1,
                    label="Chain Eval Every (S2)",
                )
                diag_backtrack_patience = gr.Slider(
                    minimum=5, maximum=100, value=30, step=5,
                    label="Backtrack Patience (S2)",
                )
                diag_max_chain_len = gr.Slider(
                    minimum=3, maximum=20, value=20, step=1,
                    label="Max Chain Len (S2)",
                )

            # --- Data Source ---
            gr.Markdown("#### Data Source")
            with gr.Row():
                diag_data_path = gr.Textbox(
                    label="SONAR Data Path",
                    value="data/squad_sequences.pt",
                )
                diag_seq_idx = gr.Slider(
                    minimum=0, maximum=999, value=0, step=1,
                    label="Sequence Index",
                )
                diag_noise_pct = gr.Slider(
                    minimum=0.1, maximum=20.0, value=5.0, step=0.1,
                    label="Noise %",
                )
                diag_run_data_btn = gr.Button("Run Inference (Data)", variant="primary")

            # --- Text Input ---
            gr.Markdown("#### Text Input (requires SONAR)")
            with gr.Row():
                diag_text_input = gr.Textbox(
                    label="Input Text",
                    lines=2,
                    placeholder="Enter text to encode with SONAR...",
                )
                diag_preset_dropdown = gr.Dropdown(
                    choices=[q["label"] for q in PRESET_QUESTIONS],
                    label="Preset Questions",
                    interactive=True,
                )
            diag_run_text_btn = gr.Button("Run Inference (Text)", variant="primary")

            # --- Results ---
            gr.Markdown("#### Results")
            diag_metrics_md = gr.Markdown("Run inference to see results.")

            with gr.Tabs():
                with gr.TabItem("Energy Trajectory"):
                    diag_energy_plot = gr.Plot(label="Energy over Steps")
                with gr.TabItem("Cosine Trajectory"):
                    diag_cos_plot = gr.Plot(label="Cosine Similarity to Target")
                with gr.TabItem("Attention Animation"):
                    with gr.Row():
                        diag_attn_layer = gr.Radio(
                            choices=["0", "1"], value="0", label="Layer",
                        )
                    diag_attn_anim = gr.Plot(label="Attention Evolution")
                with gr.TabItem("Attention Grid"):
                    with gr.Row():
                        diag_attn_grid_step = gr.Slider(
                            minimum=0, maximum=50, value=0, step=1,
                            label="Snapshot Index (0 = last)",
                        )
                        diag_attn_grid_layer = gr.Radio(
                            choices=["0", "1"], value="0", label="Layer",
                        )
                    diag_attn_grid = gr.Plot(label="All Heads at Snapshot")
                with gr.TabItem("3D Landscape"):
                    diag_landscape_3d = gr.Plot(label="Energy Surface + Trajectory")
                with gr.TabItem("2D Contour"):
                    diag_contour = gr.Plot(label="Contour + Trajectory")

            gr.Markdown("#### Export Metrics")
            with gr.Row():
                diag_export_btn = gr.Button("Export JSON")
                diag_export_csv_btn = gr.Button("Export CSV (per-step)")
                diag_export_output = gr.Textbox(label="Exported Metrics", lines=10, interactive=False)

        # === Tab 7: Context Encoder Diagnostics ===
        with gr.TabItem("Context Encoder Diagnostics"):
            gr.Markdown(
                """
            ### Context Encoder Stress Test

            Evaluate pretrained `ContextEncoder` on real sequence data with:
            - cosine / L2 / MSE / norm metrics
            - robustness sweep across input noise levels
            - length-bucket breakdown (short / medium / long contexts)
            """
            )

            gr.Markdown("#### Load Models")
            with gr.Row():
                ce_diag_config_path = gr.Textbox(
                    label="Stage2 Config Path",
                    value="configs/stage2_ce_config.json",
                )
                ce_diag_ckpt_path = gr.Textbox(
                    label="CE Checkpoint",
                    value="experiments/08_autoregressor/ce/checkpoints/best.pt",
                )
            with gr.Row():
                ce_diag_use_surprise = gr.Checkbox(
                    value=True,
                    label="Use Surprise Features (load SP)",
                )
                ce_diag_sp_ckpt_path = gr.Textbox(
                    label="SP Checkpoint (optional if present in config.init.sp_checkpoint)",
                    value="experiments/08_autoregressor/sp/checkpoints/best.pt",
                )
                ce_diag_load_btn = gr.Button("Load CE Bundle", variant="secondary")

            ce_diag_load_status = gr.Markdown("Load CE bundle to start diagnostics.")

            gr.Markdown("#### Evaluation Settings")
            with gr.Row():
                ce_diag_data_path = gr.Textbox(
                    label="SONAR Sequence Dataset Path",
                    value="data/squad_sequences.pt",
                )
                ce_diag_split_mode = gr.Radio(
                    choices=["all", "train", "val"],
                    value="val",
                    label="Split",
                )
                ce_diag_split_ratio = gr.Slider(
                    minimum=0.5,
                    maximum=0.99,
                    value=0.9,
                    step=0.01,
                    label="Train/Val Split Ratio",
                )
            with gr.Row():
                ce_diag_batch_size = gr.Slider(
                    minimum=4,
                    maximum=128,
                    value=32,
                    step=4,
                    label="Batch Size",
                )
                ce_diag_max_batches = gr.Slider(
                    minimum=0,
                    maximum=500,
                    value=0,
                    step=10,
                    label="Max Batches (0 = full split)",
                )
                ce_diag_seed = gr.Number(value=42, label="Seed")
                ce_diag_target_norm = gr.Number(value=0.2051, label="Target Norm (<=0 disables renorm)")
            with gr.Row():
                ce_diag_noise_levels = gr.Textbox(
                    label="Noise Levels (%)",
                    value="0,2,5,10,20",
                    info="Comma-separated percentages for robustness sweep.",
                )
                ce_diag_run_btn = gr.Button("Run CE Diagnostics", variant="primary")

            ce_diag_metrics_md = gr.Markdown("Run diagnostics to view metrics.")
            with gr.Row():
                ce_diag_noise_plot = gr.Plot(label="Noise Robustness")
                ce_diag_bucket_plot = gr.Plot(label="Length Bucket Metrics")

    # === Event Handlers ===

    #ÃƒÆ’Ã‚ÂÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬ÂÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â°ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â³ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã¢â‚¬ËœÃƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â·ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂºÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â° ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂµÃƒÆ’Ã‚ÂÃƒâ€šÃ‚ÂºÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¿ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¸ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â½ÃƒÆ’Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â¾ÃƒÆ’Ã‚ÂÃƒâ€šÃ‚Â²
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
        outputs=[checkpoint_summary, landscape_plot, inference_output, trajectory_plot, noise_scale_slider, num_steps_slider, lr_slider],
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
        outputs=[checkpoint_summary, landscape_plot, inference_output, trajectory_plot, noise_scale_slider, num_steps_slider, lr_slider],
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
        outputs=[checkpoint_summary, landscape_plot, inference_output, trajectory_plot, noise_scale_slider, num_steps_slider, lr_slider],
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
        outputs=[checkpoint_summary, landscape_plot, inference_output, trajectory_plot, noise_scale_slider, num_steps_slider, lr_slider],
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
        outputs=[checkpoint_summary, landscape_plot, inference_output, trajectory_plot, noise_scale_slider, num_steps_slider, lr_slider],
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
        outputs=[checkpoint_summary, landscape_plot, inference_output, trajectory_plot, noise_scale_slider, num_steps_slider, lr_slider],
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
        outputs=[checkpoint_summary, landscape_plot, inference_output, trajectory_plot, noise_scale_slider, num_steps_slider, lr_slider],
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
            start_mode_dropdown,
            far_start_scale_slider,
            manual_start_x_slider,
            manual_start_y_slider,
            project_start_norm_checkbox,
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

    # === Chain Head Diagnostics Handlers ===

    def analyze_chain_head_fn(ckpt_path, data_path, seq_idx, chain_len, layer_str):
        """Main analysis function for Chain Head diagnostics tab."""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            if not Path(ckpt_path).exists():
                return (f"Checkpoint not found: {ckpt_path}",
                        None, None, None, None, None)

            model, info = load_chain_head_from_checkpoint(ckpt_path, device)
            chain = make_sample_chain(data_path, int(chain_len), int(seq_idx), device)

            info_text = (
                f"**Chain Head loaded**: {info['params']:,} params, "
                f"epoch {info['epoch']}, best rank_acc={info['best_rank_acc']}\n\n"
                f"**Sample chain**: seq #{int(seq_idx)}, length {chain.shape[1]}\n\n"
                f"**Val metrics**: {json.dumps({k: round(v, 4) for k, v in info['val_metrics'].items()}, indent=2) if info['val_metrics'] else 'N/A'}"
            )

            # Attention
            layer_idx = int(layer_str)
            attn_maps = extract_attention_weights(model, chain)
            all_heads_fig = create_all_heads_heatmap(attn_maps, layer_idx)
            avg_fig = create_attention_heatmap(attn_maps, layer_idx, head_idx=None)

            # Swap sensitivity
            swap_data = compute_swap_sensitivity(model, chain, torch.device(device))
            swap_fig = create_swap_sensitivity_plot(swap_data)

            # Chain growth
            growth_data = compute_chain_growth_energy(model, chain, torch.device(device))
            growth_fig = create_chain_growth_plot(growth_data)

            # Pos vs neg
            comparison = compute_pos_neg_comparison(model, chain, torch.device(device))
            violin_fig = create_pos_neg_violin(comparison)

            return info_text, all_heads_fig, avg_fig, swap_fig, growth_fig, violin_fig
        except Exception as e:
            import traceback
            err = f"Error: {e}\n\n```\n{traceback.format_exc()}\n```"
            return err, None, None, None, None, None

    def analyze_training_log_fn(log_text):
        """Parse and visualize training log."""
        try:
            # Check if it's a file path
            if log_text.strip() and Path(log_text.strip()).exists():
                log_text = Path(log_text.strip()).read_text()

            parsed = parse_training_log(log_text)
            n_train = len(parsed.get("train", []))
            n_val = len(parsed.get("val", []))
            if n_train == 0 and n_val == 0:
                empty = go.Figure()
                empty.update_layout(title="No epoch data found in log")
                return empty, empty

            overfit_fig = create_overfitting_plot(parsed)
            energy_fig = create_energy_scale_plot(parsed)
            return overfit_fig, energy_fig
        except Exception as e:
            err_fig = go.Figure()
            err_fig.update_layout(title=f"Error: {e}")
            return err_fig, err_fig

    chain_analyze_btn.click(
        analyze_chain_head_fn,
        inputs=[chain_ckpt_input, chain_data_input, chain_seq_idx, chain_length, chain_layer_select],
        outputs=[chain_info_md, chain_attn_all_heads, chain_attn_single,
                 chain_swap_plot, chain_growth_plot, chain_violin_plot],
    )

    chain_log_btn.click(
        analyze_training_log_fn,
        inputs=[chain_log_input],
        outputs=[chain_overfit_plot, chain_energy_scale_plot],
    )

    # === Inference Diagnostics Handlers ===

    # Store last inference result in session state
    session_state["diag_result"] = None

    def diag_load_pairwise_fn(path):
        try:
            return diag_load_pairwise(path)
        except Exception as e:
            import traceback
            return f"Error: {e}\n{traceback.format_exc()}"

    def diag_load_chain_fn(path):
        try:
            return diag_load_chain(path)
        except Exception as e:
            import traceback
            return f"Error: {e}\n{traceback.format_exc()}"

    def diag_load_sonar_fn():
        try:
            return diag_load_sonar()
        except Exception as e:
            return f"Error: {e}"

    def diag_preset_selected(label):
        for q in PRESET_QUESTIONS:
            if q["label"] == label:
                return q["text"]
        return ""

    def diag_run_data_fn(
        data_path, seq_idx, noise_pct, mode, max_steps, lr, noise_scale,
        target_norm, chain_eval_every, backtrack_patience, max_chain_len,
        attn_layer_str,
    ):
        try:
            result = diag_run_inference(
                mode=mode, data_path=data_path, seq_idx=int(seq_idx),
                noise_pct=noise_pct, max_steps=int(max_steps), lr=lr,
                noise_scale=noise_scale, target_norm=target_norm,
                chain_eval_every=int(chain_eval_every),
                backtrack_patience=int(backtrack_patience),
                max_chain_len=int(max_chain_len),
            )
            session_state["diag_result"] = result
            return _diag_build_outputs(result, int(attn_layer_str))
        except Exception as e:
            import traceback
            err = f"Error: {e}\n```\n{traceback.format_exc()}\n```"
            empty = go.Figure()
            return err, empty, empty, empty, empty, empty, empty

    def diag_run_text_fn(
        text, mode, max_steps, lr, noise_scale, target_norm, noise_pct,
        chain_eval_every, backtrack_patience, max_chain_len,
        attn_layer_str,
    ):
        try:
            result = diag_run_text_inference(
                text=text, mode=mode, noise_pct=noise_pct,
                max_steps=int(max_steps), lr=lr,
                noise_scale=noise_scale, target_norm=target_norm,
                chain_eval_every=int(chain_eval_every),
                backtrack_patience=int(backtrack_patience),
                max_chain_len=int(max_chain_len),
            )
            session_state["diag_result"] = result
            return _diag_build_outputs(result, int(attn_layer_str))
        except Exception as e:
            import traceback
            err = f"Error: {e}\n```\n{traceback.format_exc()}\n```"
            empty = go.Figure()
            return err, empty, empty, empty, empty, empty, empty

    def _diag_build_outputs(result, attn_layer):
        md = format_metrics_markdown(result)
        energy_fig = create_energy_trajectory_plot(result)
        cos_fig = create_cosine_trajectory_plot(result)
        attn_anim = create_attention_animation(result, layer_idx=attn_layer)

        # Attention grid: use last snapshot
        attn_grid = create_attention_grid(result, snapshot_idx=-1, layer_idx=attn_layer)

        # Landscape plots (can be slow)
        try:
            landscape = create_landscape_with_trajectory(result)
        except Exception:
            landscape = go.Figure()
            landscape.update_layout(title="Landscape unavailable")

        try:
            contour = create_contour_with_trajectory(result)
        except Exception:
            contour = go.Figure()
            contour.update_layout(title="Contour unavailable")

        return md, energy_fig, cos_fig, attn_anim, attn_grid, landscape, contour

    def diag_update_attn_anim(layer_str):
        result = session_state.get("diag_result")
        if result is None:
            return go.Figure()
        return create_attention_animation(result, layer_idx=int(layer_str))

    def diag_update_attn_grid(step_idx, layer_str):
        result = session_state.get("diag_result")
        if result is None:
            return go.Figure()
        idx = int(step_idx) if int(step_idx) > 0 else -1
        return create_attention_grid(result, snapshot_idx=idx, layer_idx=int(layer_str))

    def diag_export_fn():
        result = session_state.get("diag_result")
        if result is None:
            return "No inference result to export."
        return export_metrics_json(result)

    def diag_export_csv_fn():
        result = session_state.get("diag_result")
        if result is None:
            return "No inference result to export."
        return export_metrics_csv(result)

    # Wire up event handlers
    diag_pairwise_load_btn.click(
        diag_load_pairwise_fn,
        inputs=[diag_pairwise_path],
        outputs=[diag_pairwise_status],
    )
    diag_chain_load_btn.click(
        diag_load_chain_fn,
        inputs=[diag_chain_path],
        outputs=[diag_chain_status],
    )
    diag_sonar_btn.click(
        diag_load_sonar_fn,
        outputs=[diag_sonar_status],
    )
    diag_preset_dropdown.change(
        diag_preset_selected,
        inputs=[diag_preset_dropdown],
        outputs=[diag_text_input],
    )

    diag_run_data_btn.click(
        diag_run_data_fn,
        inputs=[
            diag_data_path, diag_seq_idx, diag_noise_pct, diag_mode,
            diag_max_steps, diag_lr, diag_noise_scale, diag_target_norm,
            diag_chain_eval_every, diag_backtrack_patience, diag_max_chain_len,
            diag_attn_layer,
        ],
        outputs=[
            diag_metrics_md, diag_energy_plot, diag_cos_plot,
            diag_attn_anim, diag_attn_grid, diag_landscape_3d, diag_contour,
        ],
    )

    diag_run_text_btn.click(
        diag_run_text_fn,
        inputs=[
            diag_text_input, diag_mode, diag_max_steps, diag_lr,
            diag_noise_scale, diag_target_norm, diag_noise_pct,
            diag_chain_eval_every, diag_backtrack_patience, diag_max_chain_len,
            diag_attn_layer,
        ],
        outputs=[
            diag_metrics_md, diag_energy_plot, diag_cos_plot,
            diag_attn_anim, diag_attn_grid, diag_landscape_3d, diag_contour,
        ],
    )

    diag_attn_layer.change(
        diag_update_attn_anim,
        inputs=[diag_attn_layer],
        outputs=[diag_attn_anim],
    )
    diag_attn_grid_step.change(
        diag_update_attn_grid,
        inputs=[diag_attn_grid_step, diag_attn_grid_layer],
        outputs=[diag_attn_grid],
    )
    diag_attn_grid_layer.change(
        diag_update_attn_grid,
        inputs=[diag_attn_grid_step, diag_attn_grid_layer],
        outputs=[diag_attn_grid],
    )
    diag_export_btn.click(
        diag_export_fn,
        outputs=[diag_export_output],
    )
    diag_export_csv_btn.click(
        diag_export_csv_fn,
        outputs=[diag_export_output],
    )

    # === Context Encoder Diagnostics Handlers ===

    def ce_diag_load_fn(config_path, ce_ckpt, sp_ckpt, use_surprise):
        try:
            return ce_diag_load_models(
                config_path=str(config_path),
                ce_checkpoint_path=str(ce_ckpt),
                sp_checkpoint_path=str(sp_ckpt),
                use_surprise=bool(use_surprise),
                device="auto",
            )
        except Exception as e:
            import traceback
            return f"Error: {e}\n\n```\n{traceback.format_exc()}\n```"

    def ce_diag_run_fn(
        data_path,
        split_mode,
        split_ratio,
        batch_size,
        max_batches,
        noise_levels,
        seed,
        target_norm,
    ):
        try:
            md, noise_fig, bucket_fig = ce_diag_run(
                data_path=str(data_path),
                split_mode=str(split_mode),
                split_ratio=float(split_ratio),
                batch_size=int(batch_size),
                max_batches=int(max_batches),
                noise_levels_csv=str(noise_levels),
                seed=int(seed),
                target_norm=float(target_norm),
            )
            return md, noise_fig, bucket_fig
        except Exception as e:
            import traceback
            err = f"Error: {e}\n\n```\n{traceback.format_exc()}\n```"
            return err, go.Figure(), go.Figure()

    ce_diag_load_btn.click(
        ce_diag_load_fn,
        inputs=[ce_diag_config_path, ce_diag_ckpt_path, ce_diag_sp_ckpt_path, ce_diag_use_surprise],
        outputs=[ce_diag_load_status],
    )

    ce_diag_run_btn.click(
        ce_diag_run_fn,
        inputs=[
            ce_diag_data_path,
            ce_diag_split_mode,
            ce_diag_split_ratio,
            ce_diag_batch_size,
            ce_diag_max_batches,
            ce_diag_noise_levels,
            ce_diag_seed,
            ce_diag_target_norm,
        ],
        outputs=[ce_diag_metrics_md, ce_diag_noise_plot, ce_diag_bucket_plot],
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
