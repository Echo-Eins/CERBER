"""
Universal checkpoint loader — auto-detects SimpleEnergy vs UnconditionalEnergy.

Provides introspection: architecture params, weight stats, eval metrics, etc.
"""

import math
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class LayerStats:
    """Per-layer weight statistics."""
    name: str
    shape: tuple
    num_params: int
    mean: float
    std: float
    min_val: float
    max_val: float
    norm: float
    grad_norm: float | None = None


@dataclass
class CheckpointInfo:
    """Parsed checkpoint information."""
    path: str
    model_type: str  # "simple_energy" | "unconditional_energy"
    epoch: int
    global_step: int | None
    config: dict
    eval_metrics: dict | None
    layer_stats: list[LayerStats]
    total_params: int
    trainable_params: int
    log_energy_scale: float | None
    loss_type: str | None
    state_dict_keys: list[str]
    extra: dict = field(default_factory=dict)


def infer_model_type(state_dict: dict) -> str:
    """Auto-detect model type from state_dict keys."""
    if "_sigma_freqs" in state_dict:
        return "simple_energy"
    return "unconditional_energy"


def extract_layer_stats(state_dict: dict) -> list[LayerStats]:
    """Compute per-parameter statistics."""
    stats = []
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        t = tensor.float()
        stats.append(LayerStats(
            name=name,
            shape=tuple(tensor.shape),
            num_params=tensor.numel(),
            mean=t.mean().item(),
            std=t.std().item() if t.numel() > 1 else 0.0,
            min_val=t.min().item(),
            max_val=t.max().item(),
            norm=t.norm().item(),
        ))
    return stats


def load_checkpoint(path: str | Path) -> CheckpointInfo:
    """Load and inspect a checkpoint file."""
    path = str(path)
    ckpt = torch.load(path, weights_only=False, map_location="cpu")

    state_dict = ckpt.get("model_state", {})
    model_type = infer_model_type(state_dict)
    config = ckpt.get("config", {})
    epoch = ckpt.get("epoch", -1)
    global_step = ckpt.get("global_step")

    # Extract eval metrics
    eval_metrics = ckpt.get("final_eval") or ckpt.get("metrics")

    # Layer stats
    layer_stats = extract_layer_stats(state_dict)

    # Total params
    total_params = sum(s.num_params for s in layer_stats)

    # log_energy_scale
    log_energy_scale = None
    if "log_energy_scale" in state_dict:
        log_energy_scale = state_dict["log_energy_scale"].item()

    # Loss type
    loss_type = config.get("loss_type")

    # Extra info
    extra = {}
    if "stage1_config" in ckpt:
        extra["stage1_config"] = ckpt["stage1_config"]
    if "optimizer_state" in ckpt:
        extra["has_optimizer"] = True
    if "scheduler_state" in ckpt:
        extra["has_scheduler"] = True
    if "actor_state" in ckpt and ckpt["actor_state"] is not None:
        extra["has_actor"] = True
        extra["actor_params"] = sum(
            t.numel() for t in ckpt["actor_state"].values()
            if isinstance(t, torch.Tensor)
        )

    return CheckpointInfo(
        path=path,
        model_type=model_type,
        epoch=epoch,
        global_step=global_step,
        config=config,
        eval_metrics=eval_metrics,
        layer_stats=layer_stats,
        total_params=total_params,
        trainable_params=total_params,  # all are trainable in these models
        log_energy_scale=log_energy_scale,
        loss_type=loss_type,
        state_dict_keys=list(state_dict.keys()),
        extra=extra,
    )


def load_model_from_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, CheckpointInfo]:
    """Load and instantiate a model from checkpoint.

    Returns:
        (model, checkpoint_info)
    """
    import sys
    from pathlib import Path as P
    # Ensure project root is on path for imports
    project_root = str(P(__file__).resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from cebcm.models.energy import SimpleEnergy
    from cebcm.models.energy_unconditional import UnconditionalEnergy

    info = load_checkpoint(path)
    ckpt = torch.load(str(path), weights_only=False, map_location=device)
    state_dict = ckpt["model_state"]
    config = info.config

    dim = config.get("energy_dim", 1024)
    hidden_dims = config.get("energy_hidden_dims", [2048, 1024, 512])
    norm_mode = config.get("norm_mode", "orthonorm")
    activation = config.get("activation", "groupsort")

    if info.model_type == "simple_energy":
        model = SimpleEnergy(
            dim=dim,
            hidden_dims=hidden_dims,
            norm_mode=norm_mode,
            activation=activation,
        )
    else:
        model = UnconditionalEnergy(
            dim=dim,
            hidden_dims=hidden_dims,
            norm_mode=norm_mode,
            activation=activation,
        )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, info
