"""
Shared utilities for modular Stage 2 training pipelines.

These helpers are intentionally lightweight and script-friendly:
- config/device/AMP setup
- dataset + dataloader creation
- Stage2 model builders (SurprisePredictor, ContextEncoder, IPP)
- common schedulers/metrics/checkpoint helpers
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from cebcm.data.sequence_dataset import SONARSequenceDataset
from cebcm.models.context_encoder import ContextEncoder, ContextEncoderConfig
from cebcm.models.ipp import FlowIPP, IPPConfig, MLPIPP
from cebcm.models.surprise import SurpriseConfig, SurprisePredictor


def load_config(config_path: str | Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def resolve_device(override: str | None = None) -> torch.device:
    if override:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def setup_amp(amp_cfg: dict, device: torch.device) -> tuple[bool, torch.dtype, torch.amp.GradScaler]:
    amp_enabled = bool(amp_cfg.get("enabled", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_cfg.get("dtype", "bf16") == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16,
    )
    return amp_enabled, amp_dtype, scaler


class MetricTracker:
    def __init__(self):
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def update(self, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            self._sums[k] = self._sums.get(k, 0.0) + float(v)
            self._counts[k] = self._counts.get(k, 0) + 1

    def get(self) -> dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_factor: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(min_factor, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_factor, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_type_ids(lengths: Tensor, max_len: int, dataset_source: str) -> Tensor:
    """
    Build sequence type IDs for context encoder.

    SQuAD convention:
      [context..., question, answer]
      context -> 1, question -> 0, answer -> 1.

    Non-SQuAD corpora:
      all zeros (single generic stream type).
    """
    B = lengths.shape[0]
    type_ids = torch.zeros(B, max_len, dtype=torch.long)

    if dataset_source.lower().startswith("squad"):
        for i in range(B):
            L = int(lengths[i].item())
            if L >= 3:
                type_ids[i, : L - 2] = 1
                type_ids[i, L - 2] = 0
                type_ids[i, L - 1] = 1

    return type_ids


def build_surprise_predictor(cfg: dict) -> SurprisePredictor:
    sp_cfg = cfg.get("surprise", {})
    return SurprisePredictor(SurpriseConfig(
        d_model=sp_cfg.get("d_model", 1024),
        ssm_d_state=sp_cfg.get("ssm_d_state", 64),
        ssm_d_conv=sp_cfg.get("ssm_d_conv", 4),
        ssm_expand=sp_cfg.get("ssm_expand", 2),
        ssm_n_layers=sp_cfg.get("ssm_n_layers", 2),
        ssm_dropout=sp_cfg.get("ssm_dropout", 0.0),
        pred_hidden=sp_cfg.get("pred_hidden", 1024),
        cos_loss_weight=sp_cfg.get("cos_loss_weight", 0.5),
        threshold_mode=sp_cfg.get("threshold_mode", "percentile"),
        threshold_percentile=sp_cfg.get("threshold_percentile", 95.0),
        threshold_fixed=sp_cfg.get("threshold_fixed", 0.3),
    ))


def build_context_encoder(cfg: dict) -> ContextEncoder:
    ce_cfg = cfg.get("context_encoder", {})
    return ContextEncoder(ContextEncoderConfig(
        d_model=ce_cfg.get("d_model", 1024),
        ssm_d_state=ce_cfg.get("ssm_d_state", 64),
        ssm_d_conv=ce_cfg.get("ssm_d_conv", 4),
        ssm_expand=ce_cfg.get("ssm_expand", 2),
        ssm_n_layers=ce_cfg.get("ssm_n_layers", 2),
        ssm_dropout=ce_cfg.get("ssm_dropout", 0.05),
        n_global_heads=ce_cfg.get("n_global_heads", 8),
        global_attn_dropout=ce_cfg.get("global_attn_dropout", 0.1),
        surprise_top_k_pct=ce_cfg.get("surprise_top_k_pct", 0.05),
        surprise_top_k_min_tokens=ce_cfg.get("surprise_top_k_min_tokens", 1),
        global_include_last_token=ce_cfg.get("global_include_last_token", True),
        n_types=ce_cfg.get("n_types", 3),
        output_dim=ce_cfg.get("output_dim", 1024),
        use_alibi=ce_cfg.get("use_alibi", False),
    ))


def build_ipp(cfg: dict) -> FlowIPP | MLPIPP:
    ipp_cfg = cfg.get("ipp", {})
    ipp_config = IPPConfig(
        d_model=ipp_cfg.get("d_model", 1024),
        d_context=ipp_cfg.get("d_context", 1024),
        hidden_dims=ipp_cfg.get("hidden_dims", [2048, 2048, 1024]),
        flow_activation=ipp_cfg.get("flow_activation", "silu"),
        flow_norm=ipp_cfg.get("flow_norm", "layernorm"),
        flow_dropout=ipp_cfg.get("flow_dropout", 0.0),
        flow_zero_init_last=ipp_cfg.get("flow_zero_init_last", True),
        flow_velocity_weight=ipp_cfg.get("flow_velocity_weight", 1.0),
        # Keep defaults aligned with IPPConfig/spec for stable Flow sampling.
        n_integration_steps=ipp_cfg.get("n_integration_steps", 50),
        d_time=ipp_cfg.get("d_time", 256),
        sigma_init=ipp_cfg.get("sigma_init", 0.05),
        solver=ipp_cfg.get("solver", "midpoint"),
        endpoint_loss_weight=ipp_cfg.get("endpoint_loss_weight", 0.0),
        endpoint_cos_weight=ipp_cfg.get("endpoint_cos_weight", 0.5),
        endpoint_steps=ipp_cfg.get("endpoint_steps", 20),
        endpoint_target_norm=ipp_cfg.get("endpoint_target_norm", 0.2051),
        mlp_hidden_dims=ipp_cfg.get("mlp_hidden_dims", [2048, 1024]),
        mlp_activation=ipp_cfg.get("mlp_activation", "silu"),
        mlp_norm=ipp_cfg.get("mlp_norm", "layernorm"),
        mlp_dropout=ipp_cfg.get("mlp_dropout", 0.0),
        mlp_mse_weight=ipp_cfg.get("mlp_mse_weight", 1.0),
        mlp_cos_weight=ipp_cfg.get("mlp_cos_weight", 0.5),
        mlp_contrastive_weight=ipp_cfg.get("mlp_contrastive_weight", 0.0),
        mlp_temperature=ipp_cfg.get("mlp_temperature", 0.07),
    )
    mode = ipp_cfg.get("mode", "flow")
    if mode == "flow":
        return FlowIPP(ipp_config)
    return MLPIPP(ipp_config)


def load_train_val_datasets(data_cfg: dict, seed: int) -> tuple[SONARSequenceDataset, SONARSequenceDataset, str, str]:
    dataset = SONARSequenceDataset(
        path=data_cfg.get("train_data_path", "data/squad_sequences.pt"),
        max_seq_len=data_cfg.get("max_seq_len", 64),
        min_seq_len=data_cfg.get("min_seq_len", 3),
        legacy_window_stride=data_cfg.get("legacy_window_stride", None),
    )

    val_path = data_cfg.get("val_data_path", "")
    if val_path:
        val_dataset = SONARSequenceDataset(
            path=val_path,
            max_seq_len=data_cfg.get("max_seq_len", 64),
            min_seq_len=data_cfg.get("min_seq_len", 3),
            legacy_window_stride=data_cfg.get("legacy_window_stride", None),
        )
        train_dataset = dataset
    else:
        train_dataset, val_dataset = dataset.split(
            train_ratio=data_cfg.get("train_val_split", 0.9),
            seed=seed,
        )

    train_source = str(getattr(train_dataset, "source", "unknown"))
    val_source = str(getattr(val_dataset, "source", train_source))
    return train_dataset, val_dataset, train_source, val_source


def build_dataloaders(
    train_dataset: SONARSequenceDataset,
    val_dataset: SONARSequenceDataset,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=SONARSequenceDataset.collate,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=SONARSequenceDataset.collate,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def save_checkpoint(path: str | Path, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    return torch.load(Path(path), map_location=device, weights_only=False)
