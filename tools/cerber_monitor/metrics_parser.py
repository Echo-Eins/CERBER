"""
Parse training_metrics.json files for plotting and analysis.

Handles NaN/Inf values, extracts time series, and computes derived metrics.
"""

import json
import math
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class EpochMetric:
    """Single epoch's metrics."""
    epoch: int
    phase: str  # "nce" | "em" | ""
    train_loss: float
    train_extras: dict = field(default_factory=dict)
    eval_data: dict | None = None


@dataclass
class TrainingRun:
    """Parsed training run with time series data."""
    path: str
    total_time_seconds: float
    num_epochs: int
    best_improvement: float
    final_eval: dict
    config: dict
    epochs: list[EpochMetric]

    # Pre-computed time series for fast plotting
    loss_values: list[float] = field(default_factory=list)
    loss_epochs: list[int] = field(default_factory=list)
    phase_labels: list[str] = field(default_factory=list)

    # Eval series (sparse — only on eval epochs)
    eval_epochs: list[int] = field(default_factory=list)
    noise_levels: list[str] = field(default_factory=list)
    cos_before: dict = field(default_factory=dict)  # noise_level -> [values]
    cos_after: dict = field(default_factory=dict)
    improvements: dict = field(default_factory=dict)
    success_rates: dict = field(default_factory=dict)

    # Sample quality series
    sample_norm_mean: list[float] = field(default_factory=list)
    sample_pairwise_cos: list[float] = field(default_factory=list)
    sample_epochs: list[int] = field(default_factory=list)


def _sanitize(v: float, cap: float = 1e12) -> float:
    """Replace NaN/Inf with capped values for plotting."""
    if v is None or not math.isfinite(v):
        return cap if v is not None and v > 0 else -cap if v is not None else 0.0
    return max(-cap, min(cap, v))


def parse_metrics_file(path: str | Path) -> TrainingRun:
    """Parse a training_metrics.json file into a TrainingRun."""
    path = str(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    config = raw.get("config", {})
    epoch_metrics = raw.get("epoch_metrics", [])

    epochs: list[EpochMetric] = []
    loss_values = []
    loss_epochs = []
    phase_labels = []
    eval_epochs = []
    noise_levels_set: set[str] = set()

    cos_before: dict[str, list[float]] = {}
    cos_after: dict[str, list[float]] = {}
    improvements: dict[str, list[float]] = {}
    success_rates: dict[str, list[float]] = {}
    sample_norm_mean = []
    sample_pairwise_cos = []
    sample_epochs = []

    for em in epoch_metrics:
        epoch = em.get("epoch", 0)
        phase = em.get("phase", "")
        train = em.get("train", {})
        train_loss = _sanitize(train.get("loss", 0.0))

        train_extras = {k: _sanitize(v) for k, v in train.items() if k != "loss" and isinstance(v, (int, float))}

        eval_raw = em.get("eval")
        epoch_obj = EpochMetric(
            epoch=epoch,
            phase=phase,
            train_loss=train_loss,
            train_extras=train_extras,
            eval_data=eval_raw,
        )
        epochs.append(epoch_obj)

        loss_values.append(train_loss)
        loss_epochs.append(epoch)
        phase_labels.append(phase)

        if eval_raw:
            eval_epochs.append(epoch)

            for key, val in eval_raw.items():
                if key.startswith("noise_") and isinstance(val, dict):
                    noise_levels_set.add(key)
                    if key not in cos_before:
                        cos_before[key] = []
                        cos_after[key] = []
                        improvements[key] = []
                        success_rates[key] = []

                    cos_before[key].append(_sanitize(val.get("cos_before_mean", 0)))
                    cos_after[key].append(_sanitize(val.get("cos_after_mean", 0)))
                    improvements[key].append(_sanitize(val.get("improvement", 0)))
                    success_rates[key].append(_sanitize(val.get("success_rate", 0)))

                elif key == "samples" and isinstance(val, dict):
                    sample_epochs.append(epoch)
                    sample_norm_mean.append(_sanitize(val.get("norm_mean", 0)))
                    sample_pairwise_cos.append(_sanitize(val.get("pairwise_cos_mean", 0)))

    noise_levels = sorted(noise_levels_set, key=lambda x: float(x.split("_")[1]))

    return TrainingRun(
        path=path,
        total_time_seconds=raw.get("total_time_seconds", 0),
        num_epochs=raw.get("num_epochs", len(epochs)),
        best_improvement=_sanitize(raw.get("best_improvement", 0)),
        final_eval=raw.get("final_eval", {}),
        config=config,
        epochs=epochs,
        loss_values=loss_values,
        loss_epochs=loss_epochs,
        phase_labels=phase_labels,
        eval_epochs=eval_epochs,
        noise_levels=noise_levels,
        cos_before=cos_before,
        cos_after=cos_after,
        improvements=improvements,
        success_rates=success_rates,
        sample_norm_mean=sample_norm_mean,
        sample_pairwise_cos=sample_pairwise_cos,
        sample_epochs=sample_epochs,
    )


def load_metrics_streaming(path: str | Path) -> TrainingRun | None:
    """Load metrics file, returning None if file doesn't exist or is invalid."""
    try:
        return parse_metrics_file(path)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None
