"""
Checkpoint Analyzer — ядро системы анализа чекпоинтов CERBER.

Загрузка, парсинг, сравнение чекпоинтов и экспорт результатов.

Функции:
    load_checkpoint(path) — загрузка .pt файла с автодетекцией типа модели
    extract_metrics(checkpoint) — извлечение метаданных и метрик
    compare_checkpoints(checkpoints) — сравнение нескольких чекпоинтов
    export_comparison(results, output_path) — экспорт в JSON/CSV
"""

import json
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any

import torch
import pandas as pd


@dataclass
class CheckpointMetadata:
    """Метаданные чекпоинта."""
    path: str
    epoch: int | str
    global_step: int | None
    model_type: str  # "simple" или "unconditional"
    energy_dim: int
    energy_hidden_dims: list[int]
    norm_mode: str
    activation: str
    has_optimizer: bool
    has_scheduler: bool
    has_metrics: bool
    metrics_keys: list[str]


@dataclass
class CheckpointMetrics:
    """Извлеченные метрики из чекпоинта."""
    epoch: int | str
    global_step: int | None
    # Метрики из checkpoint["metrics"]
    train_loss: float | None
    eval_loss: float | None
    cos_sim_before: float | None
    cos_sim_after: float | None
    cos_improvement: float | None
    success_rate: float | None
    # Дополнительные метрики (динамически)
    extra_metrics: dict[str, Any]


def detect_model_type_from_state_dict(state_dict: dict) -> str:
    """
    Автодетекция типа модели из state_dict.

    Returns:
        "simple" для SimpleEnergy (pairwise, 4104 входа)
        "unconditional" для UnconditionalEnergy (1024 входа)
    """
    if "net.0.weight" in state_dict:
        input_dim = state_dict["net.0.weight"].shape[1]
        # SimpleEnergy: 4*dim + 8 = 4104 для dim=1024
        # UnconditionalEnergy: dim = 1024
        if input_dim > 2048:
            return "simple"
        else:
            return "unconditional"

    # Fallback: проверка buffer _sigma_freqs (только в SimpleEnergy)
    if "_sigma_freqs" in state_dict:
        return "simple"

    return "unconditional"


def extract_model_config(state_dict: dict, checkpoint_config: dict | None) -> dict:
    """
    Извлечение конфигурации архитектуры из state_dict.

    Вычисляет hidden_dims по количеству и размеру слоев.
    """
    config = {}

    # Определение размерности входа
    if "net.0.weight" in state_dict:
        input_dim = state_dict["net.0.weight"].shape[1]
        hidden_dim = state_dict["net.0.weight"].shape[0]
        config["energy_dim"] = 1024 if input_dim < 2048 else 1024  # SONAR dim
        config["first_hidden_dim"] = hidden_dim
    else:
        config["energy_dim"] = 1024  # default
        config["first_hidden_dim"] = 2048  # default

    # Подсчет скрытых слоев
    hidden_dims = []
    layer_idx = 0
    while f"net.{layer_idx}.weight" in state_dict:
        weight = state_dict[f"net.{layer_idx}.weight"]
        # Пропускаем финальный Linear (выход = 1)
        if weight.shape[0] == 1:
            break
        hidden_dims.append(weight.shape[0])
        layer_idx += 2  # Linear + activation

    config["energy_hidden_dims"] = hidden_dims

    # Извлечение config из чекпоинта
    if checkpoint_config:
        config["norm_mode"] = checkpoint_config.get("norm_mode", "orthonorm")
        config["activation"] = checkpoint_config.get("activation", "groupsort")
    else:
        config["norm_mode"] = "orthonorm"
        config["activation"] = "groupsort"

    return config


def load_checkpoint(path: str | Path) -> dict:
    """
    Загрузка чекпоинта из .pt файла.

    Args:
        path: Путь к файлу чекпоинта

    Returns:
        Dict с содержимым чекпоинта:
        - model_state: state_dict модели
        - optimizer_state: state_dict оптимизатора (если есть)
        - scheduler_state: state_dict scheduler (если есть)
        - epoch: номер эпохи
        - global_step: глобальный шаг (если есть)
        - metrics: метрики (если есть)
        - config: конфигурация архитектуры
        - stage1_config: Stage1Config (если есть)
        - metadata: CheckpointMetadata

    Raises:
        FileNotFoundError: если файл не найден
        ValueError: если файл не является валидным чекпоинтом
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if path.suffix not in [".pt", ".pth"]:
        raise ValueError(f"Invalid checkpoint extension: {path.suffix}")

    # Загрузка на CPU для совместимости
    device = torch.device("cpu")
    checkpoint = torch.load(path, weights_only=False, map_location=device)

    # Валидация структуры
    if "model_state" not in checkpoint:
        # Пробуем альтернативные ключи
        if "model" in checkpoint:
            checkpoint["model_state"] = checkpoint.pop("model")
        elif "state_dict" in checkpoint:
            checkpoint["model_state"] = checkpoint.pop("state_dict")
        else:
            raise ValueError(f"Invalid checkpoint format: no 'model_state' found in {path}")

    # Автодетекция типа модели
    model_type = detect_model_type_from_state_dict(checkpoint["model_state"])

    # Извлечение конфигурации
    config = checkpoint.get("config", {})
    model_config = extract_model_config(checkpoint["model_state"], config)

    # Создание метаданных
    metadata = CheckpointMetadata(
        path=str(path.absolute()),
        epoch=checkpoint.get("epoch", "?"),
        global_step=checkpoint.get("global_step"),
        model_type=model_type,
        energy_dim=model_config.get("energy_dim", 1024),
        energy_hidden_dims=model_config.get("energy_hidden_dims", []),
        norm_mode=model_config.get("norm_mode", "orthonorm"),
        activation=model_config.get("activation", "groupsort"),
        has_optimizer="optimizer_state" in checkpoint,
        has_scheduler="scheduler_state" in checkpoint,
        has_metrics="metrics" in checkpoint,
        metrics_keys=list(checkpoint.get("metrics", {}).keys()),
    )

    checkpoint["metadata"] = metadata
    checkpoint["model_type"] = model_type

    return checkpoint


def extract_metrics(checkpoint: dict) -> CheckpointMetrics:
    """
    Извлечение метрик из чекпоинта.

    Args:
        checkpoint: Загруженный чекпоинт (dict)

    Returns:
        CheckpointMetrics с извлеченными значениями
    """
    metrics_data = checkpoint.get("metrics", {})

    # Стандартные метрики
    cos_before = metrics_data.get("cos_before")
    cos_after = metrics_data.get("cos_after")
    cos_improvement = None
    if cos_before is not None and cos_after is not None:
        cos_improvement = cos_after - cos_before

    # Дополнительные метрики (все что есть в metrics)
    extra = {}
    known_keys = {"train_loss", "eval_loss", "cos_before", "cos_after", "success_rate"}
    for key, value in metrics_data.items():
        if key not in known_keys and isinstance(value, (int, float)):
            extra[key] = value

    return CheckpointMetrics(
        epoch=checkpoint.get("epoch", "?"),
        global_step=checkpoint.get("global_step"),
        train_loss=metrics_data.get("train_loss"),
        eval_loss=metrics_data.get("eval_loss"),
        cos_sim_before=cos_before,
        cos_sim_after=cos_after,
        cos_improvement=cos_improvement,
        success_rate=metrics_data.get("success_rate"),
        extra_metrics=extra,
    )


def compare_checkpoints(checkpoint_paths: list[str | Path]) -> pd.DataFrame:
    """
    Сравнение нескольких чекпоинтов.

    Args:
        checkpoint_paths: Список путей к чекпоинтам

    Returns:
        DataFrame с сравнительными метриками
    """
    results = []

    for path in checkpoint_paths:
        try:
            checkpoint = load_checkpoint(path)
            metrics = extract_metrics(checkpoint)
            metadata = checkpoint["metadata"]

            row = {
                "path": Path(path).name,
                "full_path": str(Path(path).absolute()),
                "epoch": metrics.epoch,
                "global_step": metrics.global_step,
                "model_type": metadata.model_type,
                "energy_dim": metadata.energy_dim,
                "hidden_dims": str(metadata.energy_hidden_dims),
                "norm_mode": metadata.norm_mode,
                "activation": metadata.activation,
                "train_loss": metrics.train_loss,
                "eval_loss": metrics.eval_loss,
                "cos_before": metrics.cos_sim_before,
                "cos_after": metrics.cos_sim_after,
                "cos_improvement": metrics.cos_improvement,
                "success_rate": metrics.success_rate,
                "has_optimizer": metadata.has_optimizer,
                "has_scheduler": metadata.has_scheduler,
            }

            # Добавляем экстра-метрики
            for key, value in metrics.extra_metrics.items():
                row[f"extra_{key}"] = value

            results.append(row)

        except Exception as e:
            results.append({
                "path": Path(path).name,
                "error": str(e),
            })

    return pd.DataFrame(results)


def export_comparison(
    df: pd.DataFrame,
    output_path: str | Path,
    format: str | None = None,
) -> None:
    """
    Экспорт сравнения чекпоинтов в файл.

    Args:
        df: DataFrame с результатами сравнения
        output_path: Путь к выходному файлу
        format: Формат экспорта ("json", "csv", "auto")
                Если "auto" (default), определяется по расширению

    Raises:
        ValueError: если формат не поддерживается
    """
    output_path = Path(output_path)

    if format is None or format == "auto":
        if output_path.suffix == ".json":
            format = "json"
        elif output_path.suffix == ".csv":
            format = "csv"
        else:
            format = "json"  # default

    if format == "json":
        # Конвертируем DataFrame в list of dicts
        data = df.to_dict(orient="records")
        # Обрабатываем None значения для JSON
        for row in data:
            for key, value in row.items():
                if pd.isna(value):
                    row[key] = None
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    elif format == "csv":
        df.to_csv(output_path, index=False, encoding="utf-8")

    else:
        raise ValueError(f"Unsupported export format: {format}. Use 'json' or 'csv'.")


def get_checkpoint_summary(checkpoint_path: str | Path) -> dict:
    """
    Краткая сводка по чекпоинту для UI.

    Args:
        checkpoint_path: Путь к чекпоинту

    Returns:
        Dict с краткой информацией для отображения в GUI
    """
    checkpoint = load_checkpoint(checkpoint_path)
    metadata = checkpoint["metadata"]
    metrics = extract_metrics(checkpoint)

    return {
        "file_name": Path(checkpoint_path).name,
        "epoch": metadata.epoch,
        "model_type": metadata.model_type,
        "hidden_dims": metadata.energy_hidden_dims,
        "cos_improvement": metrics.cos_improvement,
        "success_rate": metrics.success_rate,
        "has_metrics": metadata.has_metrics,
    }


def batch_load_checkpoints(checkpoint_paths: list[str | Path]) -> list[dict]:
    """
    Массовая загрузка чекпоинтов.

    Args:
        checkpoint_paths: Список путей

    Returns:
        Список загруженных чекпоинтов (пропуская ошибочные)
    """
    loaded = []
    errors = []

    for path in checkpoint_paths:
        try:
            checkpoint = load_checkpoint(path)
            loaded.append(checkpoint)
        except Exception as e:
            errors.append({"path": str(path), "error": str(e)})

    return loaded, errors
