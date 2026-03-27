"""
Metrics Viewer — визуализация метрик обучения CERBER.

Графики и анализ метрик из чекпоинтов и training_metrics.json.

Функции:
    load_training_metrics(path) — загрузка JSON с метриками тренировки
    create_loss_plot(...) — график потерь (train/eval)
    create_cosine_similarity_plot(...) — cosine similarity до/после
    create_success_rate_heatmap(...) — heatmap success rate по уровням шума
    create_learning_rate_schedule_plot(...) — график learning rate scheduler
    create_metrics_dashboard(...) — комбинированный dashboard с всеми метриками
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_training_metrics(path: str | Path) -> dict:
    """
    Загрузка метрик тренировки из JSON файла.

    Ожидается формат:
    {
        "epochs": [...],
        "train_loss": [...],
        "eval_loss": [...],
        "cos_before": [...],
        "cos_after": [...],
        "success_rate": [...],
        ...
    }

    Args:
        path: Путь к JSON файлу

    Returns:
        Dict с метриками

    Raises:
        FileNotFoundError: если файл не найден
        ValueError: если формат невалиден
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        events: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    events.append(row)
        return {"source": "jsonl", "events": events}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Валидация минимальной структуры
    if not isinstance(data, dict):
        raise ValueError(f"Invalid metrics format: expected dict, got {type(data)}")

    return data


def metrics_to_dataframe(metrics_path: str | Path) -> pd.DataFrame:
    """
    Конвертация training_metrics.json в DataFrame.

    Args:
        metrics_path: Путь к JSON файлу

    Returns:
        DataFrame с метриками по эпохам
    """
    data = load_training_metrics(metrics_path)

    events = data.get("events")
    if isinstance(events, list):
        rows: list[dict] = []
        for rec in events:
            if not isinstance(rec, dict):
                continue
            event_type = str(rec.get("event", "unknown"))
            train = rec.get("train_metrics", {})
            if not isinstance(train, dict):
                train = {}
            kill = rec.get("kill_criteria", {})
            if not isinstance(kill, dict):
                kill = {}
            agg = kill.get("aggregate", {})
            if not isinstance(agg, dict):
                agg = {}

            epoch = rec.get("epoch")
            batch_idx = rec.get("batch_idx")
            num_batches = rec.get("num_batches")
            progress = None
            try:
                if epoch is not None and batch_idx is not None and num_batches and float(num_batches) > 0:
                    progress = float(epoch) - 1.0 + float(batch_idx) / float(num_batches)
                elif epoch is not None:
                    progress = float(epoch)
            except Exception:
                progress = None

            rows.append(
                {
                    "event": event_type,
                    "epoch": float(epoch) if epoch is not None else None,
                    "progress": progress,
                    "batch_idx": float(batch_idx) if batch_idx is not None else None,
                    "num_batches": float(num_batches) if num_batches is not None else None,
                    "global_step": float(rec.get("global_step")) if rec.get("global_step") is not None else None,
                    "train_loss": float(train.get("loss")) if train.get("loss") is not None else None,
                    "critic_loss": float(train.get("critic")) if train.get("critic") is not None else None,
                    "actor_loss": float(train.get("actor")) if train.get("actor") is not None else None,
                    "rank_loss": float(train.get("rank_loss")) if train.get("rank_loss") is not None else None,
                    "rank_success": float(train.get("rank_success")) if train.get("rank_success") is not None else None,
                    "rank_clean_lt_actor": float(train.get("rank_clean_lt_actor")) if train.get("rank_clean_lt_actor") is not None else None,
                    "rank_actor_lt_hard": float(train.get("rank_actor_lt_hard")) if train.get("rank_actor_lt_hard") is not None else None,
                    "rank_clean_lt_hard": float(train.get("rank_clean_lt_hard")) if train.get("rank_clean_lt_hard") is not None else None,
                    "clean_viol": float(train.get("clean_viol")) if train.get("clean_viol") is not None else None,
                    "retrieval_cosine": float(train.get("retrieval_cosine")) if train.get("retrieval_cosine") is not None else None,
                    "mdsm": float(train.get("mdsm")) if train.get("mdsm") is not None else None,
                    "nce": float(train.get("nce")) if train.get("nce") is not None else None,
                    "cql": float(train.get("cql")) if train.get("cql") is not None else None,
                    "skip_rate": float(train.get("skip_rate")) if train.get("skip_rate") is not None else None,
                    "sec_per_batch_window": float(rec.get("sec_per_batch_window")) if rec.get("sec_per_batch_window") is not None else None,
                    "eta_epoch_sec": float(rec.get("eta_epoch_sec")) if rec.get("eta_epoch_sec") is not None else None,
                    "epoch_time_sec": float(rec.get("epoch_time_sec")) if rec.get("epoch_time_sec") is not None else None,
                    "score": float(rec.get("score")) if rec.get("score") is not None else None,
                    "success_rate": float(agg.get("mean_cos_success_rate")) if agg.get("mean_cos_success_rate") is not None else None,
                    "cos_improvement_eval": float(agg.get("mean_cos_improvement")) if agg.get("mean_cos_improvement") is not None else None,
                    "energy_success_eval": float(agg.get("mean_energy_success_rate")) if agg.get("mean_energy_success_rate") is not None else None,
                    "clean_violation_eval": float(agg.get("mean_clean_min_violation_rate")) if agg.get("mean_clean_min_violation_rate") is not None else None,
                }
            )
        if not rows:
            raise ValueError("No valid events found in JSONL metrics file")
        return pd.DataFrame(rows)

    # Находим ключи со списками одинаковой длины
    arrays = {k: v for k, v in data.items() if isinstance(v, list)}

    if not arrays:
        # Пытаемся найти вложенную структуру
        if "metrics" in data:
            return metrics_to_dataframe_from_dict(data["metrics"])
        raise ValueError("No array metrics found in file")

    # Выравнивание по минимальной длине
    min_len = min(len(v) for v in arrays.values())
    aligned = {k: v[:min_len] for k, v in arrays.items()}

    # Добавляем epoch если нет
    if "epoch" not in aligned and "epochs" not in aligned:
        aligned["epoch"] = list(range(1, min_len + 1))

    df = pd.DataFrame(aligned)

    # Переименование epochs -> epoch если нужно
    if "epochs" in df.columns and "epoch" not in df.columns:
        df = df.rename(columns={"epochs": "epoch"})

    return df


def metrics_to_dataframe_from_dict(metrics_dict: dict) -> pd.DataFrame:
    """
    Конвертация dict с метриками в DataFrame.

    Args:
        metrics_dict: Dict с метриками

    Returns:
        DataFrame
    """
    arrays = {k: v for k, v in metrics_dict.items() if isinstance(v, list)}

    if not arrays:
        # Пытаемся конвертировать скаляры в списки
        arrays = {k: [v] for k, v in metrics_dict.items() if isinstance(v, (int, float))}

    if not arrays:
        raise ValueError("No metrics found")

    min_len = min(len(v) for v in arrays.values())
    aligned = {k: v[:min_len] for k, v in arrays.items()}

    if "epoch" not in aligned:
        aligned["epoch"] = list(range(1, min_len + 1))

    return pd.DataFrame(aligned)


def create_loss_plot(
    df: pd.DataFrame,
    title: str = "Training & Evaluation Loss",
    show_grid: bool = True,
) -> go.Figure:
    """
    График потерь (train/eval) по эпохам.

    Args:
        df: DataFrame с колонками train_loss, eval_loss
        title: Заголовок графика
        show_grid: Показывать ли сетку

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Train loss
    if "train_loss" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["epoch"],
            y=df["train_loss"],
            mode="lines+markers",
            name="Train Loss",
            line=dict(color="blue", width=2),
            marker=dict(size=6),
        ))

    # Eval loss
    if "eval_loss" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["epoch"],
            y=df["eval_loss"],
            mode="lines+markers",
            name="Eval Loss",
            line=dict(color="red", width=2),
            marker=dict(size=6),
        ))

    # Gap между train и eval (признак overfitting)
    if "train_loss" in df.columns and "eval_loss" in df.columns:
        gap = df["eval_loss"] - df["train_loss"]
        fig.add_trace(go.Scatter(
            x=df["epoch"],
            y=gap,
            mode="lines",
            name="Generalization Gap",
            line=dict(color="gray", width=1, dash="dash"),
            fill="tozeroy",
            opacity=0.3,
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Epoch",
        yaxis_title="Loss",
        showlegend=True,
        legend=dict(x=0.02, y=0.98, yanchor="top"),
        width=800,
        height=500,
    )

    if show_grid:
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    return fig


def create_cosine_similarity_plot(
    df: pd.DataFrame,
    title: str = "Cosine Similarity Improvement",
) -> go.Figure:
    """
    График cosine similarity до/после денуазинга.

    Args:
        df: DataFrame с колонками cos_before, cos_after
        title: Заголовок

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Cosine before
    if "cos_before" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["epoch"],
            y=df["cos_before"],
            mode="lines+markers",
            name="Before Denoising",
            line=dict(color="red", width=2),
            marker=dict(size=6),
        ))

    # Cosine after
    if "cos_after" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["epoch"],
            y=df["cos_after"],
            mode="lines+markers",
            name="After Denoising",
            line=dict(color="green", width=2),
            marker=dict(size=6),
        ))

    # Improvement
    if "cos_before" in df.columns and "cos_after" in df.columns:
        improvement = df["cos_after"] - df["cos_before"]
        fig.add_trace(go.Scatter(
            x=df["epoch"],
            y=improvement,
            mode="lines",
            name="Improvement (Δ)",
            line=dict(color="blue", width=2),
            fill="tozeroy",
            opacity=0.3,
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Epoch",
        yaxis_title="Cosine Similarity",
        showlegend=True,
        legend=dict(x=0.02, y=0.98, yanchor="top"),
        width=800,
        height=500,
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    return fig


def create_success_rate_heatmap(
    df: pd.DataFrame,
    noise_levels: list[float] | None = None,
    title: str = "Success Rate by Noise Level",
) -> go.Figure:
    """
    Heatmap success rate по эпохам и уровням шума.

    Args:
        df: DataFrame с метриками
        noise_levels: Уровни шума (если есть в данных)
        title: Заголовок

    Returns:
        Plotly Figure с heatmap
    """
    # Проверяем наличие данных по уровням шума
    noise_columns = [col for col in df.columns if "noise" in col.lower() or "success" in col.lower()]

    if not noise_columns:
        # Создаем заглушку
        fig = go.Figure()
        fig.add_annotation(
            text="No noise level data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(
            title=title,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    # Если есть отдельные колонки для каждого noise level
    if noise_levels is None:
        # Пытаемся извлечь из названий колонок
        noise_levels = []
        for col in noise_columns:
            # Ищем числа в названии колонки
            import re
            matches = re.findall(r"[\d.]+", col)
            if matches:
                try:
                    noise_levels.append(float(matches[0]))
                except ValueError:
                    pass

    # Строим heatmap
    fig = go.Figure()

    # Собираем данные
    z_data = []
    y_labels = []

    for col in noise_columns:
        if col in df.columns:
            z_data.append(df[col].tolist())
            y_labels.append(col)

    if z_data:
        fig.add_trace(go.Heatmap(
            z=z_data,
            x=df["epoch"].tolist() if "epoch" in df.columns else list(range(len(z_data[0]))),
            y=y_labels,
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Success Rate"),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Epoch",
        yaxis_title="Noise Level",
        width=900,
        height=600,
    )

    return fig


def create_learning_rate_schedule_plot(
    df: pd.DataFrame,
    title: str = "Learning Rate Schedule",
) -> go.Figure:
    """
    График изменения learning rate по эпохам.

    Args:
        df: DataFrame с колонкой learning_rate или lr
        title: Заголовок

    Returns:
        Plotly Figure
    """
    lr_column = None
    for col in ["learning_rate", "lr", "scheduler_lr"]:
        if col in df.columns:
            lr_column = col
            break

    if lr_column is None:
        fig = go.Figure()
        fig.add_annotation(
            text="No learning rate data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(title=title)
        return fig

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["epoch"],
        y=df[lr_column],
        mode="lines+markers",
        name="Learning Rate",
        line=dict(color="purple", width=2),
        marker=dict(size=6),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Epoch",
        yaxis_title="Learning Rate",
        showlegend=True,
        width=800,
        height=400,
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    return fig


def create_energy_distribution_plot(
    energies_before: list[float],
    energies_after: list[float],
    title: str = "Energy Distribution",
) -> go.Figure:
    """
    Гистограмма распределения энергий до/после.

    Args:
        energies_before: Энергии до денуазинга
        energies_after: Энергии после денуазинга
        title: Заголовок

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=energies_before,
        name="Before Denoising",
        opacity=0.7,
        marker_color="red",
        nbinsx=30,
    ))

    fig.add_trace(go.Histogram(
        x=energies_after,
        name="After Denoising",
        opacity=0.7,
        marker_color="green",
        nbinsx=30,
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Energy",
        yaxis_title="Count",
        barmode="overlay",
        showlegend=True,
        width=800,
        height=500,
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    return fig


def create_metrics_dashboard(
    df: pd.DataFrame,
    title: str = "Training Metrics Dashboard",
) -> go.Figure:
    """
    Комбинированный dashboard с всеми основными метриками.

    Args:
        df: DataFrame с метриками
        title: Заголовок dashboard

    Returns:
        Plotly Figure с подграфиками
    """
    # Определяем какие метрики доступны
    has_loss = "train_loss" in df.columns or "eval_loss" in df.columns
    has_cosine = "cos_before" in df.columns or "cos_after" in df.columns
    has_lr = any(col in df.columns for col in ["learning_rate", "lr"])

    # Количество рядов
    n_rows = sum([has_loss, has_cosine, has_lr])
    if n_rows == 0:
        n_rows = 1

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[],
    )

    row = 1

    # Loss plot
    if has_loss:
        loss_fig = create_loss_plot(df, title="")
        for trace in loss_fig.data:
            fig.add_trace(trace, row=row, col=1)
        fig.update_yaxes(title_text="Loss", row=row, col=1)
        row += 1

    # Cosine similarity plot
    if has_cosine:
        cos_fig = create_cosine_similarity_plot(df, title="")
        for trace in cos_fig.data:
            fig.add_trace(trace, row=row, col=1)
        fig.update_yaxes(title_text="Cosine Similarity", row=row, col=1)
        row += 1

    # Learning rate plot
    if has_lr:
        lr_fig = create_learning_rate_schedule_plot(df, title="")
        for trace in lr_fig.data:
            fig.add_trace(trace, row=row, col=1)
        fig.update_yaxes(title_text="Learning Rate", row=row, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        showlegend=True,
        legend=dict(x=0.02, y=0.98, yanchor="top"),
        width=900,
        height=400 * n_rows,
        margin=dict(l=60, r=20, t=60, b=60),
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    return fig


def compute_summary_statistics(df: pd.DataFrame) -> dict:
    """
    Вычисление сводной статистики по метрикам.

    Args:
        df: DataFrame с метриками

    Returns:
        Dict со статистикой
    """
    stats = {}

    # Loss statistics
    for col in ["train_loss", "eval_loss"]:
        if col in df.columns:
            stats[f"{col}_final"] = df[col].iloc[-1]
            stats[f"{col}_min"] = df[col].min()
            stats[f"{col}_max"] = df[col].max()
            stats[f"{col}_avg"] = df[col].mean()

    # Cosine similarity statistics
    if "cos_after" in df.columns:
        stats["cos_after_final"] = df["cos_after"].iloc[-1]
        stats["cos_after_max"] = df["cos_after"].max()

    if "cos_before" in df.columns and "cos_after" in df.columns:
        improvements = df["cos_after"] - df["cos_before"]
        stats["improvement_final"] = improvements.iloc[-1]
        stats["improvement_avg"] = improvements.mean()
        stats["improvement_max"] = improvements.max()

    # Success rate
    if "success_rate" in df.columns:
        stats["success_rate_final"] = df["success_rate"].iloc[-1]
        stats["success_rate_avg"] = df["success_rate"].mean()

    # Epochs
    if "epoch" in df.columns:
        stats["total_epochs"] = len(df)
        stats["final_epoch"] = df["epoch"].iloc[-1]

    return stats
