"""
Live Monitor — реал-тайм мониторинг тренировки CERBER.

Подключение к тренировочному скрипту через:
1. File-based monitoring (watchdog за training_metrics.json)
2. WebSocket (прямая трансляция из тренировочного скрипта)

Функции:
    TrainingMetricsWatcher — класс для file-based мониторинга
    WebSocketTrainingListener — класс для WebSocket подключения
    create_live_metrics_plot(...) — обновление графика в реальном времени
    create_live_landscape_callback(...) — callback для обновления ландшафта
"""

import json
import time
import threading
from pathlib import Path
from typing import Callable, Any, Optional
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# watchdog для file system monitoring
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object


@dataclass
class TrainingStatus:
    """Текущий статус тренировки."""
    is_running: bool = False
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: float | None = None
    best_loss: float | None = None
    current_cos_sim: float | None = None
    learning_rate: float | None = None
    last_update: datetime | None = None
    error: str | None = None


@dataclass
class MetricsHistory:
    """История метрик для визуализации."""
    records: deque = field(default_factory=lambda: deque(maxlen=5000))
    epochs: deque = field(default_factory=lambda: deque(maxlen=1000))
    train_losses: deque = field(default_factory=lambda: deque(maxlen=1000))
    eval_losses: deque = field(default_factory=lambda: deque(maxlen=1000))
    cos_before: deque = field(default_factory=lambda: deque(maxlen=1000))
    cos_after: deque = field(default_factory=lambda: deque(maxlen=1000))
    success_rates: deque = field(default_factory=lambda: deque(maxlen=1000))
    learning_rates: deque = field(default_factory=lambda: deque(maxlen=1000))

    def to_dataframe(self) -> pd.DataFrame:
        """Конвертация в DataFrame для Plotly."""
        if self.records:
            return pd.DataFrame(list(self.records))
        return pd.DataFrame({
            "epoch": list(self.epochs),
            "train_loss": list(self.train_losses),
            "eval_loss": list(self.eval_losses),
            "cos_before": list(self.cos_before),
            "cos_after": list(self.cos_after),
            "success_rate": list(self.success_rates),
            "learning_rate": list(self.learning_rates),
        })


def load_metrics_payload(metrics_path: str | Path) -> dict:
    """Load monitoring payload from JSON (legacy) or JSONL (stream events)."""
    path = Path(metrics_path)
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
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid metrics payload type: {type(payload)}")
    payload["source"] = "json"
    return payload


class TrainingMetricsHandler(FileSystemEventHandler):
    """
    Обработчик событий файловой системы для мониторинга metrics файла.
    """

    def __init__(
        self,
        metrics_path: str | Path,
        callback: Callable[[dict], None] | None = None,
    ):
        super().__init__()
        self.metrics_path = Path(metrics_path)
        self.callback = callback
        self.last_modified = 0
        self.debounce_seconds = 1.0
        self._lock = threading.Lock()

    def on_modified(self, event):
        """Обработка события модификации файла."""
        if not isinstance(event, FileModifiedEvent):
            return

        if Path(event.src_path).absolute() != self.metrics_path.absolute():
            return

        current_time = time.time()
        if current_time - self.last_modified < self.debounce_seconds:
            return  # Debounce

        self.last_modified = current_time

        with self._lock:
            try:
                data = load_metrics_payload(self.metrics_path)
                if self.callback:
                    self.callback(data)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error reading metrics file: {e}")


class TrainingMetricsWatcher:
    """
    File-based мониторинг training_metrics.json через watchdog.

    Usage:
        watcher = TrainingMetricsWatcher("path/to/training_metrics.json")
        watcher.start()

        # В Gradio callback:
        def update_plot():
            df = watcher.history.to_dataframe()
            return create_loss_plot(df)
    """

    def __init__(
        self,
        metrics_path: str | Path,
        poll_interval: float = 2.0,
    ):
        """
        Args:
            metrics_path: Путь к файлу метрик
            poll_interval: Интервал опроса (секунды)
        """
        self.metrics_path = Path(metrics_path)
        self.poll_interval = poll_interval

        self.status = TrainingStatus()
        self.history = MetricsHistory()

        self._observer: Observer | None = None
        self._running = False
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[dict], None]] = []

    def add_callback(self, callback: Callable[[dict], None]) -> None:
        """Добавление callback для обновления при новых данных."""
        self._callbacks.append(callback)

    def _on_metrics_update(self, data: dict) -> None:
        """?????????? ??????? ? ??????? ??? ????? ????????."""
        with self._lock:
            self._update_history(data)
            df = self.history.to_dataframe()
            if not df.empty:
                last = df.iloc[-1]
                epoch_val = last.get("epoch", np.nan)
                if pd.notna(epoch_val):
                    self.status.current_epoch = int(float(epoch_val))

                loss_val = last.get("train_loss", np.nan)
                if pd.notna(loss_val):
                    self.status.current_loss = float(loss_val)
                    if "train_loss" in df.columns:
                        finite_losses = pd.to_numeric(df["train_loss"], errors="coerce").dropna()
                        if not finite_losses.empty:
                            self.status.best_loss = float(finite_losses.min())

                cos_after_val = last.get("cos_after", np.nan)
                if pd.notna(cos_after_val):
                    self.status.current_cos_sim = float(cos_after_val)

                lr_val = last.get("learning_rate", np.nan)
                if pd.notna(lr_val):
                    self.status.learning_rate = float(lr_val)

                self.status.last_update = datetime.now()

        # ????? callback
        for callback in self._callbacks:
            try:
                callback(data)
            except Exception as e:
                print(f"Callback error: {e}")

    def _update_history(self, data: dict) -> None:
        """?????????? ??????? ??????."""
        self.history.records.clear()
        self.history.epochs.clear()
        self.history.train_losses.clear()
        self.history.eval_losses.clear()
        self.history.cos_before.clear()
        self.history.cos_after.clear()
        self.history.success_rates.clear()
        self.history.learning_rates.clear()

        # Stage1.5 stream mode (JSONL events)
        events = data.get("events")
        if isinstance(events, list):
            for rec in events:
                if not isinstance(rec, dict):
                    continue
                event_type = str(rec.get("event", "unknown"))
                train_metrics = rec.get("train_metrics", {})
                if not isinstance(train_metrics, dict):
                    train_metrics = {}
                kill = rec.get("kill_criteria", {})
                if not isinstance(kill, dict):
                    kill = {}
                agg = kill.get("aggregate", {})
                if not isinstance(agg, dict):
                    agg = {}

                epoch = rec.get("epoch", np.nan)
                batch_idx = rec.get("batch_idx", np.nan)
                num_batches = rec.get("num_batches", np.nan)
                progress = np.nan
                try:
                    if pd.notna(epoch) and pd.notna(batch_idx) and pd.notna(num_batches) and float(num_batches) > 0:
                        progress = float(epoch) - 1.0 + float(batch_idx) / float(num_batches)
                    elif pd.notna(epoch):
                        progress = float(epoch)
                except Exception:
                    progress = np.nan

                row = {
                    "event": event_type,
                    "epoch": float(epoch) if pd.notna(epoch) else np.nan,
                    "progress": progress,
                    "batch_idx": float(batch_idx) if pd.notna(batch_idx) else np.nan,
                    "num_batches": float(num_batches) if pd.notna(num_batches) else np.nan,
                    "global_step": float(rec.get("global_step", np.nan)),
                    "train_loss": float(train_metrics.get("loss", np.nan)),
                    "critic_loss": float(train_metrics.get("critic", np.nan)),
                    "actor_loss": float(train_metrics.get("actor", np.nan)),
                    "rank_loss": float(train_metrics.get("rank_loss", np.nan)),
                    "rank_success": float(train_metrics.get("rank_success", np.nan)),
                    "rank_clean_lt_actor": float(train_metrics.get("rank_clean_lt_actor", np.nan)),
                    "rank_actor_lt_hard": float(train_metrics.get("rank_actor_lt_hard", np.nan)),
                    "rank_clean_lt_hard": float(train_metrics.get("rank_clean_lt_hard", np.nan)),
                    "clean_viol": float(train_metrics.get("clean_viol", np.nan)),
                    "retrieval_cosine": float(train_metrics.get("retrieval_cosine", np.nan)),
                    "mdsm": float(train_metrics.get("mdsm", np.nan)),
                    "nce": float(train_metrics.get("nce", np.nan)),
                    "cql": float(train_metrics.get("cql", np.nan)),
                    "skip_rate": float(train_metrics.get("skip_rate", np.nan)),
                    "sec_per_batch_window": float(rec.get("sec_per_batch_window", np.nan)),
                    "eta_epoch_sec": float(rec.get("eta_epoch_sec", np.nan)),
                    "epoch_time_sec": float(rec.get("epoch_time_sec", np.nan)),
                    "score": float(rec.get("score", np.nan)),
                    "strict_pass": (
                        float(1.0 if bool(rec.get("strict_pass", False)) else 0.0)
                        if rec.get("strict_pass", None) is not None else np.nan
                    ),
                    "cos_improvement_eval": float(agg.get("mean_cos_improvement", np.nan)),
                    "energy_success_eval": float(agg.get("mean_energy_success_rate", np.nan)),
                    "clean_violation_eval": float(agg.get("mean_clean_min_violation_rate", np.nan)),
                    # Compatibility aliases for existing plots
                    "eval_loss": np.nan,
                    "cos_before": np.nan,
                    "cos_after": np.nan,
                    "success_rate": float(agg.get("mean_cos_success_rate", np.nan)),
                    "learning_rate": np.nan,
                }
                self.history.records.append(row)
            return

        # Legacy JSON arrays mode
        epochs = data.get("epochs", data.get("epoch", []))
        train_losses = data.get("train_loss", [])
        eval_losses = data.get("eval_loss", [])
        cos_before = data.get("cos_before", [])
        cos_after = data.get("cos_after", [])
        success_rates = data.get("success_rate", [])
        learning_rates = data.get("learning_rate", data.get("lr", []))

        n = min(
            len(epochs) if isinstance(epochs, list) else 0,
            len(train_losses) if train_losses else float("inf"),
        )

        for i in range(n):
            epoch_i = epochs[i] if isinstance(epochs, list) else i + 1
            self.history.epochs.append(epoch_i)
            if train_losses:
                self.history.train_losses.append(train_losses[i])
            if eval_losses:
                self.history.eval_losses.append(eval_losses[i])
            if cos_before:
                self.history.cos_before.append(cos_before[i])
            if cos_after:
                self.history.cos_after.append(cos_after[i])
            if success_rates:
                self.history.success_rates.append(success_rates[i])
            if learning_rates:
                self.history.learning_rates.append(learning_rates[i])

            self.history.records.append(
                {
                    "event": "legacy",
                    "epoch": float(epoch_i),
                    "progress": float(epoch_i),
                    "batch_idx": np.nan,
                    "num_batches": np.nan,
                    "global_step": np.nan,
                    "train_loss": float(train_losses[i]) if train_losses else np.nan,
                    "critic_loss": np.nan,
                    "actor_loss": np.nan,
                    "rank_loss": np.nan,
                    "rank_success": np.nan,
                    "rank_clean_lt_actor": np.nan,
                    "rank_actor_lt_hard": np.nan,
                    "rank_clean_lt_hard": np.nan,
                    "clean_viol": np.nan,
                    "retrieval_cosine": np.nan,
                    "mdsm": np.nan,
                    "nce": np.nan,
                    "cql": np.nan,
                    "skip_rate": np.nan,
                    "sec_per_batch_window": np.nan,
                    "eta_epoch_sec": np.nan,
                    "epoch_time_sec": np.nan,
                    "score": np.nan,
                    "strict_pass": np.nan,
                    "cos_improvement_eval": np.nan,
                    "energy_success_eval": np.nan,
                    "clean_violation_eval": np.nan,
                    "eval_loss": float(eval_losses[i]) if eval_losses else np.nan,
                    "cos_before": float(cos_before[i]) if cos_before else np.nan,
                    "cos_after": float(cos_after[i]) if cos_after else np.nan,
                    "success_rate": float(success_rates[i]) if success_rates else np.nan,
                    "learning_rate": float(learning_rates[i]) if learning_rates else np.nan,
                }
            )

    def start(self) -> None:
        """Запуск мониторинга."""
        if not WATCHDOG_AVAILABLE:
            raise RuntimeError("watchdog not installed. Run: pip install watchdog")

        if self._running:
            return

        self._running = True

        # Создаем observer
        self._observer = Observer()
        handler = TrainingMetricsHandler(self.metrics_path, self._on_metrics_update)
        self._observer.schedule(handler, str(self.metrics_path.parent), recursive=False)

        # Начальная загрузка если файл существует
        if self.metrics_path.exists():
            try:
                data = load_metrics_payload(self.metrics_path)
                self._on_metrics_update(data)
                self.status.is_running = True
            except Exception as e:
                self.status.error = str(e)

        self._observer.start()
        self.status.is_running = True

    def stop(self) -> None:
        """Остановка мониторинга."""
        self._running = False
        if self._observer:
            self._observer.stop()
            self._observer.join()
        self.status.is_running = False

    def get_status(self) -> TrainingStatus:
        """Получение текущего статуса."""
        return self.status

    def get_history_dataframe(self) -> pd.DataFrame:
        """Получение истории метрик в DataFrame."""
        return self.history.to_dataframe()


class WebSocketTrainingListener:
    """
    WebSocket клиент для прямой трансляции метрик из тренировочного скрипта.

    Для использования требуется training script с WebSocket сервером.

    Usage:
        listener = WebSocketTrainingListener("ws://localhost:8765")
        listener.start()
    """

    def __init__(self, ws_url: str = "ws://localhost:8765"):
        self.ws_url = ws_url
        self.status = TrainingStatus()
        self.history = MetricsHistory()
        self._running = False
        self._ws = None
        self._thread: threading.Thread | None = None
        self._callbacks: list[Callable[[dict], None]] = []

    def add_callback(self, callback: Callable[[dict], None]) -> None:
        self._callbacks.append(callback)

    def start(self) -> None:
        """Запуск WebSocket подключения."""
        try:
            import websocket  # type: ignore
        except ImportError:
            raise RuntimeError("websocket-client not installed. Run: pip install websocket-client")

        self._running = True
        self._thread = threading.Thread(target=self._run_ws, daemon=True)
        self._thread.start()

    def _run_ws(self) -> None:
        """WebSocket цикл."""
        import websocket

        def on_message(ws, message):
            try:
                data = json.loads(message)
                self._update_status(data)
                for callback in self._callbacks:
                    callback(data)
            except Exception as e:
                print(f"WebSocket message error: {e}")

        def on_error(ws, error):
            self.status.error = str(error)
            self.status.is_running = False

        def on_close(ws, close_status_code, close_msg):
            self._running = False
            self.status.is_running = False

        def on_open(ws):
            self.status.is_running = True

        self._ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        self._ws.run_forever()

    def _update_status(self, data: dict) -> None:
        """Обновление статуса из WebSocket сообщения."""
        self.status.current_epoch = data.get("epoch", self.status.current_epoch)
        self.status.current_loss = data.get("train_loss", self.status.current_loss)
        self.status.current_cos_sim = data.get("cos_after", self.status.current_cos_sim)
        self.status.learning_rate = data.get("learning_rate", self.status.learning_rate)
        self.status.last_update = datetime.now()

        # Добавляем в историю
        self.history.epochs.append(self.status.current_epoch)
        if self.status.current_loss:
            self.history.train_losses.append(self.status.current_loss)
        if self.status.current_cos_sim:
            self.history.cos_after.append(self.status.current_cos_sim)

    def stop(self) -> None:
        """Остановка WebSocket подключения."""
        self._running = False
        if self._ws:
            self._ws.close()
        if self._thread:
            self._thread.join(timeout=5)


def create_live_metrics_plot(
    history_df: pd.DataFrame,
    plot_type: str = "loss",
) -> Any:
    """
    Создание графика для live обновления.

    Args:
        history_df: DataFrame с историей метрик
        plot_type: Тип графика ("loss", "cosine", "lr", "all")

    Returns:
        Plotly Figure
    """
    from cerber_gui.metrics_viewer import (
        create_loss_plot,
        create_cosine_similarity_plot,
        create_learning_rate_schedule_plot,
        create_metrics_dashboard,
    )

    if history_df is None or history_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Live Metrics",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[
                dict(
                    text="No metrics yet",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16),
                )
            ],
        )
        return fig

    is_stage15_stream = (
        "event" in history_df.columns
        and history_df["event"].astype(str).isin(["batch", "epoch", "final"]).any()
    )
    if not is_stage15_stream:
        if plot_type == "loss":
            return create_loss_plot(history_df)
        if plot_type == "cosine":
            return create_cosine_similarity_plot(history_df)
        if plot_type == "lr":
            return create_learning_rate_schedule_plot(history_df)
        return create_metrics_dashboard(history_df)

    df = history_df.copy()
    numeric_cols = [
        "progress",
        "epoch",
        "global_step",
        "batch_idx",
        "num_batches",
        "train_loss",
        "critic_loss",
        "actor_loss",
        "rank_loss",
        "rank_success",
        "rank_clean_lt_actor",
        "rank_actor_lt_hard",
        "rank_clean_lt_hard",
        "clean_viol",
        "retrieval_cosine",
        "mdsm",
        "nce",
        "cql",
        "skip_rate",
        "sec_per_batch_window",
        "eta_epoch_sec",
        "epoch_time_sec",
        "score",
        "strict_pass",
        "cos_improvement_eval",
        "energy_success_eval",
        "clean_violation_eval",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "progress" in df.columns and df["progress"].notna().any():
        x_col = "progress"
        x_title = "Progress (epoch + batch_frac)"
    elif "global_step" in df.columns and df["global_step"].notna().any():
        x_col = "global_step"
        x_title = "Global Step"
    elif "epoch" in df.columns and df["epoch"].notna().any():
        x_col = "epoch"
        x_title = "Epoch"
    else:
        df["__row_index"] = np.arange(len(df), dtype=float)
        x_col = "__row_index"
        x_title = "Record Index"

    batch_df = df[df.get("event", pd.Series(index=df.index, dtype=object)).astype(str) == "batch"]
    if batch_df.empty:
        batch_df = df
    epoch_df = df[df.get("event", pd.Series(index=df.index, dtype=object)).astype(str) == "epoch"]

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Core Losses",
            "Ranking + Violation Metrics",
            "Regularizers + Retrieval + Skip",
            "Speed + Eval/Kill Signals",
        ),
    )

    def _add_trace(dataframe: pd.DataFrame, y_col: str, row: int, name: str, color: str, dash: str = "solid") -> None:
        if y_col not in dataframe.columns:
            return
        y = pd.to_numeric(dataframe[y_col], errors="coerce")
        if not y.notna().any():
            return
        fig.add_trace(
            go.Scatter(
                x=dataframe[x_col],
                y=y,
                mode="lines",
                name=name,
                line=dict(color=color, width=2, dash=dash),
            ),
            row=row,
            col=1,
        )

    # Row 1: core losses
    _add_trace(batch_df, "train_loss", 1, "Train Loss", "#1f77b4")
    _add_trace(batch_df, "critic_loss", 1, "Critic Loss", "#2ca02c")
    _add_trace(batch_df, "actor_loss", 1, "Actor Loss", "#ff7f0e")

    # Row 2: ranking + violation
    _add_trace(batch_df, "rank_success", 2, "Rank Success", "#00cc96")
    _add_trace(batch_df, "rank_clean_lt_actor", 2, "Rank(clean<actor)", "#636efa")
    _add_trace(batch_df, "rank_actor_lt_hard", 2, "Rank(actor<hard)", "#ab63fa")
    _add_trace(batch_df, "rank_clean_lt_hard", 2, "Rank(clean<hard)", "#19d3f3")
    _add_trace(batch_df, "clean_viol", 2, "Clean Violation", "#ef553b")

    # Row 3: regularizers
    _add_trace(batch_df, "rank_loss", 3, "Rank Loss", "#d62728")
    _add_trace(batch_df, "mdsm", 3, "MDSM", "#9467bd")
    _add_trace(batch_df, "nce", 3, "NCE", "#17becf")
    _add_trace(batch_df, "cql", 3, "CQL", "#8c564b")
    _add_trace(batch_df, "retrieval_cosine", 3, "Retrieval Cosine", "#bcbd22")
    _add_trace(batch_df, "skip_rate", 3, "Skip Rate", "#7f7f7f", dash="dot")

    # Row 4: speed + eval
    _add_trace(batch_df, "sec_per_batch_window", 4, "sec/batch", "#1f77b4")
    if "eta_epoch_sec" in batch_df.columns and batch_df["eta_epoch_sec"].notna().any():
        eta_minutes = pd.to_numeric(batch_df["eta_epoch_sec"], errors="coerce") / 60.0
        fig.add_trace(
            go.Scatter(
                x=batch_df[x_col],
                y=eta_minutes,
                mode="lines",
                name="ETA (min)",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
            ),
            row=4,
            col=1,
        )
    _add_trace(epoch_df, "score", 4, "Kill Score", "#2ca02c")
    _add_trace(epoch_df, "cos_improvement_eval", 4, "Eval Cos Improvement", "#9467bd")
    _add_trace(epoch_df, "energy_success_eval", 4, "Eval Energy Success", "#00cc96")
    _add_trace(epoch_df, "clean_violation_eval", 4, "Eval Clean Violation", "#ef553b")

    # Mark epoch boundaries with light markers on row 1
    if not epoch_df.empty and x_col in epoch_df.columns:
        fig.add_trace(
            go.Scatter(
                x=epoch_df[x_col],
                y=np.zeros(len(epoch_df)),
                mode="markers",
                marker=dict(size=6, color="#444444", symbol="x"),
                name="Epoch End",
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        title="Live Stage1.5 Metrics Dashboard",
        height=1200,
        width=1200,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=60, r=30, t=90, b=60),
    )
    fig.update_xaxes(title_text=x_title, row=4, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Rate", row=2, col=1)
    fig.update_yaxes(title_text="Aux/Reg", row=3, col=1)
    fig.update_yaxes(title_text="Time / Eval", row=4, col=1)
    return fig


def get_watcher_for_path(metrics_path: str) -> TrainingMetricsWatcher:
    """
    Получение или создание watcher для пути.

    Используется для кэширования watcher в Gradio сессии.

    Args:
        metrics_path: Путь к training_metrics.json

    Returns:
        TrainingMetricsWatcher
    """
    # Простая реализация без глобального кэша
    # Для production использовать lru_cache или слабые ссылки
    watcher = TrainingMetricsWatcher(metrics_path)
    return watcher


# Пример интеграции с тренировочным скриптом
def inject_metrics_logging(
    train_script_path: str | Path,
    metrics_output_path: str | Path,
) -> None:
    """
    Внедрение логгирования метрик в тренировочный скрипт.

    Создает wrapper который периодически сохраняет метрики в JSON.

    Args:
        train_script_path: Путь к тренировочному скрипту
        metrics_output_path: Путь для output JSON
    """
    wrapper_code = f'''
# Auto-injected metrics logging
import json
from pathlib import Path

_metrics_output = Path("{metrics_output_path}")
_metrics_history = {{"epochs": [], "train_loss": [], "eval_loss": [], "cos_before": [], "cos_after": []}}

def log_training_metrics(epoch, train_loss=None, eval_loss=None, cos_before=None, cos_after=None):
    """Log metrics to JSON file for live monitoring."""
    if train_loss is not None:
        _metrics_history["epochs"].append(epoch)
        if train_loss is not None:
            _metrics_history["train_loss"].append(train_loss)
        if eval_loss is not None:
            _metrics_history["eval_loss"].append(eval_loss)
        if cos_before is not None:
            _metrics_history["cos_before"].append(cos_before)
        if cos_after is not None:
            _metrics_history["cos_after"].append(cos_after)

        with open(_metrics_output, "w") as f:
            json.dump(_metrics_history, f, indent=2)
'''

    # Чтение оригинального скрипта
    path = Path(train_script_path)
    if not path.exists():
        raise FileNotFoundError(f"Training script not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        original_code = f.read()

    # Добавление wrapper в начало
    modified_code = wrapper_code + "\n" + original_code

    # Замена print метрик на вызов log_training_metrics
    # Это простая эвристика, может потребоваться ручная настройка
    modified_code = modified_code.replace(
        'print(f"Epoch {epoch}: loss={loss}"',
        'log_training_metrics(epoch, train_loss=loss); print(f"Epoch {epoch}: loss={loss}"',
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(modified_code)

    print(f"Injected metrics logging into {path}")
