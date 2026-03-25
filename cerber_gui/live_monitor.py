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
    epochs: deque = field(default_factory=lambda: deque(maxlen=1000))
    train_losses: deque = field(default_factory=lambda: deque(maxlen=1000))
    eval_losses: deque = field(default_factory=lambda: deque(maxlen=1000))
    cos_before: deque = field(default_factory=lambda: deque(maxlen=1000))
    cos_after: deque = field(default_factory=lambda: deque(maxlen=1000))
    success_rates: deque = field(default_factory=lambda: deque(maxlen=1000))
    learning_rates: deque = field(default_factory=lambda: deque(maxlen=1000))

    def to_dataframe(self) -> pd.DataFrame:
        """Конвертация в DataFrame для Plotly."""
        return pd.DataFrame({
            "epoch": list(self.epochs),
            "train_loss": list(self.train_losses),
            "eval_loss": list(self.eval_losses),
            "cos_before": list(self.cos_before),
            "cos_after": list(self.cos_after),
            "success_rate": list(self.success_rates),
            "learning_rate": list(self.learning_rates),
        })


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
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
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
        """Обновление статуса и истории при новых метриках."""
        with self._lock:
            # Обновление статуса
            epochs = data.get("epochs", data.get("epoch", []))
            if epochs:
                self.status.current_epoch = epochs[-1] if isinstance(epochs[-1], int) else len(epochs)
                self.status.last_update = datetime.now()

            train_losses = data.get("train_loss", [])
            if train_losses:
                self.status.current_loss = train_losses[-1]
                self.status.best_loss = min(train_losses)

            cos_after = data.get("cos_after", [])
            if cos_after:
                self.status.current_cos_sim = cos_after[-1]

            # Обновление истории
            self._update_history(data)

        # Вызов callback
        for callback in self._callbacks:
            try:
                callback(data)
            except Exception as e:
                print(f"Callback error: {e}")

    def _update_history(self, data: dict) -> None:
        """Обновление истории метрик."""
        # Извлекаем массивы
        epochs = data.get("epochs", data.get("epoch", []))
        train_losses = data.get("train_loss", [])
        eval_losses = data.get("eval_loss", [])
        cos_before = data.get("cos_before", [])
        cos_after = data.get("cos_after", [])
        success_rates = data.get("success_rate", [])
        learning_rates = data.get("learning_rate", data.get("lr", []))

        # Очищаем и заполняем
        self.history.epochs.clear()
        self.history.train_losses.clear()
        self.history.eval_losses.clear()
        self.history.cos_before.clear()
        self.history.cos_after.clear()
        self.history.success_rates.clear()
        self.history.learning_rates.clear()

        n = min(
            len(epochs) if isinstance(epochs, list) else 0,
            len(train_losses) if train_losses else float("inf"),
        )

        for i in range(n):
            self.history.epochs.append(epochs[i] if isinstance(epochs, list) else i + 1)
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
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
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

    if plot_type == "loss":
        return create_loss_plot(history_df)
    elif plot_type == "cosine":
        return create_cosine_similarity_plot(history_df)
    elif plot_type == "lr":
        return create_learning_rate_schedule_plot(history_df)
    else:
        return create_metrics_dashboard(history_df)


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
