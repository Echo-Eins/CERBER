"""
CERBER GUI — универсальный инструмент мониторинга и визуализации.

Модули:
    checkpoint_analyzer — загрузка, парсинг, сравнение чекпоинтов
    metrics_viewer — визуализация метрик обучения
    landscape_3d — 3D визуализация энергетического ландшафта
    live_monitor — реал-тайм мониторинг тренировки
    app — главное Gradio приложение

Usage:
    python cerber_gui/app.py
"""

from cerber_gui.checkpoint_analyzer import (
    load_checkpoint,
    extract_metrics,
    compare_checkpoints,
    export_comparison,
    get_checkpoint_summary,
)
from cerber_gui.metrics_viewer import (
    load_training_metrics,
    metrics_to_dataframe,
    create_loss_plot,
    create_cosine_similarity_plot,
    create_metrics_dashboard,
)
from cerber_gui.landscape_3d import (
    scan_energy_landscape_3d,
    create_surface_plot,
    create_contour_plot,
    create_comparison_plot,
)
from cerber_gui.live_monitor import (
    TrainingMetricsWatcher,
    create_live_metrics_plot,
)

__all__ = [
    # Checkpoint Analyzer
    "load_checkpoint",
    "extract_metrics",
    "compare_checkpoints",
    "export_comparison",
    "get_checkpoint_summary",
    # Metrics Viewer
    "load_training_metrics",
    "metrics_to_dataframe",
    "create_loss_plot",
    "create_cosine_similarity_plot",
    "create_metrics_dashboard",
    # Landscape 3D
    "scan_energy_landscape_3d",
    "create_surface_plot",
    "create_contour_plot",
    "create_comparison_plot",
    # Live Monitor
    "TrainingMetricsWatcher",
    "create_live_metrics_plot",
]
