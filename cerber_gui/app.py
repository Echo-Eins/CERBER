"""
CERBER Model Monitor — главное Gradio приложение.

Универсальный GUI для:
- Загрузки и сравнения N чекпоинтов
- Визуализации метрик обучения
- 3D визуализации энергетического ландшафта
- Реал-тайм мониторинга тренировки

Usage:
    python cerber_gui/app.py

    # Откроется http://localhost:7860
"""

import sys
from pathlib import Path

# Добавляем корень проекта в path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import gradio as gr
import plotly.graph_objects as go

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
    create_contour_plot,
    create_comparison_plot,
    export_figure_to_html,
)
from cerber_gui.live_monitor import (
    TrainingMetricsWatcher,
    create_live_metrics_plot,
)


# Глобальное состояние сессии
session_state = {
    "checkpoints": {},  # path -> checkpoint data
    "current_checkpoint": None,
    "metrics_file": None,
    "watcher": None,
    "landscape_cache": {},  # checkpoint_path -> landscape data
}


def load_checkpoints_fn(files):
    """Загрузка чекпоинтов из uploaded файлов."""
    if not files:
        return "No files uploaded", gr.update(), gr.update()

    results = []
    errors = []

    for file in files:
        try:
            # Gradio передает файлы как tempfile
            checkpoint = load_checkpoint(file.name)
            metadata = checkpoint["metadata"]

            session_state["checkpoints"][file.name] = checkpoint

            results.append({
                "name": Path(file.name).name,
                "epoch": metadata.epoch,
                "model_type": metadata.model_type,
                "hidden_dims": metadata.energy_hidden_dims,
            })
        except Exception as e:
            errors.append(f"{Path(file.name).name}: {e}")

    # Формируем summary
    if results:
        summary = "\n".join([
            f"✓ {r['name']} — epoch {r['epoch']}, {r['model_type']}, hidden={r['hidden_dims']}"
            for r in results
        ])
    else:
        summary = ""

    if errors:
        summary += "\n\nErrors:\n" + "\n".join(errors)

    # Обновляем dropdown
    dropdown_choices = list(session_state["checkpoints"].keys())

    return summary, gr.update(choices=dropdown_choices), gr.update(choices=dropdown_choices)


def select_checkpoint_fn(checkpoint_path):
    """Выбор чекпоинта для анализа."""
    if not checkpoint_path or checkpoint_path not in session_state["checkpoints"]:
        return (
            "No checkpoint selected",
            None,
            "No metrics available",
        )

    checkpoint = session_state["checkpoints"][checkpoint_path]
    metadata = checkpoint["metadata"]
    metrics = extract_metrics(checkpoint)

    session_state["current_checkpoint"] = checkpoint_path

    # Summary
    summary = f"""
**Checkpoint:** {Path(checkpoint_path).name}
**Epoch:** {metadata.epoch}
**Model Type:** {metadata.model_type}
**Architecture:** {metadata.energy_dim} → {metadata.energy_hidden_dims} → 1
**Normalization:** {metadata.norm_mode}
**Activation:** {metadata.activation}

**Metrics:**
- Train Loss: {metrics.train_loss}
- Eval Loss: {metrics.eval_loss}
- Cosine Before: {metrics.cos_sim_before:.4f}" if metrics.cos_sim_before else "N/A"
- Cosine After: {metrics.cos_sim_after:.4f}" if metrics.cos_sim_after else "N/A"
- Improvement: {metrics.cos_improvement:.4f}" if metrics.cos_improvement else "N/A"
- Success Rate: {metrics.success_rate}" if metrics.success_rate else "N/A"
""".strip()

    # 3D ландшафт
    landscape_fig = None
    try:
        landscape_data = generate_landscape_for_checkpoint(checkpoint_path)
        landscape_fig = create_surface_plot(landscape_data, title=f"Energy Landscape — {Path(checkpoint_path).name}")
        session_state["landscape_cache"][checkpoint_path] = landscape_data
    except Exception as e:
        summary += f"\n\n**Landscape Error:** {e}"

    return summary, landscape_fig, gr.update()


def generate_landscape_for_checkpoint(checkpoint_path):
    """Генерация ландшафта для чекпоинта."""
    if checkpoint_path in session_state["landscape_cache"]:
        return session_state["landscape_cache"][checkpoint_path]

    checkpoint = session_state["checkpoints"].get(checkpoint_path)
    if not checkpoint:
        raise ValueError("Checkpoint not loaded")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем модель из чекпоинта
    from cebcm.models.energy import SimpleEnergy
    from cebcm.models.energy_unconditional import UnconditionalEnergy

    model_state = checkpoint["model_state"]
    model_type = checkpoint["model_type"]
    config = checkpoint.get("config", {})

    if model_type == "simple":
        model = SimpleEnergy(
            dim=config.get("energy_dim", 1024),
            hidden_dims=config.get("energy_hidden_dims", [2048, 1024, 512]),
            norm_mode=config.get("norm_mode", "orthonorm"),
            activation=config.get("activation", "groupsort"),
        ).to(device)
    else:
        model = UnconditionalEnergy(
            dim=config.get("energy_dim", 1024),
            hidden_dims=config.get("energy_hidden_dims", [2048, 1024, 512]),
            norm_mode=config.get("norm_mode", "orthonorm"),
            activation=config.get("activation", "groupsort"),
        ).to(device)

    model.load_state_dict(model_state)
    model.eval()

    # Генерируем тестовые векторы
    v_clean = torch.randn(1, 1024, device=device)
    v_clean = v_clean / v_clean.norm() * 10  # Нормализуем к масштабу SONAR

    noise_scale = 0.15
    v_noisy = v_clean + torch.randn_like(v_clean) * noise_scale * v_clean.norm()

    # Сканирование ландшафта
    from cerber_gui.landscape_3d import scan_energy_landscape_3d

    landscape_data = scan_energy_landscape_3d(
        energy_fn=model,
        v_clean=v_clean,
        v_noisy=v_noisy,
        grid_size=40,  # Уменьшено для скорости
        range_factor=1.0,
        model_type=model_type,
    )

    return landscape_data


def compare_selected_fn(checkpoint_paths):
    """Сравнение выбранных чекпоинтов."""
    if not checkpoint_paths or len(checkpoint_paths) < 2:
        return "Select at least 2 checkpoints", None, None

    # Загружаем чекпоинты если еще не загружены
    for path in checkpoint_paths:
        if path not in session_state["checkpoints"]:
            try:
                checkpoint = load_checkpoint(path)
                session_state["checkpoints"][path] = checkpoint
            except Exception as e:
                return f"Error loading {Path(path).name}: {e}", None, None

    # Сравниваем
    df = compare_checkpoints(checkpoint_paths)

    # Таблица
    table_md = df.to_markdown(index=False)

    # График сравнения
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

    return table_md, fig, gr.update()


def load_metrics_file_fn(file):
    """Загрузка файла training_metrics.json."""
    if not file:
        return "No file selected", None

    try:
        session_state["metrics_file"] = file.name
        df = metrics_to_dataframe(file.name)

        # Создаем dashboard
        fig = create_metrics_dashboard(df, title="Training Metrics")

        # Statistics
        stats = compute_summary_statistics(df)
        stats_text = "\n".join([f"**{k}:** {v:.4f}" if isinstance(v, float) else f"**{k}:** {v}" for k, v in stats.items()])

        return stats_text, fig
    except Exception as e:
        return f"Error: {e}", None


def start_live_monitor_fn(metrics_path):
    """Запуск live мониторинга."""
    if not metrics_path:
        return "No path specified", None

    path = Path(metrics_path)
    if not path.exists():
        return f"File not found: {path}", None

    try:
        # Создаем watcher
        watcher = TrainingMetricsWatcher(path)
        watcher.start()
        session_state["watcher"] = watcher

        return f"Monitoring started: {path}", gr.update(value=watcher.history.to_dataframe())
    except Exception as e:
        return f"Error: {e}", None


def update_live_plot_fn():
    """Обновление live графика (вызывается по таймеру)."""
    watcher = session_state.get("watcher")
    if not watcher:
        return None

    df = watcher.history.to_dataframe()
    if df.empty:
        return None

    return create_live_metrics_plot(df, plot_type="all")


def export_comparison_fn(checkpoint_paths, output_format):
    """Экспорт сравнения чекпоинтов."""
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

with gr.Blocks(title="CERBER Model Monitor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # CERBER Model Monitor

    Универсальный инструмент для анализа чекпоинтов, визуализации метрик и 3D ландшафта энергии.
    """)

    with gr.Tabs():
        # === Tab 1: Checkpoint Analysis ===
        with gr.TabItem("Checkpoint Analysis"):
            gr.Markdown("### Загрузка чекпоинтов")

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

            gr.Markdown("### Выбор чекпоинта для анализа")

            with gr.Row():
                checkpoint_dropdown = gr.Dropdown(
                    label="Select Checkpoint",
                    choices=[],
                    interactive=True,
                )

            with gr.Row():
                checkpoint_summary = gr.Markdown()
                landscape_plot = gr.Plot(label="3D Energy Landscape")

        # === Tab 2: Comparison ===
        with gr.TabItem("Comparison"):
            gr.Markdown("### Сравнение чекпоинтов")

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

        # === Tab 3: Metrics ===
        with gr.TabItem("Training Metrics"):
            gr.Markdown("### Загрузка training_metrics.json")

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
            gr.Markdown("""
            ### Реал-тайм мониторинг тренировки

            Укажите путь к `training_metrics.json` который обновляется во время тренировки.
            """)

            with gr.Row():
                live_path_input = gr.Textbox(
                    label="Path to training_metrics.json",
                    placeholder="experiments/02_energy_matching/training_metrics.json",
                )
                start_live_btn = gr.Button("Start Monitoring", variant="primary")

            live_status = gr.Textbox(label="Status")
            live_plot = gr.Plot(label="Live Metrics")

            # Auto-refresh каждые 5 секунд
            live_timer = gr.Timer(value=5, active=True)

    # === Event Handlers ===

    # Загрузка чекпоинтов
    load_btn.click(
        load_checkpoints_fn,
        inputs=[file_upload],
        outputs=[load_output, checkpoint_dropdown, compare_dropdown],
    )

    # Выбор чекпоинта
    checkpoint_dropdown.change(
        select_checkpoint_fn,
        inputs=[checkpoint_dropdown],
        outputs=[checkpoint_summary, landscape_plot, metrics_stats],
    )

    # Сравнение
    compare_btn.click(
        compare_selected_fn,
        inputs=[compare_dropdown],
        outputs=[compare_output, compare_plot, gr.update()],
    )

    # Экспорт
    export_btn.click(
        export_comparison_fn,
        inputs=[compare_dropdown, export_format],
        outputs=[export_output],
    )

    # Загрузка метрик
    metrics_upload.change(
        load_metrics_file_fn,
        inputs=[metrics_upload],
        outputs=[metrics_stats, metrics_plot],
    )

    # Live мониторинг
    start_live_btn.click(
        start_live_monitor_fn,
        inputs=[live_path_input],
        outputs=[live_status, gr.update()],
    )

    live_timer.tick(
        update_live_plot_fn,
        outputs=[live_plot],
    )


if __name__ == "__main__":
    # Создаем директорию для экспорта
    (Path(__file__).parent / "exports").mkdir(exist_ok=True)

    demo.queue(max_size=10)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
