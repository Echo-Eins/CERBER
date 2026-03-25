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
        return "No files uploaded", gr.Dropdown(choices=[]), gr.Dropdown(choices=[])

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
            "",
            None,
        )

    checkpoint = session_state["checkpoints"][checkpoint_path]
    metadata = checkpoint["metadata"]
    metrics = extract_metrics(checkpoint)

    session_state["current_checkpoint"] = checkpoint_path

    # Summary
    cos_before_str = f"{metrics.cos_sim_before:.4f}" if metrics.cos_sim_before is not None else "N/A"
    cos_after_str = f"{metrics.cos_sim_after:.4f}" if metrics.cos_sim_after is not None else "N/A"
    improvement_str = f"{metrics.cos_improvement:.4f}" if metrics.cos_improvement is not None else "N/A"
    success_rate_str = f"{metrics.success_rate}" if metrics.success_rate is not None else "N/A"

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
- Cosine Before: {cos_before_str}
- Cosine After: {cos_after_str}
- Improvement: {improvement_str}
- Success Rate: {success_rate_str}
""".strip()

    # 3D ландшафт
    landscape_fig = None
    inference_info = ""
    trajectory_plot = None

    try:
        landscape_data = generate_landscape_for_checkpoint(checkpoint_path)
        landscape_fig = create_surface_plot(landscape_data, title=f"Energy Landscape — {Path(checkpoint_path).name}")
        session_state["landscape_cache"][checkpoint_path] = landscape_data

        # Информация об инференсе
        if landscape_data.get("v_denoised") is not None:
            v_clean = landscape_data["v_clean"]
            v_noisy = landscape_data["v_noisy"]
            v_denoised = landscape_data["v_denoised"]

            cos_before = float(np.dot(v_clean, v_noisy) / (np.linalg.norm(v_clean) * np.linalg.norm(v_noisy)))
            cos_after = float(np.dot(v_clean, v_denoised) / (np.linalg.norm(v_clean) * np.linalg.norm(v_denoised)))
            improvement = cos_after - cos_before

            inference_info = f"""
**Inference Results:**
- Cosine (clean, noisy): {cos_before:.4f}
- Cosine (clean, denoised): {cos_after:.4f}
- Improvement: {improvement:+.4f}
- Trajectory steps: {len(landscape_data.get('trajectory_2d', [])) or 0}
"""
            # Создаем график траектории
            trajectory_plot = create_trajectory_plot(landscape_data)

    except Exception as e:
        summary += f"\n\n**Landscape Error:** {e}"

    return summary, landscape_fig, inference_info, trajectory_plot


def extract_hidden_dims_from_state_dict(model_state: dict, model_type: str) -> list[int]:
    """
    Извлечение hidden_dims напрямую из state_dict.
    """
    hidden_dims = []
    layer_idx = 0

    while f"net.{layer_idx}.weight" in model_state:
        weight = model_state[f"net.{layer_idx}.weight"]
        out_dim = weight.shape[0]

        # Пропускаем финальный слой (выход = 1)
        if out_dim == 1:
            break

        hidden_dims.append(out_dim)
        layer_idx += 2  # Linear + activation

    return hidden_dims


def create_trajectory_plot(landscape_data: dict) -> go.Figure:
    """
    Создание 2D графика траектории Langevin dynamics.
    """
    fig = go.Figure()

    trajectory_2d = landscape_data.get("trajectory_2d")
    if trajectory_2d:
        traj_x = [p[0] for p in trajectory_2d]
        traj_y = [p[1] for p in trajectory_2d]

        # Линия траектории
        fig.add_trace(go.Scatter(
            x=traj_x,
            y=traj_y,
            mode="lines",
            line=dict(color="cyan", width=2),
            name="Trajectory",
            opacity=0.6,
        ))

        # Точки вдоль траектории с цветовым градиентом
        fig.add_trace(go.Scatter(
            x=traj_x,
            y=traj_y,
            mode="markers",
            marker=dict(
                size=6,
                color=list(range(len(traj_x))),
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Step", thickness=10),
            ),
            name="Steps",
        ))

        # Start point (noisy)
        if trajectory_2d:
            fig.add_trace(go.Scatter(
                x=[traj_x[0]],
                y=[traj_y[0]],
                mode="markers",
                marker=dict(size=12, color="red", symbol="x", line=dict(width=2, color="white")),
                name="Start (Noisy)",
            ))

            # End point (denoised)
            fig.add_trace(go.Scatter(
                x=[traj_x[-1]],
                y=[traj_y[-1]],
                mode="markers",
                marker=dict(size=12, color="green", symbol="circle", line=dict(width=2, color="white")),
                name="End (Denoised)",
            ))

    # Clean point
    clean_point = landscape_data.get("clean_point")
    if clean_point:
        fig.add_trace(go.Scatter(
            x=[clean_point[0]],
            y=[clean_point[1]],
            mode="markers",
            marker=dict(size=15, color="white", symbol="star", line=dict(width=2, color="yellow")),
            name="Clean (Target)",
        ))

    # Noisy point
    noisy_point = landscape_data.get("noisy_point")
    if noisy_point:
        fig.add_trace(go.Scatter(
            x=[noisy_point[0]],
            y=[noisy_point[1]],
            mode="markers",
            marker=dict(size=10, color="red", symbol="x", line=dict(width=2, color="white")),
            name="Noisy (Start)",
        ))

    fig.update_layout(
        title="Langevin Dynamics Trajectory",
        xaxis_title="Direction 1 (noisy → clean)",
        yaxis_title="Direction 2 (perpendicular)",
        showlegend=True,
        legend=dict(x=1.02, y=1, yanchor="top"),
        width=500,
        height=500,
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray", scaleanchor="x", scaleratio=1)

    return fig


def run_inference_fn(checkpoint_path, noise_scale, num_steps, learning_rate):
    """
    Запуск инференса модели с показом траектории.
    """
    if not checkpoint_path or checkpoint_path not in session_state["checkpoints"]:
        return "No checkpoint selected", None

    checkpoint = session_state["checkpoints"][checkpoint_path]
    model_type = checkpoint["model_type"]
    model_state = checkpoint["model_state"]
    config = checkpoint.get("config", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем модель
    from cebcm.models.energy import SimpleEnergy
    from cebcm.models.energy_unconditional import UnconditionalEnergy

    hidden_dims = extract_hidden_dims_from_state_dict(model_state, model_type)
    if not hidden_dims:
        hidden_dims = [2048, 1024, 512]

    dim = 1024

    if model_type == "simple":
        model = SimpleEnergy(
            dim=dim,
            hidden_dims=hidden_dims,
            norm_mode=config.get("norm_mode", "orthonorm"),
            activation=config.get("activation", "groupsort"),
        ).to(device)
        model.load_state_dict(model_state, strict=False)
        if "_sigma_freqs" in model_state:
            model._sigma_freqs = model_state["_sigma_freqs"].to(device)
    else:
        model = UnconditionalEnergy(
            dim=dim,
            hidden_dims=hidden_dims,
            norm_mode=config.get("norm_mode", "orthonorm"),
            activation=config.get("activation", "groupsort"),
        ).to(device)
        model.load_state_dict(model_state)

    model.eval()

    # Генерируем тестовые векторы
    torch.manual_seed(42)
    v_clean = torch.randn(1, 1024, device=device)
    v_clean = v_clean / v_clean.norm() * 10

    v_noisy = v_clean + torch.randn_like(v_clean) * noise_scale * v_clean.norm()

    # Запускаем Langevin
    v_denoised, trajectory = run_langevin_denoise(
        model=model,
        v_clean=v_clean,
        v_noisy=v_noisy,
        model_type=model_type,
        device=device,
        max_steps=num_steps,
        lr=learning_rate,
    )

    # Вычисляем метрики
    v_clean_np = v_clean.squeeze(0).cpu().numpy()
    v_noisy_np = v_noisy.squeeze(0).cpu().numpy()
    v_denoised_np = v_denoised.squeeze(0).cpu().numpy()

    cos_before = float(np.dot(v_clean_np, v_noisy_np) / (np.linalg.norm(v_clean_np) * np.linalg.norm(v_noisy_np)))
    cos_after = float(np.dot(v_clean_np, v_denoised_np) / (np.linalg.norm(v_clean_np) * np.linalg.norm(v_denoised_np)))
    improvement = cos_after - cos_before

    info = f"""
**Inference Results:**
- Noise scale: {noise_scale}
- Steps: {num_steps}
- Learning rate: {learning_rate}

- Cosine (clean, noisy): {cos_before:.4f}
- Cosine (clean, denoised): {cos_after:.4f}
- Improvement: {improvement:+.4f}
- Trajectory steps: {len(trajectory)}
"""

    # Создаем данные для траектории в 2D
    # Используем ту же логику что и в scan_energy_landscape
    from cebcm.visualization.energy_landscape import _make_orthogonal_basis

    v_clean_flat = v_clean.squeeze(0)
    v_noisy_flat = v_noisy.squeeze(0)
    axis1, axis2 = _make_orthogonal_basis(v_clean_flat, v_noisy_flat)
    center = v_clean_flat

    def project(v):
        diff = v - center
        return (float(diff @ axis1), float(diff @ axis2))

    trajectory_2d = [project(v.squeeze(0)) for v in trajectory]

    landscape_data = {
        "clean_point": project(v_clean_flat),
        "noisy_point": project(v_noisy_flat),
        "denoised_point": project(v_denoised.squeeze(0)),
        "trajectory_2d": trajectory_2d,
    }

    trajectory_fig = create_trajectory_plot(landscape_data)

    return info, trajectory_fig


def run_langevin_denoise(
    model,
    v_clean: torch.Tensor,
    v_noisy: torch.Tensor,
    model_type: str,
    device: torch.device,
    max_steps: int = 50,
    lr: float = 0.01,
    noise_scale: float = 0.005,
    target_norm: float = 10.0,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Запуск Langevin dynamics для денуазинга с сохранением траектории.

    Args:
        model: Энергетическая модель
        v_clean: [1, D] чистый вектор (целевой)
        v_noisy: [1, D] зашумленный вектор (стартовый)
        model_type: "simple" или "unconditional"
        device: torch device
        max_steps: Количество шагов Langevin
        lr: Learning rate
        noise_scale: Множитель шума
        target_norm: Целевая норма векторов

    Returns:
        (v_denoised, trajectory) — финальный вектор и список векторов траектории
    """
    trajectory = []
    v_current = v_noisy.clone().detach()

    for step in range(max_steps):
        # Сохраняем текущую позицию в траекторию
        trajectory.append(v_current.detach().cpu().clone())

        if model_type == "unconditional":
            # UnconditionalEnergy: E(x) → scalar
            energy, grad = model.energy_and_grad(v_current)
            # Langevin step
            langevin_noise = torch.randn_like(v_current) * (2 * lr * noise_scale) ** 0.5
            v_current = v_current - lr * grad + langevin_noise
        else:
            # SimpleEnergy: E(v_query, v_candidate) → scalar
            # Для денуазинга используем v_clean как query и v_current как candidate
            v_current = v_current.requires_grad_(True)
            energy = model(v_clean, v_current)
            grad = torch.autograd.grad(energy.sum(), v_current, create_graph=False)[0]
            v_current = v_current.detach()
            # Langevin step
            langevin_noise = torch.randn_like(v_current) * (2 * lr * noise_scale) ** 0.5
            v_current = v_current - lr * grad + langevin_noise

        # Projection на сферу с target_norm
        if target_norm is not None:
            v_current = torch.nn.functional.normalize(v_current, dim=-1) * target_norm

    # Добавляем финальную позицию
    trajectory.append(v_current.detach().cpu().clone())

    return v_current, trajectory


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

    # Извлекаем hidden_dims напрямую из state_dict
    hidden_dims = extract_hidden_dims_from_state_dict(model_state, model_type)
    if not hidden_dims:
        hidden_dims = [2048, 1024, 512]  # fallback

    dim = 1024  # SONAR dim

    if model_type == "simple":
        # Для SimpleEnergy создаем модель и загружаем state_dict с strict=False
        # чтобы избежать конфликта буферов (_sigma_freqs)
        model = SimpleEnergy(
            dim=dim,
            hidden_dims=hidden_dims,
            norm_mode=config.get("norm_mode", "orthonorm"),
            activation=config.get("activation", "groupsort"),
        ).to(device)

        # Загружаем только параметры, игнорируя буферы
        model.load_state_dict(model_state, strict=False)

        # Если в state_dict есть _sigma_freqs, загружаем его вручную
        if "_sigma_freqs" in model_state:
            model._sigma_freqs = model_state["_sigma_freqs"].to(device)

    else:
        model = UnconditionalEnergy(
            dim=dim,
            hidden_dims=hidden_dims,
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

    # Запускаем Langevin dynamics для получения траектории и denoised вектора
    v_denoised, trajectory = run_langevin_denoise(
        model=model,
        v_clean=v_clean,
        v_noisy=v_noisy,
        model_type=model_type,
        device=device,
        max_steps=50,
    )

    # Сканирование ландшафта
    from cerber_gui.landscape_3d import scan_energy_landscape_3d

    landscape_data = scan_energy_landscape_3d(
        energy_fn=model,
        v_clean=v_clean,
        v_noisy=v_noisy,
        grid_size=40,  # Уменьшено для скорости
        range_factor=1.0,
        v_denoised=v_denoised,
        trajectory=trajectory,
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

    return table_md, fig, None


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

with gr.Blocks(title="CERBER Model Monitor") as demo:
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

            with gr.Row():
                landscape_plot = gr.Plot(label="3D Energy Landscape", scale=2)
                trajectory_plot = gr.Plot(label="Langevin Trajectory", scale=1)

            gr.Markdown("### Inference Settings")

            with gr.Row():
                noise_scale_slider = gr.Slider(
                    minimum=0.05, maximum=0.5, value=0.15, step=0.01,
                    label="Noise Scale"
                )
                num_steps_slider = gr.Slider(
                    minimum=10, maximum=200, value=50, step=10,
                    label="Langevin Steps"
                )
                lr_slider = gr.Slider(
                    minimum=0.001, maximum=0.1, value=0.01, step=0.001,
                    label="Learning Rate"
                )
                run_inference_btn = gr.Button("Run Inference", variant="primary")

            with gr.Row():
                inference_output = gr.Markdown()

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
            compare_status = gr.Textbox(label="Status", visible=False)

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
        outputs=[checkpoint_summary, landscape_plot, inference_output, trajectory_plot],
    )

    # Запуск инференса
    run_inference_btn.click(
        run_inference_fn,
        inputs=[checkpoint_dropdown, noise_scale_slider, num_steps_slider, lr_slider],
        outputs=[inference_output, trajectory_plot],
    )

    # Сравнение
    compare_btn.click(
        compare_selected_fn,
        inputs=[compare_dropdown],
        outputs=[compare_output, compare_plot, compare_status],
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
        outputs=[live_status, live_plot],
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
        theme=gr.themes.Soft(),
    )
