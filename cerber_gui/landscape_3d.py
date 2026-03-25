"""
3D Energy Landscape Visualizer — интерактивная визуализация энергетического ландшафта.

Использует Plotly для интерактивных 3D поверхностей, анимаций и Langevin траекторий.

Функции:
    scan_energy_landscape_3d(...) — сканирование ландшафта для 3D визуализации
    create_surface_plot(...) — создание 3D поверхности энергии
    create_contour_plot(...) — создание 2D контурного графика
    add_trajectory_to_plot(...) — добавление Langevin траектории
    create_comparison_plot(...) — сравнение "до/после" обучения
    create_animation(...) — анимация эволюции ландшафта по эпохам
"""

import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Literal


def scan_energy_landscape_3d(
    energy_fn,
    v_clean: torch.Tensor,
    v_noisy: torch.Tensor,
    grid_size: int = 50,
    range_factor: float = 1.5,
    v_denoised: torch.Tensor | None = None,
    trajectory: list[torch.Tensor] | None = None,
    model_type: str = "simple",
    batch_size: int = 100,
) -> dict:
    """
    Сканирование энергетического ландшафта в 2D плоскости для 3D визуализации.

    Args:
        energy_fn: Функция энергии (SimpleEnergy или UnconditionalEnergy)
        v_clean: [1, D] чистый вектор
        v_noisy: [1, D] зашумленный вектор
        grid_size: Размер сетки (grid_size x grid_size)
        range_factor: Множитель диапазона (относительно нормы вектора)
        v_denoised: [1, D] денуазированный вектор (опционально)
        trajectory: Список векторов траектории Langevin
        model_type: "simple" или "unconditional"
        batch_size: Размер батча для пакетного вычисления энергии

    Returns:
        Dict с данными для визуализации:
        - x_range, y_range: диапазоны осей
        - energy_grid: [grid_size, grid_size] энергия в каждой точке
        - basis: базисные векторы 2D плоскости
        - clean_point, noisy_point, denoised_point: координаты в 2D
        - trajectory_2d: траектория в 2D координатах
    """
    device = v_clean.device
    v_clean = v_clean.detach()
    v_noisy = v_noisy.detach()

    # Ensure [1, D] shape
    if v_clean.dim() == 1:
        v_clean = v_clean.unsqueeze(0)
    if v_noisy.dim() == 1:
        v_noisy = v_noisy.unsqueeze(0)

    v_clean_flat = v_clean.squeeze(0)  # [D]
    v_noisy_flat = v_noisy.squeeze(0)  # [D]

    # Нормализация для масштаба
    norm = v_clean_flat.norm()
    scale = norm * range_factor / grid_size

    # Построение 2D плоскости между clean и noisy
    direction = v_noisy_flat - v_clean_flat
    direction = direction / direction.norm()

    # Перпендикулярное направление (через проекцию)
    perp = torch.randn_like(v_clean_flat)
    perp = perp - (perp @ direction) * direction
    perp = perp / perp.norm()

    basis = torch.stack([direction, perp])  # [2, D]

    # Центр плоскости (между clean и noisy)
    center = (v_clean_flat + v_noisy_flat) / 2

    # Генерация сетки координат
    coords = torch.linspace(-grid_size // 2, grid_size // 2, grid_size, device=device)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")

    # Генерация точек сетки в пространстве embeddings
    # grid_points: [grid_size*grid_size, D]
    grid_points = center + (xx.flatten().unsqueeze(1) * basis[0] + yy.flatten().unsqueeze(1) * basis[1]) * scale

    # Вычисление энергии для каждой точки
    energies = []
    for start in range(0, len(grid_points), batch_size):
        end = start + batch_size
        batch = grid_points[start:end].to(device)

        import inspect
        sig = inspect.signature(energy_fn.forward)
        if len(sig.parameters) >= 2:
            # Pairwise model (SimpleEnergy)
            v_q = v_clean_flat.unsqueeze(0).expand(batch.shape[0], -1)
            e = energy_fn(v_q, batch)
        else:
            # Unconditional model (UnconditionalEnergy)
            e = energy_fn(batch)

        energies.append(e.detach().cpu())

    energy_grid = torch.cat(energies).reshape(grid_size, grid_size)

    # Проекция точек на 2D плоскость
    def project_to_2d(v: torch.Tensor) -> tuple[float, float]:
        v = v.squeeze(0)
        diff = v - center
        x = (diff @ basis[0]).item() / scale
        y = (diff @ basis[1]).item() / scale
        return x, y

    clean_2d = project_to_2d(v_clean_flat)
    noisy_2d = project_to_2d(v_noisy_flat)
    denoised_2d = project_to_2d(v_denoised.squeeze(0)) if v_denoised is not None else None

    # Проекция траектории
    trajectory_2d = None
    if trajectory:
        trajectory_2d = []
        for v in trajectory:
            if v.dim() == 1:
                v = v.unsqueeze(0)
            trajectory_2d.append(project_to_2d(v))

    return {
        "x_range": coords.tolist(),
        "y_range": coords.tolist(),
        "energy_grid": energy_grid.numpy(),
        "basis": basis.cpu().numpy(),
        "scale": scale.item(),
        "center": center.cpu().numpy(),
        "clean_point": clean_2d,
        "noisy_point": noisy_2d,
        "denoised_point": denoised_2d,
        "trajectory_2d": trajectory_2d,
        "v_clean": v_clean_flat.cpu().numpy(),
        "v_noisy": v_noisy_flat.cpu().numpy(),
        "v_denoised": v_denoised.cpu().numpy() if v_denoised is not None else None,
    }


def create_surface_plot(
    data: dict,
    title: str = "Energy Landscape",
    colorscale: str = "Viridis",
    show_legend: bool = True,
) -> go.Figure:
    """
    Создание 3D поверхности энергии.

    Args:
        data: Результат scan_energy_landscape_3d
        title: Заголовок графика
        colorscale: Цветовая схема Plotly
        show_legend: Показывать ли легенду

    Returns:
        Plotly Figure с 3D поверхностью
    """
    fig = go.Figure()

    # 3D поверхность
    fig.add_trace(go.Surface(
        x=data["x_range"],
        y=data["y_range"],
        z=data["energy_grid"],
        colorscale=colorscale,
        opacity=0.9,
        colorbar=dict(title="Energy", thickness=20),
        hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Energy: %{z:.4f}<extra></extra>",
    ))

    # Точки clean/noisy/denoised
    if data["clean_point"]:
        fig.add_trace(go.Scatter3d(
            x=[data["clean_point"][0]],
            y=[data["clean_point"][1]],
            z=[0],
            mode="markers",
            marker=dict(size=8, color="green", symbol="circle"),
            name="Clean",
            hovertemplate="Clean vector<extra></extra>",
        ))

    if data["noisy_point"]:
        fig.add_trace(go.Scatter3d(
            x=[data["noisy_point"][0]],
            y=[data["noisy_point"][1]],
            z=[0],
            mode="markers",
            marker=dict(size=8, color="red", symbol="circle"),
            name="Noisy",
            hovertemplate="Noisy vector<extra></extra>",
        ))

    if data["denoised_point"]:
        fig.add_trace(go.Scatter3d(
            x=[data["denoised_point"][0]],
            y=[data["denoised_point"][1]],
            z=[0],
            mode="markers",
            marker=dict(size=8, color="blue", symbol="circle"),
            name="Denoised",
            hovertemplate="Denoised vector<extra></extra>",
        ))

    # Траектория Langevin
    if data["trajectory_2d"]:
        traj_x = [p[0] for p in data["trajectory_2d"]]
        traj_y = [p[1] for p in data["trajectory_2d"]]
        traj_z = [0] * len(traj_x)

        fig.add_trace(go.Scatter3d(
            x=traj_x,
            y=traj_y,
            z=traj_z,
            mode="lines+markers",
            line=dict(color="yellow", width=4),
            marker=dict(size=3, color="yellow"),
            name="Langevin Trajectory",
            hovertemplate="Trajectory step<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        scene=dict(
            xaxis_title="Direction 1 (noisy → clean)",
            yaxis_title="Direction 2 (perpendicular)",
            zaxis_title="Energy",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        showlegend=show_legend,
        legend=dict(x=0.02, y=0.98, yanchor="top"),
        width=900,
        height=700,
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig


def create_contour_plot(
    data: dict,
    title: str = "Energy Contours",
    colorscale: str = "Viridis",
    show_trajectory: bool = True,
) -> go.Figure:
    """
    Создание 2D контурного графика энергии.

    Args:
        data: Результат scan_energy_landscape_3d
        title: Заголовок графика
        colorscale: Цветовая схема
        show_trajectory: Показывать ли траекторию

    Returns:
        Plotly Figure с контурным графиком
    """
    fig = go.Figure()

    # Контуры
    fig.add_trace(go.Contour(
        z=data["energy_grid"],
        x=data["x_range"],
        y=data["y_range"],
        colorscale=colorscale,
        contours=dict(
            coloring="heatmap",
            showlabels=True,
            labelfont=dict(size=10),
        ),
        hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Energy: %{z:.4f}<extra></extra>",
    ))

    # Точки
    if data["clean_point"]:
        fig.add_trace(go.Scatter(
            x=[data["clean_point"][0]],
            y=[data["clean_point"][1]],
            mode="markers",
            marker=dict(size=12, color="green", line=dict(width=2, color="white")),
            name="Clean",
        ))

    if data["noisy_point"]:
        fig.add_trace(go.Scatter(
            x=[data["noisy_point"][0]],
            y=[data["noisy_point"][1]],
            mode="markers",
            marker=dict(size=12, color="red", line=dict(width=2, color="white")),
            name="Noisy",
        ))

    if data["denoised_point"]:
        fig.add_trace(go.Scatter(
            x=[data["denoised_point"][0]],
            y=[data["denoised_point"][1]],
            mode="markers",
            marker=dict(size=12, color="blue", line=dict(width=2, color="white")),
            name="Denoised",
        ))

    # Траектория
    if show_trajectory and data["trajectory_2d"]:
        traj_x = [p[0] for p in data["trajectory_2d"]]
        traj_y = [p[1] for p in data["trajectory_2d"]]

        fig.add_trace(go.Scatter(
            x=traj_x,
            y=traj_y,
            mode="lines+markers",
            line=dict(color="yellow", width=3),
            marker=dict(size=5, color="yellow"),
            name="Trajectory",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Direction 1 (noisy → clean)",
        yaxis_title="Direction 2 (perpendicular)",
        width=700,
        height=600,
        showlegend=True,
        legend=dict(x=1.02, y=1, yanchor="top"),
    )

    return fig


def create_comparison_plot(
    data_before: dict,
    data_after: dict,
    title: str = "Energy Landscape Comparison",
) -> go.Figure:
    """
    Сравнение ландшафтов "до" и "после" обучения.

    Args:
        data_before: Данные ландшафта до обучения
        data_after: Данные ландшафта после обучения
        title: Заголовок

    Returns:
        Plotly Figure с двумя подграфиками
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=["Before Training", "After Training"],
    )

    # Before
    fig.add_trace(
        go.Surface(
            x=data_before["x_range"],
            y=data_before["y_range"],
            z=data_before["energy_grid"],
            colorscale="Viridis",
            opacity=0.9,
            colorbar=dict(len=0.5, y=0.25),
        ),
        row=1,
        col=1,
    )

    # After
    fig.add_trace(
        go.Surface(
            x=data_after["x_range"],
            y=data_after["y_range"],
            z=data_after["energy_grid"],
            colorscale="Viridis",
            opacity=0.9,
            colorbar=dict(len=0.5, y=0.75),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        width=1400,
        height=600,
        scene=dict(
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        scene2=dict(
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
    )

    return fig


def create_animation(
    checkpoint_data_list: list[dict],
    epoch_labels: list[str],
    title: str = "Energy Landscape Evolution",
) -> go.Figure:
    """
    Создание анимации эволюции ландшафта по эпохам.

    Args:
        checkpoint_data_list: Список данных ландшафта для каждой эпохи
        epoch_labels: Список меток эпох (для слайдера)
        title: Заголовок анимации

    Returns:
        Plotly Figure с анимацией и слайдером
    """
    frames = []
    slider_steps = []

    for i, data in enumerate(checkpoint_data_list):
        frame = go.Frame(
            data=[
                go.Surface(
                    z=data["energy_grid"],
                    x=data["x_range"],
                    y=data["y_range"],
                    colorscale="Viridis",
                    opacity=0.9,
                ),
            ],
            name=f"frame{i}",
        )
        frames.append(frame)

        slider_steps.append({
            "args": [[f"frame{i}"], {"frame": {"duration": 300, "redraw": True}, "mode": "immediate"}],
            "label": epoch_labels[i],
            "method": "animate",
        })

    # Initial frame
    initial_data = checkpoint_data_list[0]
    fig = go.Figure(
        data=[
            go.Surface(
                z=initial_data["energy_grid"],
                x=initial_data["x_range"],
                y=initial_data["y_range"],
                colorscale="Viridis",
                opacity=0.9,
            ),
        ],
        frames=frames,
    )

    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"font": {"size": 14}, "prefix": "Epoch: ", "visible": True, "xanchor": "right"},
                "len": 0.9,
                "pad": {"b": 10, "t": 60},
                "steps": slider_steps,
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "x": 0.1,
                "xanchor": "left",
                "y": 0,
                "yanchor": "top",
            }
        ],
        title=dict(text=title, font=dict(size=20)),
        scene=dict(
            xaxis_title="Direction 1",
            yaxis_title="Direction 2",
            zaxis_title="Energy",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        width=900,
        height=700,
    )

    return fig


def export_figure_to_html(fig: go.Figure, output_path: str) -> None:
    """
    Экспорт интерактивной фигуры в HTML файл.

    Args:
        fig: Plotly Figure
        output_path: Путь к выходному HTML файлу
    """
    import plotly.io as pio

    pio.write_html(fig, file=output_path, auto_open=False, include_plotlyjs="cdn")


def export_figure_to_image(
    fig: go.Figure,
    output_path: str,
    format: Literal["png", "jpeg", "svg", "pdf"] = "png",
    width: int = 1200,
    height: int = 800,
    scale: float = 2.0,
) -> None:
    """
    Экспорт фигуры в статическое изображение.

    Args:
        fig: Plotly Figure
        output_path: Путь к выходному файлу
        format: Формат изображения
        width: Ширина в пикселях
        height: Высота в пикселях
        scale: Множитель масштаба (для retina)
    """
    import plotly.io as pio

    pio.write_image(fig, file=output_path, format=format, width=width, height=height, scale=scale)
