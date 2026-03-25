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

# Переиспользуем проверенный scan_energy_landscape из cebcm.visualization
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cebcm.visualization.energy_landscape import scan_energy_landscape as _scan_energy_landscape


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

    Использует проверенную функцию из cebcm.visualization.energy_landscape.

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
    # Используем проверенную функцию из cebcm.visualization
    landscape_data = _scan_energy_landscape(
        energy_fn=energy_fn,
        v_clean=v_clean,
        v_noisy=v_noisy,
        grid_size=grid_size,
        v_denoised=v_denoised,
        trajectory=trajectory,
    )

    # Конвертируем в формат для Plotly — ЧЕСТНЫЕ значения без масштабирования
    energy_np = landscape_data.energy.numpy()
    energy_min = float(energy_np.min())
    energy_max = float(energy_np.max())

    return {
        "x_range": landscape_data.grid_x.tolist(),
        "y_range": landscape_data.grid_y.tolist(),
        "energy_grid": energy_np,  # Честные значения энергии
        "energy_min": energy_min,
        "energy_max": energy_max,
        "basis": landscape_data.basis,
        "clean_point": landscape_data.v_clean_xy,
        "noisy_point": landscape_data.v_noisy_xy,
        "denoised_point": landscape_data.v_denoised_xy,
        "trajectory_2d": landscape_data.trajectory_xy,  # Уже спроецировано в _scan_energy_landscape
        "v_clean": v_clean.squeeze(0).cpu().numpy(),
        "v_noisy": v_noisy.squeeze(0).cpu().numpy(),
        "v_denoised": v_denoised.squeeze(0).cpu().numpy() if v_denoised is not None else None,
    }


def _get_energy_at_point(data: dict, x: float, y: float) -> float:
    """
    Interpolate energy value at arbitrary (x, y) coordinates using nearest neighbor.

    Args:
        data: Landscape data with energy_grid, x_range, y_range
        x, y: Coordinates in the 2D plane

    Returns:
        Energy value at the specified point
    """
    energy_grid = data["energy_grid"]
    x_range = np.array(data["x_range"])
    y_range = np.array(data["y_range"])

    # Find nearest grid indices
    ix = np.argmin(np.abs(x_range - x))
    iy = np.argmin(np.abs(y_range - y))

    # Clamp to valid range
    ix = np.clip(ix, 0, energy_grid.shape[0] - 1)
    iy = np.clip(iy, 0, energy_grid.shape[1] - 1)

    return float(energy_grid[ix, iy])


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

    # Получаем честный диапазон энергий — БЕЗ какого-либо clamping
    energy_min = data.get("energy_min", float(data["energy_grid"].min()))
    energy_max = data.get("energy_max", float(data["energy_grid"].max()))

    # Добавляем диапазон энергий в заголовок
    full_title = f"{title}<br>Energy Range: [{energy_min:.2f}, {energy_max:.2f}]"

    # 3D поверхность с честными значениями и явным zmin/zmax для правильного масштабирования
    fig.add_trace(go.Surface(
        x=data["x_range"],
        y=data["y_range"],
        z=data["energy_grid"],
        colorscale=colorscale,
        opacity=0.9,
        # Явно устанавливаем диапазон для корректного отображения экстремальных значений
        zmin=energy_min,
        zmax=energy_max,
        colorbar=dict(
            title="Energy",
            thickness=20,
            tickformat=".2f",
        ),
        hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Energy: %{z:.4f}<extra></extra>",
    ))

    # Вычисляем энергии для ключевых точек для правильного z-позиционирования
    clean_energy = _get_energy_at_point(data, data["clean_point"][0], data["clean_point"][1]) if data["clean_point"] else 0
    noisy_energy = _get_energy_at_point(data, data["noisy_point"][0], data["noisy_point"][1]) if data["noisy_point"] else 0
    denoised_energy = _get_energy_at_point(data, data["denoised_point"][0], data["denoised_point"][1]) if data["denoised_point"] else 0

    # Точки clean/noisy/denoised — УВЕЛИЧЕННЫЙ размер, размещены на поверхности (не на z=0)
    if data["clean_point"]:
        fig.add_trace(go.Scatter3d(
            x=[data["clean_point"][0]],
            y=[data["clean_point"][1]],
            z=[clean_energy],  # Размещаем на поверхности энергии, а не на z=0
            mode="markers",
            marker=dict(
                size=15,  # Увеличено с 8 до 15 для лучшей видимости
                color="green",
                symbol="circle",
                opacity=1.0,
                line=dict(width=2, color="white"),  # Белая обводка для контраста
            ),
            name="Clean",
            hovertemplate=f"Clean vector<br>X: {data['clean_point'][0]:.2f}<br>Y: {data['clean_point'][1]:.2f}<br>Energy: {clean_energy:.4f}<extra></extra>",
        ))

    if data["noisy_point"]:
        fig.add_trace(go.Scatter3d(
            x=[data["noisy_point"][0]],
            y=[data["noisy_point"][1]],
            z=[noisy_energy],
            mode="markers",
            marker=dict(
                size=15,
                color="red",
                symbol="circle",
                opacity=1.0,
                line=dict(width=2, color="white"),
            ),
            name="Noisy",
            hovertemplate=f"Noisy vector<br>X: {data['noisy_point'][0]:.2f}<br>Y: {data['noisy_point'][1]:.2f}<br>Energy: {noisy_energy:.4f}<extra></extra>",
        ))

    if data["denoised_point"]:
        fig.add_trace(go.Scatter3d(
            x=[data["denoised_point"][0]],
            y=[data["denoised_point"][1]],
            z=[denoised_energy],
            mode="markers",
            marker=dict(
                size=15,
                color="blue",
                symbol="circle",
                opacity=1.0,
                line=dict(width=2, color="white"),
            ),
            name="Denoised",
            hovertemplate=f"Denoised vector<br>X: {data['denoised_point'][0]:.2f}<br>Y: {data['denoised_point'][1]:.2f}<br>Energy: {denoised_energy:.4f}<extra></extra>",
        ))

    # Траектория Langevin — на поверхности энергии
    if data["trajectory_2d"]:
        traj_x = [p[0] for p in data["trajectory_2d"]]
        traj_y = [p[1] for p in data["trajectory_2d"]]
        # Вычисляем энергию для каждой точки траектории
        traj_z = [_get_energy_at_point(data, x, y) for x, y in zip(traj_x, traj_y)]

        fig.add_trace(go.Scatter3d(
            x=traj_x,
            y=traj_y,
            z=traj_z,
            mode="lines+markers",
            line=dict(color="yellow", width=6),  # Увеличена ширина с 4 до 6
            marker=dict(
                size=8,  # Увеличено с 3 до 8
                color="yellow",
                symbol="circle",
                line=dict(width=1, color="black"),  # Чёрная обводка для видимости
            ),
            name="Langevin Trajectory",
            hovertemplate="Trajectory step<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Energy: %{z:.4f}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=full_title, font=dict(size=20)),
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

    # Получаем честный диапазон энергий
    energy_min = data.get("energy_min", float(data["energy_grid"].min()))
    energy_max = data.get("energy_max", float(data["energy_grid"].max()))

    # Контуры с явным диапазоном
    fig.add_trace(go.Contour(
        z=data["energy_grid"],
        x=data["x_range"],
        y=data["y_range"],
        colorscale=colorscale,
        zmin=energy_min,
        zmax=energy_max,
        contours=dict(
            coloring="heatmap",
            showlabels=True,
            labelfont=dict(size=10),
        ),
        hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Energy: %{z:.4f}<extra></extra>",
    ))

    # Вычисляем энергии для ключевых точек
    clean_energy = _get_energy_at_point(data, data["clean_point"][0], data["clean_point"][1]) if data["clean_point"] else 0
    noisy_energy = _get_energy_at_point(data, data["noisy_point"][0], data["noisy_point"][1]) if data["noisy_point"] else 0
    denoised_energy = _get_energy_at_point(data, data["denoised_point"][0], data["denoised_point"][1]) if data["denoised_point"] else 0

    # Точки с увеличенным размером и hover-информацией
    if data["clean_point"]:
        fig.add_trace(go.Scatter(
            x=[data["clean_point"][0]],
            y=[data["clean_point"][1]],
            mode="markers",
            marker=dict(size=15, color="green", line=dict(width=2, color="white")),
            name="Clean",
            hovertemplate=f"Clean vector<br>X: {data['clean_point'][0]:.2f}<br>Y: {data['clean_point'][1]:.2f}<br>Energy: {clean_energy:.4f}<extra></extra>",
        ))

    if data["noisy_point"]:
        fig.add_trace(go.Scatter(
            x=[data["noisy_point"][0]],
            y=[data["noisy_point"][1]],
            mode="markers",
            marker=dict(size=15, color="red", line=dict(width=2, color="white")),
            name="Noisy",
            hovertemplate=f"Noisy vector<br>X: {data['noisy_point'][0]:.2f}<br>Y: {data['noisy_point'][1]:.2f}<br>Energy: {noisy_energy:.4f}<extra></extra>",
        ))

    if data["denoised_point"]:
        fig.add_trace(go.Scatter(
            x=[data["denoised_point"][0]],
            y=[data["denoised_point"][1]],
            mode="markers",
            marker=dict(size=15, color="blue", line=dict(width=2, color="white")),
            name="Denoised",
            hovertemplate=f"Denoised vector<br>X: {data['denoised_point'][0]:.2f}<br>Y: {data['denoised_point'][1]:.2f}<br>Energy: {denoised_energy:.4f}<extra></extra>",
        ))

    # Траектория
    if show_trajectory and data["trajectory_2d"]:
        traj_x = [p[0] for p in data["trajectory_2d"]]
        traj_y = [p[1] for p in data["trajectory_2d"]]
        traj_z = [_get_energy_at_point(data, x, y) for x, y in zip(traj_x, traj_y)]

        fig.add_trace(go.Scatter(
            x=traj_x,
            y=traj_y,
            mode="lines+markers",
            line=dict(color="yellow", width=4),
            marker=dict(size=8, color="yellow", line=dict(width=1, color="black")),
            name="Trajectory",
            hovertemplate="Trajectory step<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Energy: %{z:.4f}<extra></extra>",
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


def create_surface_plot_matplotlib(
    data: dict,
    title: str = "Energy Landscape",
    save_path: str | None = None,
    show_trajectory: bool = True,
) -> "plt.Figure":
    """
    Создание 3D поверхности энергии через matplotlib.

    Альтернатива Plotly для случаев, когда требуется более точное контроль
    над визуализацией или когда Plotly не справляется с экстремальными значениями.

    Args:
        data: Результат scan_energy_landscape_3d
        title: Заголовок графика
        save_path: Путь для сохранения (опционально)
        show_trajectory: Показывать ли траекторию

    Returns:
        matplotlib Figure с 3D поверхностью
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from matplotlib import cm
    except ImportError:
        raise ImportError("matplotlib is required for this function. pip install matplotlib")

    # Получаем честный диапазон энергий
    energy_min = data.get("energy_min", float(data["energy_grid"].min()))
    energy_max = data.get("energy_max", float(data["energy_grid"].max()))

    # Создаем фигуру с тёмным фоном (как в reference implementation)
    fig = plt.figure(figsize=(14, 10), facecolor="#0a0a0a")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0a0a0a")

    # Создаем сетку
    X, Y = np.meshgrid(data["x_range"], data["y_range"], indexing="ij")
    Z = data["energy_grid"]

    # 3D поверхность
    surf = ax.plot_surface(
        X, Y, Z,
        cmap="inferno",
        alpha=0.95,
        edgecolor="none",
        rcount=100,
        ccount=100,
        vmin=energy_min,
        vmax=energy_max,
    )

    # Вычисляем энергии для ключевых точек
    clean_energy = _get_energy_at_point(data, data["clean_point"][0], data["clean_point"][1]) if data["clean_point"] else 0
    noisy_energy = _get_energy_at_point(data, data["noisy_point"][0], data["noisy_point"][1]) if data["noisy_point"] else 0
    denoised_energy = _get_energy_at_point(data, data["denoised_point"][0], data["denoised_point"][1]) if data["denoised_point"] else 0

    # Clean point (зелёная звезда)
    if data["clean_point"]:
        ax.scatter(
            [data["clean_point"][0]], [data["clean_point"][1]], [clean_energy],
            color="lime", s=200, marker="*", zorder=10,
            label="Clean",
            edgecolors="white",
            linewidths=2,
        )

    # Noisy point (красный X)
    if data["noisy_point"]:
        ax.scatter(
            [data["noisy_point"][0]], [data["noisy_point"][1]], [noisy_energy],
            color="red", s=150, marker="X", zorder=10,
            label="Noisy",
            edgecolors="white",
            linewidths=2,
        )

    # Denoised point (синий круг)
    if data["denoised_point"]:
        ax.scatter(
            [data["denoised_point"][0]], [data["denoised_point"][1]], [denoised_energy],
            color="blue", s=150, marker="o", zorder=10,
            label="Denoised",
            edgecolors="white",
            linewidths=2,
        )

    # Траектория
    if show_trajectory and data["trajectory_2d"]:
        traj_x = [p[0] for p in data["trajectory_2d"]]
        traj_y = [p[1] for p in data["trajectory_2d"]]
        traj_z = [_get_energy_at_point(data, x, y) for x, y in zip(traj_x, traj_y)]

        # Линия траектории
        ax.plot(traj_x, traj_y, traj_z, color="cyan", linewidth=3, alpha=0.9, zorder=9, label="Trajectory")

        # Точки вдоль траектории
        ax.scatter(traj_x, traj_y, traj_z, c=range(len(traj_x)), cmap="viridis",
                   s=50, alpha=0.8, zorder=9, edgecolors="white", linewidths=0.5)

    # Добавляем colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label("Energy", color="white", fontsize=12)
    cbar.ax.yaxis.set_tick_params(color="gray", labelcolor="gray")
    for tick in cbar.ax.get_yticklabels():
        tick.set_color("gray")

    # Заголовок с диапазоном энергий
    full_title = f"{title}\nEnergy Range: [{energy_min:.2f}, {energy_max:.2f}]"
    ax.set_title(full_title, color="white", fontsize=14, pad=20)

    # Подписи осей
    ax.set_xlabel("Direction 1 (noisy → clean)", color="white", fontsize=10, labelpad=10)
    ax.set_ylabel("Direction 2 (perpendicular)", color="white", fontsize=10, labelpad=10)
    ax.set_zlabel("Energy", color="white", fontsize=10, labelpad=10)

    # Настройка цвета осей и тиков
    ax.tick_params(colors="gray", labelsize=9)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("gray")
    ax.yaxis.pane.set_edgecolor("gray")
    ax.zaxis.pane.set_edgecolor("gray")

    # Легенда
    ax.legend(
        loc="upper left",
        facecolor="#1a1a1a",
        edgecolor="gray",
        labelcolor="white",
        fontsize=9,
    )

    # Настройка камеры
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
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
