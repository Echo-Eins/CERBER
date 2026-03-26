"""
3D Energy Landscape Visualizer â€” Ð¸Ð½Ñ‚ÐµÑ€Ð°ÐºÑ‚Ð¸Ð²Ð½Ð°Ñ Ð²Ð¸Ð·ÑƒÐ°Ð»Ð¸Ð·Ð°Ñ†Ð¸Ñ ÑÐ½ÐµÑ€Ð³ÐµÑ‚Ð¸Ñ‡ÐµÑÐºÐ¾Ð³Ð¾ Ð»Ð°Ð½Ð´ÑˆÐ°Ñ„Ñ‚Ð°.

Ð˜ÑÐ¿Ð¾Ð»ÑŒÐ·ÑƒÐµÑ‚ Plotly Ð´Ð»Ñ Ð¸Ð½Ñ‚ÐµÑ€Ð°ÐºÑ‚Ð¸Ð²Ð½Ñ‹Ñ… 3D Ð¿Ð¾Ð²ÐµÑ€Ñ…Ð½Ð¾ÑÑ‚ÐµÐ¹, Ð°Ð½Ð¸Ð¼Ð°Ñ†Ð¸Ð¹ Ð¸ Langevin Ñ‚Ñ€Ð°ÐµÐºÑ‚Ð¾Ñ€Ð¸Ð¹.

Ð¤ÑƒÐ½ÐºÑ†Ð¸Ð¸:
    scan_energy_landscape_3d(...) â€” ÑÐºÐ°Ð½Ð¸Ñ€Ð¾Ð²Ð°Ð½Ð¸Ðµ Ð»Ð°Ð½Ð´ÑˆÐ°Ñ„Ñ‚Ð° Ð´Ð»Ñ 3D Ð²Ð¸Ð·ÑƒÐ°Ð»Ð¸Ð·Ð°Ñ†Ð¸Ð¸
    create_surface_plot(...) â€” ÑÐ¾Ð·Ð´Ð°Ð½Ð¸Ðµ 3D Ð¿Ð¾Ð²ÐµÑ€Ñ…Ð½Ð¾ÑÑ‚Ð¸ ÑÐ½ÐµÑ€Ð³Ð¸Ð¸
    create_contour_plot(...) â€” ÑÐ¾Ð·Ð´Ð°Ð½Ð¸Ðµ 2D ÐºÐ¾Ð½Ñ‚ÑƒÑ€Ð½Ð¾Ð³Ð¾ Ð³Ñ€Ð°Ñ„Ð¸ÐºÐ°
    add_trajectory_to_plot(...) â€” Ð´Ð¾Ð±Ð°Ð²Ð»ÐµÐ½Ð¸Ðµ Langevin Ñ‚Ñ€Ð°ÐµÐºÑ‚Ð¾Ñ€Ð¸Ð¸
    create_comparison_plot(...) â€” ÑÑ€Ð°Ð²Ð½ÐµÐ½Ð¸Ðµ "Ð´Ð¾/Ð¿Ð¾ÑÐ»Ðµ" Ð¾Ð±ÑƒÑ‡ÐµÐ½Ð¸Ñ
    create_animation(...) â€” Ð°Ð½Ð¸Ð¼Ð°Ñ†Ð¸Ñ ÑÐ²Ð¾Ð»ÑŽÑ†Ð¸Ð¸ Ð»Ð°Ð½Ð´ÑˆÐ°Ñ„Ñ‚Ð° Ð¿Ð¾ ÑÐ¿Ð¾Ñ…Ð°Ð¼
"""

import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Literal
import inspect

# ÐŸÐµÑ€ÐµÐ¸ÑÐ¿Ð¾Ð»ÑŒÐ·ÑƒÐµÐ¼ Ð¿Ñ€Ð¾Ð²ÐµÑ€ÐµÐ½Ð½Ñ‹Ð¹ scan_energy_landscape Ð¸Ð· cebcm.visualization
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cebcm.visualization.energy_landscape import scan_energy_landscape as _scan_energy_landscape


@torch.no_grad()
def _evaluate_energy_exact(
    energy_fn,
    model_type: str,
    v_clean: torch.Tensor,
    vectors: torch.Tensor,
) -> torch.Tensor:
    """Compute exact energy values at arbitrary vectors."""
    if vectors.ndim == 1:
        vectors = vectors.unsqueeze(0)

    if model_type == "simple":
        v_query = v_clean.expand(vectors.shape[0], -1)
        return energy_fn(v_query, vectors).detach()
    if model_type == "unconditional":
        return energy_fn(vectors).detach()

    # Fallback for wrappers/adapters.
    sig = inspect.signature(energy_fn.forward)
    if len(sig.parameters) >= 2:
        v_query = v_clean.expand(vectors.shape[0], -1)
        return energy_fn(v_query, vectors).detach()
    return energy_fn(vectors).detach()


def scan_energy_landscape_3d(
    energy_fn,
    v_clean: torch.Tensor,
    v_noisy: torch.Tensor,
    grid_size: int = 50,
    range_factor: float = 1.5,
    absolute_half_range: float | None = None,
    v_denoised: torch.Tensor | None = None,
    trajectory: list[torch.Tensor] | None = None,
    model_type: str = "simple",
    batch_size: int = 100,
    basis: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict:
    """
    Ð¡ÐºÐ°Ð½Ð¸Ñ€Ð¾Ð²Ð°Ð½Ð¸Ðµ ÑÐ½ÐµÑ€Ð³ÐµÑ‚Ð¸Ñ‡ÐµÑÐºÐ¾Ð³Ð¾ Ð»Ð°Ð½Ð´ÑˆÐ°Ñ„Ñ‚Ð° Ð² 2D Ð¿Ð»Ð¾ÑÐºÐ¾ÑÑ‚Ð¸ Ð´Ð»Ñ 3D Ð²Ð¸Ð·ÑƒÐ°Ð»Ð¸Ð·Ð°Ñ†Ð¸Ð¸.

    Ð˜ÑÐ¿Ð¾Ð»ÑŒÐ·ÑƒÐµÑ‚ Ð¿Ñ€Ð¾Ð²ÐµÑ€ÐµÐ½Ð½ÑƒÑŽ Ñ„ÑƒÐ½ÐºÑ†Ð¸ÑŽ Ð¸Ð· cebcm.visualization.energy_landscape.

    Args:
        energy_fn: Ð¤ÑƒÐ½ÐºÑ†Ð¸Ñ ÑÐ½ÐµÑ€Ð³Ð¸Ð¸ (SimpleEnergy Ð¸Ð»Ð¸ UnconditionalEnergy)
        v_clean: [1, D] Ñ‡Ð¸ÑÑ‚Ñ‹Ð¹ Ð²ÐµÐºÑ‚Ð¾Ñ€
        v_noisy: [1, D] Ð·Ð°ÑˆÑƒÐ¼Ð»ÐµÐ½Ð½Ñ‹Ð¹ Ð²ÐµÐºÑ‚Ð¾Ñ€
        grid_size: Ð Ð°Ð·Ð¼ÐµÑ€ ÑÐµÑ‚ÐºÐ¸ (grid_size x grid_size)
        range_factor: ÐœÐ½Ð¾Ð¶Ð¸Ñ‚ÐµÐ»ÑŒ Ð´Ð¸Ð°Ð¿Ð°Ð·Ð¾Ð½Ð° (Ð¾Ñ‚Ð½Ð¾ÑÐ¸Ñ‚ÐµÐ»ÑŒÐ½Ð¾ Ð½Ð¾Ñ€Ð¼Ñ‹ Ð²ÐµÐºÑ‚Ð¾Ñ€Ð°)
        v_denoised: [1, D] Ð´ÐµÐ½ÑƒÐ°Ð·Ð¸Ñ€Ð¾Ð²Ð°Ð½Ð½Ñ‹Ð¹ Ð²ÐµÐºÑ‚Ð¾Ñ€ (Ð¾Ð¿Ñ†Ð¸Ð¾Ð½Ð°Ð»ÑŒÐ½Ð¾)
        trajectory: Ð¡Ð¿Ð¸ÑÐ¾Ðº Ð²ÐµÐºÑ‚Ð¾Ñ€Ð¾Ð² Ñ‚Ñ€Ð°ÐµÐºÑ‚Ð¾Ñ€Ð¸Ð¸ Langevin
        model_type: "simple" Ð¸Ð»Ð¸ "unconditional"
        batch_size: Ð Ð°Ð·Ð¼ÐµÑ€ Ð±Ð°Ñ‚Ñ‡Ð° Ð´Ð»Ñ Ð¿Ð°ÐºÐµÑ‚Ð½Ð¾Ð³Ð¾ Ð²Ñ‹Ñ‡Ð¸ÑÐ»ÐµÐ½Ð¸Ñ ÑÐ½ÐµÑ€Ð³Ð¸Ð¸

    Returns:
        Dict Ñ Ð´Ð°Ð½Ð½Ñ‹Ð¼Ð¸ Ð´Ð»Ñ Ð²Ð¸Ð·ÑƒÐ°Ð»Ð¸Ð·Ð°Ñ†Ð¸Ð¸:
        - x_range, y_range: Ð´Ð¸Ð°Ð¿Ð°Ð·Ð¾Ð½Ñ‹ Ð¾ÑÐµÐ¹
        - energy_grid: [grid_size, grid_size] ÑÐ½ÐµÑ€Ð³Ð¸Ñ Ð² ÐºÐ°Ð¶Ð´Ð¾Ð¹ Ñ‚Ð¾Ñ‡ÐºÐµ
        - basis: Ð±Ð°Ð·Ð¸ÑÐ½Ñ‹Ðµ Ð²ÐµÐºÑ‚Ð¾Ñ€Ñ‹ 2D Ð¿Ð»Ð¾ÑÐºÐ¾ÑÑ‚Ð¸
        - clean_point, noisy_point, denoised_point: ÐºÐ¾Ð¾Ñ€Ð´Ð¸Ð½Ð°Ñ‚Ñ‹ Ð² 2D
        - trajectory_2d: Ñ‚Ñ€Ð°ÐµÐºÑ‚Ð¾Ñ€Ð¸Ñ Ð² 2D ÐºÐ¾Ð¾Ñ€Ð´Ð¸Ð½Ð°Ñ‚Ð°Ñ…
    """
    # Ð˜ÑÐ¿Ð¾Ð»ÑŒÐ·ÑƒÐµÐ¼ Ð¿Ñ€Ð¾Ð²ÐµÑ€ÐµÐ½Ð½ÑƒÑŽ Ñ„ÑƒÐ½ÐºÑ†Ð¸ÑŽ Ð¸Ð· cebcm.visualization
    grid_range = None
    if absolute_half_range is not None and float(absolute_half_range) > 0:
        grid_range = float(absolute_half_range)
    elif range_factor is not None and range_factor > 0:
        with torch.no_grad():
            dist = (v_noisy - v_clean).norm().item()
        grid_range = dist * float(range_factor)

    landscape_data = _scan_energy_landscape(
        energy_fn=energy_fn,
        v_clean=v_clean,
        v_noisy=v_noisy,
        grid_size=grid_size,
        grid_range=grid_range,
        v_denoised=v_denoised,
        trajectory=trajectory,
        basis=basis,
    )

    # ÐšÐ¾Ð½Ð²ÐµÑ€Ñ‚Ð¸Ñ€ÑƒÐµÐ¼ Ð² Ñ„Ð¾Ñ€Ð¼Ð°Ñ‚ Ð´Ð»Ñ Plotly â€” Ð§Ð•Ð¡Ð¢ÐÐ«Ð• Ð·Ð½Ð°Ñ‡ÐµÐ½Ð¸Ñ Ð±ÐµÐ· Ð¼Ð°ÑÑˆÑ‚Ð°Ð±Ð¸Ñ€Ð¾Ð²Ð°Ð½Ð¸Ñ
    energy_np = landscape_data.energy.numpy()
    energy_min = float(energy_np.min())
    energy_max = float(energy_np.max())

    clean_energy = float(_evaluate_energy_exact(energy_fn, model_type, v_clean, v_clean)[0].item())
    noisy_energy = float(_evaluate_energy_exact(energy_fn, model_type, v_clean, v_noisy)[0].item())
    denoised_energy = None
    if v_denoised is not None:
        denoised_energy = float(_evaluate_energy_exact(energy_fn, model_type, v_clean, v_denoised)[0].item())

    trajectory_energy = None
    if trajectory:
        traj_stack = torch.cat([t.reshape(1, -1) for t in trajectory], dim=0).to(v_clean.device)
        trajectory_energy = _evaluate_energy_exact(
            energy_fn, model_type, v_clean, traj_stack
        ).cpu().numpy().tolist()

    if model_type == "unconditional":
        point_labels = {
            "clean": "Reference Data",
            "noisy": "Noisy Start",
            "denoised": "Refined",
        }
        axis1_label = "Direction 1 (reference -> noisy)"
    else:
        point_labels = {
            "clean": "Clean Target",
            "noisy": "Noisy Input",
            "denoised": "Denoised",
        }
        axis1_label = "Direction 1 (clean -> noisy)"

    return {
        "x_range": landscape_data.grid_x.tolist(),
        "y_range": landscape_data.grid_y.tolist(),
        "energy_grid": energy_np,  # Ð§ÐµÑÑ‚Ð½Ñ‹Ðµ Ð·Ð½Ð°Ñ‡ÐµÐ½Ð¸Ñ ÑÐ½ÐµÑ€Ð³Ð¸Ð¸
        "energy_min": energy_min,
        "energy_max": energy_max,
        "basis": landscape_data.basis,
        "clean_point": landscape_data.v_clean_xy,
        "noisy_point": landscape_data.v_noisy_xy,
        "denoised_point": landscape_data.v_denoised_xy,
        "trajectory_2d": landscape_data.trajectory_xy,  # Ð£Ð¶Ðµ ÑÐ¿Ñ€Ð¾ÐµÑ†Ð¸Ñ€Ð¾Ð²Ð°Ð½Ð¾ Ð² _scan_energy_landscape
        "v_clean": v_clean.squeeze(0).cpu().numpy(),
        "v_noisy": v_noisy.squeeze(0).cpu().numpy(),
        "v_denoised": v_denoised.squeeze(0).cpu().numpy() if v_denoised is not None else None,
        "noise_scale": landscape_data.noise_scale,
        "point_energies": {
            "clean": clean_energy,
            "noisy": noisy_energy,
            "denoised": denoised_energy,
        },
        "trajectory_energy": trajectory_energy,
        "point_labels": point_labels,
        "axis1_label": axis1_label,
        "axis2_label": "Direction 2 (orthogonal)",
        "model_type": model_type,
    }


def _get_energy_at_point(
    data: dict,
    x: float,
    y: float,
    *,
    grid: np.ndarray | None = None,
    order: Literal["xy", "yx"] = "xy",
) -> float:
    """
    Interpolate energy value at arbitrary (x, y) coordinates using nearest neighbor.

    Args:
        data: Landscape data with energy_grid, x_range, y_range
        x, y: Coordinates in the 2D plane

    Returns:
        Energy value at the specified point
    """
    energy_grid = np.asarray(grid if grid is not None else data["energy_grid"])
    x_range = np.array(data["x_range"])
    y_range = np.array(data["y_range"])

    # Find nearest grid indices
    ix = np.argmin(np.abs(x_range - x))
    iy = np.argmin(np.abs(y_range - y))

    if order == "xy":
        ix = np.clip(ix, 0, energy_grid.shape[0] - 1)
        iy = np.clip(iy, 0, energy_grid.shape[1] - 1)
        return float(energy_grid[ix, iy])

    # order == "yx": first axis is y, second axis is x
    ix = np.clip(ix, 0, energy_grid.shape[1] - 1)
    iy = np.clip(iy, 0, energy_grid.shape[0] - 1)
    return float(energy_grid[iy, ix])


def create_surface_plot(
    data: dict,
    title: str = "Energy Landscape",
    colorscale: str = "Viridis",
    show_legend: bool = True,
) -> go.Figure:
    """
    Ð¡Ð¾Ð·Ð´Ð°Ð½Ð¸Ðµ 3D Ð¿Ð¾Ð²ÐµÑ€Ñ…Ð½Ð¾ÑÑ‚Ð¸ ÑÐ½ÐµÑ€Ð³Ð¸Ð¸.

    Args:
        data: Ð ÐµÐ·ÑƒÐ»ÑŒÑ‚Ð°Ñ‚ scan_energy_landscape_3d
        title: Ð—Ð°Ð³Ð¾Ð»Ð¾Ð²Ð¾Ðº Ð³Ñ€Ð°Ñ„Ð¸ÐºÐ°
        colorscale: Ð¦Ð²ÐµÑ‚Ð¾Ð²Ð°Ñ ÑÑ…ÐµÐ¼Ð° Plotly
        show_legend: ÐŸÐ¾ÐºÐ°Ð·Ñ‹Ð²Ð°Ñ‚ÑŒ Ð»Ð¸ Ð»ÐµÐ³ÐµÐ½Ð´Ñƒ

    Returns:
        Plotly Figure Ñ 3D Ð¿Ð¾Ð²ÐµÑ€Ñ…Ð½Ð¾ÑÑ‚ÑŒÑŽ
    """
    fig = go.Figure()
    # Plotly expects z[y_idx, x_idx] for 1D x/y arrays.
    z_plot = np.asarray(data["energy_grid"]).T

    # ÐŸÐ¾Ð»ÑƒÑ‡Ð°ÐµÐ¼ Ñ‡ÐµÑÑ‚Ð½Ñ‹Ð¹ Ð´Ð¸Ð°Ð¿Ð°Ð·Ð¾Ð½ ÑÐ½ÐµÑ€Ð³Ð¸Ð¹ â€” Ð‘Ð•Ð— ÐºÐ°ÐºÐ¾Ð³Ð¾-Ð»Ð¸Ð±Ð¾ clamping
    energy_min = data.get("energy_min", float(data["energy_grid"].min()))
    energy_max = data.get("energy_max", float(data["energy_grid"].max()))

    # Ð”Ð¾Ð±Ð°Ð²Ð»ÑÐµÐ¼ Ð´Ð¸Ð°Ð¿Ð°Ð·Ð¾Ð½ ÑÐ½ÐµÑ€Ð³Ð¸Ð¹ Ð² Ð·Ð°Ð³Ð¾Ð»Ð¾Ð²Ð¾Ðº
    full_title = f"{title}<br>Energy Range: [{energy_min:.2f}, {energy_max:.2f}]"

    # 3D Ð¿Ð¾Ð²ÐµÑ€Ñ…Ð½Ð¾ÑÑ‚ÑŒ Ñ Ñ‡ÐµÑÑ‚Ð½Ñ‹Ð¼Ð¸ Ð·Ð½Ð°Ñ‡ÐµÐ½Ð¸ÑÐ¼Ð¸ Ð¸ ÑÐ²Ð½Ñ‹Ð¼ cmin/cmax Ð´Ð»Ñ Ð¿Ñ€Ð°Ð²Ð¸Ð»ÑŒÐ½Ð¾Ð³Ð¾ Ð¼Ð°ÑÑˆÑ‚Ð°Ð±Ð¸Ñ€Ð¾Ð²Ð°Ð½Ð¸Ñ
    fig.add_trace(go.Surface(
        x=data["x_range"],
        y=data["y_range"],
        z=z_plot,
        uid="surface",
        colorscale=colorscale,
        opacity=0.9,
        # Ð¯Ð²Ð½Ð¾ ÑƒÑÑ‚Ð°Ð½Ð°Ð²Ð»Ð¸Ð²Ð°ÐµÐ¼ Ð´Ð¸Ð°Ð¿Ð°Ð·Ð¾Ð½ Ð´Ð»Ñ ÐºÐ¾Ñ€Ñ€ÐµÐºÑ‚Ð½Ð¾Ð³Ð¾ Ð¾Ñ‚Ð¾Ð±Ñ€Ð°Ð¶ÐµÐ½Ð¸Ñ ÑÐºÑÑ‚Ñ€ÐµÐ¼Ð°Ð»ÑŒÐ½Ñ‹Ñ… Ð·Ð½Ð°Ñ‡ÐµÐ½Ð¸Ð¹
        cmin=energy_min,
        cmax=energy_max,
        colorbar=dict(
            title="Energy",
            thickness=20,
            tickformat=".2f",
        ),
        hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Energy: %{z:.4f}<extra></extra>",
    ))

    # Snap marker energies to the rendered mesh so markers visually lie on surface.
    clean_energy = (
        _get_energy_at_point(data, data["clean_point"][0], data["clean_point"][1], grid=z_plot, order="yx")
        if data["clean_point"] else 0.0
    )
    noisy_energy = (
        _get_energy_at_point(data, data["noisy_point"][0], data["noisy_point"][1], grid=z_plot, order="yx")
        if data["noisy_point"] else 0.0
    )
    denoised_energy = (
        _get_energy_at_point(data, data["denoised_point"][0], data["denoised_point"][1], grid=z_plot, order="yx")
        if data["denoised_point"] else 0.0
    )
    labels = data.get("point_labels", {})
    clean_label = labels.get("clean", "Clean")
    noisy_label = labels.get("noisy", "Noisy")
    denoised_label = labels.get("denoised", "Denoised")

    # Ð¢Ð¾Ñ‡ÐºÐ¸ clean/noisy/denoised â€” Ð£Ð’Ð•Ð›Ð˜Ð§Ð•ÐÐÐ«Ð™ Ñ€Ð°Ð·Ð¼ÐµÑ€, Ñ€Ð°Ð·Ð¼ÐµÑ‰ÐµÐ½Ñ‹ Ð½Ð° Ð¿Ð¾Ð²ÐµÑ€Ñ…Ð½Ð¾ÑÑ‚Ð¸ (Ð½Ðµ Ð½Ð° z=0)
    if data.get("clean_point") is not None:
        fig.add_trace(go.Scatter3d(
            x=[data["clean_point"][0]],
            y=[data["clean_point"][1]],
            z=[clean_energy],
            uid="clean_point",  # Ð Ð°Ð·Ð¼ÐµÑ‰Ð°ÐµÐ¼ Ð½Ð° Ð¿Ð¾Ð²ÐµÑ€Ñ…Ð½Ð¾ÑÑ‚Ð¸ ÑÐ½ÐµÑ€Ð³Ð¸Ð¸, Ð° Ð½Ðµ Ð½Ð° z=0
            mode="markers",
            marker=dict(
                size=15,  # Ð£Ð²ÐµÐ»Ð¸Ñ‡ÐµÐ½Ð¾ Ñ 8 Ð´Ð¾ 15 Ð´Ð»Ñ Ð»ÑƒÑ‡ÑˆÐµÐ¹ Ð²Ð¸Ð´Ð¸Ð¼Ð¾ÑÑ‚Ð¸
                color="green",
                symbol="circle",
                opacity=1.0,
            ),
            name=clean_label,
            hovertemplate=f"{clean_label}<br>X: {data['clean_point'][0]:.2f}<br>Y: {data['clean_point'][1]:.2f}<br>Energy: {clean_energy:.4f}<extra></extra>",
        ))

    if data.get("noisy_point") is not None:
        fig.add_trace(go.Scatter3d(
            x=[data["noisy_point"][0]],
            y=[data["noisy_point"][1]],
            z=[noisy_energy],
            uid="noisy_point",
            mode="markers",
            marker=dict(
                size=15,
                color="red",
                symbol="circle",
                opacity=1.0,
            ),
            name=noisy_label,
            hovertemplate=f"{noisy_label}<br>X: {data['noisy_point'][0]:.2f}<br>Y: {data['noisy_point'][1]:.2f}<br>Energy: {noisy_energy:.4f}<extra></extra>",
        ))

    if data.get("denoised_point") is not None:
        fig.add_trace(go.Scatter3d(
            x=[data["denoised_point"][0]],
            y=[data["denoised_point"][1]],
            z=[denoised_energy],
            uid="denoised_point",
            mode="markers",
            marker=dict(
                size=15,
                color="blue",
                symbol="circle",
                opacity=1.0,
            ),
            name=denoised_label,
            hovertemplate=f"{denoised_label}<br>X: {data['denoised_point'][0]:.2f}<br>Y: {data['denoised_point'][1]:.2f}<br>Energy: {denoised_energy:.4f}<extra></extra>",
        ))

    # Ð¢Ñ€Ð°ÐµÐºÑ‚Ð¾Ñ€Ð¸Ñ Langevin â€” Ð½Ð° Ð¿Ð¾Ð²ÐµÑ€Ñ…Ð½Ð¾ÑÑ‚Ð¸ ÑÐ½ÐµÑ€Ð³Ð¸Ð¸
    if data.get("trajectory_2d"):
        traj_x = [p[0] for p in data["trajectory_2d"]]
        traj_y = [p[1] for p in data["trajectory_2d"]]
        # Keep trajectory exactly on plotted mesh.
        traj_z = [_get_energy_at_point(data, x, y, grid=z_plot, order="yx") for x, y in zip(traj_x, traj_y)]

        fig.add_trace(go.Scatter3d(
            x=traj_x,
            y=traj_y,
            z=traj_z,
            uid="langevin_traj",
            mode="lines+markers",
            line=dict(color="yellow", width=6),  # Ð£Ð²ÐµÐ»Ð¸Ñ‡ÐµÐ½Ð° ÑˆÐ¸Ñ€Ð¸Ð½Ð° Ñ 4 Ð´Ð¾ 6
            marker=dict(
                size=8,  # Ð£Ð²ÐµÐ»Ð¸Ñ‡ÐµÐ½Ð¾ Ñ 3 Ð´Ð¾ 8
                color="yellow",
                symbol="circle",
            ),
            name="Langevin Trajectory",
            hovertemplate="Trajectory step<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Energy: %{z:.4f}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=full_title, font=dict(size=20)),
        scene=dict(
            xaxis_title=data.get("axis1_label", "Direction 1 (clean -> noisy)"),
            yaxis_title=data.get("axis2_label", "Direction 2 (orthogonal)"),
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
    Ð¡Ð¾Ð·Ð´Ð°Ð½Ð¸Ðµ 2D ÐºÐ¾Ð½Ñ‚ÑƒÑ€Ð½Ð¾Ð³Ð¾ Ð³Ñ€Ð°Ñ„Ð¸ÐºÐ° ÑÐ½ÐµÑ€Ð³Ð¸Ð¸.

    Args:
        data: Ð ÐµÐ·ÑƒÐ»ÑŒÑ‚Ð°Ñ‚ scan_energy_landscape_3d
        title: Ð—Ð°Ð³Ð¾Ð»Ð¾Ð²Ð¾Ðº Ð³Ñ€Ð°Ñ„Ð¸ÐºÐ°
        colorscale: Ð¦Ð²ÐµÑ‚Ð¾Ð²Ð°Ñ ÑÑ…ÐµÐ¼Ð°
        show_trajectory: ÐŸÐ¾ÐºÐ°Ð·Ñ‹Ð²Ð°Ñ‚ÑŒ Ð»Ð¸ Ñ‚Ñ€Ð°ÐµÐºÑ‚Ð¾Ñ€Ð¸ÑŽ

    Returns:
        Plotly Figure Ñ ÐºÐ¾Ð½Ñ‚ÑƒÑ€Ð½Ñ‹Ð¼ Ð³Ñ€Ð°Ñ„Ð¸ÐºÐ¾Ð¼
    """
    fig = go.Figure()
    # Plotly expects z[y_idx, x_idx] for 1D x/y arrays.
    z_plot = np.asarray(data["energy_grid"]).T

    # ÐŸÐ¾Ð»ÑƒÑ‡Ð°ÐµÐ¼ Ñ‡ÐµÑÑ‚Ð½Ñ‹Ð¹ Ð´Ð¸Ð°Ð¿Ð°Ð·Ð¾Ð½ ÑÐ½ÐµÑ€Ð³Ð¸Ð¹
    energy_min = data.get("energy_min", float(data["energy_grid"].min()))
    energy_max = data.get("energy_max", float(data["energy_grid"].max()))

    # ÐšÐ¾Ð½Ñ‚ÑƒÑ€Ñ‹ Ñ ÑÐ²Ð½Ñ‹Ð¼ Ð´Ð¸Ð°Ð¿Ð°Ð·Ð¾Ð½Ð¾Ð¼
    fig.add_trace(go.Contour(
        z=z_plot,
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

    clean_energy = (
        _get_energy_at_point(data, data["clean_point"][0], data["clean_point"][1], grid=z_plot, order="yx")
        if data["clean_point"] else 0.0
    )
    noisy_energy = (
        _get_energy_at_point(data, data["noisy_point"][0], data["noisy_point"][1], grid=z_plot, order="yx")
        if data["noisy_point"] else 0.0
    )
    denoised_energy = (
        _get_energy_at_point(data, data["denoised_point"][0], data["denoised_point"][1], grid=z_plot, order="yx")
        if data["denoised_point"] else 0.0
    )
    labels = data.get("point_labels", {})
    clean_label = labels.get("clean", "Clean")
    noisy_label = labels.get("noisy", "Noisy")
    denoised_label = labels.get("denoised", "Denoised")

    # Ð¢Ð¾Ñ‡ÐºÐ¸ Ñ ÑƒÐ²ÐµÐ»Ð¸Ñ‡ÐµÐ½Ð½Ñ‹Ð¼ Ñ€Ð°Ð·Ð¼ÐµÑ€Ð¾Ð¼ Ð¸ hover-Ð¸Ð½Ñ„Ð¾Ñ€Ð¼Ð°Ñ†Ð¸ÐµÐ¹
    if data.get("clean_point") is not None:
        fig.add_trace(go.Scatter(
            x=[data["clean_point"][0]],
            y=[data["clean_point"][1]],
            mode="markers",
            marker=dict(size=15, color="green"),
            name=clean_label,
            hovertemplate=f"{clean_label}<br>X: {data['clean_point'][0]:.2f}<br>Y: {data['clean_point'][1]:.2f}<br>Energy: {clean_energy:.4f}<extra></extra>",
        ))

    if data.get("noisy_point") is not None:
        fig.add_trace(go.Scatter(
            x=[data["noisy_point"][0]],
            y=[data["noisy_point"][1]],
            mode="markers",
            marker=dict(size=15, color="red"),
            name=noisy_label,
            hovertemplate=f"{noisy_label}<br>X: {data['noisy_point'][0]:.2f}<br>Y: {data['noisy_point'][1]:.2f}<br>Energy: {noisy_energy:.4f}<extra></extra>",
        ))

    if data.get("denoised_point") is not None:
        fig.add_trace(go.Scatter(
            x=[data["denoised_point"][0]],
            y=[data["denoised_point"][1]],
            mode="markers",
            marker=dict(size=15, color="blue"),
            name=denoised_label,
            hovertemplate=f"{denoised_label}<br>X: {data['denoised_point'][0]:.2f}<br>Y: {data['denoised_point'][1]:.2f}<br>Energy: {denoised_energy:.4f}<extra></extra>",
        ))

    # Ð¢Ñ€Ð°ÐµÐºÑ‚Ð¾Ñ€Ð¸Ñ
    if show_trajectory and data.get("trajectory_2d"):
        traj_x = [p[0] for p in data["trajectory_2d"]]
        traj_y = [p[1] for p in data["trajectory_2d"]]
        traj_z = [_get_energy_at_point(data, x, y, grid=z_plot, order="yx") for x, y in zip(traj_x, traj_y)]
        traj_energy = np.asarray(traj_z, dtype=np.float32).reshape(-1, 1)

        fig.add_trace(go.Scatter(
            x=traj_x,
            y=traj_y,
            customdata=traj_energy,
            mode="lines+markers",
            line=dict(color="yellow", width=4),
            marker=dict(size=8, color="yellow"),
            name="Trajectory",
            hovertemplate=(
                "Trajectory step<br>X: %{x:.2f}<br>Y: %{y:.2f}"
                "<br>Energy: %{customdata[0]:.4f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title=data.get("axis1_label", "Direction 1 (clean -> noisy)"),
        yaxis_title=data.get("axis2_label", "Direction 2 (orthogonal)"),
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
    Ð¡Ñ€Ð°Ð²Ð½ÐµÐ½Ð¸Ðµ Ð»Ð°Ð½Ð´ÑˆÐ°Ñ„Ñ‚Ð¾Ð² "Ð´Ð¾" Ð¸ "Ð¿Ð¾ÑÐ»Ðµ" Ð¾Ð±ÑƒÑ‡ÐµÐ½Ð¸Ñ.

    Args:
        data_before: Ð”Ð°Ð½Ð½Ñ‹Ðµ Ð»Ð°Ð½Ð´ÑˆÐ°Ñ„Ñ‚Ð° Ð´Ð¾ Ð¾Ð±ÑƒÑ‡ÐµÐ½Ð¸Ñ
        data_after: Ð”Ð°Ð½Ð½Ñ‹Ðµ Ð»Ð°Ð½Ð´ÑˆÐ°Ñ„Ñ‚Ð° Ð¿Ð¾ÑÐ»Ðµ Ð¾Ð±ÑƒÑ‡ÐµÐ½Ð¸Ñ
        title: Ð—Ð°Ð³Ð¾Ð»Ð¾Ð²Ð¾Ðº

    Returns:
        Plotly Figure Ñ Ð´Ð²ÑƒÐ¼Ñ Ð¿Ð¾Ð´Ð³Ñ€Ð°Ñ„Ð¸ÐºÐ°Ð¼Ð¸
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
            z=np.asarray(data_before["energy_grid"]).T,
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
            z=np.asarray(data_after["energy_grid"]).T,
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
    Ð¡Ð¾Ð·Ð´Ð°Ð½Ð¸Ðµ Ð°Ð½Ð¸Ð¼Ð°Ñ†Ð¸Ð¸ ÑÐ²Ð¾Ð»ÑŽÑ†Ð¸Ð¸ Ð»Ð°Ð½Ð´ÑˆÐ°Ñ„Ñ‚Ð° Ð¿Ð¾ ÑÐ¿Ð¾Ñ…Ð°Ð¼.

    Args:
        checkpoint_data_list: Ð¡Ð¿Ð¸ÑÐ¾Ðº Ð´Ð°Ð½Ð½Ñ‹Ñ… Ð»Ð°Ð½Ð´ÑˆÐ°Ñ„Ñ‚Ð° Ð´Ð»Ñ ÐºÐ°Ð¶Ð´Ð¾Ð¹ ÑÐ¿Ð¾Ñ…Ð¸
        epoch_labels: Ð¡Ð¿Ð¸ÑÐ¾Ðº Ð¼ÐµÑ‚Ð¾Ðº ÑÐ¿Ð¾Ñ… (Ð´Ð»Ñ ÑÐ»Ð°Ð¹Ð´ÐµÑ€Ð°)
        title: Ð—Ð°Ð³Ð¾Ð»Ð¾Ð²Ð¾Ðº Ð°Ð½Ð¸Ð¼Ð°Ñ†Ð¸Ð¸

    Returns:
        Plotly Figure Ñ Ð°Ð½Ð¸Ð¼Ð°Ñ†Ð¸ÐµÐ¹ Ð¸ ÑÐ»Ð°Ð¹Ð´ÐµÑ€Ð¾Ð¼
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
    Ð¡Ð¾Ð·Ð´Ð°Ð½Ð¸Ðµ 3D Ð¿Ð¾Ð²ÐµÑ€Ñ…Ð½Ð¾ÑÑ‚Ð¸ ÑÐ½ÐµÑ€Ð³Ð¸Ð¸ Ñ‡ÐµÑ€ÐµÐ· matplotlib.

    ÐÐ»ÑŒÑ‚ÐµÑ€Ð½Ð°Ñ‚Ð¸Ð²Ð° Plotly Ð´Ð»Ñ ÑÐ»ÑƒÑ‡Ð°ÐµÐ², ÐºÐ¾Ð³Ð´Ð° Ñ‚Ñ€ÐµÐ±ÑƒÐµÑ‚ÑÑ Ð±Ð¾Ð»ÐµÐµ Ñ‚Ð¾Ñ‡Ð½Ð¾Ðµ ÐºÐ¾Ð½Ñ‚Ñ€Ð¾Ð»ÑŒ
    Ð½Ð°Ð´ Ð²Ð¸Ð·ÑƒÐ°Ð»Ð¸Ð·Ð°Ñ†Ð¸ÐµÐ¹ Ð¸Ð»Ð¸ ÐºÐ¾Ð³Ð´Ð° Plotly Ð½Ðµ ÑÐ¿Ñ€Ð°Ð²Ð»ÑÐµÑ‚ÑÑ Ñ ÑÐºÑÑ‚Ñ€ÐµÐ¼Ð°Ð»ÑŒÐ½Ñ‹Ð¼Ð¸ Ð·Ð½Ð°Ñ‡ÐµÐ½Ð¸ÑÐ¼Ð¸.

    Args:
        data: Ð ÐµÐ·ÑƒÐ»ÑŒÑ‚Ð°Ñ‚ scan_energy_landscape_3d
        title: Ð—Ð°Ð³Ð¾Ð»Ð¾Ð²Ð¾Ðº Ð³Ñ€Ð°Ñ„Ð¸ÐºÐ°
        save_path: ÐŸÑƒÑ‚ÑŒ Ð´Ð»Ñ ÑÐ¾Ñ…Ñ€Ð°Ð½ÐµÐ½Ð¸Ñ (Ð¾Ð¿Ñ†Ð¸Ð¾Ð½Ð°Ð»ÑŒÐ½Ð¾)
        show_trajectory: ÐŸÐ¾ÐºÐ°Ð·Ñ‹Ð²Ð°Ñ‚ÑŒ Ð»Ð¸ Ñ‚Ñ€Ð°ÐµÐºÑ‚Ð¾Ñ€Ð¸ÑŽ

    Returns:
        matplotlib Figure Ñ 3D Ð¿Ð¾Ð²ÐµÑ€Ñ…Ð½Ð¾ÑÑ‚ÑŒÑŽ
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from matplotlib import cm
    except ImportError:
        raise ImportError("matplotlib is required for this function. pip install matplotlib")

    # ÐŸÐ¾Ð»ÑƒÑ‡Ð°ÐµÐ¼ Ñ‡ÐµÑÑ‚Ð½Ñ‹Ð¹ Ð´Ð¸Ð°Ð¿Ð°Ð·Ð¾Ð½ ÑÐ½ÐµÑ€Ð³Ð¸Ð¹
    energy_min = data.get("energy_min", float(data["energy_grid"].min()))
    energy_max = data.get("energy_max", float(data["energy_grid"].max()))

    # Ð¡Ð¾Ð·Ð´Ð°ÐµÐ¼ Ñ„Ð¸Ð³ÑƒÑ€Ñƒ Ñ Ñ‚Ñ‘Ð¼Ð½Ñ‹Ð¼ Ñ„Ð¾Ð½Ð¾Ð¼ (ÐºÐ°Ðº Ð² reference implementation)
    fig = plt.figure(figsize=(14, 10), facecolor="#0a0a0a")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0a0a0a")

    # Ð¡Ð¾Ð·Ð´Ð°ÐµÐ¼ ÑÐµÑ‚ÐºÑƒ
    X, Y = np.meshgrid(data["x_range"], data["y_range"], indexing="ij")
    Z = data["energy_grid"]

    # 3D Ð¿Ð¾Ð²ÐµÑ€Ñ…Ð½Ð¾ÑÑ‚ÑŒ
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

    # Keep markers attached to rendered mesh (same interpolation as surface).
    clean_energy = _get_energy_at_point(data, data["clean_point"][0], data["clean_point"][1]) if data["clean_point"] else 0.0
    noisy_energy = _get_energy_at_point(data, data["noisy_point"][0], data["noisy_point"][1]) if data["noisy_point"] else 0.0
    denoised_energy = _get_energy_at_point(data, data["denoised_point"][0], data["denoised_point"][1]) if data["denoised_point"] else 0.0

    # Clean point (Ð·ÐµÐ»Ñ‘Ð½Ð°Ñ Ð·Ð²ÐµÐ·Ð´Ð°)
    if data.get("clean_point") is not None:
        ax.scatter(
            [data["clean_point"][0]], [data["clean_point"][1]], [clean_energy],
            color="lime", s=200, marker="*", zorder=10,
            label="Clean",
            edgecolors="white",
            linewidths=2,
        )

    # Noisy point (ÐºÑ€Ð°ÑÐ½Ñ‹Ð¹ X)
    if data.get("noisy_point") is not None:
        ax.scatter(
            [data["noisy_point"][0]], [data["noisy_point"][1]], [noisy_energy],
            color="red", s=150, marker="X", zorder=10,
            label="Noisy",
            edgecolors="white",
            linewidths=2,
        )

    # Denoised point (ÑÐ¸Ð½Ð¸Ð¹ ÐºÑ€ÑƒÐ³)
    if data.get("denoised_point") is not None:
        ax.scatter(
            [data["denoised_point"][0]], [data["denoised_point"][1]], [denoised_energy],
            color="blue", s=150, marker="o", zorder=10,
            label="Denoised",
            edgecolors="white",
            linewidths=2,
        )

    # Ð¢Ñ€Ð°ÐµÐºÑ‚Ð¾Ñ€Ð¸Ñ
    if show_trajectory and data.get("trajectory_2d"):
        traj_x = [p[0] for p in data["trajectory_2d"]]
        traj_y = [p[1] for p in data["trajectory_2d"]]
        traj_z = [_get_energy_at_point(data, x, y) for x, y in zip(traj_x, traj_y)]

        # Ð›Ð¸Ð½Ð¸Ñ Ñ‚Ñ€Ð°ÐµÐºÑ‚Ð¾Ñ€Ð¸Ð¸
        ax.plot(traj_x, traj_y, traj_z, color="cyan", linewidth=3, alpha=0.9, zorder=9, label="Trajectory")

        # Ð¢Ð¾Ñ‡ÐºÐ¸ Ð²Ð´Ð¾Ð»ÑŒ Ñ‚Ñ€Ð°ÐµÐºÑ‚Ð¾Ñ€Ð¸Ð¸
        ax.scatter(traj_x, traj_y, traj_z, c=range(len(traj_x)), cmap="viridis",
                   s=50, alpha=0.8, zorder=9, edgecolors="white", linewidths=0.5)

    # Ð”Ð¾Ð±Ð°Ð²Ð»ÑÐµÐ¼ colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label("Energy", color="white", fontsize=12)
    cbar.ax.yaxis.set_tick_params(color="gray", labelcolor="gray")
    for tick in cbar.ax.get_yticklabels():
        tick.set_color("gray")

    # Ð—Ð°Ð³Ð¾Ð»Ð¾Ð²Ð¾Ðº Ñ Ð´Ð¸Ð°Ð¿Ð°Ð·Ð¾Ð½Ð¾Ð¼ ÑÐ½ÐµÑ€Ð³Ð¸Ð¹
    full_title = f"{title}\nEnergy Range: [{energy_min:.2f}, {energy_max:.2f}]"
    ax.set_title(full_title, color="white", fontsize=14, pad=20)

    # ÐŸÐ¾Ð´Ð¿Ð¸ÑÐ¸ Ð¾ÑÐµÐ¹
    ax.set_xlabel(data.get("axis1_label", "Direction 1 (clean -> noisy)"), color="white", fontsize=10, labelpad=10)
    ax.set_ylabel("Direction 2 (perpendicular)", color="white", fontsize=10, labelpad=10)
    ax.set_zlabel("Energy", color="white", fontsize=10, labelpad=10)

    # ÐÐ°ÑÑ‚Ñ€Ð¾Ð¹ÐºÐ° Ñ†Ð²ÐµÑ‚Ð° Ð¾ÑÐµÐ¹ Ð¸ Ñ‚Ð¸ÐºÐ¾Ð²
    ax.tick_params(colors="gray", labelsize=9)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("gray")
    ax.yaxis.pane.set_edgecolor("gray")
    ax.zaxis.pane.set_edgecolor("gray")

    # Ð›ÐµÐ³ÐµÐ½Ð´Ð°
    ax.legend(
        loc="upper left",
        facecolor="#1a1a1a",
        edgecolor="gray",
        labelcolor="white",
        fontsize=9,
    )

    # ÐÐ°ÑÑ‚Ñ€Ð¾Ð¹ÐºÐ° ÐºÐ°Ð¼ÐµÑ€Ñ‹
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
    Ð­ÐºÑÐ¿Ð¾Ñ€Ñ‚ Ð¸Ð½Ñ‚ÐµÑ€Ð°ÐºÑ‚Ð¸Ð²Ð½Ð¾Ð¹ Ñ„Ð¸Ð³ÑƒÑ€Ñ‹ Ð² HTML Ñ„Ð°Ð¹Ð».

    Args:
        fig: Plotly Figure
        output_path: ÐŸÑƒÑ‚ÑŒ Ðº Ð²Ñ‹Ñ…Ð¾Ð´Ð½Ð¾Ð¼Ñƒ HTML Ñ„Ð°Ð¹Ð»Ñƒ
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
    Ð­ÐºÑÐ¿Ð¾Ñ€Ñ‚ Ñ„Ð¸Ð³ÑƒÑ€Ñ‹ Ð² ÑÑ‚Ð°Ñ‚Ð¸Ñ‡ÐµÑÐºÐ¾Ðµ Ð¸Ð·Ð¾Ð±Ñ€Ð°Ð¶ÐµÐ½Ð¸Ðµ.

    Args:
        fig: Plotly Figure
        output_path: ÐŸÑƒÑ‚ÑŒ Ðº Ð²Ñ‹Ñ…Ð¾Ð´Ð½Ð¾Ð¼Ñƒ Ñ„Ð°Ð¹Ð»Ñƒ
        format: Ð¤Ð¾Ñ€Ð¼Ð°Ñ‚ Ð¸Ð·Ð¾Ð±Ñ€Ð°Ð¶ÐµÐ½Ð¸Ñ
        width: Ð¨Ð¸Ñ€Ð¸Ð½Ð° Ð² Ð¿Ð¸ÐºÑÐµÐ»ÑÑ…
        height: Ð’Ñ‹ÑÐ¾Ñ‚Ð° Ð² Ð¿Ð¸ÐºÑÐµÐ»ÑÑ…
        scale: ÐœÐ½Ð¾Ð¶Ð¸Ñ‚ÐµÐ»ÑŒ Ð¼Ð°ÑÑˆÑ‚Ð°Ð±Ð° (Ð´Ð»Ñ retina)
    """
    import plotly.io as pio

    pio.write_image(fig, file=output_path, format=format, width=width, height=height, scale=scale)

