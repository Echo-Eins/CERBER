"""
Energy landscape visualization — 2D slices of the 1024d energy surface.

Produces 3D surface plots, contour maps, and Langevin trajectory overlays
to visualize how the energy function shapes the landscape around data points.

Two visualization modes:
    1. Energy surface:  E(V_query, V_candidate) as V_candidate moves in a 2D plane
    2. Cosine map:      cos_sim(V_clean, V_point) at the same grid points

The 2D plane is defined by two orthogonal directions in 1024d space:
    - axis1: direction from V_clean to V_noisy (the denoising direction)
    - axis2: a random direction orthogonal to axis1

This lets you see:
    - Whether the energy gradient points toward V_clean (correct behavior)
    - How smooth/rough the landscape is
    - Where Langevin dynamics actually walks

Spec reference: §5.6 (energy landscape smoothness), §10.4 (oscillation)
"""

import torch
import torch.nn.functional as F
from torch import Tensor
import math
from pathlib import Path
from dataclasses import dataclass

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


@dataclass
class LandscapeData:
    """Raw data from a landscape scan — can be plotted or saved."""
    grid_x: Tensor          # [G] axis1 coordinates
    grid_y: Tensor          # [G] axis2 coordinates
    energy: Tensor           # [G, G] energy values
    cosine_sim: Tensor       # [G, G] cosine similarity to v_clean
    v_clean_xy: tuple[float, float]   # (x, y) of clean vector in grid coords
    v_noisy_xy: tuple[float, float]   # (x, y) of noisy vector
    v_denoised_xy: tuple[float, float] | None  # (x, y) of denoised result
    trajectory_xy: list[tuple[float, float]] | None  # Langevin path
    noise_scale: float
    grid_range: float
    basis: tuple[Tensor, Tensor] | None = None  # (axis1, axis2) for reuse


def _make_orthogonal_basis(
    v_clean: Tensor,
    v_noisy: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Build two orthonormal directions in 1024d space.

    axis1: normalized direction from v_clean to v_noisy
    axis2: random direction orthogonalized against axis1

    Returns:
        (axis1 [D], axis2 [D]) — both unit vectors
    """
    # axis1: denoising direction
    diff = v_noisy - v_clean  # [D]
    axis1 = F.normalize(diff, dim=-1)

    # axis2: deterministic orthogonal direction (stable across repeated scans)
    # Pick canonical basis vector least aligned with axis1, then Gram-Schmidt.
    idx_order = torch.argsort(axis1.abs())
    axis2 = None
    for idx in idx_order.tolist():
        candidate = torch.zeros_like(axis1)
        candidate[idx] = 1.0
        candidate = candidate - (candidate @ axis1) * axis1
        cand_norm = candidate.norm()
        if cand_norm > 1e-8:
            axis2 = candidate / cand_norm
            break

    if axis2 is None:
        # Numerical fallback (rare): build a cyclicly shifted vector.
        candidate = torch.roll(axis1, shifts=1, dims=0)
        candidate = candidate - (candidate @ axis1) * axis1
        axis2 = F.normalize(candidate, dim=-1)

    return axis1, axis2


def _project_to_2d(
    v: Tensor,
    v_center: Tensor,
    axis1: Tensor,
    axis2: Tensor,
) -> tuple[float, float]:
    """Project a 1024d vector onto the 2D plane defined by (axis1, axis2) centered at v_center."""
    delta = v - v_center  # [D]
    x = (delta @ axis1).item()
    y = (delta @ axis2).item()
    return (x, y)


@torch.no_grad()
def scan_energy_landscape(
    energy_fn: torch.nn.Module,
    v_clean: Tensor,
    v_noisy: Tensor,
    grid_size: int = 80,
    grid_range: float | None = None,
    v_denoised: Tensor | None = None,
    trajectory: list[Tensor] | None = None,
    basis: tuple[Tensor, Tensor] | None = None,
) -> LandscapeData:
    """
    Scan energy values on a 2D grid slice through 1024d space.

    The grid is centered at v_clean, with axis1 pointing toward v_noisy.

    Args:
        energy_fn:   E(v_query, v_candidate) → scalar
        v_clean:     [1, D] clean embedding (query / anchor)
        v_noisy:     [1, D] noisy embedding
        grid_size:   Number of points per axis (total = grid_size²)
        grid_range:  Half-width of grid in each direction.
                     If None, auto-set to 1.5× distance(clean, noisy).
        v_denoised:  [1, D] optional denoised result
        trajectory:  List of [1, D] tensors from Langevin steps
        basis:       Optional (axis1, axis2) tuple to reuse the same 2D plane
                     across multiple scans (e.g. before/after comparison).

    Returns:
        LandscapeData with energy/cosine grids and point coordinates.
    """
    device = v_clean.device
    v_clean_flat = v_clean.squeeze(0)  # [D]
    v_noisy_flat = v_noisy.squeeze(0)  # [D]

    if basis is not None:
        axis1, axis2 = basis
    else:
        axis1, axis2 = _make_orthogonal_basis(v_clean_flat, v_noisy_flat)

    # Auto grid range: 1.5× the clean-noisy distance
    if grid_range is None:
        dist = (v_noisy_flat - v_clean_flat).norm().item()
        grid_range = dist * 1.5

    # Ensure scanned plane covers key points (noisy/denoised/trajectory), so markers
    # always lie on the visible surface domain.
    required_radius = 0.0
    noisy_xy = _project_to_2d(v_noisy_flat, v_clean_flat, axis1, axis2)
    required_radius = max(required_radius, abs(noisy_xy[0]), abs(noisy_xy[1]))

    if v_denoised is not None:
        den_xy = _project_to_2d(v_denoised.squeeze(0).to(device), v_clean_flat, axis1, axis2)
        required_radius = max(required_radius, abs(den_xy[0]), abs(den_xy[1]))

    if trajectory is not None:
        for t in trajectory:
            t_xy = _project_to_2d(t.squeeze(0).to(device), v_clean_flat, axis1, axis2)
            required_radius = max(required_radius, abs(t_xy[0]), abs(t_xy[1]))

    if required_radius > 0:
        grid_range = max(float(grid_range), float(required_radius) * 1.1)

    # Create grid coordinates
    coords = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    energy_grid = torch.zeros(grid_size, grid_size)
    cosine_grid = torch.zeros(grid_size, grid_size)

    # Scan grid in batches for efficiency
    batch_points = []
    for i, x in enumerate(coords):
        for j, y in enumerate(coords):
            point = v_clean_flat + x * axis1 + y * axis2  # [D]
            batch_points.append(point)

    batch_points = torch.stack(batch_points)  # [G*G, D]

    # Compute energy in batches
    batch_size = 256
    all_energies = []
    all_cosines = []

    for start in range(0, len(batch_points), batch_size):
        end = min(start + batch_size, len(batch_points))
        batch = batch_points[start:end]  # [B, D]

        # Support both pairwise (SimpleEnergy) and unconditional (UnconditionalEnergy) models
        # SimpleEnergy: energy_fn(v_query, v_candidate) -> [B]
        # UnconditionalEnergy: energy_fn(v_candidate) -> [B]
        import inspect
        _callable = getattr(energy_fn, 'forward', None) or energy_fn.__call__
        sig = inspect.signature(_callable)
        # Exclude 'self' from parameter count
        params = [p for p in sig.parameters.values() if p.name != 'self']
        if len(params) >= 2:
            # Pairwise model (SimpleEnergy)
            v_q = v_clean.expand(batch.shape[0], -1)
            e = energy_fn(v_q, batch)  # [B]
        else:
            # Unconditional model (UnconditionalEnergy)
            e = energy_fn(batch)  # [B]

        cos = F.cosine_similarity(v_clean_flat.unsqueeze(0), batch, dim=-1)  # [B]

        all_energies.append(e.cpu())
        all_cosines.append(cos.cpu())

    all_energies = torch.cat(all_energies)  # [G*G]
    all_cosines = torch.cat(all_cosines)    # [G*G]

    energy_grid = all_energies.reshape(grid_size, grid_size)
    cosine_grid = all_cosines.reshape(grid_size, grid_size)

    # Project key points to 2D
    clean_xy = (0.0, 0.0)  # center by construction
    noisy_xy = _project_to_2d(v_noisy_flat, v_clean_flat, axis1, axis2)

    denoised_xy = None
    if v_denoised is not None:
        denoised_xy = _project_to_2d(v_denoised.squeeze(0).to(device), v_clean_flat, axis1, axis2)

    trajectory_xy = None
    if trajectory is not None:
        trajectory_xy = [
            _project_to_2d(t.squeeze(0).to(device), v_clean_flat, axis1, axis2)
            for t in trajectory
        ]

    noise_scale = (v_noisy_flat - v_clean_flat).norm().item() / v_clean_flat.norm().item()

    return LandscapeData(
        grid_x=coords.cpu(),
        grid_y=coords.cpu(),
        energy=energy_grid,
        cosine_sim=cosine_grid,
        v_clean_xy=clean_xy,
        v_noisy_xy=noisy_xy,
        v_denoised_xy=denoised_xy,
        trajectory_xy=trajectory_xy,
        noise_scale=noise_scale,
        grid_range=grid_range,
        basis=(axis1, axis2),
    )


def plot_landscape(
    data: LandscapeData,
    title: str = "",
    save_path: str | Path | None = None,
    show_3d: bool = True,
) -> None:
    """
    Plot energy landscape: 3D surface + contour + cosine map.

    Args:
        data: LandscapeData from scan_energy_landscape.
        title: Plot title prefix.
        save_path: If set, save figure to this path.
        show_3d: Include 3D surface plot (left panel).
    """
    if not HAS_MPL:
        raise ImportError("matplotlib is required for plotting. pip install matplotlib")

    X, Y = torch.meshgrid(data.grid_x, data.grid_y, indexing="ij")
    X_np = X.numpy()
    Y_np = Y.numpy()
    E_np = data.energy.numpy()
    C_np = data.cosine_sim.numpy()

    n_cols = 3 if show_3d else 2
    fig = plt.figure(figsize=(7 * n_cols, 6), facecolor="black")
    fig.patch.set_facecolor("#0a0a0a")

    col_idx = 0

    # ── Panel 1: 3D surface ──
    if show_3d:
        col_idx += 1
        ax3d = fig.add_subplot(1, n_cols, col_idx, projection="3d", facecolor="#0a0a0a")
        ax3d.plot_surface(
            X_np, Y_np, E_np,
            cmap="inferno", alpha=0.85,
            edgecolor="none", rcount=80, ccount=80,
        )
        # Overlay trajectory on surface
        if data.trajectory_xy:
            tx = [p[0] for p in data.trajectory_xy]
            ty = [p[1] for p in data.trajectory_xy]
            # Interpolate energy at trajectory points
            te = _interp_grid(data, tx, ty)
            ax3d.plot(tx, ty, te, color="cyan", linewidth=1.5, alpha=0.9, zorder=10)
            ax3d.scatter([tx[0]], [ty[0]], [te[0]], color="magenta", s=40, zorder=11)
            ax3d.scatter([tx[-1]], [ty[-1]], [te[-1]], color="lime", s=40, zorder=11)

        # Mark clean point
        e_clean = _interp_grid(data, [data.v_clean_xy[0]], [data.v_clean_xy[1]])
        ax3d.scatter(
            [data.v_clean_xy[0]], [data.v_clean_xy[1]], e_clean,
            color="white", s=60, marker="*", zorder=12,
        )

        ax3d.set_xlabel("axis₁ (→ noisy)", color="white", fontsize=8)
        ax3d.set_ylabel("axis₂ (⊥)", color="white", fontsize=8)
        ax3d.set_zlabel("E(V_q, V)", color="white", fontsize=8)
        ax3d.set_title("Energy Surface", color="white", fontsize=10, pad=10)
        ax3d.tick_params(colors="gray", labelsize=7)
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False

    # ── Panel 2: Energy contour + trajectory ──
    col_idx += 1
    ax_cont = fig.add_subplot(1, n_cols, col_idx, facecolor="#0a0a0a")

    contour = ax_cont.contourf(X_np, Y_np, E_np, levels=40, cmap="inferno")
    ax_cont.contour(X_np, Y_np, E_np, levels=20, colors="white", alpha=0.15, linewidths=0.3)
    plt.colorbar(contour, ax=ax_cont, label="Energy", shrink=0.8)

    # Key points
    ax_cont.scatter(*data.v_clean_xy, color="white", s=80, marker="*", zorder=10, label="clean")
    ax_cont.scatter(*data.v_noisy_xy, color="magenta", s=60, marker="x", zorder=10, label="noisy", linewidths=2)
    if data.v_denoised_xy:
        ax_cont.scatter(*data.v_denoised_xy, color="lime", s=60, marker="o", zorder=10, label="denoised")

    # Trajectory
    if data.trajectory_xy:
        tx = [p[0] for p in data.trajectory_xy]
        ty = [p[1] for p in data.trajectory_xy]
        ax_cont.plot(tx, ty, color="cyan", linewidth=1.0, alpha=0.7, zorder=9)
        # Arrows every N steps
        step = max(1, len(tx) // 15)
        for k in range(0, len(tx) - 1, step):
            dx = tx[k + 1] - tx[k]
            dy = ty[k + 1] - ty[k]
            ax_cont.annotate(
                "", xy=(tx[k + 1], ty[k + 1]), xytext=(tx[k], ty[k]),
                arrowprops=dict(arrowstyle="->", color="cyan", lw=0.8),
            )

    ax_cont.set_xlabel("axis₁ (→ noisy)", color="white", fontsize=9)
    ax_cont.set_ylabel("axis₂ (⊥)", color="white", fontsize=9)
    ax_cont.set_title("Energy Contour + Trajectory", color="white", fontsize=10)
    ax_cont.legend(fontsize=7, loc="upper right", facecolor="#1a1a1a", edgecolor="gray", labelcolor="white")
    ax_cont.tick_params(colors="gray")

    # ── Panel 3: Cosine similarity map ──
    col_idx += 1
    ax_cos = fig.add_subplot(1, n_cols, col_idx, facecolor="#0a0a0a")

    cos_contour = ax_cos.contourf(X_np, Y_np, C_np, levels=40, cmap="viridis")
    ax_cos.contour(X_np, Y_np, C_np, levels=20, colors="white", alpha=0.15, linewidths=0.3)
    plt.colorbar(cos_contour, ax=ax_cos, label="cos(V_clean, V)", shrink=0.8)

    ax_cos.scatter(*data.v_clean_xy, color="white", s=80, marker="*", zorder=10)
    ax_cos.scatter(*data.v_noisy_xy, color="magenta", s=60, marker="x", zorder=10, linewidths=2)
    if data.v_denoised_xy:
        ax_cos.scatter(*data.v_denoised_xy, color="lime", s=60, marker="o", zorder=10)

    if data.trajectory_xy:
        tx = [p[0] for p in data.trajectory_xy]
        ty = [p[1] for p in data.trajectory_xy]
        ax_cos.plot(tx, ty, color="cyan", linewidth=1.0, alpha=0.7, zorder=9)

    ax_cos.set_xlabel("axis₁ (→ noisy)", color="white", fontsize=9)
    ax_cos.set_ylabel("axis₂ (⊥)", color="white", fontsize=9)
    ax_cos.set_title("Latent Space (cosine to clean)", color="white", fontsize=10)
    ax_cos.tick_params(colors="gray")

    # ── Suptitle ──
    noise_pct = data.noise_scale * 100
    suptitle = f"{title}  |  noise={noise_pct:.0f}%"
    if data.v_denoised_xy:
        suptitle += f"  |  trajectory: {len(data.trajectory_xy or [])} steps"
    fig.suptitle(suptitle, color="white", fontsize=12, y=1.02)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor(), edgecolor="none",
        )
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_comparison(
    data_before: LandscapeData,
    data_after: LandscapeData,
    save_path: str | Path | None = None,
) -> None:
    """
    Side-by-side comparison: before training vs after training.

    Shows energy contours for both, with trajectory overlay on the after panel.
    """
    if not HAS_MPL:
        raise ImportError("matplotlib is required for plotting.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0a0a0a")

    for ax, data, label in zip(axes, [data_before, data_after], ["BEFORE Training", "AFTER Training"]):
        ax.set_facecolor("#0a0a0a")
        X, Y = torch.meshgrid(data.grid_x, data.grid_y, indexing="ij")
        E_np = data.energy.numpy()

        contour = ax.contourf(X.numpy(), Y.numpy(), E_np, levels=40, cmap="inferno")
        ax.contour(X.numpy(), Y.numpy(), E_np, levels=20, colors="white", alpha=0.15, linewidths=0.3)
        plt.colorbar(contour, ax=ax, label="Energy", shrink=0.8)

        ax.scatter(*data.v_clean_xy, color="white", s=80, marker="*", zorder=10)
        ax.scatter(*data.v_noisy_xy, color="magenta", s=60, marker="x", zorder=10, linewidths=2)
        if data.v_denoised_xy:
            ax.scatter(*data.v_denoised_xy, color="lime", s=60, marker="o", zorder=10)

        if data.trajectory_xy:
            tx = [p[0] for p in data.trajectory_xy]
            ty = [p[1] for p in data.trajectory_xy]
            ax.plot(tx, ty, color="cyan", linewidth=1.0, alpha=0.7, zorder=9)

        ax.set_title(label, color="white", fontsize=12)
        ax.set_xlabel("axis₁ (→ noisy)", color="white", fontsize=9)
        ax.set_ylabel("axis₂ (⊥)", color="white", fontsize=9)
        ax.tick_params(colors="gray")

    fig.suptitle(
        f"Energy Landscape  |  noise={data_after.noise_scale * 100:.0f}%",
        color="white", fontsize=13, y=1.02,
    )
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor(), edgecolor="none",
        )
        print(f"Saved: {save_path}")
    plt.close(fig)


def _interp_grid(
    data: LandscapeData,
    xs: list[float],
    ys: list[float],
) -> list[float]:
    """Nearest-neighbor interpolation of energy values at arbitrary (x, y) points."""
    gx = data.grid_x
    gy = data.grid_y
    result = []
    for x, y in zip(xs, ys):
        # Find nearest grid indices
        ix = (gx - x).abs().argmin().item()
        iy = (gy - y).abs().argmin().item()
        ix = max(0, min(ix, data.energy.shape[0] - 1))
        iy = max(0, min(iy, data.energy.shape[1] - 1))
        result.append(data.energy[ix, iy].item())
    return result
