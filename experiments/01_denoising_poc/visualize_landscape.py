"""
Visualize energy landscape — standalone CLI tool.

Can run in parallel with training: just point it at the latest checkpoint.

Usage:
    # Single checkpoint — 3-panel plot (3D surface + contour + cosine map)
    python experiments/01_denoising_poc/visualize_landscape.py \
        --data data/wikitext_sonar_10k.pt \
        --checkpoint experiments/01_denoising_poc/checkpoints/best.pt

    # Before/after comparison (two checkpoints)
    python experiments/01_denoising_poc/visualize_landscape.py \
        --data data/wikitext_sonar_10k.pt \
        --checkpoint experiments/01_denoising_poc/checkpoints/best.pt \
        --checkpoint_before experiments/01_denoising_poc/checkpoints/epoch_001.pt

    # Custom settings
    python experiments/01_denoising_poc/visualize_landscape.py \
        --data data/wikitext_sonar_10k.pt \
        --checkpoint experiments/01_denoising_poc/checkpoints/best.pt \
        --noise 0.2 --grid 100 --samples 5 --langevin_steps 50

    # Without any checkpoint (random init — useful as "before training" baseline)
    python experiments/01_denoising_poc/visualize_landscape.py \
        --data data/wikitext_sonar_10k.pt --no_checkpoint
"""

import argparse
import sys
from dataclasses import is_dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.base import Stage1Config
from cebcm.models.energy import SimpleEnergy
from cebcm.models.energy_unconditional import UnconditionalEnergy
from cebcm.inference.langevin import run_langevin
from cebcm.data.dataset import SONARVectorDataset
from cebcm.visualization.energy_landscape import (
    scan_energy_landscape,
    plot_landscape,
    plot_comparison,
)


def update_dataclass(target, updates: dict) -> None:
    for key, value in updates.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            update_dataclass(current, value)
        else:
            setattr(target, key, value)


def maybe_load_stage1_config(checkpoint_path: str | None, config: Stage1Config, device: torch.device) -> None:
    if checkpoint_path is None:
        return
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    stage1_cfg = ckpt.get("stage1_config")
    if isinstance(stage1_cfg, dict):
        update_dataclass(config, stage1_cfg)


def add_relative_noise(v: torch.Tensor, scale: float) -> torch.Tensor:
    norms = v.norm(dim=-1, keepdim=True)
    return v + torch.randn_like(v) * scale * norms


def detect_model_type(state_dict: dict) -> str:
    """
    Detect model type from state_dict keys.

    Returns: "simple" for SimpleEnergy (pairwise), "unconditional" for UnconditionalEnergy
    """
    first_key = next(iter(state_dict.keys()), "")

    # SimpleEnergy has pairwise features: net.0.weight shape [H, 4D+8]
    # UnconditionalEnergy has single input: net.0.weight shape [H, D]
    if "net.0.weight" in state_dict:
        input_dim = state_dict["net.0.weight"].shape[1]
        # SimpleEnergy: 4*dim + 8 (sigma_embed) = 4104 for dim=1024
        # UnconditionalEnergy: dim = 1024
        if input_dim > 2048:  # Has pairwise features + sigma embedding
            return "simple"
        else:
            return "unconditional"

    # Fallback: check for _sigma_freqs buffer (SimpleEnergy specific)
    if "_sigma_freqs" in state_dict:
        return "simple"

    return "unconditional"


def load_model(checkpoint_path: str | None, config: Stage1Config, device: torch.device):
    """
    Load model from checkpoint, or create random-init model.
    Auto-detects model type (SimpleEnergy vs UnconditionalEnergy) from checkpoint.

    Returns:
        Tuple of (model, model_type) where model_type is "simple" or "unconditional"
    """
    if checkpoint_path is None:
        model = SimpleEnergy(
            dim=config.energy_dim,
            hidden_dims=config.energy_hidden_dims,
            norm_mode=config.norm_mode,
            activation=config.activation,
            energy_output_clamp=None,
        ).to(device)
        print("Created random-init SimpleEnergy (no checkpoint)")
        return model, "simple"

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model_state = ckpt["model_state"]

    # Auto-detect model type from state_dict
    model_type = detect_model_type(model_state)

    model_config = ckpt.get("config", {})

    if model_type == "simple":
        model = SimpleEnergy(
            dim=model_config.get("energy_dim", config.energy_dim),
            hidden_dims=model_config.get("energy_hidden_dims", config.energy_hidden_dims),
            norm_mode=model_config.get("norm_mode", config.norm_mode),
            activation=model_config.get("activation", config.activation),
            energy_output_clamp=None,
        ).to(device)
    else:  # unconditional
        model = UnconditionalEnergy(
            dim=model_config.get("energy_dim", config.energy_dim),
            hidden_dims=model_config.get("energy_hidden_dims", config.energy_hidden_dims),
            norm_mode=model_config.get("norm_mode", config.norm_mode),
            activation=model_config.get("activation", config.activation),
        ).to(device)

    model.load_state_dict(model_state)
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {epoch}, model_type={model_type})")
    return model, model_type


def run_langevin_with_trajectory(
    model,
    v_query: torch.Tensor,
    v_noisy: torch.Tensor,
    config: Stage1Config,
    model_type: str = "simple",
    max_steps: int = 100,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Run Langevin dynamics and collect the full trajectory for visualization.

    Args:
        model: Energy model (SimpleEnergy or UnconditionalEnergy)
        v_query: Query/target vector (clean)
        v_noisy: Starting noisy vector
        config: Langevin configuration
        model_type: "simple" for pairwise, "unconditional" for scalar energy
        max_steps: Number of Langevin steps
    """
    method = config.langevin.method
    method_kwargs = {}
    if method == "pid":
        method_kwargs = dict(
            kp=config.langevin.pid_kp,
            ki=config.langevin.pid_ki,
            kd=config.langevin.pid_kd,
            integral_decay=config.langevin.pid_integral_decay,
        )
    elif method == "underdamped":
        method_kwargs = dict(
            friction=config.langevin.underdamped_friction,
            mass=config.langevin.underdamped_mass,
        )
    elif method == "overdamped":
        method_kwargs = dict(momentum_beta=config.langevin.momentum_beta)

    # For UnconditionalEnergy, use the model's built-in Langevin with manual trajectory tracking
    if model_type == "unconditional":
        # Manual trajectory tracking for unconditional model
        trajectory = []
        v_trajectory = []
        v_current = v_noisy.clone().detach()
        target_norm = config.langevin.target_norm

        lr = config.langevin.lr
        noise_scale = config.langevin.noise_scale

        for step in range(max_steps):
            energy, grad = model.energy_and_grad(v_current)
            e_mean = energy.mean().item()
            trajectory.append(e_mean)
            v_trajectory.append(v_current.detach().cpu().clone())

            # Langevin step
            langevin_noise = torch.randn_like(v_current) * (2 * lr * noise_scale) ** 0.5
            v_current = v_current - lr * grad + langevin_noise

            # OOD projection
            if target_norm is not None:
                v_current = torch.nn.functional.normalize(v_current, dim=-1) * target_norm

        v_trajectory.append(v_current.detach().cpu().clone())
        return v_current, v_trajectory

    else:
        result = run_langevin(
            method=method,
            energy_fn=model,
            v_query=v_query,
            v_init=v_noisy,
            lr=config.langevin.lr,
            noise_scale=config.langevin.noise_scale,
            max_steps=max_steps,
            target_norm=config.langevin.target_norm,
            plateau_patience=config.langevin.plateau_patience,
            plateau_delta=config.langevin.plateau_delta,
            v_target=v_query,
            track_vectors=True,
            **method_kwargs,
        )
        trajectory = result.v_trajectory
        return result.v_final, trajectory


def main():
    parser = argparse.ArgumentParser(description="Visualize energy landscape")
    parser.add_argument("--data", type=str, required=True, help="Path to .pt dataset")
    parser.add_argument("--checkpoint", type=str, default=None, help="Trained model checkpoint")
    parser.add_argument("--checkpoint_before", type=str, default=None,
                        help="Earlier checkpoint for before/after comparison")
    parser.add_argument("--no_checkpoint", action="store_true",
                        help="Use random-init model (no checkpoint)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--noise", type=float, default=0.15,
                        help="Noise scale for noisy vector (fraction of norm)")
    parser.add_argument("--grid", type=int, default=80, help="Grid resolution per axis")
    parser.add_argument("--samples", type=int, default=3,
                        help="Number of sample vectors to visualize")
    parser.add_argument("--langevin_steps", type=int, default=50,
                        help="Langevin steps for trajectory")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/01_denoising_poc/landscape_plots")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_3d", action="store_true", help="Skip 3D surface plot")
    args = parser.parse_args()

    config = Stage1Config()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    print(f"Device: {device}")

    # Align visualization dynamics with checkpoint train-time config when available
    if args.checkpoint and not args.no_checkpoint:
        maybe_load_stage1_config(args.checkpoint, config, device)

    # Load data
    full_dataset = SONARVectorDataset(args.data)
    n_test = min(config.num_test_sentences, len(full_dataset) // 10)
    test_dataset = full_dataset.subset(len(full_dataset) - n_test, len(full_dataset))
    print(f"Dataset: {len(full_dataset)} total, {len(test_dataset)} test")

    # Load model(s)
    if args.no_checkpoint:
        model, model_type = load_model(None, config, device)
    elif args.checkpoint:
        model, model_type = load_model(args.checkpoint, config, device)
    else:
        parser.error("Provide --checkpoint or --no_checkpoint")

    model.eval()

    model_before = None
    model_before_type = None
    if args.checkpoint_before:
        model_before, model_before_type = load_model(args.checkpoint_before, config, device)
        model_before.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pick random samples
    indices = torch.randperm(len(test_dataset))[:args.samples]

    for sample_idx, idx in enumerate(indices):
        v_clean = test_dataset[idx.item()].unsqueeze(0).to(device)
        v_noisy = add_relative_noise(v_clean, args.noise)

        cos_before = F.cosine_similarity(v_clean, v_noisy, dim=-1).item()
        print(f"\nSample {sample_idx} (idx={idx.item()}): cos(clean, noisy)={cos_before:.4f}")

        # Run Langevin and collect trajectory
        v_denoised, trajectory = run_langevin_with_trajectory(
            model, v_clean, v_noisy, config, model_type=model_type, max_steps=args.langevin_steps,
        )
        cos_after = F.cosine_similarity(v_clean, v_denoised.unsqueeze(0) if v_denoised.dim() == 1 else v_denoised, dim=-1).item()
        print(f"  cos(clean, denoised)={cos_after:.4f} (Δ={cos_after - cos_before:+.4f})")

        # Scan landscape
        print(f"  Scanning {args.grid}x{args.grid} grid...")
        data = scan_energy_landscape(
            energy_fn=model,
            v_clean=v_clean,
            v_noisy=v_noisy,
            grid_size=args.grid,
            v_denoised=v_denoised.unsqueeze(0) if v_denoised.dim() == 1 else v_denoised,
            trajectory=trajectory,
        )

        # Plot
        save_path = output_dir / f"landscape_sample{sample_idx:02d}.png"
        plot_landscape(
            data,
            title=f"Sample {sample_idx}",
            save_path=save_path,
            show_3d=not args.no_3d,
        )

        # Before/after comparison
        if model_before is not None:
            print(f"  Scanning before-training landscape...")
            data_before = scan_energy_landscape(
                energy_fn=model_before,
                v_clean=v_clean,
                v_noisy=v_noisy,
                grid_size=args.grid,
                grid_range=data.grid_range,  # same range for fair comparison
                basis=data.basis,  # same 2D plane for fair comparison
            )
            comp_path = output_dir / f"comparison_sample{sample_idx:02d}.png"
            plot_comparison(data_before, data, save_path=comp_path)
            print(f"  Saved comparison: {comp_path}")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
