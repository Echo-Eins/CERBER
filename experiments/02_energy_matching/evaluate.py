"""
Evaluate a trained Energy Matching model.

Standalone evaluation script — can run on any checkpoint from
experiments/02_energy_matching/train.py.

Evaluates:
1. Denoising quality (gradient following from noisy → clean)
2. Sample generation (ODE + Langevin)
3. Energy landscape statistics
4. Comparison with Stage 1 SimpleEnergy (if checkpoint provided)

Usage:
    # Basic evaluation:
    python experiments/02_energy_matching/evaluate.py \\
        --data data/wikitext_sonar_10k.pt \\
        --checkpoint experiments/02_energy_matching/checkpoints/best.pt

    # With sample generation and SONAR decoding:
    python experiments/02_energy_matching/evaluate.py \\
        --data data/wikitext_sonar_10k.pt \\
        --checkpoint experiments/02_energy_matching/checkpoints/best.pt \\
        --generate 50 --decode

    # Compare with Stage 1 model:
    python experiments/02_energy_matching/evaluate.py \\
        --data data/wikitext_sonar_10k.pt \\
        --checkpoint experiments/02_energy_matching/checkpoints/best.pt \\
        --baseline experiments/01_denoising_poc/checkpoints/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.energy_matching import EnergyMatchingConfig
from cebcm.models.energy_unconditional import UnconditionalEnergy
from cebcm.training.energy_matching import (
    generate_samples_ode,
    generate_samples_langevin,
)
from cebcm.data.dataset import SONARVectorDataset


def load_model(checkpoint_path: str, device: torch.device) -> UnconditionalEnergy:
    """Load UnconditionalEnergy from checkpoint."""
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model_config = ckpt.get("config", {})
    model = UnconditionalEnergy(
        dim=model_config.get("energy_dim", 1024),
        hidden_dims=model_config.get("energy_hidden_dims", [2048, 1024, 512]),
        norm_mode=model_config.get("norm_mode", "orthonorm"),
        activation=model_config.get("activation", "groupsort"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded: {checkpoint_path} (epoch {epoch})")
    return model


@torch.no_grad()
def evaluate_denoising(
    model: UnconditionalEnergy,
    dataset: SONARVectorDataset,
    device: torch.device,
    noise_scales: list[float],
    num_samples: int = 200,
    denoise_steps: int = 50,
    denoise_lr: float = 0.01,
    target_norm: float = 0.2051,
) -> dict:
    """Evaluate denoising via gradient following."""
    model.eval()
    results = {}

    indices = torch.randperm(len(dataset))[:num_samples]
    v_clean = torch.stack([dataset[i.item()] for i in indices]).to(device)

    for noise_scale in noise_scales:
        norms = v_clean.norm(dim=-1, keepdim=True)
        v_noisy = v_clean + torch.randn_like(v_clean) * noise_scale * norms
        cos_before = F.cosine_similarity(v_clean, v_noisy, dim=-1)

        # Gradient following (deterministic)
        v_current = v_noisy.clone()
        for step in range(denoise_steps):
            v_grad = v_current.detach().requires_grad_(True)
            energy = model(v_grad)
            grad = torch.autograd.grad(energy.sum(), v_grad)[0]
            v_current = v_current - denoise_lr * grad
            v_current = F.normalize(v_current, dim=-1) * target_norm

        cos_after = F.cosine_similarity(v_clean, v_current, dim=-1)

        results[f"noise_{noise_scale}"] = {
            "cos_before_mean": cos_before.mean().item(),
            "cos_before_std": cos_before.std().item(),
            "cos_after_mean": cos_after.mean().item(),
            "cos_after_std": cos_after.std().item(),
            "improvement": (cos_after - cos_before).mean().item(),
            "success_rate": (cos_after > cos_before).float().mean().item(),
        }

    return results


@torch.no_grad()
def evaluate_samples(
    model: UnconditionalEnergy,
    dataset: SONARVectorDataset,
    device: torch.device,
    num_generate: int = 500,
    target_norm: float = 0.2051,
    prior_std: float = 0.00641,
) -> dict:
    """Evaluate quality of generated samples."""
    model.eval()

    # Generate via ODE
    samples_ode = generate_samples_ode(
        energy_fn=model,
        num_samples=num_generate,
        dim=1024,
        num_steps=100,
        prior_std=prior_std,
        target_norm=target_norm,
        device=device,
    )

    # Generate via Langevin
    samples_lang = generate_samples_langevin(
        energy_fn=model,
        num_samples=num_generate,
        dim=1024,
        num_steps=200,
        lr=0.01,
        noise_scale=0.005,
        target_norm=target_norm,
        device=device,
    )

    # Get data reference
    n_ref = min(num_generate, len(dataset))
    indices = torch.randperm(len(dataset))[:n_ref]
    v_data = torch.stack([dataset[i.item()] for i in indices]).to(device)

    results = {}
    for name, samples in [("ode", samples_ode), ("langevin", samples_lang)]:
        norms = samples.norm(dim=-1)
        data_norms = v_data.norm(dim=-1)

        # Pairwise cosine (diversity measure)
        s_norm = F.normalize(samples, dim=-1)
        pairwise = (s_norm @ s_norm.T).fill_diagonal_(0)
        n = s_norm.shape[0]

        d_norm = F.normalize(v_data, dim=-1)
        data_pairwise = (d_norm @ d_norm.T).fill_diagonal_(0)

        # Cross-similarity (samples vs data)
        cross = (s_norm @ d_norm.T)

        # Energies
        e_samples = model(samples)
        e_data = model(v_data)

        results[name] = {
            "norm_mean": norms.mean().item(),
            "norm_std": norms.std().item(),
            "data_norm_mean": data_norms.mean().item(),
            "pairwise_cos_mean": pairwise.sum().item() / max(1, n * (n - 1)),
            "data_pairwise_cos_mean": data_pairwise.sum().item() / max(1, n_ref * (n_ref - 1)),
            "cross_cos_mean": cross.mean().item(),
            "cross_cos_max_per_sample": cross.max(dim=1).values.mean().item(),
            "energy_mean": e_samples.mean().item(),
            "energy_std": e_samples.std().item(),
            "data_energy_mean": e_data.mean().item(),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Energy Matching model")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--generate", type=int, default=200, help="Number of samples to generate")
    parser.add_argument("--denoise_samples", type=int, default=200)
    parser.add_argument("--decode", action="store_true", help="Decode generated samples via SONAR")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    full_dataset = SONARVectorDataset(args.data)
    n_test = min(500, len(full_dataset) // 10)
    test_dataset = full_dataset.subset(len(full_dataset) - n_test, len(full_dataset))
    print(f"Dataset: {len(full_dataset)} total, {len(test_dataset)} test")

    # Load model
    model = load_model(args.checkpoint, device)
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    config = EnergyMatchingConfig()
    results = {}

    # 1. Denoising evaluation
    print("\n--- Denoising Evaluation ---")
    denoise_results = evaluate_denoising(
        model, test_dataset, device,
        noise_scales=config.eval_noise_scales,
        num_samples=args.denoise_samples,
        target_norm=config.target_norm,
    )
    results["denoising"] = denoise_results
    for key, val in denoise_results.items():
        print(
            f"  {key}: cos {val['cos_before_mean']:.4f} → {val['cos_after_mean']:.4f} "
            f"(Δ={val['improvement']:+.4f}, success={val['success_rate']:.0%})"
        )

    # 2. Sample generation
    print(f"\n--- Sample Generation ({args.generate} samples) ---")
    sample_results = evaluate_samples(
        model, test_dataset, device,
        num_generate=args.generate,
        target_norm=config.target_norm,
        prior_std=config.prior_std,
    )
    results["samples"] = sample_results
    for method, stats in sample_results.items():
        print(f"  {method}:")
        print(f"    norm={stats['norm_mean']:.4f}±{stats['norm_std']:.4f} (data={stats['data_norm_mean']:.4f})")
        print(f"    pairwise_cos={stats['pairwise_cos_mean']:.4f} (data={stats['data_pairwise_cos_mean']:.4f})")
        print(f"    cross_cos_mean={stats['cross_cos_mean']:.4f}, max_per_sample={stats['cross_cos_max_per_sample']:.4f}")
        print(f"    energy={stats['energy_mean']:.4f}±{stats['energy_std']:.4f} (data={stats['data_energy_mean']:.4f})")

    # 3. Decode samples (optional)
    if args.decode:
        print("\n--- Decoding Generated Samples ---")
        try:
            from cebcm.models.sonar_wrapper import SONARWrapper
            sonar = SONARWrapper(device=str(device))

            samples_ode = generate_samples_ode(
                energy_fn=model, num_samples=10, dim=1024,
                num_steps=100, prior_std=config.prior_std,
                target_norm=config.target_norm, device=device,
            )

            print("\n  ODE-generated samples:")
            for i in range(min(10, samples_ode.shape[0])):
                text = sonar.decode_safe(samples_ode[i:i+1])
                norm = samples_ode[i].norm().item()
                energy = model(samples_ode[i:i+1]).item()
                print(f"    [{i}] (||v||={norm:.4f}, E={energy:.4f}): {text}")

            results["decoded_samples"] = []
            for i in range(min(10, samples_ode.shape[0])):
                text = sonar.decode_safe(samples_ode[i:i+1])
                results["decoded_samples"].append({
                    "index": i,
                    "text": text,
                    "norm": samples_ode[i].norm().item(),
                    "energy": model(samples_ode[i:i+1]).item(),
                })
        except Exception as e:
            print(f"  Failed to decode: {e}")

    # Save results
    output_path = args.output or str(Path(args.checkpoint).parent.parent / "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
