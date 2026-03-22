"""
Stage 1: Evaluation — Detailed denoising quality assessment with SONAR decode.

Loads a trained SimpleEnergy checkpoint, runs Langevin denoising on test vectors,
decodes both noisy and denoised versions, and compares them to originals.

Usage:
    python experiments/01_denoising_poc/evaluate.py \
        --data data/wikitext_sonar_10k.pt \
        --checkpoint experiments/01_denoising_poc/checkpoints/best.pt

Spec reference: §13.1, IMPLEMENTATION_PLAN §4.2
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.base import Stage1Config
from cebcm.models.energy import SimpleEnergy
from cebcm.models.sonar_wrapper import SONARWrapper
from cebcm.inference.langevin import langevin_dynamics
from cebcm.data.dataset import SONARVectorDataset


def add_relative_noise(v: torch.Tensor, scale: float) -> torch.Tensor:
    norms = v.norm(dim=-1, keepdim=True)
    noise = torch.randn_like(v) * scale * norms
    return v + noise


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Evaluate denoising quality")
    parser.add_argument("--data", type=str, required=True, help="Path to encoded .pt dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to evaluate")
    parser.add_argument("--decode", action="store_true", help="Decode vectors to text (requires SONAR decoder VRAM)")
    parser.add_argument("--noise_scales", type=float, nargs="+", default=None)
    args = parser.parse_args()

    config = Stage1Config()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load data (use test split — last portion)
    full_dataset = SONARVectorDataset(args.data)
    n_test = min(config.num_test_sentences, len(full_dataset) // 10)
    test_dataset = full_dataset.subset(len(full_dataset) - n_test, len(full_dataset))
    print(f"Test set: {len(test_dataset)} vectors")

    # Load model
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    model_config = ckpt.get("config", {})
    model = SimpleEnergy(
        dim=model_config.get("energy_dim", config.energy_dim),
        hidden_dims=model_config.get("energy_hidden_dims", config.energy_hidden_dims),
        spectral_norm=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    noise_scales = args.noise_scales or config.eval_noise_scales

    # Load SONAR decoder if needed
    sonar = None
    if args.decode:
        sonar = SONARWrapper(device=args.device)
        print("SONAR decoder loaded for text decoding")

    # ============================================================
    # Evaluation
    # ============================================================
    indices = torch.randperm(len(test_dataset))[: args.num_samples]
    all_results = {}

    for noise_scale in noise_scales:
        print(f"\n{'='*70}")
        print(f"Noise scale: {noise_scale} ({noise_scale*100:.0f}% of norm)")
        print(f"{'='*70}")

        results = []
        cos_before_list = []
        cos_after_list = []

        for i, idx in enumerate(indices):
            v_orig = test_dataset[idx.item()].unsqueeze(0).to(device)
            v_noisy = add_relative_noise(v_orig, noise_scale)

            cos_before = F.cosine_similarity(v_orig, v_noisy, dim=-1).item()

            # Langevin denoising
            with torch.no_grad():
                result = langevin_dynamics(
                    energy_fn=model,
                    v_query=v_orig,
                    v_init=v_noisy,
                    lr=config.langevin.lr,
                    noise_scale=config.langevin.noise_scale,
                    max_steps=config.langevin.max_steps,
                    target_norm=config.langevin.target_norm,
                    v_target=v_orig,
                )

            cos_after = F.cosine_similarity(v_orig, result.v_final, dim=-1).item()
            improved = cos_after > cos_before

            cos_before_list.append(cos_before)
            cos_after_list.append(cos_after)

            sample_result = {
                "idx": idx.item(),
                "cos_before": cos_before,
                "cos_after": cos_after,
                "improvement": cos_after - cos_before,
                "steps": result.num_steps,
                "early_stopped": result.stopped_early,
                "original_text": test_dataset.get_text(idx.item()),
            }

            # Decode if requested
            if sonar is not None:
                texts_clean = sonar.decode_safe(v_orig.cpu())
                sample_result["decoded_clean"] = texts_clean[0]
                texts_noisy = sonar.decode_safe(v_noisy.cpu())
                texts_denoised = sonar.decode_safe(result.v_final.cpu())
                sample_result["decoded_noisy"] = texts_noisy[0]
                sample_result["decoded_denoised"] = texts_denoised[0]

            results.append(sample_result)

            # Print sample
            marker = "✓" if improved else "✗"
            print(f"\n  [{i}] {marker} cos: {cos_before:.4f} → {cos_after:.4f} (Δ={cos_after - cos_before:+.4f}, {result.num_steps} steps)")
            print(f"       orig:     {sample_result['original_text'][:100]}")
            if sonar is not None:
                print(f"       clean:    {sample_result['decoded_clean'][:100]}")
                print(f"       noisy:    {sample_result['decoded_noisy'][:100]}")
                print(f"       denoised: {sample_result['decoded_denoised'][:100]}")

        # Summary for this noise scale
        cos_before_mean = sum(cos_before_list) / len(cos_before_list)
        cos_after_mean = sum(cos_after_list) / len(cos_after_list)
        success_rate = sum(1 for a, b in zip(cos_after_list, cos_before_list) if a > b) / len(cos_after_list)

        print(f"\n  Summary (noise={noise_scale}):")
        print(f"    cos_sim:  {cos_before_mean:.4f} → {cos_after_mean:.4f} (Δ={cos_after_mean - cos_before_mean:+.4f})")
        print(f"    Success rate: {success_rate:.0%}")

        all_results[f"noise_{noise_scale}"] = {
            "cos_before_mean": cos_before_mean,
            "cos_after_mean": cos_after_mean,
            "improvement": cos_after_mean - cos_before_mean,
            "success_rate": success_rate,
            "samples": results,
        }

    # ============================================================
    # Save results
    # ============================================================
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evaluation_results.json"

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    # ============================================================
    # Kill criterion
    # ============================================================
    print(f"\n{'='*60}")
    print("KILL CRITERION CHECK (§14.2 criterion #2)")
    print(f"{'='*60}")
    all_pass = True
    for noise_key, data in all_results.items():
        passed = data["improvement"] > 0 and data["success_rate"] > 0.5
        status = "PASS" if passed else "FAIL"
        print(f"  {noise_key}: Δcos={data['improvement']:+.4f}, success={data['success_rate']:.0%} → {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n  VERDICT: PASS — denoising works, proceed to Stage 2")
    else:
        print("\n  VERDICT: FAIL — energy function does not denoise effectively")


if __name__ == "__main__":
    main()
