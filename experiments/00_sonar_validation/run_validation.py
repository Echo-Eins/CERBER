"""
Stage 0: SONAR Space Validation

Validates that SONAR's latent space is suitable for gradient-based navigation.

Experiments:
    A. Noise Robustness — decode(V + noise) preserves meaning?
    B. Interpolation — decode(αV₁ + (1-α)V₂) produces meaningful intermediates?
    C. Distribution Analysis — norms, cosine distances, clustering
    D. Gradient Flow — ∂output/∂V exists through decoder?

Usage:
    python experiments/00_sonar_validation/run_validation.py [--device cuda] [--num-sentences 1000]

Kill criteria:
    - 5% relative noise causes cos_sim < 0.85 → SONAR space too fragile
    - Interpolation produces gibberish → space not smooth
    - Norms collapse or cluster → space not expressive enough
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cebcm.models.sonar_wrapper import SONARWrapper


# ─────────────────────────────────────────────
# Experiment A: Noise Robustness
# ─────────────────────────────────────────────

def experiment_a_noise_robustness(
    sonar: SONARWrapper,
    noise_scales: list[float],
    output_dir: Path,
):
    """
    Add RELATIVE Gaussian noise to embeddings and check if decoded text preserves meaning.

    Noise is scaled relative to each embedding's norm:
        V_noisy = V + N(0,1) * noise_scale * ||V||

    This ensures noise_scale=0.05 means "5% of the signal magnitude"
    regardless of the absolute norm of SONAR embeddings (~0.2).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Noise Robustness (RELATIVE noise)")
    print("=" * 70)

    test_sentences = [
        "The cat sat on the mat.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "The weather forecast predicts rain tomorrow afternoon.",
        "Quantum computing may revolutionize cryptography in the future.",
        "She walked to the park and sat under a large oak tree.",
        "The stock market experienced significant volatility this week.",
        "Renewable energy sources are becoming more cost-effective.",
        "The neural network achieved state-of-the-art accuracy on the benchmark.",
        "He opened the book and began reading the first chapter.",
    ]

    print(f"\nEncoding {len(test_sentences)} test sentences...")
    V_orig = sonar.encode(test_sentences)
    norms = V_orig.norm(dim=-1, keepdim=True)  # [N, 1]
    print(f"  Shape: {V_orig.shape}, dtype: {V_orig.dtype}")
    print(f"  Norms: mean={norms.mean().item():.4f}, min={norms.min().item():.4f}, max={norms.max().item():.4f}")

    # First: decode originals to see baseline reconstruction quality
    print(f"\n--- Baseline (no noise) ---")
    decoded_orig = sonar.decode_safe(V_orig)
    for i, (orig, dec) in enumerate(zip(test_sentences, decoded_orig)):
        print(f"  [{i}] orig:    {orig}")
        print(f"       decoded: {dec}")

    results = []

    for noise_scale in noise_scales:
        print(f"\n--- relative noise_scale = {noise_scale} ({noise_scale*100:.0f}% of norm) ---")
        # Relative noise: scale by per-vector norm
        noise = torch.randn_like(V_orig) * noise_scale * norms
        V_noisy = V_orig + noise

        noise_norm = noise.norm(dim=-1)
        signal_norm = V_orig.norm(dim=-1)
        snr = signal_norm / (noise_norm + 1e-8)

        cos_sims = F.cosine_similarity(V_orig, V_noisy, dim=-1)
        decoded_noisy = sonar.decode_safe(V_noisy)

        print(f"  SNR: mean={snr.mean().item():.2f}")
        print(f"  cos_sim: mean={cos_sims.mean().item():.4f}, min={cos_sims.min().item():.4f}")

        scale_results = {
            "noise_scale": noise_scale,
            "noise_type": "relative",
            "cos_sim_mean": cos_sims.mean().item(),
            "cos_sim_min": cos_sims.min().item(),
            "cos_sim_max": cos_sims.max().item(),
            "snr_mean": snr.mean().item(),
            "pairs": [],
        }

        for i, (orig, noisy_text) in enumerate(zip(test_sentences, decoded_noisy)):
            pair = {
                "original": orig,
                "noisy_decoded": noisy_text,
                "cos_sim": cos_sims[i].item(),
            }
            scale_results["pairs"].append(pair)
            print(f"  [{i}] cos_sim={cos_sims[i].item():.4f}")
            print(f"       orig:  {orig}")
            print(f"       noisy: {noisy_text}")

        results.append(scale_results)
        torch.cuda.empty_cache()

    # Save results
    out_file = output_dir / "experiment_a_noise.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")

    return results


# ─────────────────────────────────────────────
# Experiment B: Interpolation
# ─────────────────────────────────────────────

def experiment_b_interpolation(
    sonar: SONARWrapper,
    alphas: list[float],
    output_dir: Path,
):
    """
    Interpolate between pairs of sentence embeddings and decode intermediates.

    Checks if the space is smooth: intermediate points should produce
    meaningful sentences that blend meanings of the endpoints.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Interpolation")
    print("=" * 70)

    sentence_pairs = [
        ("I love programming in Python.", "Machine learning is fascinating."),
        ("The cat sat on the mat.", "The dog ran through the park."),
        ("It is raining heavily outside.", "The sun is shining brightly today."),
        ("She is a brilliant scientist.", "He is an excellent musician."),
        ("The economy is growing rapidly.", "Unemployment rates are falling."),
    ]

    results = []

    for pair_idx, (text_a, text_b) in enumerate(sentence_pairs):
        print(f"\n--- Pair {pair_idx}: ---")
        print(f"  A: {text_a}")
        print(f"  B: {text_b}")

        V_a = sonar.encode([text_a])
        V_b = sonar.encode([text_b])

        cos_ab = F.cosine_similarity(V_a, V_b, dim=-1).item()
        print(f"  cos_sim(A, B) = {cos_ab:.4f}")

        pair_results = {
            "text_a": text_a,
            "text_b": text_b,
            "cos_sim_ab": cos_ab,
            "interpolations": [],
        }

        for alpha in alphas:
            V_interp = (1 - alpha) * V_a + alpha * V_b
            decoded = sonar.decode_safe(V_interp)
            norm = V_interp.norm(dim=-1).item()

            interp = {
                "alpha": alpha,
                "decoded": decoded[0],
                "norm": norm,
            }
            pair_results["interpolations"].append(interp)
            print(f"  alpha={alpha:.1f}: {decoded[0]}  (norm={norm:.2f})")

        results.append(pair_results)

    out_file = output_dir / "experiment_b_interpolation.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")

    return results


# ─────────────────────────────────────────────
# Experiment C: Distribution Analysis
# ─────────────────────────────────────────────

def experiment_c_distribution(
    sonar: SONARWrapper,
    num_sentences: int,
    output_dir: Path,
):
    """
    Analyze the distribution of SONAR embeddings:
    - Norm distribution (mean, std, min, max)
    - Pairwise cosine similarity distribution
    - Per-dimension statistics
    - Check for collapse or clustering

    Extracts target_norm for OOD projection in Langevin dynamics.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT C: Distribution Analysis")
    print("=" * 70)

    # Generate diverse sentences
    print(f"\nGenerating {num_sentences} diverse sentences...")
    sentences = _generate_diverse_sentences(num_sentences)

    print(f"Encoding {len(sentences)} sentences...")
    V = sonar.encode_batched(sentences, batch_size=64)
    print(f"  Shape: {V.shape}")

    # Norm statistics
    norms = V.norm(dim=-1)
    norm_stats = {
        "mean": norms.mean().item(),
        "std": norms.std().item(),
        "min": norms.min().item(),
        "max": norms.max().item(),
        "median": norms.median().item(),
    }
    print(f"\nNorm statistics:")
    for k, v in norm_stats.items():
        print(f"  {k}: {v:.4f}")

    # Pairwise cosine similarity (sample to avoid O(N^2) for large N)
    sample_size = min(500, len(V))
    V_sample = V[:sample_size]
    V_norm = F.normalize(V_sample, dim=-1)
    cos_matrix = V_norm @ V_norm.T

    # Extract upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones_like(cos_matrix, dtype=torch.bool), diagonal=1)
    pairwise_cos = cos_matrix[mask]

    cos_stats = {
        "mean": pairwise_cos.mean().item(),
        "std": pairwise_cos.std().item(),
        "min": pairwise_cos.min().item(),
        "max": pairwise_cos.max().item(),
        "median": pairwise_cos.median().item(),
        "pct_above_0.9": (pairwise_cos > 0.9).float().mean().item(),
        "pct_above_0.95": (pairwise_cos > 0.95).float().mean().item(),
    }
    print(f"\nPairwise cosine similarity ({sample_size} samples):")
    for k, v in cos_stats.items():
        print(f"  {k}: {v:.4f}")

    # Per-dimension statistics
    dim_means = V.mean(dim=0)
    dim_stds = V.std(dim=0)
    dim_stats = {
        "dim_mean_of_means": dim_means.mean().item(),
        "dim_std_of_means": dim_means.std().item(),
        "dim_mean_of_stds": dim_stds.mean().item(),
        "dim_std_of_stds": dim_stds.std().item(),
        "dead_dims_count": (dim_stds < 1e-6).sum().item(),
    }
    print(f"\nPer-dimension statistics:")
    for k, v in dim_stats.items():
        print(f"  {k}: {v:.6f}")

    # target_norm for Langevin OOD projection
    target_norm = norms.mean().item()
    print(f"\n>>> target_norm (for Langevin OOD projection): {target_norm:.4f}")

    results = {
        "num_sentences": len(sentences),
        "embedding_dim": V.shape[1],
        "norm_stats": norm_stats,
        "cosine_similarity_stats": cos_stats,
        "dimension_stats": dim_stats,
        "target_norm": target_norm,
    }

    out_file = output_dir / "experiment_c_distribution.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")

    # Save target_norm separately for easy loading by other stages
    torch.save({"target_norm": target_norm, "norm_stats": norm_stats},
               output_dir / "target_norm.pt")
    print(f"target_norm saved to {output_dir / 'target_norm.pt'}")

    return results


# ─────────────────────────────────────────────
# Experiment D: Gradient Flow
# ─────────────────────────────────────────────

def experiment_d_gradient_flow(
    sonar: SONARWrapper,
    output_dir: Path,
):
    """
    Check if gradients flow through SONAR embeddings.

    This verifies that ∂L/∂V is non-zero — critical for Langevin dynamics.
    Note: we don't need gradients through the SONAR decoder itself
    (EBT has its own gradient path), but it's useful to know.

    We test:
    1. Can we compute gradients of cosine_similarity w.r.t. V?
    2. Does a simple optimization step (V -= lr * grad) improve cosine similarity?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT D: Gradient Flow")
    print("=" * 70)

    test_sentences = [
        "The cat sat on the mat.",
        "Machine learning is transforming industry.",
        "Quantum physics describes nature at the smallest scales.",
    ]

    V_orig = sonar.encode(test_sentences)
    results = []

    for i, text in enumerate(test_sentences):
        print(f"\n--- Sentence {i}: {text} ---")
        V_target = V_orig[i : i + 1].detach()

        # Start from noisy version (relative noise: 10% of norm)
        target_norm = V_target.norm()
        V_start = V_target + torch.randn_like(V_target) * 0.1 * target_norm
        V_current = V_start.clone().requires_grad_(True)

        cos_before = F.cosine_similarity(V_target, V_start, dim=-1).item()
        print(f"  cos_sim before optimization: {cos_before:.4f}")

        # Simple gradient descent: minimize 1 - cos_sim
        trajectory = [cos_before]
        lr = 0.01

        for step in range(100):
            loss = 1 - F.cosine_similarity(V_target, V_current, dim=-1)
            loss.backward()

            grad = V_current.grad
            grad_norm = grad.norm().item()

            with torch.no_grad():
                V_current = V_current - lr * grad
            V_current = V_current.detach().requires_grad_(True)

            cos_now = F.cosine_similarity(V_target, V_current, dim=-1).item()
            trajectory.append(cos_now)

            if step % 10 == 0:
                print(f"  step {step}: cos_sim={cos_now:.4f}, grad_norm={grad_norm:.6f}")

        cos_after = trajectory[-1]
        print(f"  cos_sim after optimization: {cos_after:.4f}")
        print(f"  Improvement: {cos_after - cos_before:.4f}")

        # Decode the optimized vector
        decoded_start = sonar.decode_safe(V_start)
        decoded_optimized = sonar.decode_safe(V_current.detach())
        print(f"  Decoded start:     {decoded_start[0]}")
        print(f"  Decoded optimized: {decoded_optimized[0]}")

        results.append({
            "sentence": text,
            "cos_sim_before": cos_before,
            "cos_sim_after": cos_after,
            "improvement": cos_after - cos_before,
            "decoded_start": decoded_start[0],
            "decoded_optimized": decoded_optimized[0],
            "trajectory": trajectory,
        })

    out_file = output_dir / "experiment_d_gradient.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")

    return results


# ─────────────────────────────────────────────
# VRAM Estimation
# ─────────────────────────────────────────────

def estimate_vram(sonar: SONARWrapper, output_dir: Path):
    """Estimate VRAM usage for SONAR + future EBT."""
    print("\n" + "=" * 70)
    print("VRAM ESTIMATION")
    print("=" * 70)

    # Force load both models
    sonar.encode(["test"])
    sonar.decode_safe(sonar.encode(["test"]))

    vram = sonar.estimate_vram()
    print(f"\nSONAR encoder: {vram.get('encoder', 'N/A'):.1f} MB")
    print(f"SONAR decoder: {vram.get('decoder', 'N/A'):.1f} MB")

    # Estimate EBT (simple calculation)
    ebt_pairwise_params = (4096 * 2048 + 2048 * 1024 + 1024 * 1)  # ~10M
    ebt_pairwise_mb = ebt_pairwise_params * 4 / (1024 ** 2)
    print(f"\nEBT Pairwise (estimate): {ebt_pairwise_mb:.1f} MB")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"\nGPU Memory:")
        print(f"  Allocated: {allocated:.1f} MB")
        print(f"  Reserved:  {reserved:.1f} MB")
        print(f"  Total:     {total:.1f} MB")
        print(f"  Available: {total - reserved:.1f} MB")

        vram["gpu_allocated_mb"] = allocated
        vram["gpu_reserved_mb"] = reserved
        vram["gpu_total_mb"] = total

    out_file = output_dir / "vram_estimation.json"
    with open(out_file, "w") as f:
        json.dump(vram, f, indent=2)
    print(f"\nResults saved to {out_file}")

    return vram


# ─────────────────────────────────────────────
# GO/NO-GO Analysis
# ─────────────────────────────────────────────

def go_nogo_analysis(
    noise_results: list[dict],
    interp_results: list[dict],
    dist_results: dict,
    grad_results: list[dict],
    output_dir: Path,
):
    """
    Analyze all experiment results and produce GO/NO-GO recommendation.
    """
    print("\n" + "=" * 70)
    print("GO / NO-GO ANALYSIS")
    print("=" * 70)

    issues = []
    warnings = []

    # Check A: Noise robustness (relative noise scales)
    # Find the "safe threshold": largest noise scale with cos_sim > 0.90
    safe_threshold = 0.0
    for r in noise_results:
        scale = r["noise_scale"]
        cos_mean = r["cos_sim_mean"]
        if cos_mean > 0.90:
            safe_threshold = max(safe_threshold, scale)

    print(f"  Safe noise threshold (cos_sim > 0.90): {safe_threshold*100:.1f}% of norm")

    if safe_threshold < 0.005:
        issues.append(
            f"KILL: No noise scale preserves cos_sim > 0.90 "
            f"— space too fragile for gradient navigation"
        )
    elif safe_threshold < 0.01:
        warnings.append(
            f"WARNING: Safe noise threshold is only {safe_threshold*100:.1f}% "
            f"— Langevin step size must be very small"
        )

    # Check if there's a steep cliff (common with beam search decoders)
    for i in range(len(noise_results) - 1):
        s1 = noise_results[i]["cos_sim_mean"]
        s2 = noise_results[i + 1]["cos_sim_mean"]
        sc1 = noise_results[i]["noise_scale"]
        sc2 = noise_results[i + 1]["noise_scale"]
        if s1 > 0.85 and s2 < 0.60:
            warnings.append(
                f"WARNING: Steep cliff between {sc1*100:.1f}% (cos={s1:.3f}) "
                f"and {sc2*100:.1f}% (cos={s2:.3f}) — "
                f"Langevin must stay below {sc1*100:.1f}% relative step"
            )

    # Check B: Interpolation smoothness
    for r in interp_results:
        interps = r["interpolations"]
        # Check if any intermediate is empty or garbage
        for interp in interps:
            if not interp["decoded"] or len(interp["decoded"]) < 3:
                warnings.append(
                    f"WARNING: Empty interpolation at alpha={interp['alpha']}"
                )

    # Check C: Distribution health
    if dist_results["cosine_similarity_stats"]["mean"] > 0.9:
        issues.append(
            f"KILL: Mean pairwise cosine similarity = "
            f"{dist_results['cosine_similarity_stats']['mean']:.3f} > 0.9 "
            f"— space collapsed, all vectors nearly identical"
        )
    if dist_results["dimension_stats"]["dead_dims_count"] > 100:
        warnings.append(
            f"WARNING: {dist_results['dimension_stats']['dead_dims_count']} dead "
            f"dimensions (std < 1e-6)"
        )

    # Check D: Gradient flow
    for r in grad_results:
        if r["improvement"] < 0.01:
            warnings.append(
                f"WARNING: Gradient optimization gave only {r['improvement']:.4f} "
                f"improvement for '{r['sentence'][:30]}...'"
            )

    # Verdict
    print()
    if issues:
        verdict = "NO-GO"
        print("🔴 VERDICT: NO-GO")
        for issue in issues:
            print(f"  {issue}")
    elif warnings:
        verdict = "GO (with warnings)"
        print("🟡 VERDICT: GO (with warnings)")
        for w in warnings:
            print(f"  {w}")
    else:
        verdict = "GO"
        print("🟢 VERDICT: GO")

    # Compute recommended Langevin step size
    target_norm = dist_results["target_norm"]
    # Langevin lr should keep perturbation within the safe zone
    # safe_threshold is the max relative noise that preserves cos_sim > 0.90
    # Each Langevin step perturbs by ~lr * grad_norm; keep this < safe_threshold * target_norm
    recommended_lr = safe_threshold * target_norm * 0.5  # 50% safety margin
    print(f"\n  target_norm = {target_norm:.4f}")
    print(f"  embedding_dim = {dist_results['embedding_dim']}")
    print(f"  safe_noise_threshold = {safe_threshold*100:.1f}% of norm")
    print(f"  recommended_langevin_lr ≈ {recommended_lr:.6f} (absolute)")
    print(f"  recommended_langevin_lr ≈ {safe_threshold * 0.5:.4f} (relative to norm)")

    report = {
        "verdict": verdict,
        "issues": issues,
        "warnings": warnings,
        "target_norm": target_norm,
        "safe_noise_threshold": safe_threshold,
        "recommended_langevin_lr_abs": recommended_lr,
        "recommended_langevin_lr_rel": safe_threshold * 0.5,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_file = output_dir / "go_nogo_report.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {out_file}")

    return report


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def _generate_diverse_sentences(num: int) -> list[str]:
    """
    Generate diverse sentences for distribution analysis.
    Uses a mix of hardcoded templates + random combinations.
    """
    templates = [
        # Science
        "Quantum {thing} is a field of {field}.",
        "The {adj} experiment demonstrated {result}.",
        "Researchers at {place} discovered {thing}.",
        # Technology
        "The new {thing} algorithm improves {metric} by {number} percent.",
        "Cloud {thing} enables scalable {field} applications.",
        "{thing} learning models require large amounts of {resource}.",
        # Daily life
        "She walked to the {place} and bought some {thing}.",
        "The {adj} weather made everyone stay {place}.",
        "He enjoys {activity} every {time}.",
        # Abstract
        "The concept of {thing} has evolved over {time}.",
        "Understanding {thing} requires knowledge of {field}.",
        "The relationship between {thing} and {thing2} is {adj}.",
    ]

    things = [
        "computing", "gravity", "evolution", "democracy", "entropy", "software",
        "architecture", "blockchain", "database", "network", "algorithm",
        "intelligence", "optimization", "physics", "chemistry", "biology",
        "mathematics", "economics", "philosophy", "language", "music",
    ]
    fields = [
        "computer science", "physics", "biology", "medicine", "engineering",
        "mathematics", "economics", "sociology", "psychology", "art",
    ]
    adjs = [
        "revolutionary", "classical", "modern", "fundamental", "complex",
        "simple", "elegant", "practical", "theoretical", "innovative",
    ]
    places = [
        "MIT", "Stanford", "CERN", "the university", "the library",
        "the store", "the park", "home", "the office", "the lab",
    ]
    metrics = [
        "accuracy", "efficiency", "throughput", "latency", "reliability",
    ]
    numbers = ["10", "20", "30", "50", "75"]
    resources = ["data", "compute", "memory", "time", "energy"]
    activities = [
        "reading", "running", "coding", "painting", "cooking",
        "hiking", "swimming", "writing", "gardening", "meditating",
    ]
    times = [
        "morning", "evening", "weekend", "summer", "century", "decade",
    ]

    rng = np.random.RandomState(42)
    sentences = []
    for i in range(num):
        template = templates[i % len(templates)]
        sentence = template.format(
            thing=rng.choice(things),
            thing2=rng.choice(things),
            field=rng.choice(fields),
            adj=rng.choice(adjs),
            place=rng.choice(places),
            result=f"a {rng.choice(adjs)} {rng.choice(things)} effect",
            metric=rng.choice(metrics),
            number=rng.choice(numbers),
            resource=rng.choice(resources),
            activity=rng.choice(activities),
            time=rng.choice(times),
        )
        sentences.append(sentence)

    return sentences


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 0: SONAR Validation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--num-sentences", type=int, default=1000,
                        help="Number of sentences for distribution analysis")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/00_sonar_validation",
                        help="Output directory for results")
    parser.add_argument("--skip-vram", action="store_true",
                        help="Skip VRAM estimation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    noise_scales = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5]
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print("=" * 70)
    print("CEBCM Stage 0: SONAR Space Validation")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")

    # Init SONAR
    sonar = SONARWrapper(device=args.device)

    # Run experiments
    t0 = time.time()

    noise_results = experiment_a_noise_robustness(sonar, noise_scales, output_dir)
    interp_results = experiment_b_interpolation(sonar, alphas, output_dir)
    dist_results = experiment_c_distribution(sonar, args.num_sentences, output_dir)
    grad_results = experiment_d_gradient_flow(sonar, output_dir)

    if not args.skip_vram:
        estimate_vram(sonar, output_dir)

    # GO/NO-GO
    report = go_nogo_analysis(
        noise_results, interp_results, dist_results, grad_results, output_dir
    )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"\nAll results saved in {output_dir}/")

    # Exit with non-zero code if NO-GO
    if report["verdict"] == "NO-GO":
        sys.exit(1)


if __name__ == "__main__":
    main()
