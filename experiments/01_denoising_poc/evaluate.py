"""
Stage 1: detailed denoising evaluation with optional SONAR decoding.

This script now prefers checkpoint-embedded Stage1 config to avoid train/eval drift.
"""

import argparse
import json
import math
import random
import sys
from dataclasses import is_dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.base import Stage1Config
from cebcm.models.actor import LatentDenoiseActor
from cebcm.models.energy import SimpleEnergy
from cebcm.models.sonar_wrapper import SONARWrapper
from cebcm.inference.langevin import run_langevin
from cebcm.data.dataset import SONARVectorDataset
from cebcm.training.kill_criteria import summarize_conditional_eval


def add_relative_noise(v: torch.Tensor, scale: float) -> torch.Tensor:
    norms = v.norm(dim=-1, keepdim=True)
    noise = torch.randn_like(v) * scale * norms
    return v + noise


def cosine_to_geodesic(cos_value: float) -> float:
    return math.acos(max(-1.0, min(1.0, float(cos_value))))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def update_dataclass(target, updates: dict) -> None:
    for key, value in updates.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            update_dataclass(current, value)
        else:
            setattr(target, key, value)


def actor_refine(
    actor: LatentDenoiseActor,
    v_query: torch.Tensor,
    v_init: torch.Tensor,
    config: Stage1Config,
) -> torch.Tensor:
    v_current = v_init
    with torch.no_grad():
        for _ in range(max(1, config.actor_eval_steps)):
            v_current, _ = actor.predict_step(
                v_query=v_query,
                v_current=v_current,
                sigma=None,
                step_size=config.actor_eval_step_size,
                target_norm=config.langevin.target_norm,
                tangent_projection=config.actor_tangent_projection,
            )
    return v_current


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Evaluate denoising quality")
    parser.add_argument("--data", type=str, required=True, help="Path to encoded .pt dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to evaluate")
    parser.add_argument("--decode", action="store_true", help="Decode vectors to text")
    parser.add_argument("--noise_scales", type=float, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint first so we can restore exact train-time config
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)

    config = Stage1Config()
    if "stage1_config" in ckpt and isinstance(ckpt["stage1_config"], dict):
        update_dataclass(config, ckpt["stage1_config"])

    if args.seed is not None:
        config.seed = args.seed
    set_seed(config.seed)

    full_dataset = SONARVectorDataset(args.data)
    n_test = min(config.num_test_sentences, len(full_dataset) // 10)
    test_dataset = full_dataset.subset(len(full_dataset) - n_test, len(full_dataset))
    print(f"Test set: {len(test_dataset)} vectors")

    model_config = ckpt.get("config", {})
    model = SimpleEnergy(
        dim=model_config.get("energy_dim", config.energy_dim),
        hidden_dims=model_config.get("energy_hidden_dims", config.energy_hidden_dims),
        norm_mode=model_config.get("norm_mode", config.norm_mode),
        activation=model_config.get("activation", config.activation),
        ortho_n_iters=config.ortho_n_iters,
        groupsort_size=config.groupsort_size,
        spline_num_knots=config.spline_num_knots,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    actor = None
    if ckpt.get("actor_state") is not None:
        actor = LatentDenoiseActor(
            dim=model_config.get("energy_dim", config.energy_dim),
            hidden_dims=model_config.get("actor_hidden_dims", config.actor_hidden_dims),
            norm_mode=model_config.get("actor_norm_mode", config.actor_norm_mode),
            activation=model_config.get("actor_activation", config.actor_activation),
            ortho_n_iters=config.ortho_n_iters,
            groupsort_size=config.groupsort_size,
            spline_num_knots=config.spline_num_knots,
        ).to(device)
        actor.load_state_dict(ckpt["actor_state"])
        actor.eval()
        print("Loaded actor state (Actor+Critic checkpoint).")

    noise_scales = args.noise_scales or config.eval_noise_scales

    sonar = None
    if args.decode:
        sonar = SONARWrapper(device=args.device)
        print("SONAR decoder loaded for text decoding")

    indices_gen = torch.Generator()
    indices_gen.manual_seed(config.seed + 77)
    indices = torch.randperm(len(test_dataset), generator=indices_gen)[: args.num_samples]

    all_results = {}

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

    for noise_scale in noise_scales:
        print(f"\n{'='*70}")
        print(f"Noise scale: {noise_scale} ({noise_scale*100:.0f}% of norm)")
        print(f"{'='*70}")

        results = []
        cos_before_list = []
        cos_after_list = []
        geo_before_list = []
        geo_after_list = []
        l2_before_list = []
        l2_after_list = []
        energy_before_list = []
        energy_after_list = []
        energy_clean_list = []
        energy_improvement_list = []
        energy_success_flags = []
        clean_min_violation_flags = []
        step_norm_list = []

        for i, idx in enumerate(indices):
            sample_idx = int(idx.item())
            v_orig = test_dataset[sample_idx].unsqueeze(0).to(device)
            v_noisy = add_relative_noise(v_orig, noise_scale)
            v_init = v_noisy
            if actor is not None:
                v_init = actor_refine(actor, v_orig, v_noisy, config)

            cos_before = F.cosine_similarity(v_orig, v_noisy, dim=-1).item()
            if actor is not None and config.critic_eval_langevin_steps <= 0:
                v_final = v_init
                num_steps = 0
                early_stopped = False
            else:
                max_steps = (
                    config.critic_eval_langevin_steps
                    if actor is not None
                    else config.langevin.max_steps
                )
                result = run_langevin(
                    method=method,
                    energy_fn=model,
                    v_query=v_orig,
                    v_init=v_init,
                    lr=config.langevin.lr,
                    noise_scale=config.langevin.noise_scale,
                    max_steps=max_steps,
                    target_norm=config.langevin.target_norm,
                    energy_threshold=config.langevin.energy_threshold,
                    plateau_patience=config.langevin.plateau_patience,
                    plateau_delta=config.langevin.plateau_delta,
                    v_target=v_orig,
                    **method_kwargs,
                )
                v_final = result.v_last
                num_steps = result.num_steps
                early_stopped = result.stopped_early

            cos_after = F.cosine_similarity(v_orig, v_final, dim=-1).item()
            geo_before = cosine_to_geodesic(cos_before)
            geo_after = cosine_to_geodesic(cos_after)
            l2_before = torch.norm(v_orig - v_noisy, dim=-1).item()
            l2_after = torch.norm(v_orig - v_final, dim=-1).item()
            step_norm = torch.norm(v_final - v_noisy, dim=-1).item()

            with torch.no_grad():
                e_clean = model(v_orig, v_orig).item()
                e_before = model(v_orig, v_noisy).item()
                e_after = model(v_orig, v_final).item()
            e_improvement = e_before - e_after
            improved = cos_after > cos_before

            cos_before_list.append(cos_before)
            cos_after_list.append(cos_after)
            geo_before_list.append(geo_before)
            geo_after_list.append(geo_after)
            l2_before_list.append(l2_before)
            l2_after_list.append(l2_after)
            energy_clean_list.append(e_clean)
            energy_before_list.append(e_before)
            energy_after_list.append(e_after)
            energy_improvement_list.append(e_improvement)
            energy_success_flags.append(1.0 if e_after < e_before else 0.0)
            clean_min_violation_flags.append(1.0 if e_after < e_clean else 0.0)
            step_norm_list.append(step_norm)

            sample_result = {
                "idx": sample_idx,
                "cos_before": cos_before,
                "cos_after": cos_after,
                "improvement": cos_after - cos_before,
                "geodesic_before": geo_before,
                "geodesic_after": geo_after,
                "geodesic_improvement": geo_before - geo_after,
                "l2_before": l2_before,
                "l2_after": l2_after,
                "l2_improvement": l2_before - l2_after,
                "energy_clean": e_clean,
                "energy_before": e_before,
                "energy_after": e_after,
                "energy_improvement": e_improvement,
                "energy_success": bool(e_after < e_before),
                "clean_min_violation": bool(e_after < e_clean),
                "step_norm": step_norm,
                "steps": num_steps,
                "early_stopped": early_stopped,
                "original_text": test_dataset.get_text(sample_idx),
            }

            if sonar is not None:
                sample_result["decoded_clean"] = sonar.decode_safe(v_orig.cpu())[0]
                sample_result["decoded_noisy"] = sonar.decode_safe(v_noisy.cpu())[0]
                sample_result["decoded_denoised"] = sonar.decode_safe(v_final.cpu())[0]

            results.append(sample_result)

            marker = "OK" if improved else "NO"
            print(
                f"\n  [{i}] {marker} cos: {cos_before:.4f} -> {cos_after:.4f} "
                f"(d={cos_after - cos_before:+.4f}, {num_steps} steps)"
            )
            print(f"       orig:     {sample_result['original_text'][:100]}")
            if sonar is not None:
                print(f"       clean:    {sample_result['decoded_clean'][:100]}")
                print(f"       noisy:    {sample_result['decoded_noisy'][:100]}")
                print(f"       denoised: {sample_result['decoded_denoised'][:100]}")

        cos_before_mean = sum(cos_before_list) / len(cos_before_list)
        cos_after_mean = sum(cos_after_list) / len(cos_after_list)
        geo_before_mean = sum(geo_before_list) / len(geo_before_list)
        geo_after_mean = sum(geo_after_list) / len(geo_after_list)
        l2_before_mean = sum(l2_before_list) / len(l2_before_list)
        l2_after_mean = sum(l2_after_list) / len(l2_after_list)
        success_rate = sum(
            1 for a, b in zip(cos_after_list, cos_before_list) if a > b
        ) / len(cos_after_list)
        energy_success_rate = sum(energy_success_flags) / len(energy_success_flags)
        clean_min_violation_rate = sum(clean_min_violation_flags) / len(clean_min_violation_flags)

        print(f"\n  Summary (noise={noise_scale}):")
        print(
            f"    cos_sim:  {cos_before_mean:.4f} -> {cos_after_mean:.4f} "
            f"(d={cos_after_mean - cos_before_mean:+.4f})"
        )
        print(
            f"    geodesic: {geo_before_mean:.4f} -> {geo_after_mean:.4f} "
            f"(d={geo_before_mean - geo_after_mean:+.4f})"
        )
        print(
            f"    l2:       {l2_before_mean:.4f} -> {l2_after_mean:.4f} "
            f"(d={l2_before_mean - l2_after_mean:+.4f})"
        )
        print(
            f"    energy:   {sum(energy_before_list)/len(energy_before_list):.4f} -> "
            f"{sum(energy_after_list)/len(energy_after_list):.4f} "
            f"(d={sum(energy_improvement_list)/len(energy_improvement_list):+.4f}, "
            f"success={energy_success_rate:.0%})"
        )
        print(f"    Success rate: {success_rate:.0%}")

        all_results[f"noise_{noise_scale}"] = {
            "cos_before_mean": cos_before_mean,
            "cos_after_mean": cos_after_mean,
            "improvement": cos_after_mean - cos_before_mean,
            "success_rate": success_rate,
            "geodesic_before_mean": geo_before_mean,
            "geodesic_after_mean": geo_after_mean,
            "geodesic_improvement": geo_before_mean - geo_after_mean,
            "l2_before_mean": l2_before_mean,
            "l2_after_mean": l2_after_mean,
            "l2_improvement": l2_before_mean - l2_after_mean,
            "energy_clean_mean": sum(energy_clean_list) / len(energy_clean_list),
            "energy_before_mean": sum(energy_before_list) / len(energy_before_list),
            "energy_after_mean": sum(energy_after_list) / len(energy_after_list),
            "energy_improvement": sum(energy_improvement_list) / len(energy_improvement_list),
            "energy_success_rate": energy_success_rate,
            "clean_min_violation_rate": clean_min_violation_rate,
            "step_norm_mean": sum(step_norm_list) / len(step_norm_list),
            "samples": results,
        }

    print(f"\n{'='*60}")
    print("STRICT KILL CRITERION CHECK (SOTA)")
    print(f"{'='*60}")
    kill_report = summarize_conditional_eval(all_results)
    all_results["kill_criteria"] = kill_report
    for noise_key, block in kill_report["per_noise"].items():
        status = "PASS" if block["passed"] else "FAIL"
        m = block["metrics"]
        print(
            f"  {noise_key}: d_cos={m['cos_improvement']:+.4f}, "
            f"d_geo={m['geodesic_improvement']:+.4f}, "
            f"d_l2={m['l2_improvement']:+.4f}, "
            f"energy_success={m['energy_success_rate']:.0%}, "
            f"clean_violation={m['clean_min_violation_rate']:.0%}, "
            f"step={m['step_norm_mean']:.6f} -> {status}"
        )
    print("  Global gates:")
    for gate_name, gate_ok in kill_report["global_gates"].items():
        print(f"    - {gate_name}: {'PASS' if gate_ok else 'FAIL'}")
    print(
        f"  Composite score: {kill_report['score']:+.6f} "
        f"(noise_pass_rate={kill_report['aggregate'].get('noise_pass_rate', float('nan')):.0%})"
    )

    if kill_report["passed"]:
        print("\n  VERDICT: PASS (strict) - denoising is acceptable for promotion checks")
    else:
        print("\n  VERDICT: FAIL (strict) - denoising is not reliable")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
