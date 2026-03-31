"""
Shared Stage1 kill-criteria logic.

Purpose:
- Remove weak single-metric PASS decisions.
- Provide strict, explicit multi-gate checks for:
  1) conditional Stage1 (SimpleEnergy / Actor+Critic),
  2) unconditional energy models (Energy Matching track).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _safe_float(x: float | int | None, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _noise_keys(eval_metrics: dict) -> list[str]:
    return sorted(k for k in eval_metrics.keys() if k.startswith("noise_"))


@dataclass
class ConditionalThresholds:
    min_cos_improvement: float = 0.01
    min_cos_success_rate: float = 0.60
    min_geodesic_improvement: float = 0.01
    min_l2_improvement: float = 0.0
    min_energy_success_rate: float = 0.55
    max_clean_min_violation: float = 0.25
    clean_min_violation_key: str = "clean_min_violation_rate"
    min_step_norm: float = 1e-6


@dataclass
class UnconditionalThresholds:
    min_energy_improvement: float = 0.0
    min_energy_success_rate: float = 0.65
    min_step_norm: float = 1e-5
    max_c2st_acc: float = 0.85
    max_mmd_rbf: float = 0.25
    min_prdc_precision: float = 0.01
    min_prdc_coverage: float = 0.01


def summarize_conditional_eval(
    eval_metrics: dict,
    thresholds: ConditionalThresholds | None = None,
) -> dict:
    t = thresholds or ConditionalThresholds()
    keys = _noise_keys(eval_metrics)
    if not keys:
        return {
            "status": "invalid_eval",
            "score": float("-inf"),
            "passed": False,
            "reason": "no noise_* entries",
            "global_gates": {},
            "aggregate": {},
            "per_noise": {},
        }

    per_noise: dict[str, dict] = {}
    cos_improvements = []
    cos_success_rates = []
    geodesic_improvements = []
    l2_improvements = []
    energy_success_rates = []
    clean_min_violations = []
    step_norms = []

    for key in keys:
        m = eval_metrics.get(key, {})
        cos_imp = _safe_float(m.get("improvement"))
        cos_success = _safe_float(m.get("cos_success_rate", m.get("success_rate")))
        geo_imp = _safe_float(m.get("geodesic_improvement"))
        l2_imp = _safe_float(m.get("l2_improvement"))
        energy_success = _safe_float(m.get("energy_success_rate"))
        clean_viol = _safe_float(
            m.get(
                t.clean_min_violation_key,
                m.get("clean_min_violation_rate"),
            )
        )
        step_norm = _safe_float(m.get("step_norm_mean"))

        gates = {
            "cos_improvement": cos_imp >= t.min_cos_improvement,
            "cos_success_rate": cos_success >= t.min_cos_success_rate,
            "geodesic_improvement": geo_imp >= t.min_geodesic_improvement,
            "l2_improvement": l2_imp >= t.min_l2_improvement,
            "energy_success_rate": energy_success >= t.min_energy_success_rate,
            "clean_min_violation": clean_viol <= t.max_clean_min_violation,
            "step_norm": step_norm >= t.min_step_norm,
        }

        per_noise[key] = {
            "metrics": {
                "cos_improvement": cos_imp,
                "cos_success_rate": cos_success,
                "geodesic_improvement": geo_imp,
                "l2_improvement": l2_imp,
                "energy_success_rate": energy_success,
                "clean_min_violation_rate": clean_viol,
                "step_norm_mean": step_norm,
            },
            "gates": gates,
            "passed": all(gates.values()),
        }

        cos_improvements.append(cos_imp)
        cos_success_rates.append(cos_success)
        geodesic_improvements.append(geo_imp)
        l2_improvements.append(l2_imp)
        energy_success_rates.append(energy_success)
        clean_min_violations.append(clean_viol)
        step_norms.append(step_norm)

    n = float(len(keys))
    agg = {
        "mean_cos_improvement": sum(cos_improvements) / n,
        "mean_cos_success_rate": sum(cos_success_rates) / n,
        "mean_geodesic_improvement": sum(geodesic_improvements) / n,
        "mean_l2_improvement": sum(l2_improvements) / n,
        "mean_energy_success_rate": sum(energy_success_rates) / n,
        "mean_clean_min_violation_rate": sum(clean_min_violations) / n,
        "mean_step_norm": sum(step_norms) / n,
        "noise_pass_rate": sum(1.0 for k in per_noise.values() if k["passed"]) / n,
    }

    # Composite score for checkpoint selection (higher is better).
    score = (
        1.00 * agg["mean_cos_improvement"]
        + 0.50 * agg["mean_geodesic_improvement"]
        + 0.25 * agg["mean_l2_improvement"]
        + 0.25 * (agg["mean_cos_success_rate"] - 0.5)
        + 0.10 * (agg["mean_energy_success_rate"] - 0.5)
        - 0.50 * agg["mean_clean_min_violation_rate"]
    )

    global_gates = {
        "all_noise_scales_passed": agg["noise_pass_rate"] >= 1.0,
        "mean_cos_improvement": agg["mean_cos_improvement"] >= t.min_cos_improvement,
        "mean_cos_success_rate": agg["mean_cos_success_rate"] >= t.min_cos_success_rate,
        "mean_geodesic_improvement": agg["mean_geodesic_improvement"] >= t.min_geodesic_improvement,
        "mean_l2_improvement": agg["mean_l2_improvement"] >= t.min_l2_improvement,
        "mean_energy_success_rate": agg["mean_energy_success_rate"] >= t.min_energy_success_rate,
        "mean_clean_min_violation_rate": agg["mean_clean_min_violation_rate"] <= t.max_clean_min_violation,
        "mean_step_norm": agg["mean_step_norm"] >= t.min_step_norm,
    }

    return {
        "status": "evaluated",
        "score": float(score),
        "passed": all(global_gates.values()),
        "global_gates": global_gates,
        "aggregate": agg,
        "per_noise": per_noise,
    }


def summarize_unconditional_eval(
    eval_metrics: dict,
    thresholds: UnconditionalThresholds | None = None,
) -> dict:
    t = thresholds or UnconditionalThresholds()
    keys = _noise_keys(eval_metrics)
    if not keys:
        return {
            "status": "invalid_eval",
            "score": float("-inf"),
            "passed": False,
            "reason": "no noise_* entries",
            "global_gates": {},
            "aggregate": {},
            "per_noise": {},
            "distribution_gates": {},
        }

    per_noise: dict[str, dict] = {}
    energy_improvements = []
    energy_success_rates = []
    step_norms = []
    cos_improvements = []

    for key in keys:
        m = eval_metrics.get(key, {})
        energy_imp = _safe_float(m.get("energy_improvement"))
        energy_success = _safe_float(m.get("energy_success_rate"))
        step_norm = _safe_float(m.get("step_norm_mean"))
        cos_imp = _safe_float(m.get("improvement"))

        gates = {
            "energy_improvement": energy_imp > t.min_energy_improvement,
            "energy_success_rate": energy_success >= t.min_energy_success_rate,
            "step_norm": step_norm >= t.min_step_norm,
        }
        per_noise[key] = {
            "metrics": {
                "energy_improvement": energy_imp,
                "energy_success_rate": energy_success,
                "step_norm_mean": step_norm,
                "cos_improvement": cos_imp,
            },
            "gates": gates,
            "passed": all(gates.values()),
        }

        energy_improvements.append(energy_imp)
        energy_success_rates.append(energy_success)
        step_norms.append(step_norm)
        cos_improvements.append(cos_imp)

    n = float(len(keys))
    agg = {
        "mean_energy_improvement": sum(energy_improvements) / n,
        "mean_energy_success_rate": sum(energy_success_rates) / n,
        "mean_step_norm": sum(step_norms) / n,
        "mean_cos_improvement_diagnostic": sum(cos_improvements) / n,
        "noise_pass_rate": sum(1.0 for k in per_noise.values() if k["passed"]) / n,
    }

    dist = eval_metrics.get("distribution", {})
    c2st = _safe_float(dist.get("c2st_acc"), default=1.0)
    mmd = _safe_float(dist.get("mmd_rbf"), default=float("inf"))
    prdc_precision = _safe_float(dist.get("prdc_precision"), default=0.0)
    prdc_coverage = _safe_float(dist.get("prdc_coverage"), default=0.0)

    dist_gates = {
        "c2st_acc": c2st <= t.max_c2st_acc,
        "mmd_rbf": mmd <= t.max_mmd_rbf,
        "prdc_precision": prdc_precision >= t.min_prdc_precision,
        "prdc_coverage": prdc_coverage >= t.min_prdc_coverage,
    }

    # Composite score for checkpoint selection.
    energy_term = math.tanh(20.0 * agg["mean_energy_improvement"])
    success_term = agg["mean_energy_success_rate"]
    c2st_similarity = max(0.0, 1.0 - 2.0 * abs(c2st - 0.5))
    mmd_score = math.exp(-5.0 * max(0.0, mmd)) if math.isfinite(mmd) else 0.0
    prdc_term = 0.5 * (prdc_precision + prdc_coverage)
    step_term = math.tanh(10.0 * agg["mean_step_norm"])

    score = (
        0.30 * success_term
        + 0.25 * energy_term
        + 0.15 * c2st_similarity
        + 0.15 * mmd_score
        + 0.10 * prdc_term
        + 0.05 * step_term
    )

    global_gates = {
        "all_noise_scales_passed": agg["noise_pass_rate"] >= 1.0,
        "mean_energy_improvement": agg["mean_energy_improvement"] > t.min_energy_improvement,
        "mean_energy_success_rate": agg["mean_energy_success_rate"] >= t.min_energy_success_rate,
        "mean_step_norm": agg["mean_step_norm"] >= t.min_step_norm,
    }

    passed = all(global_gates.values()) and all(dist_gates.values())
    return {
        "status": "evaluated",
        "score": float(score),
        "passed": passed,
        "global_gates": global_gates,
        "distribution_gates": dist_gates,
        "aggregate": agg,
        "distribution": {
            "c2st_acc": c2st,
            "mmd_rbf": mmd,
            "prdc_precision": prdc_precision,
            "prdc_coverage": prdc_coverage,
        },
        "per_noise": per_noise,
    }
