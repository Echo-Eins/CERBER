"""
SOTA evaluation metrics for latent-space generative refinement.

Metrics implemented:
- MMD (RBF kernel, median heuristic)
- C2ST (linear probe classifier two-sample test)
- PRDC (precision/recall/density/coverage)
- Manifold kNN proximity improvements (cosine and euclidean)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SOTAEvalConfig:
    prdc_k: int = 5
    manifold_k: int = 10
    c2st_steps: int = 120
    c2st_lr: float = 0.1
    c2st_train_frac: float = 0.8
    mmd_subsample: int = 256


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def _pairwise_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cdist(a, b, p=2)


def _pairwise_cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_n = _normalize(a)
    b_n = _normalize(b)
    sim = a_n @ b_n.T
    return (1.0 - sim).clamp(min=0.0)


def _safe_float(x: torch.Tensor | float) -> float:
    if isinstance(x, float):
        return float(x)
    if x.numel() == 0:
        return float("nan")
    return float(torch.nan_to_num(x.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).item())


def estimate_rbf_sigma(real: torch.Tensor, fake: torch.Tensor, max_points: int = 256) -> float:
    """Median heuristic on a mixed subset."""
    n_real = real.shape[0]
    n_fake = fake.shape[0]
    take_real = min(n_real, max_points // 2 if max_points >= 2 else 1)
    take_fake = min(n_fake, max_points - take_real)

    real_sub = real[:take_real]
    fake_sub = fake[:take_fake]
    mixed = torch.cat([real_sub, fake_sub], dim=0)

    if mixed.shape[0] < 2:
        return 1.0

    dists = _pairwise_l2(mixed, mixed)
    mask = ~torch.eye(dists.shape[0], dtype=torch.bool, device=dists.device)
    vals = dists[mask]
    vals = vals[torch.isfinite(vals)]
    if vals.numel() == 0:
        return 1.0
    sigma = vals.median().clamp(min=1e-4)
    return float(sigma.item())


def compute_mmd_rbf(real: torch.Tensor, fake: torch.Tensor, sigma: float | None = None) -> dict:
    """Biased MMD^2 with RBF kernel."""
    if real.numel() == 0 or fake.numel() == 0:
        return {"mmd_rbf": float("nan"), "sigma": float("nan")}

    if sigma is None:
        sigma = estimate_rbf_sigma(real, fake)
    sigma_sq = max(float(sigma) ** 2, 1e-8)

    d_xx = _pairwise_l2(real, real).pow(2)
    d_yy = _pairwise_l2(fake, fake).pow(2)
    d_xy = _pairwise_l2(real, fake).pow(2)

    k_xx = torch.exp(-d_xx / (2.0 * sigma_sq))
    k_yy = torch.exp(-d_yy / (2.0 * sigma_sq))
    k_xy = torch.exp(-d_xy / (2.0 * sigma_sq))

    mmd2 = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    return {"mmd_rbf": _safe_float(mmd2), "sigma": float(sigma)}


def compute_c2st_linear(
    real: torch.Tensor,
    fake: torch.Tensor,
    *,
    steps: int = 120,
    lr: float = 0.1,
    train_frac: float = 0.8,
) -> dict:
    """
    Classifier two-sample test with a linear probe.

    Output is held-out accuracy. Closer to 0.5 means harder to separate.
    """
    n = min(real.shape[0], fake.shape[0])
    if n < 4:
        return {"c2st_acc": float("nan")}

    x = torch.cat([real[:n], fake[:n]], dim=0)
    y = torch.cat([
        torch.zeros(n, device=x.device),
        torch.ones(n, device=x.device),
    ], dim=0)

    perm = torch.randperm(x.shape[0], device=x.device)
    x = x[perm]
    y = y[perm]

    n_train = max(2, int(train_frac * x.shape[0]))
    n_train = min(n_train, x.shape[0] - 1)
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    probe = torch.nn.Linear(x.shape[1], 1, bias=True).to(x.device)
    opt = torch.optim.SGD(probe.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(max(10, int(steps))):
        logits = probe(x_train).squeeze(-1)
        loss = loss_fn(logits, y_train)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        test_logits = probe(x_test).squeeze(-1)
        pred = (torch.sigmoid(test_logits) > 0.5).float()
        acc = (pred == y_test).float().mean()
    return {"c2st_acc": _safe_float(acc)}


def _kth_radius(dist_self: torch.Tensor, k: int) -> torch.Tensor:
    """
    k-th nearest neighbor radius for each point.
    dist_self is NxN with pairwise distances.
    """
    n = dist_self.shape[0]
    if n <= 1:
        return torch.zeros(n, device=dist_self.device)

    k_eff = max(1, min(int(k), n - 1))
    dist = dist_self.clone()
    eye = torch.eye(n, dtype=torch.bool, device=dist.device)
    dist[eye] = float("inf")
    vals, _ = dist.topk(k=k_eff, largest=False, dim=1)
    return vals[:, -1]


def compute_prdc(real: torch.Tensor, fake: torch.Tensor, k: int = 5) -> dict:
    """
    PRDC metrics using euclidean distances.
    """
    n_real = real.shape[0]
    n_fake = fake.shape[0]
    if n_real < 2 or n_fake < 2:
        return {
            "prdc_precision": float("nan"),
            "prdc_recall": float("nan"),
            "prdc_density": float("nan"),
            "prdc_coverage": float("nan"),
        }

    real_real = _pairwise_l2(real, real)
    fake_fake = _pairwise_l2(fake, fake)
    fake_real = _pairwise_l2(fake, real)  # [N_fake, N_real]
    real_fake = fake_real.T               # [N_real, N_fake]

    real_k = _kth_radius(real_real, k=k)
    fake_k = _kth_radius(fake_fake, k=k)
    k_eff = float(max(1, min(int(k), n_real - 1)))

    precision_mask = fake_real <= real_k.unsqueeze(0)
    precision = precision_mask.any(dim=1).float().mean()
    density = precision_mask.float().sum(dim=1).mean() / k_eff

    recall_mask = real_fake <= fake_k.unsqueeze(0)
    recall = recall_mask.any(dim=1).float().mean()

    coverage = (real_fake.min(dim=1).values <= real_k).float().mean()

    return {
        "prdc_precision": _safe_float(precision),
        "prdc_recall": _safe_float(recall),
        "prdc_density": _safe_float(density),
        "prdc_coverage": _safe_float(coverage),
    }


def compute_manifold_knn_metrics(
    ref_bank: torch.Tensor,
    noisy: torch.Tensor,
    denoised: torch.Tensor,
    k: int = 10,
) -> dict:
    """
    kNN manifold proximity via cosine and euclidean neighborhoods.
    """
    if ref_bank.numel() == 0 or noisy.numel() == 0 or denoised.numel() == 0:
        return {
            "knn_cos_noisy_top1": float("nan"),
            "knn_cos_denoised_top1": float("nan"),
            "knn_cos_improvement": float("nan"),
            "knn_l2_noisy_top1": float("nan"),
            "knn_l2_denoised_top1": float("nan"),
            "knn_l2_improvement": float("nan"),
        }

    cos_noisy = _normalize(noisy) @ _normalize(ref_bank).T
    cos_denoised = _normalize(denoised) @ _normalize(ref_bank).T

    top1_cos_noisy = cos_noisy.max(dim=1).values.mean()
    top1_cos_denoised = cos_denoised.max(dim=1).values.mean()

    k_eff = max(1, min(int(k), ref_bank.shape[0]))
    topk_noisy = cos_noisy.topk(k=k_eff, dim=1).values.mean()
    topk_denoised = cos_denoised.topk(k=k_eff, dim=1).values.mean()

    l2_noisy = _pairwise_l2(noisy, ref_bank).min(dim=1).values.mean()
    l2_denoised = _pairwise_l2(denoised, ref_bank).min(dim=1).values.mean()

    return {
        "knn_cos_noisy_top1": _safe_float(top1_cos_noisy),
        "knn_cos_denoised_top1": _safe_float(top1_cos_denoised),
        "knn_cos_improvement": _safe_float(top1_cos_denoised - top1_cos_noisy),
        "knn_cos_noisy_topk": _safe_float(topk_noisy),
        "knn_cos_denoised_topk": _safe_float(topk_denoised),
        "knn_cos_topk_improvement": _safe_float(topk_denoised - topk_noisy),
        "knn_l2_noisy_top1": _safe_float(l2_noisy),
        "knn_l2_denoised_top1": _safe_float(l2_denoised),
        "knn_l2_improvement": _safe_float(l2_noisy - l2_denoised),
    }


def compute_distribution_suite(
    real: torch.Tensor,
    fake: torch.Tensor,
    cfg: SOTAEvalConfig,
) -> dict:
    """
    Combined distribution metrics: MMD + C2ST + PRDC.
    """
    out: dict[str, float] = {}
    out.update(compute_mmd_rbf(real, fake))
    out.update(
        compute_c2st_linear(
            real,
            fake,
            steps=cfg.c2st_steps,
            lr=cfg.c2st_lr,
            train_frac=cfg.c2st_train_frac,
        )
    )
    out.update(compute_prdc(real, fake, k=cfg.prdc_k))
    return out
