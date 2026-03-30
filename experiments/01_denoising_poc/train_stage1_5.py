from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, fields
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint as grad_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cebcm.data.dataset import SONARVectorDataset
from cebcm.inference.langevin import run_langevin
from cebcm.inference.sigma_schedule import AdaptiveSigmaEnergyWrapper, SigmaScheduleConfig
from cebcm.models.actor import LatentDenoiseActor
from cebcm.models.energy import SimpleEnergy
from cebcm.models.energy_unconditional import UnconditionalEnergy
from cebcm.models.normalization import OrthoLinear
from cebcm.training.kill_criteria import ConditionalThresholds, summarize_conditional_eval
from cebcm.training.losses import gradient_penalty
from configs.base import LangevinConfig, Stage1_5Config


def migrate_bjorck_state_dict(model: torch.nn.Module, old_sd: dict) -> dict:
    """
    Migrate old Björck (OrthoLinear) checkpoint to Cayley parametrization format.

    Old keys: "net.0.weight", "net.0.bias"
    New keys: "net.0.parametrizations.weight.original", "net.0.parametrizations.weight.0.base", "net.0.bias"

    If old_sd already has "parametrizations" keys, returns it unchanged.
    """
    has_parametrizations = any("parametrizations" in k for k in old_sd)
    if has_parametrizations:
        return old_sd

    # Build mapping from current model's expected keys
    new_sd = model.state_dict()
    migrated = {}
    for new_key in new_sd:
        if "parametrizations.weight.original" in new_key:
            # Map from old bare weight: strip "parametrizations.weight.original" → "weight"
            old_key = new_key.replace("parametrizations.weight.original", "weight")
            if old_key in old_sd:
                migrated[new_key] = old_sd[old_key]
            else:
                migrated[new_key] = new_sd[new_key]
        elif "parametrizations.weight.0.base" in new_key:
            # The base matrix is initialized by the parametrization, keep model default
            migrated[new_key] = new_sd[new_key]
        elif new_key in old_sd:
            migrated[new_key] = old_sd[new_key]
        else:
            migrated[new_key] = new_sd[new_key]

    return migrated


def append_jsonl_record(path: Path, payload: dict) -> None:
    """Append one JSON record (single line) for live monitoring."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_amp_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16


def add_relative_noise(v: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(scale):
        scale = torch.tensor(scale, device=v.device, dtype=v.dtype)
    scale = scale.to(device=v.device, dtype=v.dtype)
    if scale.ndim == 0:
        scale = scale.view(1, 1)
    elif scale.ndim == 1:
        scale = scale.unsqueeze(-1)
    return v + torch.randn_like(v) * scale * v.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def sample_sigma(cfg: Stage1_5Config, batch: int, device: torch.device) -> torch.Tensor:
    if cfg.sigma_sampling == "loguniform":
        lo, hi = math.log(cfg.sigma_curriculum_start), math.log(cfg.sigma_curriculum_end)
        return (torch.rand(batch, 1, device=device) * (hi - lo) + lo).exp()
    if cfg.sigma_sampling == "edm":
        return torch.exp(
            torch.randn(batch, 1, device=device) * cfg.edm_p_std + cfg.edm_p_mean
        ).clamp(min=cfg.sigma_curriculum_start, max=cfg.sigma_curriculum_end)
    raise ValueError(f"Unknown sigma sampling: {cfg.sigma_sampling}")


def sanitize_grads(mods: list[torch.nn.Module]) -> None:
    for m in mods:
        for p in m.parameters():
            if p.grad is not None:
                torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0, out=p.grad)


def clip_grads(mods: list[torch.nn.Module], max_norm: float) -> torch.Tensor:
    params: list[torch.nn.Parameter] = []
    for m in mods:
        params.extend([p for p in m.parameters() if p.requires_grad])
    return torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)


def set_ortho_n_iters(model: torch.nn.Module, n_iters: int) -> None:
    """Legacy no-op. Cayley parametrization provides exact orthogonality."""
    # With Cayley parametrization, n_iters is irrelevant.
    # Kept for backward compatibility with old OrthoLinear checkpoints.
    for module in model.modules():
        if isinstance(module, OrthoLinear):
            module.n_iters = n_iters


def resolve_ortho_n_iters(cfg: Stage1_5Config, epoch_idx: int, total_epochs: int) -> int:
    """Legacy no-op. Returns 0 — Cayley parametrization needs no iterations."""
    return 0


def build_bank(
    ds: SONARVectorDataset,
    device: torch.device,
    bank_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    emb = ds.embeddings
    idx = torch.arange(len(emb), dtype=torch.long)
    if 0 < bank_size < len(emb):
        idx = torch.randperm(len(emb))[:bank_size]
        emb = emb[idx]
    bank = emb.to(device)
    bank_n = F.normalize(bank, dim=-1)
    return bank, bank_n, idx.to(device=device, dtype=torch.long)


def retrieve_pos_hard(
    q: torch.Tensor,
    bank: torch.Tensor,
    bank_n: torch.Tensor,
    topk_pos: int,
    hard_start: int,
    hard_end: int,
    self_sim_exclude: float,
    min_pos_similarity: float,
    q_indices: torch.Tensor | None = None,
    bank_indices: torch.Tensor | None = None,
    strict_index_exclusion: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    qn = F.normalize(q, dim=-1)
    k = min(bank.shape[0], max(2, topk_pos, hard_end))
    sim = qn @ bank_n.T
    vals, idx = torch.topk(sim, k=k, dim=-1)
    bsz = q.shape[0]
    batch = torch.arange(bsz, device=q.device)

    same_index = torch.zeros_like(idx, dtype=torch.bool)
    strict_has_index = (
        strict_index_exclusion
        and q_indices is not None
        and bank_indices is not None
    )
    if strict_has_index:
        same_index = bank_indices[idx] == q_indices.unsqueeze(1)

    valid_pos = (
        (~same_index)
        & (vals < float(self_sim_exclude))
        & (vals >= float(min_pos_similarity))
    )
    has_valid_pos = valid_pos.any(dim=1)
    first_valid_col = valid_pos.to(torch.int64).argmax(dim=1)
    pos = idx[batch, first_valid_col]

    # If there is no < self_sim_exclude candidate, fall back to first non-self candidate.
    fallback_mask = ~same_index if strict_has_index else torch.ones_like(idx, dtype=torch.bool)
    fallback_col = fallback_mask.to(torch.int64).argmax(dim=1)
    pos_fallback = idx[batch, fallback_col]
    pos = torch.where(has_valid_pos, pos, pos_fallback)

    hs = min(max(0, hard_start), k - 1)
    he = min(max(hard_start + 1, hard_end), k)
    win = idx[:, hs:he]
    if win.shape[1] == 0:
        win = idx[:, -1:]
    rand_col = torch.randint(0, win.shape[1], (bsz,), device=q.device)
    hard = win[batch, rand_col]

    if strict_has_index:
        hard_same = bank_indices[hard] == q_indices
    else:
        hard_same = torch.zeros_like(hard, dtype=torch.bool)
    need_fix = (hard == pos) | hard_same

    rev_idx = idx.flip(dims=[1])
    rev_same = same_index.flip(dims=[1])
    rev_is_pos = rev_idx == pos.unsqueeze(1)
    valid_alt = ~rev_is_pos
    if strict_has_index:
        valid_alt = valid_alt & (~rev_same)
    has_alt = valid_alt.any(dim=1)
    alt_col = valid_alt.to(torch.int64).argmax(dim=1)
    alt = rev_idx[batch, alt_col]
    hard = torch.where(need_fix & has_alt, alt, hard)

    pos_vec = bank[pos]
    if (~has_valid_pos).any():
        # No reliable retrieval positive for this query:
        # fall back to identity target to avoid training on semantically wrong pairs.
        pos_vec = pos_vec.clone()
        pos_vec[~has_valid_pos] = q[~has_valid_pos]

    return pos_vec, bank[hard]


def seed_actor(q: torch.Tensor, hard: torch.Tensor, sigma: torch.Tensor, cfg: Stage1_5Config) -> torch.Tensor:
    mix = cfg.actor_seed_mix_query * q + (1.0 - cfg.actor_seed_mix_query) * hard
    return add_relative_noise(mix, sigma * cfg.actor_seed_noise_scale)


def conditional_mdsm(
    critic: SimpleEnergy,
    q: torch.Tensor,
    pos: torch.Tensor,
    sigma: torch.Tensor,
    cfg: Stage1_5Config,
) -> torch.Tensor:
    noise = torch.randn_like(pos)
    nrm = pos.norm(dim=-1, keepdim=True).clamp(min=cfg.mdsm_norm_floor)
    noisy = pos + noise * sigma * nrm
    sigma_eff_sq = ((sigma * nrm) ** 2).clamp(min=1e-6)
    noisy_req = noisy.detach().requires_grad_(True)
    if cfg.mdsm_gradient_checkpointing:
        def _forward(inp: torch.Tensor) -> torch.Tensor:
            return critic(q, inp, sigma=sigma.detach())
        e = grad_checkpoint(_forward, noisy_req, use_reentrant=False)
    else:
        e = critic(q, noisy_req, sigma=sigma.detach())
    g = torch.autograd.grad(e.sum(), noisy_req, create_graph=True)[0]
    tgt = (noisy.detach() - pos) / sigma_eff_sq
    g = torch.nan_to_num(g, nan=0.0, posinf=1e4, neginf=-1e4)
    tgt = torch.nan_to_num(tgt, nan=0.0, posinf=1e4, neginf=-1e4)
    if cfg.mdsm_tangent_projection:
        vh = F.normalize(noisy.detach(), dim=-1)
        g = g - (g * vh).sum(dim=-1, keepdim=True) * vh
        tgt = tgt - (tgt * vh).sum(dim=-1, keepdim=True) * vh
    if cfg.mdsm_directional:
        c = F.cosine_similarity(g, tgt, dim=-1, eps=cfg.mdsm_cosine_eps).clamp(-1.0, 1.0)
        loss = 1.0 - c
        if cfg.mdsm_magnitude_aux_weight > 0:
            gn = g.norm(dim=-1).clamp(min=cfg.mdsm_norm_floor, max=1e4)
            tn = tgt.norm(dim=-1).clamp(min=cfg.mdsm_norm_floor, max=1e4)
            loss = loss + cfg.mdsm_magnitude_aux_weight * F.smooth_l1_loss(
                torch.log(gn), torch.log(tn), reduction="none"
            )
    else:
        loss = ((g - tgt) ** 2).sum(dim=-1)
    if cfg.sigma_weighting == "sigma2":
        w = sigma_eff_sq.squeeze(-1)
    elif cfg.sigma_weighting == "inv_sigma2":
        w = 1.0 / sigma_eff_sq.squeeze(-1).clamp(min=1e-8)
    elif cfg.sigma_weighting == "uniform":
        w = torch.ones_like(loss)
    else:
        raise ValueError(f"Unknown sigma_weighting: {cfg.sigma_weighting}")
    loss = torch.nan_to_num(loss, nan=1e4, posinf=1e4, neginf=1e4)
    w = w / w.mean().clamp(min=1e-8)
    return (w * loss).mean()


def normalize_triplet_energies(
    e_pos: torch.Tensor,
    e_actor: torch.Tensor,
    e_hard: torch.Tensor,
    enabled: bool,
    std_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize triplet energies by detached batch statistics to make ranking margins
    scale-invariant. This prevents impossible absolute-margin constraints when
    energy range is narrow.
    """
    if not enabled:
        return e_pos, e_actor, e_hard
    with torch.no_grad():
        stacked = torch.cat([e_pos, e_actor, e_hard], dim=0)
        mu = stacked.mean()
        std = stacked.std(unbiased=False).clamp(min=float(std_floor))
    mu = mu.detach()
    std = std.detach()
    return (e_pos - mu) / std, (e_actor - mu) / std, (e_hard - mu) / std


def conditional_nce_loss(
    critic: SimpleEnergy,
    q: torch.Tensor,
    pos: torch.Tensor,
    hard: torch.Tensor,
    bank: torch.Tensor,
    sigma: torch.Tensor,
    cfg: Stage1_5Config,
) -> torch.Tensor:
    bsz = q.shape[0]
    num_neg = max(0, int(cfg.nce_num_random_negatives))
    temp = max(1e-6, float(cfg.nce_temperature))

    e_pos = critic(q, pos, sigma=sigma.detach())
    e_hard = critic(q, hard, sigma=sigma.detach())

    logits_parts = [(-e_pos / temp).unsqueeze(1), (-e_hard / temp).unsqueeze(1)]
    if num_neg > 0:
        ridx = torch.randint(0, bank.shape[0], (bsz, num_neg), device=q.device)
        rand_neg = bank[ridx]  # [B, K, D]
        q_rep = q.unsqueeze(1).expand(-1, num_neg, -1).reshape(bsz * num_neg, -1)
        neg_flat = rand_neg.reshape(bsz * num_neg, -1)
        sigma_rep = sigma.repeat_interleave(num_neg, dim=0).detach()
        e_rand = critic(q_rep, neg_flat, sigma=sigma_rep).view(bsz, num_neg)
        logits_parts.append(-e_rand / temp)

    logits = torch.cat(logits_parts, dim=1)
    labels = torch.zeros(bsz, dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)


def prior_nce_loss(
    prior: UnconditionalEnergy,
    pos: torch.Tensor,
    hard: torch.Tensor,
    bank: torch.Tensor,
    cfg: Stage1_5Config,
) -> torch.Tensor:
    bsz = pos.shape[0]
    num_neg = max(0, int(cfg.nce_num_random_negatives))
    temp = max(1e-6, float(cfg.nce_temperature))

    e_pos = prior(pos)
    e_hard = prior(hard)
    logits_parts = [(-e_pos / temp).unsqueeze(1), (-e_hard / temp).unsqueeze(1)]
    if num_neg > 0:
        ridx = torch.randint(0, bank.shape[0], (bsz, num_neg), device=pos.device)
        rand_neg = bank[ridx].reshape(bsz * num_neg, -1)
        e_rand = prior(rand_neg).view(bsz, num_neg)
        logits_parts.append(-e_rand / temp)
    logits = torch.cat(logits_parts, dim=1)
    labels = torch.zeros(bsz, dtype=torch.long, device=pos.device)
    return F.cross_entropy(logits, labels)


def clean_minimum_penalty(
    e_clean: torch.Tensor,
    e_actor: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """P0.1: Penalize when critic assigns lower energy to actor output than to clean target.

    This prevents sub-clean attractors — energy minima that are NOT at the clean
    target, which cause Langevin to overshoot past clean and degrade cosine.

    L = relu(E_clean - E_actor + margin).mean()
    When E_actor < E_clean (violation), this produces a positive penalty.
    """
    return F.relu(e_clean - e_actor + margin).mean()


def gradient_direction_loss(
    critic: SimpleEnergy,
    q: torch.Tensor,
    pos: torch.Tensor,
    sigma: torch.Tensor,
    cfg: Stage1_5Config,
) -> torch.Tensor:
    """P0.2: Explicit gradient direction loss — teaches critic WHERE to point.

    At noisy points, -grad_E should point toward clean target (pos).
    This is AUXILIARY to MDSM (first-order only, no create_graph needed for target),
    providing a direct supervision signal on gradient direction.

    L = (1 - cos_sim(F.normalize(-grad_E), F.normalize(v_clean - v_noisy))).mean()

    When direction_num_samples > 1, generates multiple noise perturbations per
    clean sample and averages the loss — more gradient supervision per step.
    """
    num_samples = getattr(cfg, "direction_num_samples", 1)
    nrm = pos.norm(dim=-1, keepdim=True).clamp(min=cfg.mdsm_norm_floor)
    cos_accum = []
    for _ in range(num_samples):
        noise = torch.randn_like(pos)
        noisy = pos + noise * sigma * nrm
        noisy_req = noisy.detach().requires_grad_(True)
        e = critic(q, noisy_req, sigma=sigma.detach())
        # create_graph=True needed: backprop through g to update critic params θ
        g = torch.autograd.grad(e.sum(), noisy_req, create_graph=True)[0]
        g = torch.nan_to_num(g, nan=0.0, posinf=1e4, neginf=-1e4)
        # Target direction: from noisy toward clean
        target_dir = pos - noisy.detach()
        # Tangent projection on sphere if configured
        if cfg.mdsm_tangent_projection:
            vh = F.normalize(noisy.detach(), dim=-1)
            g = g - (g * vh).sum(dim=-1, keepdim=True) * vh
            target_dir = target_dir - (target_dir * vh).sum(dim=-1, keepdim=True) * vh
        # Cosine similarity between negative gradient and target direction
        neg_g = -g
        cos = F.cosine_similarity(
            F.normalize(neg_g, dim=-1, eps=1e-8),
            F.normalize(target_dir, dim=-1, eps=1e-8),
            dim=-1,
            eps=cfg.mdsm_cosine_eps,
        ).clamp(-1.0, 1.0)
        cos_accum.append(cos)
    return (1.0 - torch.stack(cos_accum).mean())


def inbatch_cross_negative_nce(
    critic: SimpleEnergy,
    q: torch.Tensor,
    pos: torch.Tensor,
    sigma: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """P1.3: In-batch cross-negative contrastive loss.

    For each query q_i, the positive is pos_i and negatives are ALL other pos_j (j≠i)
    in the batch. This is a dense, balanced NCE that scales with batch size without
    extra bank lookups.

    Uses symmetric InfoNCE: for each q_i, compute energy to all pos_j,
    then cross-entropy with label=i.
    """
    bsz = q.shape[0]
    if bsz < 2:
        return torch.tensor(0.0, device=q.device)
    temp = max(1e-6, temperature)
    # Compute pairwise energies: E(q_i, pos_j) for all i,j
    # Expand q: [B, 1, D] -> [B, B, D], pos: [1, B, D] -> [B, B, D]
    q_exp = q.unsqueeze(1).expand(bsz, bsz, -1).reshape(bsz * bsz, -1)
    pos_exp = pos.unsqueeze(0).expand(bsz, bsz, -1).reshape(bsz * bsz, -1)
    # sigma shape: [B, 1] from sample_sigma. Each q_i needs its sigma_i for all B pos_j.
    # q_exp layout: [q_0]*B, [q_1]*B, ... so sigma must follow same interleave pattern.
    if sigma.dim() == 1:
        sigma_exp = sigma.unsqueeze(1).expand(bsz, bsz).reshape(bsz * bsz, 1)
    else:
        # sigma: [B, 1] → repeat_interleave to get [B*B, 1] matching q_exp layout
        sigma_exp = sigma.repeat_interleave(bsz, dim=0)
    e_all = critic(q_exp, pos_exp, sigma=sigma_exp.detach()).view(bsz, bsz)
    # Logits: lower energy = higher probability
    logits = -e_all / temp
    labels = torch.arange(bsz, device=q.device)
    return F.cross_entropy(logits, labels)


def knn_support_penalty(
    actor_output: torch.Tensor,
    bank: torch.Tensor,
    k: int = 5,
    threshold: float | None = None,
) -> tuple[torch.Tensor, float]:
    """P1.2: Support/manifold proximity penalty via kNN distance to retrieval bank.

    Penalizes actor outputs that are far from the data manifold (measured by
    distance to k-th nearest neighbor in the bank). This prevents the actor
    from drifting into low-density regions of embedding space.

    Returns (penalty, mean_knn_dist) for logging.
    """
    # Compute cosine distances to bank (1 - cosine_sim)
    # actor_output: [B, D], bank: [N, D]
    actor_normed = F.normalize(actor_output, dim=-1)
    bank_normed = F.normalize(bank, dim=-1)
    # [B, N] cosine similarity matrix
    sim = torch.mm(actor_normed, bank_normed.t())
    dist = 1.0 - sim  # cosine distance
    # k-th nearest neighbor distance (k-th smallest distance)
    topk_dist, _ = dist.topk(k, dim=1, largest=False)
    knn_dist = topk_dist[:, -1]  # k-th nearest = last of top-k smallest
    mean_knn = float(knn_dist.mean().item())
    if threshold is None:
        # No penalty if no threshold set
        return torch.tensor(0.0, device=actor_output.device), mean_knn
    # Penalty: penalize distances beyond threshold (soft hinge)
    penalty = F.relu(knn_dist - threshold).pow(2).mean()
    return penalty, mean_knn


class TwinHybridEnergy:
    def __init__(
        self,
        c1: SimpleEnergy,
        c2: SimpleEnergy,
        prior: UnconditionalEnergy | None,
        lambda_prior: float,
        aggregate: str,
        softmax_temperature: float,
    ):
        self.c1 = c1
        self.c2 = c2
        self.prior = prior
        self.lambda_prior = lambda_prior
        self.aggregate = aggregate
        self.softmax_temperature = softmax_temperature

    def cond(self, q: torch.Tensor, v: torch.Tensor, sigma: torch.Tensor | None = None) -> torch.Tensor:
        e1, e2 = self.c1(q, v, sigma=sigma), self.c2(q, v, sigma=sigma)
        if self.aggregate == "mean":
            return 0.5 * (e1 + e2)
        if self.aggregate == "softmax":
            tau = max(1e-6, float(self.softmax_temperature))
            stacked = torch.stack([e1, e2], dim=0)
            return tau * torch.logsumexp(stacked / tau, dim=0)
        if self.aggregate == "max":
            return torch.maximum(e1, e2)
        raise ValueError(f"Unknown twin aggregation mode: {self.aggregate}")

    def __call__(self, q: torch.Tensor, v: torch.Tensor, sigma: torch.Tensor | None = None) -> torch.Tensor:
        e = self.cond(q, v, sigma=sigma)
        if self.prior is not None and self.lambda_prior > 0:
            e = e + self.lambda_prior * self.prior(v)
        return e

    def energy_and_grad(
        self, q: torch.Tensor, v: torch.Tensor, sigma: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v_req = v.detach().requires_grad_(True)
        e = self(q, v_req, sigma=sigma)
        g = torch.autograd.grad(e.sum(), v_req, create_graph=False)[0]
        return e.detach(), g.detach()


class SigmaBoundEnergy:
    """
    Bind a fixed sigma tensor to an energy callable for Langevin API parity.
    """

    def __init__(self, energy_fn: TwinHybridEnergy, sigma: torch.Tensor):
        self.energy_fn = energy_fn
        self.sigma = sigma

    def __call__(self, v_query: torch.Tensor, v_candidate: torch.Tensor) -> torch.Tensor:
        return self.energy_fn(v_query, v_candidate, sigma=self.sigma)

    def energy_and_grad(
        self,
        v_query: torch.Tensor,
        v_candidate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.energy_fn.energy_and_grad(v_query, v_candidate, sigma=self.sigma)


def make_thresholds(cfg: Stage1_5Config) -> ConditionalThresholds:
    return ConditionalThresholds(
        min_cos_improvement=cfg.min_cosine_improvement,
        min_cos_success_rate=cfg.min_cosine_success_rate,
        min_geodesic_improvement=cfg.min_geodesic_improvement,
        min_l2_improvement=cfg.min_l2_improvement,
        min_energy_success_rate=cfg.min_energy_success_rate,
        max_clean_min_violation=cfg.max_clean_min_violation_rate,
        min_step_norm=cfg.min_step_norm,
    )


def eval_model(
    ef: TwinHybridEnergy,
    actor: LatentDenoiseActor,
    ds: SONARVectorDataset,
    bank: torch.Tensor,
    bank_n: torch.Tensor,
    bank_idx: torch.Tensor,
    cfg: Stage1_5Config,
    device: torch.device,
    eval_ids: torch.Tensor | None = None,
) -> dict:
    ef.c1.eval()
    ef.c2.eval()
    if ef.prior is not None:
        ef.prior.eval()
    actor.eval()
    if eval_ids is None:
        ids = torch.randperm(len(ds))[: min(cfg.eval_num_samples, len(ds))]
    else:
        ids = eval_ids.to(dtype=torch.long).cpu()
    kw = {}
    if cfg.langevin.method == "pid":
        kw = dict(
            kp=cfg.langevin.pid_kp,
            ki=cfg.langevin.pid_ki,
            kd=cfg.langevin.pid_kd,
            integral_decay=cfg.langevin.pid_integral_decay,
        )
    elif cfg.langevin.method == "underdamped":
        kw = dict(friction=cfg.langevin.underdamped_friction, mass=cfg.langevin.underdamped_mass)
    else:
        kw = dict(momentum_beta=cfg.langevin.momentum_beta)
    out = {}
    eval_batch = max(1, int(cfg.eval_langevin_batch_size))
    for ns in cfg.eval_noise_scales:
        cb, ca, lb, la, eb, ea, ep, step, succ, viol, rcos = [], [], [], [], [], [], [], [], [], [], []
        for start in range(0, len(ids), eval_batch):
            batch_ids = ids[start:start + eval_batch]
            q_idx = batch_ids.to(device=device, dtype=torch.long)
            q = ds.embeddings[batch_ids].to(device=device, dtype=bank.dtype)
            pos, hard = retrieve_pos_hard(
                q,
                bank,
                bank_n,
                cfg.retrieval_topk_pos,
                cfg.retrieval_hard_start,
                cfg.retrieval_hard_end,
                cfg.retrieval_self_sim_exclude,
                cfg.retrieval_min_pos_similarity,
                q_indices=q_idx,
                bank_indices=bank_idx,
                strict_index_exclusion=cfg.retrieval_strict_index_exclusion,
            )
            rcos.extend(F.cosine_similarity(q, pos, dim=-1).detach().cpu().tolist())
            sigma = torch.full((q.shape[0], 1), float(ns), device=device, dtype=q.dtype)
            noisy = seed_actor(q, hard, sigma, cfg)
            with torch.no_grad():
                v = noisy
                for _ in range(max(1, cfg.actor_eval_steps)):
                    v, _ = actor.predict_step(
                        q,
                        v,
                        sigma=sigma,
                        step_size=cfg.actor_step_size,
                        target_norm=cfg.langevin.target_norm,
                        tangent_projection=cfg.actor_tangent_projection,
                    )
            # Use adaptive sigma if enabled, else fixed sigma
            if getattr(cfg.langevin, 'sigma_anneal', False):
                sigma_sched_cfg = SigmaScheduleConfig(
                    enabled=True,
                    mode=getattr(cfg.langevin, 'sigma_anneal_mode', 'hybrid'),
                    sigma_max=getattr(cfg.langevin, 'sigma_anneal_max', 0.3),
                    sigma_min=getattr(cfg.langevin, 'sigma_anneal_min', 0.01),
                    adaptive_blend=getattr(cfg.langevin, 'sigma_anneal_blend', 0.5),
                )
                sigma_bound_ef = AdaptiveSigmaEnergyWrapper(
                    ef, sigma_sched_cfg, max_steps=cfg.critic_eval_langevin_steps,
                )
            else:
                sigma_bound_ef = SigmaBoundEnergy(ef, sigma=sigma.detach())
            if cfg.critic_eval_langevin_steps > 0:
                # Keep eval math sample-independent when batched: fixed-step rollout,
                # no batch-coupled early stop by mean energy.
                res = run_langevin(
                    method=cfg.langevin.method,
                    energy_fn=sigma_bound_ef,
                    v_query=q,
                    v_init=v,
                    lr=cfg.langevin.lr,
                    noise_scale=cfg.langevin.noise_scale,
                    max_steps=cfg.critic_eval_langevin_steps,
                    target_norm=cfg.langevin.target_norm,
                    energy_threshold=None,
                    plateau_patience=max(cfg.critic_eval_langevin_steps + 1, cfg.langevin.plateau_patience),
                    plateau_delta=cfg.langevin.plateau_delta,
                    tangent_noise=cfg.langevin_tangent_noise,
                    v_target=None,
                    tamed=getattr(cfg.langevin, 'tamed', False),
                    **kw,
                )
                final = res.v_last if res.v_last is not None else res.v_final
            else:
                final = v

            cb_batch = F.cosine_similarity(pos, noisy, dim=-1)
            ca_batch = F.cosine_similarity(pos, final, dim=-1)
            lb_batch = torch.norm(pos - noisy, dim=-1)
            la_batch = torch.norm(pos - final, dim=-1)
            step_batch = torch.norm(final - noisy, dim=-1)
            with torch.no_grad():
                ep_batch = ef(q, pos, sigma=sigma.detach()).detach()
                eb_batch = ef(q, noisy, sigma=sigma.detach()).detach()
                ea_batch = ef(q, final, sigma=sigma.detach()).detach()

            cb.extend(cb_batch.detach().cpu().tolist())
            ca.extend(ca_batch.detach().cpu().tolist())
            lb.extend(lb_batch.detach().cpu().tolist())
            la.extend(la_batch.detach().cpu().tolist())
            ep.extend(ep_batch.detach().cpu().tolist())
            eb.extend(eb_batch.detach().cpu().tolist())
            ea.extend(ea_batch.detach().cpu().tolist())
            step.extend(step_batch.detach().cpu().tolist())
            succ.extend((ea_batch < eb_batch).to(torch.float32).detach().cpu().tolist())
            viol.extend((ea_batch < ep_batch).to(torch.float32).detach().cpu().tolist())
        n = float(len(cb))
        out[f"noise_{ns}"] = {
            "cos_before_mean": sum(cb) / n,
            "cos_after_mean": sum(ca) / n,
            "improvement": (sum(ca) - sum(cb)) / n,
            "geodesic_before_mean": sum(math.acos(max(-1.0, min(1.0, x))) for x in cb) / n,
            "geodesic_after_mean": sum(math.acos(max(-1.0, min(1.0, x))) for x in ca) / n,
            "geodesic_improvement": (
                sum(math.acos(max(-1.0, min(1.0, x))) for x in cb)
                - sum(math.acos(max(-1.0, min(1.0, x))) for x in ca)
            ) / n,
            "l2_before_mean": sum(lb) / n,
            "l2_after_mean": sum(la) / n,
            "l2_improvement": (sum(lb) - sum(la)) / n,
            "energy_clean_mean": sum(ep) / n,
            "energy_before_mean": sum(eb) / n,
            "energy_after_mean": sum(ea) / n,
            "energy_improvement": (sum(eb) - sum(ea)) / n,
            "energy_success_rate": sum(succ) / n,
            "clean_min_violation_rate": sum(viol) / n,
            "step_norm_mean": sum(step) / n,
            # Backward-compatible alias:
            # `success_rate` historically meant cosine-gain success.
            "success_rate": sum(1.0 for b, a in zip(cb, ca) if a > b) / n,
            "cos_success_rate": sum(1.0 for b, a in zip(cb, ca) if a > b) / n,
            "retrieval_cosine_mean": sum(rcos) / n,
        }
    return out


def _not_evaluated_kill_stub() -> dict:
    """Stable schema for epochs where eval is intentionally skipped."""
    return {
        "status": "not_evaluated",
        "score": None,
        "passed": None,
        "global_gates": {},
        "aggregate": {},
        "per_noise": {},
    }


def _validate_stage15_config(cfg: Stage1_5Config) -> None:
    def _must_be_non_negative(name: str, value: float) -> None:
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")

    def _must_be_positive(name: str, value: float) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")

    for name in [
        "critic_lr",
        "actor_lr",
        "prior_critic_lr",
        "clip_grad_norm",
        "loss_spike_factor",
        "sigma_curriculum_end",
        "sigma_max",
        "nce_temperature",
        "actor_step_size",
        "ortho_n_iters",
        "eval_langevin_batch_size",
        "checkpoint_every_epochs",
    ]:
        _must_be_positive(name, float(getattr(cfg, name)))

    for name in [
        "weight_decay",
        "lambda_mdsm",
        "lambda_rank",
        "lambda_nce",
        "lambda_cql",
        "lambda_shell",
        "lambda_geo",
        "lambda_align",
        "lambda_bc_reg",
        "lambda_prior",
        "lambda_prior_nce",
        "lambda_actor_barrier",
        "lambda_actor_descent",
        "gradient_penalty_lambda",
        "shell_barrier_margin",
        "cql_noise_scale",
        "nce_num_random_negatives",
        "sigma_curriculum_start",
        "sigma_min",
        "mdsm_magnitude_aux_weight",
        "critic_margin_clean_actor",
        "critic_margin_actor_noisy",
        "critic_margin_clean_noisy",
        "rank_std_floor",
        "actor_energy_margin_pos",
        "actor_energy_margin_hard",
        "retrieval_self_sim_exclude",
        "retrieval_min_pos_similarity",
        "non_finite_backoff_streak_trigger",
        "max_consecutive_non_finite_batches",
        "param_finite_check_interval",
    ]:
        _must_be_non_negative(name, float(getattr(cfg, name)))
    if int(cfg.critic_steps_per_actor) < 1:
        raise ValueError(f"critic_steps_per_actor must be >= 1, got {cfg.critic_steps_per_actor}")

    if cfg.sigma_curriculum_start > cfg.sigma_curriculum_end:
        raise ValueError(
            "sigma_curriculum_start must be <= sigma_curriculum_end, "
            f"got {cfg.sigma_curriculum_start} > {cfg.sigma_curriculum_end}"
        )
    if cfg.sigma_curriculum_start <= 0.0:
        raise ValueError(
            f"sigma_curriculum_start must be > 0 for loguniform sampling, got {cfg.sigma_curriculum_start}"
        )
    if cfg.sigma_min > cfg.sigma_max:
        raise ValueError(f"sigma_min must be <= sigma_max, got {cfg.sigma_min} > {cfg.sigma_max}")
    if cfg.eval_every_epochs < 1:
        raise ValueError(f"eval_every_epochs must be >= 1, got {cfg.eval_every_epochs}")
    if cfg.num_epochs < 1:
        raise ValueError(f"num_epochs must be >= 1, got {cfg.num_epochs}")
    if cfg.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {cfg.batch_size}")
    if cfg.retrieval_self_sim_exclude > 1.0:
        raise ValueError(
            f"retrieval_self_sim_exclude must be <= 1.0 for cosine similarity, got {cfg.retrieval_self_sim_exclude}"
        )
    if cfg.retrieval_min_pos_similarity < -1.0 or cfg.retrieval_min_pos_similarity > 1.0:
        raise ValueError(
            f"retrieval_min_pos_similarity must be in [-1,1], got {cfg.retrieval_min_pos_similarity}"
        )
    if cfg.rank_std_floor <= 0:
        raise ValueError(f"rank_std_floor must be > 0, got {cfg.rank_std_floor}")
    if cfg.sigma_weighting not in {"sigma2", "uniform", "inv_sigma2"}:
        raise ValueError(
            "sigma_weighting must be one of {'sigma2','uniform','inv_sigma2'}, "
            f"got {cfg.sigma_weighting}"
        )
    if cfg.twin_aggregate not in {"max", "mean", "softmax"}:
        raise ValueError(
            "twin_aggregate must be one of {'max','mean','softmax'}, "
            f"got {cfg.twin_aggregate}"
        )
    if cfg.actor_norm_mode not in {"orthonorm", "spectral_norm", "none"}:
        raise ValueError(
            "actor_norm_mode must be one of {'orthonorm','spectral_norm','none'}, "
            f"got {cfg.actor_norm_mode}"
        )
    if cfg.actor_activation not in {"silu", "gelu", "relu", "groupsort", "lipschitz_spline"}:
        raise ValueError(
            "actor_activation must be one of {'silu','gelu','relu','groupsort','lipschitz_spline'}, "
            f"got {cfg.actor_activation}"
        )
    if cfg.compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
        raise ValueError(
            "compile_mode must be one of {'default','reduce-overhead','max-autotune'}, "
            f"got {cfg.compile_mode}"
        )
    if not str(cfg.rolling_checkpoint_name).strip():
        raise ValueError("rolling_checkpoint_name must be non-empty")

    if cfg.ortho_schedule_enabled:
        if not cfg.ortho_schedule_iters:
            raise ValueError("ortho_schedule_enabled requires non-empty ortho_schedule_iters")
        if len(cfg.ortho_schedule_boundaries) != max(0, len(cfg.ortho_schedule_iters) - 1):
            raise ValueError(
                "len(ortho_schedule_boundaries) must equal len(ortho_schedule_iters)-1, "
                f"got {len(cfg.ortho_schedule_boundaries)} vs {len(cfg.ortho_schedule_iters)-1}"
            )
        prev = -float("inf")
        for b in cfg.ortho_schedule_boundaries:
            if not (0.0 < float(b) < 1.0):
                raise ValueError(f"ortho_schedule_boundaries values must be in (0,1), got {b}")
            if float(b) <= prev:
                raise ValueError("ortho_schedule_boundaries must be strictly increasing")
            prev = float(b)

    for name in [
        "lr",
        "noise_scale",
        "max_steps",
        "pid_kp",
        "pid_ki",
        "pid_kd",
        "pid_integral_decay",
        "momentum_beta",
    ]:
        value = float(getattr(cfg.langevin, name))
        _must_be_non_negative(f"langevin.{name}", value)
    if cfg.langevin.method == "underdamped":
        if float(cfg.langevin.underdamped_mass) <= 0.0:
            raise ValueError(
                f"langevin.underdamped_mass must be > 0, got {cfg.langevin.underdamped_mass}"
            )
        fr = float(cfg.langevin.underdamped_friction)
        if not (0.0 < fr <= 1.0):
            raise ValueError(
                f"langevin.underdamped_friction must be in (0,1], got {cfg.langevin.underdamped_friction}"
            )


def load_config(path: str) -> Stage1_5Config:
    raw = json.load(open(path, "r", encoding="utf-8"))
    raw = {k: v for k, v in raw.items() if not str(k).startswith("_")}
    kill = raw.pop("kill_criteria", None)
    if isinstance(raw.get("langevin"), dict):
        allowed = {f.name for f in fields(LangevinConfig)}
        raw["langevin"] = LangevinConfig(**{k: v for k, v in raw["langevin"].items() if k in allowed})
    if isinstance(raw.get("sigma_weighting"), bool):
        raw["sigma_weighting"] = "sigma2" if raw["sigma_weighting"] else "uniform"
    allowed = {f.name for f in fields(Stage1_5Config)}
    cfg = Stage1_5Config(**{k: v for k, v in raw.items() if k in allowed})
    if isinstance(kill, dict):
        cfg.min_cosine_improvement = float(kill.get("min_cosine_improvement", cfg.min_cosine_improvement))
        cfg.min_cosine_success_rate = float(kill.get("min_cosine_success_rate", cfg.min_cosine_success_rate))
        cfg.min_energy_success_rate = float(kill.get("min_energy_success_rate", cfg.min_energy_success_rate))
        cfg.max_clean_min_violation_rate = float(kill.get("max_clean_min_violation_rate", cfg.max_clean_min_violation_rate))
    _validate_stage15_config(cfg)
    return cfg


def _build_lr_schedulers(
    opt_c: torch.optim.Optimizer,
    opt_a: torch.optim.Optimizer,
    cfg,
) -> tuple:
    """Build LR schedulers for critic and actor optimizers.

    Modes:
      - "none": no scheduling (default)
      - "cosine_warmup": linear warmup + cosine decay to lr_min_factor
      - "cosine_warm_restarts": linear warmup + cosine annealing with warm restarts
    """
    mode = getattr(cfg, "lr_scheduler", "none")
    if mode == "none":
        return None, None

    warmup = getattr(cfg, "lr_warmup_epochs", 5)
    start_factor = getattr(cfg, "lr_warmup_start_factor", 0.1)
    min_factor = getattr(cfg, "lr_min_factor", 0.01)
    total = cfg.num_epochs

    def _make(opt):
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, SequentialLR
        if warmup > 0:
            warmup_sched = LinearLR(opt, start_factor=start_factor, end_factor=1.0, total_iters=warmup)
        if mode == "cosine_warmup":
            cos_epochs = max(1, total - warmup)
            cos_sched = CosineAnnealingLR(opt, T_max=cos_epochs, eta_min=min_factor * opt.defaults["lr"])
            if warmup > 0:
                return SequentialLR(opt, schedulers=[warmup_sched, cos_sched], milestones=[warmup])
            return cos_sched
        elif mode == "cosine_warm_restarts":
            t0 = getattr(cfg, "lr_restart_period", 10)
            t_mult = getattr(cfg, "lr_restart_mult", 2)
            cos_sched = CosineAnnealingWarmRestarts(opt, T_0=t0, T_mult=t_mult, eta_min=min_factor * opt.defaults["lr"])
            if warmup > 0:
                return SequentialLR(opt, schedulers=[warmup_sched, cos_sched], milestones=[warmup])
            return cos_sched
        else:
            raise ValueError(f"Unknown lr_scheduler: {mode}")

    sc = _make(opt_c)
    sa = _make(opt_a)
    print(f"  LR scheduler: {mode} (warmup={warmup}, min_factor={min_factor})")
    return sc, sa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage1_5_config.json")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.amp_enabled and device.type == "cuda"
    autocast = (lambda: torch.amp.autocast("cuda", dtype=resolve_amp_dtype(cfg.amp_dtype))) if use_amp else nullcontext
    ds = SONARVectorDataset(cfg.train_data_path)
    if cfg.val_data_path and Path(cfg.val_data_path).exists():
        ds_val = SONARVectorDataset(cfg.val_data_path)
    else:
        n_val = max(1, min(cfg.eval_num_samples * 2, len(ds) // 10))
        ds, ds_val = ds.subset(0, max(1, len(ds) - n_val)), ds.subset(max(1, len(ds) - n_val), len(ds))
    train_index_ds = torch.arange(len(ds), dtype=torch.long)
    loader = DataLoader(
        train_index_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    bank_tr, bankn_tr, bankidx_tr = build_bank(ds, device, cfg.retrieval_bank_size)
    bank_va, bankn_va, bankidx_va = build_bank(ds_val, device, cfg.retrieval_bank_size)
    eval_gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed) + 1009)
    eval_ids = torch.randperm(len(ds_val), generator=eval_gen)[: min(cfg.eval_num_samples, len(ds_val))]

    c1_base = SimpleEnergy(cfg.energy_dim, cfg.energy_hidden_dims, cfg.norm_mode, cfg.activation, ortho_n_iters=cfg.ortho_n_iters, energy_output_clamp=None).to(device)
    c2_base = SimpleEnergy(cfg.energy_dim, cfg.energy_hidden_dims, cfg.norm_mode, cfg.activation, ortho_n_iters=cfg.ortho_n_iters, energy_output_clamp=None).to(device)
    actor_base = LatentDenoiseActor(
        cfg.energy_dim,
        cfg.actor_hidden_dims,
        cfg.actor_norm_mode,
        cfg.actor_activation,
        ortho_n_iters=cfg.ortho_n_iters,
    ).to(device)
    prior_base = UnconditionalEnergy(cfg.energy_dim, cfg.energy_hidden_dims, cfg.norm_mode, cfg.activation, ortho_n_iters=cfg.ortho_n_iters).to(device) if cfg.use_prior_critic else None

    # Runtime n_iters is epoch-scheduled; initialize to first-epoch value.
    init_ortho_iters = resolve_ortho_n_iters(cfg, epoch_idx=0, total_epochs=cfg.num_epochs)
    for m in [c1_base, c2_base, actor_base] + ([prior_base] if prior_base is not None else []):
        set_ortho_n_iters(m, init_ortho_iters)

    c1, c2, actor, prior = c1_base, c2_base, actor_base, prior_base
    compile_enabled = bool(
        cfg.enable_compile
        and device.type == "cuda"
        and hasattr(torch, "compile")
    )
    if compile_enabled:
        try:
            c1 = torch.compile(c1_base, mode=cfg.compile_mode, fullgraph=cfg.compile_fullgraph)
            c2 = torch.compile(c2_base, mode=cfg.compile_mode, fullgraph=cfg.compile_fullgraph)
            actor = torch.compile(actor_base, mode=cfg.compile_mode, fullgraph=cfg.compile_fullgraph)
            if prior_base is not None:
                prior = torch.compile(prior_base, mode=cfg.compile_mode, fullgraph=cfg.compile_fullgraph)
            print(f"torch.compile: enabled (mode={cfg.compile_mode}, fullgraph={cfg.compile_fullgraph})")
        except Exception as ex:
            c1, c2, actor, prior = c1_base, c2_base, actor_base, prior_base
            compile_enabled = False
            print(f"[WARN] torch.compile disabled due to runtime error: {ex}")
    ef = TwinHybridEnergy(
        c1,
        c2,
        prior,
        cfg.lambda_prior if cfg.use_prior_critic else 0.0,
        cfg.twin_aggregate,
        cfg.twin_softmax_temperature,
    )

    groups = [{"params": list(c1_base.parameters()), "lr": cfg.critic_lr}, {"params": list(c2_base.parameters()), "lr": cfg.critic_lr}]
    if prior_base is not None:
        groups.append({"params": list(prior_base.parameters()), "lr": cfg.prior_critic_lr})
    opt_c = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    opt_a = torch.optim.AdamW(actor_base.parameters(), lr=cfg.actor_lr, weight_decay=cfg.weight_decay)

    # LR scheduler
    sched_c, sched_a = _build_lr_schedulers(opt_c, opt_a, cfg)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ckpt_dir, log_dir = Path(cfg.checkpoint_dir), Path(cfg.logs_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)
    stream_path = log_dir / "training_metrics.jsonl"
    best_score, start_epoch, global_step = float("-inf"), 0, 0
    plateau_counter = 0  # for LR plateau boost
    if args.resume:
        ck = torch.load(args.resume, weights_only=False, map_location=device)
        c1_base.load_state_dict(migrate_bjorck_state_dict(c1_base, ck["critic1_state"]))
        c2_base.load_state_dict(migrate_bjorck_state_dict(c2_base, ck["critic2_state"]))
        actor_base.load_state_dict(migrate_bjorck_state_dict(actor_base, ck["actor_state"]))
        if prior_base is not None and ck.get("prior_state") is not None:
            prior_base.load_state_dict(migrate_bjorck_state_dict(prior_base, ck["prior_state"]))
        opt_c.load_state_dict(ck["opt_c_state"]); opt_a.load_state_dict(ck["opt_a_state"]); scaler.load_state_dict(ck["scaler_state"])
        if sched_c is not None and ck.get("sched_c_state") is not None:
            sched_c.load_state_dict(ck["sched_c_state"])
        if sched_a is not None and ck.get("sched_a_state") is not None:
            sched_a.load_state_dict(ck["sched_a_state"])
        best_score, start_epoch, global_step = float(ck.get("best_score", best_score)), int(ck.get("epoch", 0)) + 1, int(ck.get("global_step", 0))
        plateau_counter = int(ck.get("plateau_counter", 0))
    elif stream_path.exists():
        # Fresh run: avoid mixing with previous monitoring session.
        stream_path.write_text("", encoding="utf-8")

    th = make_thresholds(cfg)

    # P1.2: Calibrate kNN support threshold from bank inter-point distances
    support_threshold: float | None = None
    if cfg.use_support_penalty and cfg.lambda_support > 0:
        with torch.no_grad():
            # Sample a subset of bank to estimate typical kNN distances
            n_cal = min(512, bank_tr.shape[0])
            cal_idx = torch.randperm(bank_tr.shape[0], device=device)[:n_cal]
            cal_pts = F.normalize(bank_tr[cal_idx], dim=-1)
            cal_sim = torch.mm(cal_pts, bankn_tr.t())
            cal_dist = 1.0 - cal_sim
            cal_topk, _ = cal_dist.topk(cfg.support_k + 1, dim=1, largest=False)  # +1 for self
            cal_knn = cal_topk[:, -1]  # k-th neighbor (excluding self which is ~0)
            support_threshold = float(
                torch.quantile(cal_knn, cfg.support_threshold_percentile / 100.0).item()
            )
        print(f"  kNN support threshold (p{cfg.support_threshold_percentile:.0f}): {support_threshold:.6f}")

    print(f"Device: {device} | Train: {len(ds)} | Val: {len(ds_val)} | compile={compile_enabled} | grad_checkpointing={cfg.mdsm_gradient_checkpointing}")
    print(
        "Checkpoint policy: "
        f"periodic_every={int(cfg.checkpoint_every_epochs)} "
        f"rolling='{cfg.rolling_checkpoint_name}'"
    )
    for epoch_idx in range(start_epoch, cfg.num_epochs):
        t0 = time.time()
        epoch_timer_start = time.perf_counter()
        last_window_time = epoch_timer_start
        last_window_batch = 0
        active_ortho_iters = resolve_ortho_n_iters(cfg, epoch_idx=epoch_idx, total_epochs=cfg.num_epochs)
        for m in [c1_base, c2_base, actor_base] + ([prior_base] if prior_base is not None else []):
            set_ortho_n_iters(m, active_ortho_iters)
        # MDSM warmup: pure ranking for first N epochs, then linear ramp-up
        mdsm_warmup = getattr(cfg, "mdsm_warmup_epochs", 0)
        if mdsm_warmup > 0 and epoch_idx < mdsm_warmup:
            mdsm_scale = float(epoch_idx) / float(mdsm_warmup)
        else:
            mdsm_scale = 1.0
        effective_lambda_mdsm = cfg.lambda_mdsm * mdsm_scale
        print(f"  Ortho schedule: n_iters={active_ortho_iters}")
        if mdsm_warmup > 0:
            print(f"  MDSM warmup: scale={mdsm_scale:.3f} effective_lambda={effective_lambda_mdsm:.4f}")
        c1.train(); c2.train(); actor.train(); prior.train() if prior is not None else None
        sums = {
            "loss": 0.0,
            "critic": 0.0,
            "actor": 0.0,
            "rank": 0.0,
            "mdsm": 0.0,
            "cql": 0.0,
            "nce": 0.0,
            "rank_success": 0.0,
            "rank_clean_lt_actor": 0.0,
            "rank_actor_lt_hard": 0.0,
            "rank_clean_lt_hard": 0.0,
            "retrieval_cosine": 0.0,
            "clean_viol": 0.0,
            "actor_barrier": 0.0,
            "actor_descent": 0.0,
            "clean_min": 0.0,
            "direction": 0.0,
            "inbatch_nce": 0.0,
            "support": 0.0,
            "knn_dist": 0.0,
            "energy_reg": 0.0,
            "energy_floor": 0.0,
            "interp_gp": 0.0,
            "e_pos_mean": 0.0,
            "e_actor_mean": 0.0,
            "e_hard_mean": 0.0,
            "e_spread": 0.0,
            "cos_pos_actor": 0.0,
            "cos_pos_hard": 0.0,
            "cos_actor_hard": 0.0,
        }
        n_ok, n_skip, bad_streak = 0, 0, 0
        ema_c, ema_a = None, None

        def on_bad() -> None:
            nonlocal n_skip, bad_streak
            n_skip += 1; bad_streak += 1; opt_c.zero_grad(set_to_none=True); opt_a.zero_grad(set_to_none=True)
            if bad_streak >= cfg.non_finite_backoff_streak_trigger:
                for opt in (opt_c, opt_a):
                    for g in opt.param_groups: g["lr"] = max(float(g["lr"]) * cfg.non_finite_lr_backoff, 1e-8)
            if bad_streak >= cfg.max_consecutive_non_finite_batches:
                raise RuntimeError("Too many consecutive bad batches.")

        for bi, batch_idx in enumerate(loader):
            if (
                cfg.param_finite_check_interval > 0
                and (bi % cfg.param_finite_check_interval == 0)
            ):
                for module in [c1_base, c2_base, actor_base] + ([prior_base] if prior_base is not None else []):
                    for p in module.parameters():
                        if not torch.isfinite(p).all():
                            raise RuntimeError("Non-finite model parameter detected.")
            q_idx_cpu = batch_idx.long()
            q = ds.embeddings[q_idx_cpu].to(device, non_blocking=True)
            q_idx = q_idx_cpu.to(device=device, dtype=torch.long, non_blocking=True)
            pos, hard = retrieve_pos_hard(
                q,
                bank_tr,
                bankn_tr,
                cfg.retrieval_topk_pos,
                cfg.retrieval_hard_start,
                cfg.retrieval_hard_end,
                cfg.retrieval_self_sim_exclude,
                cfg.retrieval_min_pos_similarity,
                q_indices=q_idx,
                bank_indices=bankidx_tr,
                strict_index_exclusion=cfg.retrieval_strict_index_exclusion,
            )
            retrieval_cos = float(F.cosine_similarity(q, pos, dim=-1).mean().item())

            c_loss_acc, rank_acc, mdsm_acc, cql_acc, nce_acc, rank_ok_acc, viol_acc = (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            rank_clean_actor_acc = 0.0
            rank_actor_hard_acc = 0.0
            rank_clean_hard_acc = 0.0
            clean_min_acc = 0.0
            direction_acc = 0.0
            inbatch_nce_acc = 0.0
            energy_reg_acc = 0.0
            energy_floor_acc = 0.0
            interp_gp_acc = 0.0
            e_pos_mean_acc = 0.0
            e_actor_mean_acc = 0.0
            e_hard_mean_acc = 0.0
            e_spread_acc = 0.0
            cos_pos_actor_acc = 0.0
            cos_pos_hard_acc = 0.0
            cos_actor_hard_acc = 0.0
            critic_failed = False
            for cstep in range(max(1, cfg.critic_steps_per_actor)):
                try:
                    crit = c1 if (cstep % 2 == 0) else c2
                    crit_base = c1_base if (cstep % 2 == 0) else c2_base
                    sigma = sample_sigma(cfg, q.shape[0], device)
                    seed = seed_actor(q, hard, sigma, cfg)
                    with torch.no_grad():
                        a_init, _ = actor.predict_step(
                            q,
                            seed,
                            sigma=sigma,
                            step_size=cfg.actor_step_size,
                            target_norm=cfg.langevin.target_norm,
                            tangent_projection=cfg.actor_tangent_projection,
                        )
                    with (nullcontext() if cfg.mdsm_force_fp32 else autocast()):
                        if effective_lambda_mdsm > 0:
                            mdsm = conditional_mdsm(crit, q, pos, sigma, cfg)
                        else:
                            mdsm = torch.tensor(0.0, device=device)
                    with autocast():
                        e_pos = crit(q, pos, sigma=sigma.detach())
                        e_actor = crit(q, a_init.detach(), sigma=sigma.detach())
                        e_hard = crit(q, hard.detach(), sigma=sigma.detach())
                        e_pos_rank, e_actor_rank, e_hard_rank = normalize_triplet_energies(
                            e_pos=e_pos,
                            e_actor=e_actor,
                            e_hard=e_hard,
                            enabled=bool(cfg.rank_normalize_by_std),
                            std_floor=float(cfg.rank_std_floor),
                        )
                        rank = (
                            F.relu(e_pos_rank - e_actor_rank + cfg.critic_margin_clean_actor)
                            + F.relu(e_actor_rank - e_hard_rank + cfg.critic_margin_actor_noisy)
                            + F.relu(e_pos_rank - e_hard_rank + cfg.critic_margin_clean_noisy)
                        ).mean()
                        cql = torch.tensor(0.0, device=device)
                        if cfg.use_cql:
                            ood = add_relative_noise(hard.detach(), cfg.cql_noise_scale)
                            if cfg.langevin.target_norm is not None:
                                ood = F.normalize(ood, dim=-1) * cfg.langevin.target_norm
                            cql = F.softplus(-crit(q, ood, sigma=sigma.detach())).mean()
                        nce = torch.tensor(0.0, device=device)
                        if cfg.use_nce and cfg.lambda_nce > 0:
                            nce = conditional_nce_loss(
                                critic=crit,
                                q=q,
                                pos=pos,
                                hard=hard,
                                bank=bank_tr,
                                sigma=sigma.detach(),
                                cfg=cfg,
                            )
                        prior_nce = torch.tensor(0.0, device=device)
                        if (
                            prior is not None
                            and cfg.use_prior_nce
                            and cfg.lambda_prior_nce > 0
                        ):
                            prior_nce = prior_nce_loss(
                                prior=prior,
                                pos=pos,
                                hard=hard,
                                bank=bank_tr,
                                cfg=cfg,
                            )
                        gp = torch.tensor(0.0, device=device)
                        if cfg.use_gradient_penalty:
                            gp = gradient_penalty(crit, q, a_init.detach(), sigma=sigma.detach())
                        shell = torch.tensor(0.0, device=device)
                        if cfg.use_shell_barrier and cfg.langevin.target_norm is not None:
                            nrm = a_init.norm(dim=-1)
                            r = cfg.langevin.target_norm
                            m = cfg.shell_barrier_margin
                            shell = (
                                F.relu(r * (1 - m) - nrm).pow(2)
                                + F.relu(nrm - r * (1 + m)).pow(2)
                            ).mean()
                        loss_c = (
                            effective_lambda_mdsm * mdsm
                            + cfg.lambda_rank * rank
                            + cfg.lambda_nce * nce
                            + cfg.lambda_cql * cql
                            + cfg.gradient_penalty_lambda * gp
                            + cfg.lambda_shell * shell
                        )
                        if (
                            prior is not None
                            and cfg.use_prior_nce
                            and cfg.lambda_prior_nce > 0
                        ):
                            loss_c = loss_c + cfg.lambda_prior_nce * prior_nce
                        if prior is not None and cfg.lambda_prior > 0:
                            loss_c = loss_c + cfg.lambda_prior * F.relu(
                                prior(pos) - prior(hard.detach()) + cfg.critic_margin_clean_noisy
                            ).mean()

                        # P0.1: Clean-minimum penalty
                        l_clean_min = torch.tensor(0.0, device=device)
                        if cfg.use_clean_min_penalty and cfg.lambda_clean_min > 0:
                            l_clean_min = clean_minimum_penalty(
                                e_clean=e_pos,
                                e_actor=e_actor,
                                margin=cfg.clean_min_margin,
                            )
                            loss_c = loss_c + cfg.lambda_clean_min * l_clean_min

                        # P0.2: Gradient direction loss (auxiliary to MDSM)
                        l_direction = torch.tensor(0.0, device=device)
                        if cfg.use_direction_loss and cfg.lambda_direction > 0:
                            with (nullcontext() if cfg.mdsm_force_fp32 else autocast()):
                                l_direction = gradient_direction_loss(
                                    critic=crit,
                                    q=q,
                                    pos=pos,
                                    sigma=sigma,
                                    cfg=cfg,
                                )
                            loss_c = loss_c + cfg.lambda_direction * l_direction

                        # P1.3: In-batch cross-negative NCE
                        l_inbatch = torch.tensor(0.0, device=device)
                        if cfg.use_inbatch_negatives and cfg.lambda_inbatch_nce > 0:
                            l_inbatch = inbatch_cross_negative_nce(
                                critic=crit,
                                q=q,
                                pos=pos,
                                sigma=sigma,
                                temperature=cfg.inbatch_nce_temperature,
                            )
                            loss_c = loss_c + cfg.lambda_inbatch_nce * l_inbatch

                        # Energy scale regularization — penalize large absolute energies
                        l_energy_reg = torch.tensor(0.0, device=device)
                        if getattr(cfg, 'use_energy_reg', False):
                            if getattr(cfg, 'energy_reg_universal', False):
                                # Universal: penalize all energies, not just clean
                                w = getattr(cfg, 'energy_reg_actor_weight', 2.0)
                                l_energy_reg = (
                                    (e_pos ** 2).mean()
                                    + w * (e_actor ** 2).mean()
                                    + w * (e_hard ** 2).mean()
                                ) / (1.0 + 2.0 * w)
                            else:
                                l_energy_reg = (e_pos ** 2).mean()
                            loss_c = loss_c + cfg.lambda_energy_reg * l_energy_reg

                        # Energy floor: softplus penalty for spurious deep wells
                        l_energy_floor = torch.tensor(0.0, device=device)
                        if getattr(cfg, 'use_energy_floor', False) and getattr(cfg, 'lambda_energy_floor', 0) > 0:
                            threshold = getattr(cfg, 'energy_floor_threshold', 5.0)
                            sharpness = getattr(cfg, 'energy_floor_sharpness', 2.0)
                            # Penalize ALL energies that go below -threshold
                            all_e = torch.cat([e_pos, e_actor, e_hard], dim=0)
                            l_energy_floor = F.softplus((-all_e - threshold) * sharpness).mean()
                            loss_c = loss_c + cfg.lambda_energy_floor * l_energy_floor

                        # Interpolated gradient penalty (WGAN-GP style)
                        l_interp_gp = torch.tensor(0.0, device=device)
                        if getattr(cfg, 'use_interp_gp', False) and getattr(cfg, 'lambda_interp_gp', 0) > 0:
                            alpha = torch.rand(q.shape[0], 1, device=device)
                            interp = alpha * pos.detach() + (1.0 - alpha) * hard.detach()
                            tn = cfg.langevin.target_norm
                            if tn is not None:
                                interp = F.normalize(interp, dim=-1) * tn
                            l_interp_gp = gradient_penalty(crit, q, interp, sigma=sigma.detach())
                            loss_c = loss_c + cfg.lambda_interp_gp * l_interp_gp

                    lc = float(loss_c.detach().item())
                    if (
                        (not math.isfinite(lc))
                        or (
                            cfg.guard_loss_spikes
                            and ema_c is not None
                            and global_step >= cfg.loss_spike_warmup_steps
                            and lc > cfg.loss_spike_factor * max(ema_c, 1e-8)
                        )
                    ):
                        on_bad()
                        critic_failed = True
                        break
                    ema_c = lc if ema_c is None else 0.98 * ema_c + 0.02 * lc
                    mods_c = [crit_base] + ([prior_base] if prior_base is not None else [])
                    opt_c.zero_grad(set_to_none=True)
                    if scaler.is_enabled():
                        scaler.scale(loss_c).backward()
                        scaler.unscale_(opt_c)
                        sanitize_grads(mods_c)
                        gn = clip_grads(mods_c, cfg.clip_grad_norm)
                        if not torch.isfinite(gn):
                            on_bad()
                            scaler.update()
                            critic_failed = True
                            break
                        scaler.step(opt_c)
                        scaler.update()
                    else:
                        loss_c.backward()
                        sanitize_grads(mods_c)
                        gn = clip_grads(mods_c, cfg.clip_grad_norm)
                        if not torch.isfinite(gn):
                            on_bad()
                            critic_failed = True
                            break
                        opt_c.step()

                    c_loss_acc += lc
                    rank_acc += float(rank.item())
                    mdsm_acc += float(mdsm.item())
                    cql_acc += float(cql.item())
                    nce_acc += float(nce.item())
                    clean_min_acc += float(l_clean_min.item())
                    direction_acc += float(l_direction.item())
                    inbatch_nce_acc += float(l_inbatch.item())
                    energy_reg_acc += float(l_energy_reg.item())
                    energy_floor_acc += float(l_energy_floor.item())
                    interp_gp_acc += float(l_interp_gp.item())
                    rank_clean_actor = (e_pos < e_actor).float().mean().item()
                    rank_actor_hard = (e_actor < e_hard).float().mean().item()
                    rank_clean_hard = (e_pos < e_hard).float().mean().item()
                    rank_ok_acc += float(
                        ((e_pos < e_actor) & (e_actor < e_hard) & (e_pos < e_hard)).float().mean().item()
                    )
                    rank_clean_actor_acc += float(rank_clean_actor)
                    rank_actor_hard_acc += float(rank_actor_hard)
                    rank_clean_hard_acc += float(rank_clean_hard)
                    viol_acc += float((e_actor < e_pos).float().mean().item())
                    e_pos_mean_acc += float(e_pos.detach().mean().item())
                    e_actor_mean_acc += float(e_actor.detach().mean().item())
                    e_hard_mean_acc += float(e_hard.detach().mean().item())
                    e_spread_acc += float((e_hard.detach().mean() - e_pos.detach().mean()).item())
                    # Diagnostic: cosine similarity between inputs
                    cos_pos_actor = F.cosine_similarity(pos, a_init.detach(), dim=-1).mean().item()
                    cos_pos_hard = F.cosine_similarity(pos, hard.detach(), dim=-1).mean().item()
                    cos_actor_hard = F.cosine_similarity(a_init.detach(), hard.detach(), dim=-1).mean().item()
                    cos_pos_actor_acc += cos_pos_actor
                    cos_pos_hard_acc += cos_pos_hard
                    cos_actor_hard_acc += cos_actor_hard
                except torch.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    on_bad()
                    critic_failed = True
                    break
            if critic_failed:
                continue

            try:
                sigma_a = sample_sigma(cfg, q.shape[0], device)
                seed_a = seed_actor(q, hard, sigma_a, cfg)
                with autocast():
                    nxt, delta = actor.predict_step(
                        q,
                        seed_a,
                        sigma=sigma_a.detach(),
                        step_size=cfg.actor_step_size,
                        target_norm=cfg.langevin.target_norm,
                        tangent_projection=cfg.actor_tangent_projection,
                    )
                    l_geo = (1.0 - F.cosine_similarity(nxt, pos, dim=-1).clamp(-1.0, 1.0)).mean()
                    l_bc = F.mse_loss(nxt, pos) if cfg.use_bc else torch.tensor(0.0, device=device)
                    if cfg.use_grad_align:
                        with torch.enable_grad():
                            _, g_seed = ef.energy_and_grad(q, seed_a, sigma=sigma_a.detach())
                        if cfg.actor_tangent_projection and cfg.langevin.target_norm is not None:
                            seed_hat = F.normalize(seed_a, dim=-1)
                            g_seed = g_seed - (g_seed * seed_hat).sum(dim=-1, keepdim=True) * seed_hat
                        l_align = (
                            1.0
                            - F.cosine_similarity(
                                delta, -g_seed, dim=-1, eps=cfg.mdsm_cosine_eps
                            ).clamp(-1.0, 1.0)
                        ).mean()
                    else:
                        l_align = torch.tensor(0.0, device=device)
                    e_next = ef(q, nxt, sigma=sigma_a.detach())
                    if cfg.langevin.target_norm is not None:
                        seed_ref = F.normalize(seed_a, dim=-1) * cfg.langevin.target_norm
                    else:
                        seed_ref = seed_a
                    e_seed = ef(q, seed_ref, sigma=sigma_a.detach()).detach()
                    e_pos_ref = ef(q, pos, sigma=sigma_a.detach()).detach()
                    e_hard_ref = ef(q, hard.detach(), sigma=sigma_a.detach()).detach()
                    e_pos_bar, e_next_bar, e_hard_bar = normalize_triplet_energies(
                        e_pos=e_pos_ref,
                        e_actor=e_next,
                        e_hard=e_hard_ref,
                        enabled=bool(cfg.actor_barrier_normalize_by_std),
                        std_floor=float(cfg.rank_std_floor),
                    )

                    l_bar = torch.tensor(0.0, device=device)
                    if cfg.lambda_actor_barrier > 0:
                        # Keep actor proposal in a valid critic band:
                        # E(pos)+m_pos <= E(actor) <= E(hard)-m_hard
                        low = F.relu((e_pos_bar + cfg.actor_energy_margin_pos) - e_next_bar)
                        high = F.relu(e_next_bar - (e_hard_bar - cfg.actor_energy_margin_hard))
                        l_bar = (low + high).mean()

                    l_desc = torch.tensor(0.0, device=device)
                    if cfg.lambda_actor_descent > 0:
                        # Actor step should not raise energy from its own seed.
                        l_desc = F.relu(e_next - e_seed).mean()

                    # P1.2: kNN support/manifold proximity penalty
                    l_support = torch.tensor(0.0, device=device)
                    knn_dist_val = 0.0
                    if cfg.use_support_penalty and cfg.lambda_support > 0 and support_threshold is not None:
                        l_support, knn_dist_val = knn_support_penalty(
                            actor_output=nxt,
                            bank=bank_tr,
                            k=cfg.support_k,
                            threshold=support_threshold,
                        )

                    loss_a = (
                        cfg.lambda_geo * l_geo
                        + cfg.lambda_align * l_align
                        + cfg.lambda_bc_reg * l_bc
                        + cfg.lambda_actor_barrier * l_bar
                        + cfg.lambda_actor_descent * l_desc
                        + cfg.lambda_support * l_support
                    )
                la = float(loss_a.detach().item())
                if (
                    (not math.isfinite(la))
                    or (
                        cfg.guard_loss_spikes
                        and ema_a is not None
                        and global_step >= cfg.loss_spike_warmup_steps
                        and la > cfg.loss_spike_factor * max(ema_a, 1e-8)
                    )
                ):
                    on_bad()
                    continue
                ema_a = la if ema_a is None else 0.98 * ema_a + 0.02 * la
                opt_a.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(loss_a).backward()
                    scaler.unscale_(opt_a)
                    sanitize_grads([actor_base])
                    gn = clip_grads([actor_base], cfg.clip_grad_norm)
                    if not torch.isfinite(gn):
                        on_bad()
                        scaler.update()
                        continue
                    scaler.step(opt_a)
                    scaler.update()
                else:
                    loss_a.backward()
                    sanitize_grads([actor_base])
                    gn = clip_grads([actor_base], cfg.clip_grad_norm)
                    if not torch.isfinite(gn):
                        on_bad()
                        continue
                    opt_a.step()
            except torch.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                on_bad()
                continue
            n_ok += 1; global_step += 1; bad_streak = 0
            c_avg = c_loss_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["critic"] += c_avg; sums["actor"] += la; sums["loss"] += c_avg + la
            sums["rank"] += rank_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["mdsm"] += mdsm_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["cql"] += cql_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["nce"] += nce_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["rank_success"] += rank_ok_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["rank_clean_lt_actor"] += rank_clean_actor_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["rank_actor_lt_hard"] += rank_actor_hard_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["rank_clean_lt_hard"] += rank_clean_hard_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["clean_viol"] += viol_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["retrieval_cosine"] += retrieval_cos
            sums["actor_barrier"] += float(l_bar.item())
            sums["actor_descent"] += float(l_desc.item())
            sums["clean_min"] += clean_min_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["direction"] += direction_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["inbatch_nce"] += inbatch_nce_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["support"] += float(l_support.item())
            sums["knn_dist"] += knn_dist_val
            sums["energy_reg"] += energy_reg_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["energy_floor"] += energy_floor_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["interp_gp"] += interp_gp_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["e_pos_mean"] += e_pos_mean_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["e_actor_mean"] += e_actor_mean_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["e_hard_mean"] += e_hard_mean_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["e_spread"] += e_spread_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["cos_pos_actor"] += cos_pos_actor_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["cos_pos_hard"] += cos_pos_hard_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["cos_actor_hard"] += cos_actor_hard_acc / float(max(1, cfg.critic_steps_per_actor))
            if cfg.log_every > 0 and (bi + 1) % cfg.log_every == 0 and n_ok > 0:
                now = time.perf_counter()
                window_batches = max(1, (bi + 1) - last_window_batch)
                sec_per_batch = (now - last_window_time) / float(window_batches)
                eta_epoch_sec = max(0.0, float(len(loader) - (bi + 1)) * sec_per_batch)
                last_window_time = now
                last_window_batch = bi + 1
                print(
                    f"  [{bi+1}/{len(loader)}] "
                    f"loss={sums['loss']/n_ok:.4f} "
                    f"critic={sums['critic']/n_ok:.4f} "
                    f"actor={sums['actor']/n_ok:.4f} "
                    f"rank_loss={sums['rank']/n_ok:.4f} "
                    f"mdsm={sums['mdsm']/n_ok:.4f} "
                    f"rank_success={sums['rank_success']/n_ok:.3f} "
                    f"rank(c<a)={sums['rank_clean_lt_actor']/n_ok:.3f} "
                    f"rank(a<h)={sums['rank_actor_lt_hard']/n_ok:.3f} "
                    f"rank(c<h)={sums['rank_clean_lt_hard']/n_ok:.3f} "
                    f"viol={sums['clean_viol']/n_ok:.3f} "
                    f"a_bar={sums['actor_barrier']/n_ok:.3f} "
                    f"a_desc={sums['actor_descent']/n_ok:.3f} "
                    f"cmin={sums['clean_min']/n_ok:.3f} "
                    f"dir={sums['direction']/n_ok:.3f} "
                    f"ibnce={sums['inbatch_nce']/n_ok:.3f} "
                    f"supp={sums['support']/n_ok:.4f} "
                    f"ereg={sums['energy_reg']/n_ok:.3f} "
                    f"efloor={sums['energy_floor']/n_ok:.3f} "
                    f"igp={sums['interp_gp']/n_ok:.3f} "
                    f"E[c/a/h]={sums['e_pos_mean']/n_ok:.2f}/{sums['e_actor_mean']/n_ok:.2f}/{sums['e_hard_mean']/n_ok:.2f} "
                    f"spread={sums['e_spread']/n_ok:.3f} "
                    f"cos(p/a)={cos_pos_actor:.3f} "
                    f"cos(p/h)={cos_pos_hard:.3f} "
                    f"cos(a/h)={cos_actor_hard:.3f} "
                    f"sec/batch={sec_per_batch:.3f} "
                    f"eta={eta_epoch_sec/60.0:.1f}m"
                )
                append_jsonl_record(
                    stream_path,
                    {
                        "event": "batch",
                        "epoch": int(epoch_idx + 1),
                        "batch_idx": int(bi + 1),
                        "num_batches": int(len(loader)),
                        "global_step": int(global_step),
                        "sec_per_batch_window": float(sec_per_batch),
                        "eta_epoch_sec": float(eta_epoch_sec),
                        "train_metrics": {
                            "loss": float(sums["loss"] / n_ok),
                            "critic": float(sums["critic"] / n_ok),
                            "actor": float(sums["actor"] / n_ok),
                            "rank_loss": float(sums["rank"] / n_ok),
                            "rank_success": float(sums["rank_success"] / n_ok),
                            "rank_clean_lt_actor": float(sums["rank_clean_lt_actor"] / n_ok),
                            "rank_actor_lt_hard": float(sums["rank_actor_lt_hard"] / n_ok),
                            "rank_clean_lt_hard": float(sums["rank_clean_lt_hard"] / n_ok),
                            "clean_viol": float(sums["clean_viol"] / n_ok),
                            "retrieval_cosine": float(sums["retrieval_cosine"] / n_ok),
                            "actor_barrier": float(sums["actor_barrier"] / n_ok),
                            "actor_descent": float(sums["actor_descent"] / n_ok),
                            "mdsm": float(sums["mdsm"] / n_ok),
                            "nce": float(sums["nce"] / n_ok),
                            "cql": float(sums["cql"] / n_ok),
                            "clean_min": float(sums["clean_min"] / n_ok),
                            "direction": float(sums["direction"] / n_ok),
                            "inbatch_nce": float(sums["inbatch_nce"] / n_ok),
                            "support": float(sums["support"] / n_ok),
                            "knn_dist": float(sums["knn_dist"] / n_ok),
                            "energy_reg": float(sums["energy_reg"] / n_ok),
                            "interp_gp": float(sums["interp_gp"] / n_ok),
                            "energy_floor": float(sums["energy_floor"] / n_ok),
                            "e_pos_mean": float(sums["e_pos_mean"] / n_ok),
                            "e_actor_mean": float(sums["e_actor_mean"] / n_ok),
                            "e_hard_mean": float(sums["e_hard_mean"] / n_ok),
                            "e_spread": float(sums["e_spread"] / n_ok),
                            "cos_pos_actor": float(sums["cos_pos_actor"] / n_ok),
                            "cos_pos_hard": float(sums["cos_pos_hard"] / n_ok),
                            "cos_actor_hard": float(sums["cos_actor_hard"] / n_ok),
                            "skip_rate": float(n_skip / max(len(loader), 1)),
                        },
                    },
                )

        train = {k: (v / max(n_ok, 1)) for k, v in sums.items()}
        train["skip_rate"] = n_skip / max(len(loader), 1)
        do_eval = ((epoch_idx + 1) % max(1, cfg.eval_every_epochs) == 0) or (epoch_idx == cfg.num_epochs - 1)
        if do_eval:
            eval_m = eval_model(
                ef,
                actor,
                ds_val,
                bank_va,
                bankn_va,
                bankidx_va,
                cfg,
                device,
                eval_ids=eval_ids,
            )
            kill = summarize_conditional_eval(eval_m, th)
            eval_m["kill_criteria"] = kill
            score = float(kill["score"])
            print(
                f"Epoch {epoch_idx+1}: train={train['loss']:.4f} score={score:+.6f} "
                f"strict_pass={kill['passed']} skip={train['skip_rate']:.2%} ({time.time()-t0:.1f}s)"
            )
        else:
            eval_m = {"status": "not_evaluated"}
            kill = _not_evaluated_kill_stub()
            score = None
            print(
                f"Epoch {epoch_idx+1}: train={train['loss']:.4f} "
                f"skip={train['skip_rate']:.2%} eval=skipped ({time.time()-t0:.1f}s)"
            )
        is_new_best = bool(do_eval and score is not None and score > best_score)
        if is_new_best:
            best_score = score
        epoch_time_sec = float(time.perf_counter() - epoch_timer_start)

        # LR scheduler step + plateau boost
        if sched_c is not None:
            sched_c.step()
            sched_a.step()
            cur_lr = opt_c.param_groups[0]["lr"]
            print(f"  LR: {cur_lr:.6f}")
        plat_patience = getattr(cfg, "lr_plateau_patience", 0)
        if plat_patience > 0 and do_eval and score is not None:
            if is_new_best:
                plateau_counter = 0
            else:
                plateau_counter += 1
            if plateau_counter >= plat_patience:
                boost = getattr(cfg, "lr_plateau_boost", 3.0)
                for opt in [opt_c, opt_a]:
                    for pg in opt.param_groups:
                        pg["lr"] = min(pg["lr"] * boost, cfg.critic_lr * 2.0)
                plateau_counter = 0
                print(f"  Plateau boost! LR *= {boost} → {opt_c.param_groups[0]['lr']:.6f}")

        payload = {
            "epoch": epoch_idx, "global_step": global_step, "best_score": best_score,
            "critic1_state": c1_base.state_dict(), "critic2_state": c2_base.state_dict(), "actor_state": actor_base.state_dict(),
            "prior_state": prior_base.state_dict() if prior_base is not None else None,
            "opt_c_state": opt_c.state_dict(), "opt_a_state": opt_a.state_dict(), "scaler_state": scaler.state_dict(),
            "sched_c_state": sched_c.state_dict() if sched_c is not None else None,
            "sched_a_state": sched_a.state_dict() if sched_a is not None else None,
            "plateau_counter": plateau_counter,
            "train_metrics": train, "eval_metrics": eval_m, "config": asdict(cfg),
        }
        current_epoch_1b = int(epoch_idx + 1)
        checkpoint_period = max(1, int(cfg.checkpoint_every_epochs))
        is_periodic_checkpoint = (current_epoch_1b % checkpoint_period) == 0
        is_last_epoch = current_epoch_1b == int(cfg.num_epochs)
        if is_periodic_checkpoint or is_last_epoch:
            torch.save(payload, ckpt_dir / f"epoch_{current_epoch_1b}.pt")

        # Rolling checkpoint for live landscape inspection.
        rolling_ckpt = ckpt_dir / str(cfg.rolling_checkpoint_name)
        torch.save(payload, rolling_ckpt)

        # Cleanup stale non-periodic epoch checkpoints (from legacy runs).
        epoch_file_current = ckpt_dir / f"epoch_{current_epoch_1b}.pt"
        if (not is_periodic_checkpoint) and (not is_last_epoch) and epoch_file_current.exists():
            epoch_file_current.unlink(missing_ok=True)

        prev_epoch = current_epoch_1b - 1
        if prev_epoch > 0 and (prev_epoch % checkpoint_period) != 0:
            prev_epoch_file = ckpt_dir / f"epoch_{prev_epoch}.pt"
            if prev_epoch_file.exists():
                prev_epoch_file.unlink(missing_ok=True)

        if is_new_best:
            payload["best_score"] = best_score
            torch.save(payload, ckpt_dir / "best.pt")
        append_jsonl_record(
            stream_path,
            {
                "event": "epoch",
                "epoch": int(epoch_idx + 1),
                "global_step": int(global_step),
                "score": None if score is None else float(score),
                "best_score": float(best_score),
                "eval_ran": bool(do_eval),
                "strict_pass": None if kill.get("passed", None) is None else bool(kill.get("passed")),
                "train_metrics": {k: float(v) for k, v in train.items()},
                "kill_criteria": kill,
                "epoch_time_sec": epoch_time_sec,
            },
        )

    final_eval = eval_model(
        ef,
        actor,
        ds_val,
        bank_va,
        bankn_va,
        bankidx_va,
        cfg,
        device,
        eval_ids=eval_ids,
    )
    final_kill = summarize_conditional_eval(final_eval, th); final_eval["kill_criteria"] = final_kill
    append_jsonl_record(
        stream_path,
        {
            "event": "final",
            "epoch": int(cfg.num_epochs),
            "best_score": float(best_score),
            "final_kill": final_kill,
        },
    )
    torch.save(
        {
            "critic1_state": c1_base.state_dict(), "critic2_state": c2_base.state_dict(), "actor_state": actor_base.state_dict(),
            "prior_state": prior_base.state_dict() if prior_base is not None else None,
            "eval_metrics": final_eval, "config": asdict(cfg), "best_score": best_score,
        },
        ckpt_dir / "final.pt",
    )
    with open(log_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump({"best_score": best_score, "final_kill": final_kill, "config": asdict(cfg)}, f, indent=2)
    print(f"Final strict pass: {final_kill['passed']} | best_score={best_score:+.6f}")


if __name__ == "__main__":
    main()
