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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cebcm.data.dataset import SONARVectorDataset
from cebcm.inference.langevin import run_langevin
from cebcm.models.actor import LatentDenoiseActor
from cebcm.models.energy import SimpleEnergy
from cebcm.models.energy_unconditional import UnconditionalEnergy
from cebcm.models.normalization import OrthoLinear
from cebcm.training.kill_criteria import ConditionalThresholds, summarize_conditional_eval
from cebcm.training.losses import gradient_penalty
from configs.base import LangevinConfig, Stage1_5Config


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
    for module in model.modules():
        if isinstance(module, OrthoLinear):
            module.n_iters = n_iters


def build_bank(ds: SONARVectorDataset, device: torch.device, bank_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    emb = ds.embeddings
    if 0 < bank_size < len(emb):
        idx = torch.randperm(len(emb))[:bank_size]
        emb = emb[idx]
    bank = emb.to(device)
    bank_n = F.normalize(bank, dim=-1)
    return bank, bank_n


def retrieve_pos_hard(
    q: torch.Tensor,
    bank: torch.Tensor,
    bank_n: torch.Tensor,
    topk_pos: int,
    hard_start: int,
    hard_end: int,
    self_sim_exclude: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    qn = F.normalize(q, dim=-1)
    k = min(bank.shape[0], max(2, topk_pos, hard_end))
    sim = qn @ bank_n.T
    vals, idx = torch.topk(sim, k=k, dim=-1)
    pos_ids, hard_ids = [], []
    for r in range(q.shape[0]):
        cand_idx, cand_val = idx[r], vals[r]
        pos = int(cand_idx[0].item())
        for j in range(k):
            if float(cand_val[j].item()) < self_sim_exclude:
                pos = int(cand_idx[j].item())
                break
        hs, he = min(max(0, hard_start), k - 1), min(max(hard_start + 1, hard_end), k)
        win = cand_idx[hs:he]
        if win.numel() == 0:
            hard = int(cand_idx[-1].item())
        else:
            hard = int(win[torch.randint(0, win.numel(), (1,), device=q.device)].item())
        if hard == pos:
            for j in range(k - 1, -1, -1):
                alt = int(cand_idx[j].item())
                if alt != pos:
                    hard = alt
                    break
        pos_ids.append(pos)
        hard_ids.append(hard)
    return bank[torch.tensor(pos_ids, device=q.device)], bank[torch.tensor(hard_ids, device=q.device)]


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
    e = critic(q, noisy_req, sigma=sigma.detach())
    g = torch.autograd.grad(e.sum(), noisy_req, create_graph=True)[0]
    tgt = (noisy.detach() - pos) / sigma_eff_sq
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
    else:
        w = torch.ones_like(loss)
    w = w / w.mean().clamp(min=1e-8)
    return (w * loss).mean()


class TwinHybridEnergy:
    def __init__(
        self,
        c1: SimpleEnergy,
        c2: SimpleEnergy,
        prior: UnconditionalEnergy | None,
        lambda_prior: float,
        aggregate: str,
    ):
        self.c1 = c1
        self.c2 = c2
        self.prior = prior
        self.lambda_prior = lambda_prior
        self.aggregate = aggregate

    def cond(self, q: torch.Tensor, v: torch.Tensor, sigma: torch.Tensor | None = None) -> torch.Tensor:
        e1, e2 = self.c1(q, v, sigma=sigma), self.c2(q, v, sigma=sigma)
        return 0.5 * (e1 + e2) if self.aggregate == "mean" else torch.maximum(e1, e2)

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
    cfg: Stage1_5Config,
    device: torch.device,
) -> dict:
    ef.c1.eval()
    ef.c2.eval()
    if ef.prior is not None:
        ef.prior.eval()
    actor.eval()
    ids = torch.randperm(len(ds))[: min(cfg.eval_num_samples, len(ds))]
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
    for ns in cfg.eval_noise_scales:
        cb, ca, lb, la, eb, ea, ep, step, succ, viol, rcos = [], [], [], [], [], [], [], [], [], [], []
        for i in ids:
            q = ds[int(i.item())].unsqueeze(0).to(device)
            pos, hard = retrieve_pos_hard(
                q, bank, bank_n, cfg.retrieval_topk_pos, cfg.retrieval_hard_start, cfg.retrieval_hard_end, cfg.retrieval_self_sim_exclude
            )
            rcos.append(float(F.cosine_similarity(q, pos, dim=-1).item()))
            sigma = torch.full((1, 1), float(ns), device=device)
            noisy = seed_actor(q, hard, sigma, cfg)
            with torch.no_grad():
                v = noisy
                for _ in range(max(1, cfg.actor_eval_steps)):
                    v, _ = actor.predict_step(q, v, sigma=sigma, step_size=cfg.actor_step_size, target_norm=cfg.langevin.target_norm, tangent_projection=cfg.actor_tangent_projection)
            if cfg.critic_eval_langevin_steps > 0:
                res = run_langevin(
                    method=cfg.langevin.method, energy_fn=ef, v_query=q, v_init=v,
                    lr=cfg.langevin.lr, noise_scale=cfg.langevin.noise_scale, max_steps=cfg.critic_eval_langevin_steps,
                    target_norm=cfg.langevin.target_norm, energy_threshold=cfg.langevin.energy_threshold,
                    plateau_patience=cfg.langevin.plateau_patience, plateau_delta=cfg.langevin.plateau_delta, v_target=pos, **kw
                )
                final = res.v_last if res.v_last is not None else res.v_final
            else:
                final = v
            cb.append(float(F.cosine_similarity(pos, noisy, dim=-1).item()))
            ca.append(float(F.cosine_similarity(pos, final, dim=-1).item()))
            lb.append(float(torch.norm(pos - noisy, dim=-1).item()))
            la.append(float(torch.norm(pos - final, dim=-1).item()))
            with torch.no_grad():
                ep_i = float(ef(q, pos).item()); eb_i = float(ef(q, noisy).item()); ea_i = float(ef(q, final).item())
            ep.append(ep_i); eb.append(eb_i); ea.append(ea_i)
            step.append(float(torch.norm(final - noisy, dim=-1).item()))
            succ.append(1.0 if ea_i < eb_i else 0.0)
            viol.append(1.0 if ea_i < ep_i else 0.0)
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
            "success_rate": sum(1.0 for b, a in zip(cb, ca) if a > b) / n,
            "retrieval_cosine_mean": sum(rcos) / n,
        }
    return out


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
    return cfg


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
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))
    bank_tr, bankn_tr = build_bank(ds, device, cfg.retrieval_bank_size)
    bank_va, bankn_va = build_bank(ds_val, device, cfg.retrieval_bank_size)

    c1 = SimpleEnergy(cfg.energy_dim, cfg.energy_hidden_dims, cfg.norm_mode, cfg.activation, ortho_n_iters=cfg.ortho_n_iters, energy_output_clamp=None).to(device)
    c2 = SimpleEnergy(cfg.energy_dim, cfg.energy_hidden_dims, cfg.norm_mode, cfg.activation, ortho_n_iters=cfg.ortho_n_iters, energy_output_clamp=None).to(device)
    actor = LatentDenoiseActor(cfg.energy_dim, cfg.actor_hidden_dims, cfg.norm_mode, "silu", ortho_n_iters=cfg.ortho_n_iters).to(device)
    prior = UnconditionalEnergy(cfg.energy_dim, cfg.energy_hidden_dims, cfg.norm_mode, cfg.activation, ortho_n_iters=cfg.ortho_n_iters).to(device) if cfg.use_prior_critic else None
    for m in [c1, c2, actor] + ([prior] if prior is not None else []):
        set_ortho_n_iters(m, cfg.ortho_n_iters)
    ef = TwinHybridEnergy(c1, c2, prior, cfg.lambda_prior if cfg.use_prior_critic else 0.0, cfg.twin_aggregate)

    groups = [{"params": list(c1.parameters()), "lr": cfg.critic_lr}, {"params": list(c2.parameters()), "lr": cfg.critic_lr}]
    if prior is not None:
        groups.append({"params": list(prior.parameters()), "lr": cfg.prior_critic_lr})
    opt_c = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    opt_a = torch.optim.AdamW(actor.parameters(), lr=cfg.actor_lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ckpt_dir, log_dir = Path(cfg.checkpoint_dir), Path(cfg.logs_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)
    best_score, start_epoch, global_step = float("-inf"), 0, 0
    if args.resume:
        ck = torch.load(args.resume, weights_only=False, map_location=device)
        c1.load_state_dict(ck["critic1_state"]); c2.load_state_dict(ck["critic2_state"]); actor.load_state_dict(ck["actor_state"])
        if prior is not None and ck.get("prior_state") is not None: prior.load_state_dict(ck["prior_state"])
        opt_c.load_state_dict(ck["opt_c_state"]); opt_a.load_state_dict(ck["opt_a_state"]); scaler.load_state_dict(ck["scaler_state"])
        best_score, start_epoch, global_step = float(ck.get("best_score", best_score)), int(ck.get("epoch", 0)) + 1, int(ck.get("global_step", 0))

    th = make_thresholds(cfg)
    print(f"Device: {device} | Train: {len(ds)} | Val: {len(ds_val)}")
    for ep in range(start_epoch, cfg.num_epochs):
        t0 = time.time(); c1.train(); c2.train(); actor.train(); prior.train() if prior is not None else None
        sums = {"loss": 0.0, "critic": 0.0, "actor": 0.0, "rank_success": 0.0, "retrieval_cosine": 0.0, "clean_viol": 0.0}
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

        for bi, q in enumerate(loader):
            q = q.to(device, non_blocking=True)
            pos, hard = retrieve_pos_hard(q, bank_tr, bankn_tr, cfg.retrieval_topk_pos, cfg.retrieval_hard_start, cfg.retrieval_hard_end, cfg.retrieval_self_sim_exclude)
            retrieval_cos = float(F.cosine_similarity(q, pos, dim=-1).mean().item())

            c_loss_acc, rank_ok_acc, viol_acc = 0.0, 0.0, 0.0
            critic_failed = False
            for _ in range(max(1, cfg.critic_steps_per_actor)):
                sigma = sample_sigma(cfg, q.shape[0], device); seed = seed_actor(q, hard, sigma, cfg)
                with torch.no_grad():
                    a_init, _ = actor.predict_step(q, seed, sigma=sigma, step_size=cfg.actor_step_size, target_norm=cfg.langevin.target_norm, tangent_projection=cfg.actor_tangent_projection)
                with (nullcontext() if cfg.mdsm_force_fp32 else autocast()):
                    mdsm = 0.5 * (conditional_mdsm(c1, q, pos, sigma, cfg) + conditional_mdsm(c2, q, pos, sigma, cfg))
                with autocast():
                    ep1, ea1, eh1 = c1(q, pos, sigma=sigma.detach()), c1(q, a_init.detach(), sigma=sigma.detach()), c1(q, hard.detach(), sigma=sigma.detach())
                    ep2, ea2, eh2 = c2(q, pos, sigma=sigma.detach()), c2(q, a_init.detach(), sigma=sigma.detach()), c2(q, hard.detach(), sigma=sigma.detach())
                    rank = 0.5 * (
                        (F.relu(ep1 - ea1 + cfg.critic_margin_clean_actor) + F.relu(ea1 - eh1 + cfg.critic_margin_actor_noisy) + F.relu(ep1 - eh1 + cfg.critic_margin_clean_noisy)).mean()
                        + (F.relu(ep2 - ea2 + cfg.critic_margin_clean_actor) + F.relu(ea2 - eh2 + cfg.critic_margin_actor_noisy) + F.relu(ep2 - eh2 + cfg.critic_margin_clean_noisy)).mean()
                    )
                    cql = torch.tensor(0.0, device=device)
                    if cfg.use_cql:
                        ood = add_relative_noise(hard.detach(), cfg.cql_noise_scale)
                        if cfg.langevin.target_norm is not None: ood = F.normalize(ood, dim=-1) * cfg.langevin.target_norm
                        cql = 0.5 * (F.softplus(-c1(q, ood, sigma=sigma.detach())).mean() + F.softplus(-c2(q, ood, sigma=sigma.detach())).mean())
                    gp = torch.tensor(0.0, device=device)
                    if cfg.use_gradient_penalty:
                        gp = 0.5 * (gradient_penalty(c1, q, a_init.detach()) + gradient_penalty(c2, q, a_init.detach()))
                    loss_c = cfg.lambda_mdsm * mdsm + cfg.lambda_rank * rank + cfg.lambda_cql * cql + cfg.gradient_penalty_lambda * gp
                    if prior is not None and cfg.lambda_prior > 0:
                        loss_c = loss_c + cfg.lambda_prior * F.relu(prior(pos) - prior(hard.detach()) + cfg.critic_margin_clean_noisy).mean()

                lc = float(loss_c.detach().item())
                if (not math.isfinite(lc)) or (cfg.guard_loss_spikes and ema_c is not None and global_step >= cfg.loss_spike_warmup_steps and lc > cfg.loss_spike_factor * max(ema_c, 1e-8)):
                    on_bad(); critic_failed = True; break
                ema_c = lc if ema_c is None else 0.98 * ema_c + 0.02 * lc
                mods_c = [c1, c2] + ([prior] if prior is not None else [])
                opt_c.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(loss_c).backward(); scaler.unscale_(opt_c); sanitize_grads(mods_c); gn = clip_grads(mods_c, cfg.clip_grad_norm)
                    if not torch.isfinite(gn): on_bad(); scaler.update(); critic_failed = True; break
                    scaler.step(opt_c); scaler.update()
                else:
                    loss_c.backward(); sanitize_grads(mods_c); gn = clip_grads(mods_c, cfg.clip_grad_norm)
                    if not torch.isfinite(gn): on_bad(); critic_failed = True; break
                    opt_c.step()
                c_loss_acc += lc
                rank_ok_acc += 0.5 * (((ep1 < eh1).float().mean().item()) + ((ep2 < eh2).float().mean().item()))
                viol_acc += 0.5 * (((ea1 < ep1).float().mean().item()) + ((ea2 < ep2).float().mean().item()))
            if critic_failed:
                continue

            sigma_a = sample_sigma(cfg, q.shape[0], device); seed_a = seed_actor(q, hard, sigma_a, cfg)
            with autocast():
                nxt, delta = actor.predict_step(q, seed_a, sigma=sigma_a.detach(), step_size=cfg.actor_step_size, target_norm=cfg.langevin.target_norm, tangent_projection=cfg.actor_tangent_projection)
                l_geo = (1.0 - F.cosine_similarity(nxt, pos, dim=-1).clamp(-1.0, 1.0)).mean()
                l_bc = F.mse_loss(nxt, pos) if cfg.use_bc else torch.tensor(0.0, device=device)
                if cfg.use_grad_align:
                    with torch.enable_grad():
                        _, g_seed = ef.energy_and_grad(q, seed_a, sigma=sigma_a.detach())
                    l_align = (1.0 - F.cosine_similarity(delta, -g_seed, dim=-1, eps=cfg.mdsm_cosine_eps).clamp(-1.0, 1.0)).mean()
                else:
                    l_align = torch.tensor(0.0, device=device)
                l_bar = torch.tensor(0.0, device=device)
                if cfg.lambda_actor_barrier > 0:
                    l_bar = F.softplus(ef(q, pos, sigma=sigma_a.detach()).detach() - ef(q, nxt, sigma=sigma_a.detach())).mean()
                loss_a = cfg.lambda_geo * l_geo + cfg.lambda_align * l_align + cfg.lambda_bc_reg * l_bc + cfg.lambda_actor_barrier * l_bar
            la = float(loss_a.detach().item())
            if (not math.isfinite(la)) or (cfg.guard_loss_spikes and ema_a is not None and global_step >= cfg.loss_spike_warmup_steps and la > cfg.loss_spike_factor * max(ema_a, 1e-8)):
                on_bad(); continue
            ema_a = la if ema_a is None else 0.98 * ema_a + 0.02 * la
            opt_a.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss_a).backward(); scaler.unscale_(opt_a); sanitize_grads([actor]); gn = clip_grads([actor], cfg.clip_grad_norm)
                if not torch.isfinite(gn): on_bad(); scaler.update(); continue
                scaler.step(opt_a); scaler.update()
            else:
                loss_a.backward(); sanitize_grads([actor]); gn = clip_grads([actor], cfg.clip_grad_norm)
                if not torch.isfinite(gn): on_bad(); continue
                opt_a.step()
            n_ok += 1; global_step += 1; bad_streak = 0
            c_avg = c_loss_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["critic"] += c_avg; sums["actor"] += la; sums["loss"] += c_avg + la
            sums["rank_success"] += rank_ok_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["clean_viol"] += viol_acc / float(max(1, cfg.critic_steps_per_actor))
            sums["retrieval_cosine"] += retrieval_cos
            if cfg.log_every > 0 and (bi + 1) % cfg.log_every == 0 and n_ok > 0:
                print(f"  [{bi+1}/{len(loader)}] loss={sums['loss']/n_ok:.4f} critic={sums['critic']/n_ok:.4f} actor={sums['actor']/n_ok:.4f} rank={sums['rank_success']/n_ok:.3f} viol={sums['clean_viol']/n_ok:.3f}")

        train = {k: (v / max(n_ok, 1)) for k, v in sums.items()}; train["skip_rate"] = n_skip / max(len(loader), 1)
        eval_m = eval_model(ef, actor, ds_val, bank_va, bankn_va, cfg, device)
        kill = summarize_conditional_eval(eval_m, th); eval_m["kill_criteria"] = kill; score = float(kill["score"])
        print(f"Epoch {ep+1}: train={train['loss']:.4f} score={score:+.6f} strict_pass={kill['passed']} skip={train['skip_rate']:.2%} ({time.time()-t0:.1f}s)")
        payload = {
            "epoch": ep, "global_step": global_step, "best_score": best_score,
            "critic1_state": c1.state_dict(), "critic2_state": c2.state_dict(), "actor_state": actor.state_dict(),
            "prior_state": prior.state_dict() if prior is not None else None,
            "opt_c_state": opt_c.state_dict(), "opt_a_state": opt_a.state_dict(), "scaler_state": scaler.state_dict(),
            "train_metrics": train, "eval_metrics": eval_m, "config": asdict(cfg),
        }
        torch.save(payload, ckpt_dir / f"epoch_{ep+1}.pt")
        if score > best_score:
            best_score = score; payload["best_score"] = best_score; torch.save(payload, ckpt_dir / "best.pt")
        with open(log_dir / "training_metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": ep + 1, "score": score, "train_metrics": train, "kill_criteria": kill}) + "\n")

    final_eval = eval_model(ef, actor, ds_val, bank_va, bankn_va, cfg, device)
    final_kill = summarize_conditional_eval(final_eval, th); final_eval["kill_criteria"] = final_kill
    torch.save(
        {
            "critic1_state": c1.state_dict(), "critic2_state": c2.state_dict(), "actor_state": actor.state_dict(),
            "prior_state": prior.state_dict() if prior is not None else None,
            "eval_metrics": final_eval, "config": asdict(cfg), "best_score": best_score,
        },
        ckpt_dir / "final.pt",
    )
    with open(log_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump({"best_score": best_score, "final_kill": final_kill, "config": asdict(cfg)}, f, indent=2)
    print(f"Final strict pass: {final_kill['passed']} | best_score={best_score:+.6f}")


if __name__ == "__main__":
    main()
