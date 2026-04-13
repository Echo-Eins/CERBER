"""
ChainGenerator Diagnostics — GUI backend for autoregressive QA analysis.

Provides:
  - Model loading (ChainGenerator, CompositeCritic, SONAR)
  - Autoregressive generation with attention extraction
  - Critic energy analysis along reasoning chain
  - 3D energy landscape around chain path
  - Full metrics & visualization suite
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cebcm.models.chain_head import _apply_rope


# ── Data Classes ─────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """Full diagnostics from one ChainGenerator run."""

    # Mode
    mode: str = "system1"  # "system1" or "system2"
    num_steps: int = 1
    requested_steps: int = 1
    step_cap_applied: bool = False
    step_cap_reason: str | None = None

    # Input / Output
    v_query: Tensor | None = None
    v_chain: Tensor | None = None       # [N, D] generated chain
    v_target: Tensor | None = None      # [D] ground-truth answer (if known)

    # Text
    input_text: str | None = None
    chain_texts: list[str] = field(default_factory=list)  # decoded chain steps
    response_text: str | None = None  # assembled clean response (deduplicated chain)
    target_text: str | None = None

    # Per-step metrics
    step_cos_to_target: list[float] = field(default_factory=list)
    step_norms: list[float] = field(default_factory=list)
    step_critic_energies: list[float] = field(default_factory=list)

    # Attention maps: list of dicts per layer
    # Each dict: {"self_attn": [H, L, L], "cross_attn": [H, L, K]}
    attention_maps: list[dict[str, Tensor]] = field(default_factory=list)
    context_labels: list[str] = field(default_factory=list)

    # Critic analysis
    critic_rank_acc: float | None = None
    critic_energy_answer: float | None = None
    critic_energy_query: float | None = None

    # 3D landscape data
    landscape_data: dict | None = None  # {xs, ys, zs, chain_points}

    # Reranking
    rerank_candidates: list[dict] | None = None  # [{chain, cos, energy}]
    rerank_best_idx: int | None = None

    # Anti-loop diagnostics
    early_stop: bool = False
    early_stop_step: int = -1
    repeat_resamples: int = 0
    diversity_cos_mean: float | None = None

    # Timing
    elapsed_ms: float = 0.0


@dataclass
class DiagState:
    """Module-level state for loaded models."""

    generator: nn.Module | None = None
    generator_cfg: Any = None
    generator_train_max_steps: int | None = None
    generator_context_bank_size: int | None = None
    critic: nn.Module | None = None
    critic_cfg: Any = None
    sonar: Any = None
    device: str = "cpu"
    last_result: GenerationResult | None = None


_state = DiagState()


def get_state() -> DiagState:
    return _state


class _LegacyTwinCriticAdapter:
    """
    Runtime adapter for legacy Stage1.5 twin critics.

    Supports both:
      - radial_angular (AngularEnergyCritic + RadialEnergyCritic)
      - homogeneous twin (SimpleEnergy + SimpleEnergy)
    """

    def __init__(
        self,
        critic1: nn.Module,
        critic2: nn.Module,
        aggregate: str = "max",
        softmax_temperature: float = 0.1,
        critic_architecture: str = "homogeneous",
        sigma_min: float = 0.01,
        sigma_max: float = 0.3,
        sigma_head_weighting_enabled: bool = False,
        angular_weight_low_sigma: float = 0.5,
        angular_weight_high_sigma: float = 0.5,
        head_weight_power: float = 1.0,
    ):
        self.critic1 = critic1
        self.critic2 = critic2
        self.aggregate = str(aggregate)
        self.softmax_temperature = float(softmax_temperature)
        self.critic_architecture = str(critic_architecture)
        self.sigma_min = float(max(sigma_min, 1e-8))
        self.sigma_max = float(max(sigma_max, self.sigma_min + 1e-8))
        self.sigma_head_weighting_enabled = bool(sigma_head_weighting_enabled)
        self.angular_weight_low_sigma = float(angular_weight_low_sigma)
        self.angular_weight_high_sigma = float(angular_weight_high_sigma)
        self.head_weight_power = float(max(head_weight_power, 1e-6))

    def eval(self):
        self.critic1.eval()
        self.critic2.eval()
        return self

    def _conditional_energy(
        self,
        v_query: Tensor,
        v_candidate: Tensor,
        sigma: Tensor | None = None,
    ) -> Tensor:
        e1 = self.critic1(v_query, v_candidate, sigma=sigma)
        e2 = self.critic2(v_query, v_candidate, sigma=sigma)

        # Stage1.5 radial_angular weighting by sigma.
        if self.critic_architecture == "radial_angular":
            sigma_use = sigma
            if sigma_use is None:
                with torch.no_grad():
                    d = (v_candidate - v_query).norm(dim=-1, keepdim=True)
                    qn = v_query.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    sigma_use = (d / qn).clamp(min=self.sigma_min, max=self.sigma_max)

            if self.sigma_head_weighting_enabled:
                log_s = sigma_use.clamp(min=self.sigma_min, max=self.sigma_max).log()
                denom = max(float(torch.log(torch.tensor(self.sigma_max / self.sigma_min))), 1e-8)
                t = ((log_s - torch.log(torch.tensor(self.sigma_min, device=log_s.device))) / denom)
                t = t.clamp(min=0.0, max=1.0).pow(self.head_weight_power)
                w_ang = self.angular_weight_low_sigma + (
                    self.angular_weight_high_sigma - self.angular_weight_low_sigma
                ) * t
            else:
                w_const = 0.5 * (self.angular_weight_low_sigma + self.angular_weight_high_sigma)
                w_ang = torch.full_like(sigma_use, w_const)
            w_ang = w_ang.clamp(min=0.0, max=1.0).squeeze(-1)
            return w_ang * e1 + (1.0 - w_ang) * e2

        if self.aggregate == "mean":
            return 0.5 * (e1 + e2)
        if self.aggregate == "max":
            return torch.maximum(e1, e2)
        if self.aggregate == "softmax":
            tau = max(1e-6, self.softmax_temperature)
            stacked = torch.stack([e1, e2], dim=0)
            return tau * torch.logsumexp(stacked / tau, dim=0)
        raise ValueError(f"Unknown twin aggregate mode: {self.aggregate}")

    def __call__(
        self,
        v_query: Tensor,
        v_candidate: Tensor,
        sigma: Tensor | None = None,
    ) -> Tensor:
        return self._conditional_energy(v_query, v_candidate, sigma=sigma)

    def energy_and_grad(
        self,
        v_query: Tensor,
        v_candidate: Tensor,
        sigma: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        v_req = v_candidate.detach().requires_grad_(True)
        energy = self(v_query, v_req, sigma=sigma)
        grad = torch.autograd.grad(energy.sum(), v_req, create_graph=False)[0]
        return energy.detach(), grad.detach()


# ── Model Loading ────────────────────────────────────────────────

def _resolve_device(override: str | None = None) -> torch.device:
    if override and override != "auto":
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_generator(ckpt_path: str, device: str = "auto") -> str:
    """Load ChainGenerator from checkpoint."""
    from cebcm.models.chain_generator import ChainGenerator, ChainGeneratorConfig

    dev = _resolve_device(device)
    path = Path(ckpt_path)
    if not path.exists():
        return f"File not found: {path}"

    try:
        ckpt = torch.load(str(path), map_location=dev, weights_only=False)

        # Extract config
        if "config" in ckpt and "generator" in ckpt["config"]:
            cfg = ChainGeneratorConfig(**ckpt["config"]["generator"])
        else:
            cfg = ChainGeneratorConfig()

        train_max_steps = None
        context_bank_size = None
        if "config" in ckpt and isinstance(ckpt["config"], dict):
            tr_cfg = ckpt["config"].get("training", {}) or {}
            if isinstance(tr_cfg, dict) and tr_cfg.get("max_chain_steps") is not None:
                train_max_steps = int(tr_cfg["max_chain_steps"])
            if isinstance(tr_cfg, dict) and tr_cfg.get("context_bank_size") is not None:
                context_bank_size = int(tr_cfg["context_bank_size"])

        model = ChainGenerator(cfg).to(dev)

        # Load state dict
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        elif "generator" in ckpt:
            model.load_state_dict(ckpt["generator"])
        else:
            model.load_state_dict(ckpt)

        model.eval()
        _state.generator = model
        _state.generator_cfg = cfg
        _state.generator_train_max_steps = train_max_steps
        _state.generator_context_bank_size = context_bank_size
        _state.device = str(dev)

        epoch = ckpt.get("epoch", "?")
        best = ckpt.get("best_metric", "?")
        train_steps_text = (
            str(train_max_steps) if train_max_steps is not None else "unknown"
        )
        return (
            f"ChainGenerator loaded: {model.num_params:,} params\n"
            f"  Epoch: {epoch}, Best cos: {best}\n"
            f"  Layers: {cfg.n_layers}, Heads: {cfg.n_heads}, FFN: {cfg.dim_feedforward}\n"
            f"  Train max steps: {train_steps_text}, Arch max len: {cfg.max_chain_len}\n"
            f"  Context bank size: {context_bank_size if context_bank_size is not None else 'unknown'}\n"
            f"  Device: {dev}"
        )
    except Exception as e:
        import traceback
        return f"Error loading generator: {e}\n{traceback.format_exc()}"


def load_critic(ckpt_path: str, device: str = "auto") -> str:
    """Load CompositeCritic from checkpoint."""
    from cebcm.models.composite_critic import CompositeCritic, CompositeCriticConfig
    from cebcm.models.conditional_angular_critic import ConditionalAngularCriticConfig
    from cebcm.models.radial_guard import RadialGuardConfig
    from cebcm.models.energy import SimpleEnergy
    from cebcm.models.energy_decomposed import AngularEnergyCritic, RadialEnergyCritic

    dev = _resolve_device(device)
    path = Path(ckpt_path)
    if not path.exists():
        return f"File not found: {path}"

    try:
        ckpt = torch.load(str(path), map_location=dev, weights_only=False)

        def _load_compat_state(model_obj: nn.Module, state_dict: dict, label: str) -> None:
            """
            Load with backward-compatible migration:
              - plain -> parametrized
              - parametrized -> plain
            """
            def _mismatch(load_res) -> tuple[list[str], list[str]]:
                missing_keys = [k for k in load_res.missing_keys if k != "_sigma_freqs"]
                unexpected_keys = [k for k in load_res.unexpected_keys]
                return missing_keys, unexpected_keys

            def _prepare_migration(model_target: nn.Module, source_sd: dict) -> tuple[dict | None, str | None]:
                model_sd = model_target.state_dict()
                model_has_param = any("parametrizations.weight.original" in k for k in model_sd.keys())
                source_has_param = any("parametrizations.weight.original" in k for k in source_sd.keys())

                if model_has_param and not source_has_param:
                    migrated = {}
                    for new_key, new_val in model_sd.items():
                        if new_key in source_sd:
                            migrated[new_key] = source_sd[new_key]
                            continue
                        if "parametrizations.weight.original" in new_key:
                            old_key = new_key.replace("parametrizations.weight.original", "weight")
                            if old_key in source_sd:
                                migrated[new_key] = source_sd[old_key]
                                continue
                        migrated[new_key] = new_val
                    return migrated, "plain->parametrized"

                if (not model_has_param) and source_has_param:
                    migrated = {}
                    for key, value in source_sd.items():
                        if "parametrizations.weight.original" in key:
                            bare_key = key.replace("parametrizations.weight.original", "weight")
                            migrated[bare_key] = value
                            continue
                        if ".parametrizations.weight." in key:
                            continue
                        migrated[key] = value
                    return migrated, "parametrized->plain"

                return None, None

            load_result = model_obj.load_state_dict(state_dict, strict=False)
            missing, unexpected = _mismatch(load_result)
            if not (missing or unexpected):
                return

            migrated, tag = _prepare_migration(model_obj, state_dict)
            if migrated is not None:
                retry_result = model_obj.load_state_dict(migrated, strict=False)
                missing_retry, unexpected_retry = _mismatch(retry_result)
                if not (missing_retry or unexpected_retry):
                    return
                missing, unexpected = missing_retry, unexpected_retry

            missing_preview = ", ".join(missing[:10])
            unexpected_preview = ", ".join(unexpected[:10])
            migration_note = f" Migration attempted: {tag}." if tag else ""
            raise RuntimeError(
                f"Checkpoint/model mismatch ({label}). "
                f"Missing({len(missing)}): {missing_preview}. "
                f"Unexpected({len(unexpected)}): {unexpected_preview}."
                f"{migration_note}"
            )

        # Legacy Stage1.5 checkpoint format (twin critics + training payload).
        if ("critic1_state" in ckpt) or ("critic2_state" in ckpt):
            cfg = ckpt.get("config", {}) or {}
            dim = int(cfg.get("energy_dim", 1024))
            arch = str(cfg.get("critic_architecture", "homogeneous"))
            norm_mode = str(cfg.get("norm_mode", "none"))
            activation = str(cfg.get("activation", "silu"))

            if arch == "radial_angular":
                c1 = AngularEnergyCritic(
                    dim=dim,
                    hidden_dims=[int(x) for x in cfg.get("angular_hidden_dims", [2048, 1024, 512])],
                    norm_mode=str(cfg.get("angular_norm_mode", norm_mode)),
                    activation=str(cfg.get("angular_activation", activation)),
                    energy_output_clamp=None,
                ).to(dev)
                target_norm = cfg.get("target_norm", cfg.get("langevin", {}).get("target_norm", 0.2051))
                c2 = RadialEnergyCritic(
                    dim=dim,
                    hidden_dims=[int(x) for x in cfg.get("radial_hidden_dims", [512, 256, 128])],
                    norm_mode=str(cfg.get("radial_norm_mode", norm_mode)),
                    activation=str(cfg.get("radial_activation", activation)),
                    target_norm=float(target_norm) if target_norm is not None else None,
                    energy_output_clamp=None,
                ).to(dev)
            else:
                hidden_dims = [int(x) for x in cfg.get("energy_hidden_dims", [2048, 1024, 512])]
                c1 = SimpleEnergy(
                    dim=dim,
                    hidden_dims=hidden_dims,
                    norm_mode=norm_mode,
                    activation=activation,
                    energy_output_clamp=None,
                ).to(dev)
                c2 = SimpleEnergy(
                    dim=dim,
                    hidden_dims=hidden_dims,
                    norm_mode=norm_mode,
                    activation=activation,
                    energy_output_clamp=None,
                ).to(dev)

            c1_state = ckpt.get("critic1_state", ckpt.get("critic_state", {}))
            c2_state = ckpt.get("critic2_state", c1_state)
            _load_compat_state(c1, c1_state, "legacy_critic1_state")
            _load_compat_state(c2, c2_state, "legacy_critic2_state")

            aggregate = str(cfg.get("twin_aggregate", "max"))
            softmax_temperature = float(cfg.get("twin_softmax_temperature", 0.1))
            model = _LegacyTwinCriticAdapter(
                critic1=c1,
                critic2=c2,
                aggregate=aggregate,
                softmax_temperature=softmax_temperature,
                critic_architecture=arch,
                sigma_min=float(cfg.get("sigma_min", 0.01)),
                sigma_max=float(cfg.get("sigma_max", 0.3)),
                sigma_head_weighting_enabled=bool(cfg.get("sigma_head_weighting_enabled", False)),
                angular_weight_low_sigma=float(cfg.get("angular_weight_low_sigma", 0.5)),
                angular_weight_high_sigma=float(cfg.get("angular_weight_high_sigma", 0.5)),
                head_weight_power=float(cfg.get("head_weight_power", 1.0)),
            )
            model.eval()

            _state.critic = model
            _state.critic_cfg = {
                "legacy": True,
                "critic_architecture": arch,
                "aggregate": aggregate,
            }
            _state.device = str(dev)

            c1_params = sum(p.numel() for p in c1.parameters())
            c2_params = sum(p.numel() for p in c2.parameters())
            return (
                f"Legacy twin critic loaded ({arch})\n"
                f"  Critic1 params: {c1_params:,}\n"
                f"  Critic2 params: {c2_params:,}\n"
                f"  Aggregate: {aggregate}\n"
                f"  Device: {dev}"
            )

        critic_raw = {}
        if "config" in ckpt and "critic" in ckpt["config"]:
            critic_raw = ckpt["config"]["critic"]

        ang_cfg = ConditionalAngularCriticConfig(**critic_raw.get("angular", {}))
        rad_cfg = RadialGuardConfig(**critic_raw.get("radial", {}))
        comp_cfg = CompositeCriticConfig(
            angular=ang_cfg, radial=rad_cfg,
            lambda_radial=critic_raw.get("lambda_radial", 5.0),
        )
        model = CompositeCritic(comp_cfg).to(dev)

        if "model" in ckpt:
            _load_compat_state(model, ckpt["model"], "model")
        elif "critic" in ckpt:
            _load_compat_state(model, ckpt["critic"], "critic")
        else:
            _load_compat_state(model, ckpt, "raw_checkpoint")

        model.eval()
        _state.critic = model
        _state.critic_cfg = comp_cfg
        _state.device = str(dev)

        return (
            f"CompositeCritic loaded: {model.angular.num_params:,} angular params\n"
            f"  Radial: analytical (no params), λ={comp_cfg.lambda_radial}\n"
            f"  Target norm: {rad_cfg.target_norm}\n"
            f"  Device: {dev}"
        )
    except Exception as e:
        import traceback
        return f"Error loading critic: {e}\n{traceback.format_exc()}"


def load_sonar(device: str = "auto") -> str:
    """Load SONAR encoder/decoder."""
    try:
        from cebcm.models.sonar_wrapper import SONARWrapper
        dev = _resolve_device(device)
        _state.sonar = SONARWrapper(device=str(dev))
        # Trigger lazy load
        _state.sonar._load_encoder()
        return f"SONAR loaded on {dev}"
    except Exception as e:
        return f"Error loading SONAR: {e}"


# ── Attention Extraction ─────────────────────────────────────────

def _extract_attention_maps(
    model: nn.Module,
    v_query: Tensor,
    chain_input: Tensor,
    v_context_bank: Tensor | None = None,
    context_mask: Tensor | None = None,
) -> list[dict]:
    """
    Extract self-attention and cross-attention maps from all decoder layers.

    Returns:
        List of dicts per layer: {"self_attn": [H, L, L], "cross_attn": [H, L, K]}
    """
    attention_maps: list[dict] = []
    hooks = []
    captured: dict[int, dict[str, dict[str, Tensor]]] = {}

    for layer_idx, layer in enumerate(model.layers):
        sa = layer.self_attn
        ca = layer.cross_attn
        self_cap: dict[str, Tensor] = {}
        cross_cap: dict[str, Tensor] = {}

        def make_hook(storage: dict[str, Tensor], key: str):
            def _hook(_, __, out):
                storage[key] = out.detach()

            return _hook

        hooks.append(sa.q_proj.register_forward_hook(make_hook(self_cap, "q_out")))
        hooks.append(sa.k_proj.register_forward_hook(make_hook(self_cap, "k_out")))
        hooks.append(ca.q_proj.register_forward_hook(make_hook(cross_cap, "q_out")))
        hooks.append(ca.k_proj.register_forward_hook(make_hook(cross_cap, "k_out")))

        captured[layer_idx] = {"self": self_cap, "cross": cross_cap}

    if v_context_bank is None:
        context = v_query.unsqueeze(0).unsqueeze(0)  # [1,1,D]
        ctx_mask = torch.ones((1, 1), device=context.device, dtype=torch.bool)
    else:
        context = v_context_bank.to(chain_input.device)
        if context.dim() == 2:
            context = context.unsqueeze(0)
        if context_mask is None:
            ctx_mask = torch.ones((context.shape[0], context.shape[1]), device=context.device, dtype=torch.bool)
        else:
            ctx_mask = context_mask.to(device=context.device, dtype=torch.bool)

    x = chain_input
    with torch.no_grad():
        for layer in model.layers:
            x = layer(x, context, context_mask=ctx_mask)

    for layer_idx in range(len(model.layers)):
        sa = model.layers[layer_idx].self_attn
        ca = model.layers[layer_idx].cross_attn
        sc = captured[layer_idx]["self"]
        cc = captured[layer_idx]["cross"]
        layer_map: dict[str, Tensor] = {}

        if "q_out" in sc and "k_out" in sc:
            n_heads = sa.n_heads
            head_dim = sa.head_dim
            q = sc["q_out"].view(1, -1, n_heads, head_dim).transpose(1, 2)
            k = sc["k_out"].view(1, -1, n_heads, head_dim).transpose(1, 2)

            seq_len = q.shape[2]
            rope = sa._get_rope(seq_len, q.device)
            q = _apply_rope(q, rope)
            k = _apply_rope(k, rope)

            scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
            causal = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1,
            )
            scores = scores + causal
            attn = F.softmax(scores, dim=-1)
            layer_map["self_attn"] = attn[0].cpu()

        if "q_out" in cc and "k_out" in cc:
            n_heads = ca.n_heads
            head_dim = ca.head_dim
            q = cc["q_out"].view(1, -1, n_heads, head_dim).transpose(1, 2)
            k = cc["k_out"].view(1, -1, n_heads, head_dim).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
            if ctx_mask is not None:
                invalid = ~ctx_mask.bool()
                scores = scores.masked_fill(invalid[:, None, None, :], float("-inf"))
            attn = F.softmax(scores, dim=-1)
            layer_map["cross_attn"] = attn[0].cpu()

        attention_maps.append(layer_map)

    for h in hooks:
        h.remove()

    return attention_maps



# ── Generation with Full Diagnostics ─────────────────────────────

def _generate_stochastic_chain(
    model: nn.Module,
    v_query: Tensor,
    num_steps: int,
    v_context_bank: Tensor | None = None,
    context_mask: Tensor | None = None,
    temperature: float = 1.0,
    latent_noise_std: float = 0.0,
    start_noise_std: float = 0.0,
    repeat_penalty: float = 0.0,
    repeat_cos_threshold: float = 0.98,
    repeat_ban_threshold: float = 0.995,
    repeat_ban_retries: int = 2,
    energy_fn=None,
    target_vec: Tensor | None = None,
    stagnation_patience: int = 0,
    stagnation_delta_energy: float = 1e-4,
    stagnation_delta_cos: float = 1e-4,
    convergence_cos: float = 0.0,
    convergence_window: int = 2,
    num_candidates: int = 1,
) -> tuple[Tensor, dict]:
    """Generate one candidate chain with stochastic + anti-loop controls."""
    chain, info = model.generate(
        v_query,
        num_steps=num_steps,
        v_context_bank=v_context_bank,
        context_mask=context_mask,
        temperature=temperature,
        latent_noise_std=latent_noise_std,
        start_noise_std=start_noise_std,
        repeat_penalty=repeat_penalty,
        repeat_cos_threshold=repeat_cos_threshold,
        repeat_ban_threshold=repeat_ban_threshold,
        repeat_ban_max_retries=repeat_ban_retries,
        energy_fn=energy_fn,
        target_vec=target_vec,
        stagnation_patience=stagnation_patience,
        stagnation_delta_energy=stagnation_delta_energy,
        stagnation_delta_cos=stagnation_delta_cos,
        convergence_cos=convergence_cos,
        convergence_window=convergence_window,
        return_info=True,
        num_candidates=num_candidates,
    )
    return chain, info


def _resolve_num_steps(requested_steps: int) -> tuple[int, bool, str | None]:
    """
    Clamp requested chain length to safe limits from loaded checkpoint/config.
    Prevent OOD rollout when inference horizon exceeds trained horizon.
    """
    req = max(1, int(requested_steps))
    caps: list[int] = []
    reasons: list[str] = []

    if _state.generator_cfg is not None and hasattr(_state.generator_cfg, "max_chain_len"):
        arch_cap = int(_state.generator_cfg.max_chain_len)
        if arch_cap > 0:
            caps.append(arch_cap)
            reasons.append(f"arch_max={arch_cap}")

    if _state.generator_train_max_steps is not None:
        tr_cap = int(_state.generator_train_max_steps)
        if tr_cap > 0:
            caps.append(tr_cap)
            reasons.append(f"train_max={tr_cap}")

    if not caps:
        return req, False, None

    cap = min(caps)
    eff = min(req, cap)
    if eff == req:
        return eff, False, None
    return eff, True, f"requested={req}, cap={cap} ({', '.join(reasons)})"


def _call_critic_energy(
    critic: nn.Module,
    v_query: Tensor,
    v_candidate: Tensor,
    v_context: Tensor | None = None,
) -> Tensor:
    """Call critic with context when supported; fallback for legacy adapters."""
    if v_context is not None:
        try:
            return critic(v_query, v_candidate, v_context=v_context)
        except TypeError:
            pass
    return critic(v_query, v_candidate)


@torch.no_grad()
def run_generation(
    v_query: Tensor,
    num_steps: int = 1,
    v_target: Tensor | None = None,
    num_candidates: int = 1,
    beam_width: int = 1,
    temperature: float = 1.0,
    noise_std: float = 0.01,
    v_context: Tensor | None = None,
    v_context_bank: Tensor | None = None,
    context_labels: list[str] | None = None,
) -> GenerationResult:
    """
    Run ChainGenerator with full diagnostics using Active Inference.
    """
    if _state.generator is None:
        raise RuntimeError("ChainGenerator not loaded. Load checkpoint first.")

    model = _state.generator
    device = torch.device(_state.device)
    t0 = time.time()

    effective_steps, _, _ = _resolve_num_steps(num_steps)

    v_q = v_query.unsqueeze(0).to(device)  # [1, D]
    v_ctx = v_q if v_context is None else v_context.unsqueeze(0).to(device)

    if v_context_bank is None:
        gen_ctx = v_q.unsqueeze(1)  # [1,1,D]
        gen_ctx_mask = torch.ones((1, 1), device=device, dtype=torch.bool)
        labels = context_labels or ["query"]
    else:
        gen_ctx = v_context_bank.to(device)
        if gen_ctx.dim() == 2:
            gen_ctx = gen_ctx.unsqueeze(0)
        gen_ctx_mask = torch.ones((gen_ctx.shape[0], gen_ctx.shape[1]), device=device, dtype=torch.bool)
        labels = context_labels[: gen_ctx.shape[1]] if context_labels else [f"ctx_{i}" for i in range(gen_ctx.shape[1])]

    def _energy_fn(q_batch: Tensor, c_batch: Tensor) -> Tensor:
        if _state.critic is None:
            return torch.zeros(c_batch.shape[0], device=c_batch.device)
        return _call_critic_energy(_state.critic, q_batch, c_batch, v_context=v_ctx)

    target_for_stop = None if v_target is None else v_target.to(device).unsqueeze(0)

    latent_noise_std = noise_std
    start_noise_std = noise_std * 0.5
    repeat_penalty = 0.2 if effective_steps > 1 else 0.0
    repeat_cos_thr = 0.985
    repeat_ban_thr = 0.997
    repeat_ban_retries = 3

    stagnation_patience = 4 if effective_steps > 1 else 0
    stagnation_delta_energy = 5e-4
    stagnation_delta_cos = 5e-4
    convergence_cos = 0.995 if effective_steps > 1 else 0.0
    convergence_window = 2

    if beam_width > 1:
        chain = model.beam_generate(
            v_query=v_q,
            num_steps=effective_steps,
            beam_width=int(beam_width),
            num_candidates=int(num_candidates),
            v_context_bank=gen_ctx,
            context_mask=gen_ctx_mask,
            temperature=temperature,
            noise_std=noise_std,
            energy_fn=_energy_fn if _state.critic is not None else None,
        )
        info = {
            "steps_generated": chain.shape[1] - 1,
            "early_stop": False,
            "early_stop_reason": "",
            "mode": "beam_search",
            "repeat_resamples": 0,
        }
    else:
        chain, info = _generate_stochastic_chain(
            model,
            v_q,
            num_steps=effective_steps,
            v_context_bank=gen_ctx,
            context_mask=gen_ctx_mask,
            temperature=temperature,
            latent_noise_std=latent_noise_std,
            start_noise_std=start_noise_std,
            repeat_penalty=repeat_penalty,
            repeat_cos_threshold=repeat_cos_thr,
            repeat_ban_threshold=repeat_ban_thr,
            repeat_ban_retries=repeat_ban_retries,
            energy_fn=_energy_fn if _state.critic is not None else None,
            target_vec=target_for_stop,
            stagnation_patience=stagnation_patience,
            stagnation_delta_energy=stagnation_delta_energy,
            stagnation_delta_cos=stagnation_delta_cos,
            convergence_cos=convergence_cos,
            convergence_window=convergence_window,
            num_candidates=num_candidates,
        )

    candidates = [chain[0]]
    candidate_infos = [
        {
            "idx": 0,
            "temperature": float(temperature),
            "latent_noise_std": float(latent_noise_std),
            "start_noise_std": float(start_noise_std),
            "early_stop": bool(info.get("early_stop", False)),
            "early_stop_reason": str(info.get("early_stop_reason", "")),
            "steps_generated": int(info.get("steps_generated", chain.shape[1] - 1)),
            "repeat_resamples": int(info.get("repeat_resamples", 0)),
        }
    ]

    diversity_cos_mean = None

    # Rerank by critic energy on final step when critic is available.
    cand_info: list[dict] | None = None
    best_idx = 0
    if _state.critic is not None:
        best_energy = float("inf")
        cand_info = []
        for i, c in enumerate(candidates):
            v_ans = c[-1:].to(device)
            energy = float(_call_critic_energy(_state.critic, v_q[0:1], v_ans, v_context=v_ctx).item())
            cos = (
                float(F.cosine_similarity(v_ans, v_target.unsqueeze(0).to(device), dim=-1).item())
                if v_target is not None
                else 0.0
            )
            merged = {
                "idx": i,
                "energy": energy,
                "cos": cos,
                **candidate_infos[i],
            }
            cand_info.append(merged)
            if energy < best_energy:
                best_energy = energy
                best_idx = i

    v_chain = candidates[best_idx]
    selected_info = candidate_infos[best_idx]

    result = GenerationResult(
        mode="system1" if effective_steps == 1 else "system2",
        num_steps=int(v_chain.shape[0]),
        requested_steps=int(num_steps),
        step_cap_applied=step_cap_applied,
        step_cap_reason=step_cap_reason,
        v_query=v_query.cpu(),
        v_chain=v_chain.cpu(),
        v_target=v_target.cpu() if v_target is not None else None,
        context_labels=labels,
        early_stop=bool(selected_info.get("early_stop", False)),
        early_stop_step=int(selected_info.get("steps_generated", v_chain.shape[0])),
        repeat_resamples=int(selected_info.get("repeat_resamples", 0)),
        diversity_cos_mean=diversity_cos_mean,
    )

    # Per-step metrics.
    for i in range(v_chain.shape[0]):
        result.step_norms.append(float(v_chain[i].norm().item()))
        if v_target is not None:
            cos = F.cosine_similarity(v_chain[i:i+1].to(device), v_target.unsqueeze(0).to(device), dim=-1).item()
            result.step_cos_to_target.append(float(cos))

    # Critic energies along trajectory.
    if _state.critic is not None:
        for i in range(v_chain.shape[0]):
            v_step = v_chain[i : i + 1].to(device)
            e = _call_critic_energy(_state.critic, v_q[0:1], v_step, v_context=v_ctx).item()
            result.step_critic_energies.append(float(e))

        result.critic_energy_query = float(
            _call_critic_energy(_state.critic, v_q[0:1], v_q[0:1], v_context=v_ctx).item()
        )
        if v_target is not None:
            result.critic_energy_answer = float(
                _call_critic_energy(_state.critic, v_q[0:1], v_target.unsqueeze(0).to(device), v_context=v_ctx).item()
            )

    # Attention extraction on selected chain.
    start = model.start_token.expand(1, -1, -1)
    decoder_input = torch.cat([start, v_chain[:-1].unsqueeze(0).to(device)], dim=1)
    result.attention_maps = _extract_attention_maps(
        model,
        v_q[0],
        decoder_input,
        v_context_bank=gen_ctx,
        context_mask=gen_ctx_mask,
    )

    if cand_info is not None:
        result.rerank_candidates = cand_info
        result.rerank_best_idx = best_idx

    result.elapsed_ms = (time.time() - t0) * 1000.0
    _state.last_result = result
    return result



# ── Critic Analysis on Chain ─────────────────────────────────────

@torch.no_grad()
def analyze_critic_on_chain(
    v_query: Tensor,
    v_chain: Tensor,
    v_target: Tensor | None = None,
    v_context: Tensor | None = None,
    num_random_negatives: int = 20,
) -> dict:
    """
    Analyze CompositeCritic behavior on a generated chain.

    Returns dict with:
        - per_step_energy: energies along chain
        - energy_target: energy at ground truth
        - energy_random: energies at random points
        - rank_acc: fraction of random negatives with higher energy than answer
    """
    if _state.critic is None:
        return {"error": "Critic not loaded"}

    device = torch.device(_state.device)
    critic = _state.critic
    v_q = v_query.unsqueeze(0).to(device)
    v_ctx = v_q if v_context is None else v_context.unsqueeze(0).to(device)

    # Per-step energy
    per_step = []
    for i in range(v_chain.shape[0]):
        v = v_chain[i:i+1].to(device)
        e = _call_critic_energy(critic, v_q, v, v_context=v_ctx).item()
        per_step.append(e)

    # Target energy
    e_target = None
    if v_target is not None:
        e_target = _call_critic_energy(
            critic, v_q, v_target.unsqueeze(0).to(device), v_context=v_ctx
        ).item()

    # Random negative energies
    target_norm = v_chain.norm(dim=-1).mean().item()
    random_vecs = F.normalize(torch.randn(num_random_negatives, v_query.shape[-1]), dim=-1) * target_norm
    random_energies = []
    for i in range(num_random_negatives):
        v = random_vecs[i:i+1].to(device)
        e = _call_critic_energy(critic, v_q, v, v_context=v_ctx).item()
        random_energies.append(e)

    # Rank accuracy: how many random negatives have higher energy than answer?
    rank_acc = None
    if e_target is not None:
        rank_acc = sum(1 for e in random_energies if e > e_target) / len(random_energies)

    return {
        "per_step_energy": per_step,
        "energy_target": e_target,
        "energy_random": random_energies,
        "rank_acc": rank_acc,
        "energy_query": _call_critic_energy(critic, v_q, v_q, v_context=v_ctx).item(),
    }


# ── 3D Energy Landscape Around Chain Path ────────────────────────

@torch.no_grad()
def compute_chain_landscape(
    v_query: Tensor,
    v_chain: Tensor,
    v_context: Tensor | None = None,
    grid_size: int = 30,
    spread: float = 0.05,
) -> dict:
    """
    Compute 3D energy landscape around the reasoning chain path.

    Uses PCA on the chain to find the 2D plane of variation,
    then samples energy on a grid around each chain step.

    Returns:
        dict with xs, ys, energy_grid, chain_positions for 3D plotting
    """
    if _state.critic is None:
        return {"error": "Critic not loaded"}

    device = torch.device(_state.device)
    critic = _state.critic
    v_q = v_query.unsqueeze(0).to(device)
    v_ctx = v_q if v_context is None else v_context.unsqueeze(0).to(device)
    chain = v_chain.to(device)  # [N, D]
    N, D = chain.shape

    # PCA on chain to find 2 principal directions
    chain_centered = chain - chain.mean(dim=0, keepdim=True)
    if N < 2:
        # Single step — use random orthogonal directions
        d1 = F.normalize(torch.randn(D, device=device), dim=0)
        d2 = F.normalize(torch.randn(D, device=device), dim=0)
        d2 = F.normalize(d2 - (d2 @ d1) * d1, dim=0)
    else:
        U, S, Vh = torch.linalg.svd(chain_centered, full_matrices=False)
        d1 = Vh[0]  # first principal direction
        d2 = Vh[1] if N > 2 else F.normalize(torch.randn(D, device=device), dim=0)
        # Ensure d2 is orthogonal to d1
        d2 = F.normalize(d2 - (d2 @ d1) * d1, dim=0)

    # Project chain steps onto PCA plane
    chain_x = (chain @ d1).cpu().tolist()
    chain_y = (chain @ d2).cpu().tolist()

    # Grid bounds
    cx_min, cx_max = min(chain_x), max(chain_x)
    cy_min, cy_max = min(chain_y), max(chain_y)
    pad_x = max((cx_max - cx_min) * 0.3, spread * 5)
    pad_y = max((cy_max - cy_min) * 0.3, spread * 5)

    xs = torch.linspace(cx_min - pad_x, cx_max + pad_x, grid_size, device=device)
    ys = torch.linspace(cy_min - pad_y, cy_max + pad_y, grid_size, device=device)

    # Compute energy on grid
    # Base point: center of chain
    base = chain.mean(dim=0)  # [D]
    base_x = (base @ d1).item()
    base_y = (base @ d2).item()

    energy_grid = torch.zeros(grid_size, grid_size)
    target_norm = chain.norm(dim=-1).mean().item()

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            # Reconstruct point in D-dimensional space
            v = base + (x - base_x) * d1 + (y - base_y) * d2
            v = F.normalize(v, dim=-1) * target_norm  # project to sphere
            e = _call_critic_energy(
                critic, v_q, v.unsqueeze(0), v_context=v_ctx
            ).item()
            energy_grid[i, j] = e

    # Chain step energies
    chain_energies = []
    for i in range(N):
        e = _call_critic_energy(critic, v_q, chain[i:i+1], v_context=v_ctx).item()
        chain_energies.append(e)

    return {
        "xs": xs.cpu().tolist(),
        "ys": ys.cpu().tolist(),
        "energy_grid": energy_grid.cpu().tolist(),
        "chain_x": chain_x,
        "chain_y": chain_y,
        "chain_energies": chain_energies,
        "d1": d1.cpu(),
        "d2": d2.cpu(),
    }


# ── Plotting Functions ───────────────────────────────────────────

def create_self_attention_plot(result: GenerationResult, layer_idx: int = 0) -> go.Figure:
    """Heatmap of self-attention for a specific layer (averaged over heads)."""
    if not result.attention_maps or layer_idx >= len(result.attention_maps):
        fig = go.Figure()
        fig.update_layout(title="No attention data available")
        return fig

    layer_map = result.attention_maps[layer_idx]
    if "self_attn" not in layer_map:
        fig = go.Figure()
        fig.update_layout(title="Self-attention not captured")
        return fig

    attn = layer_map["self_attn"]  # [H, L, L]
    avg_attn = attn.mean(dim=0).numpy()  # [L, L]
    L = avg_attn.shape[0]

    labels = ["[START]"] + [f"step_{i+1}" for i in range(L - 1)]

    fig = go.Figure(data=go.Heatmap(
        z=avg_attn,
        x=labels, y=labels,
        colorscale="Viridis",
        colorbar=dict(title="Weight"),
    ))
    fig.update_layout(
        title=f"Self-Attention (Layer {layer_idx}, avg over heads)",
        xaxis_title="Key",
        yaxis_title="Query",
        height=500, width=550,
        template="plotly_dark",
    )
    return fig


def create_all_heads_attention_plot(result: GenerationResult, layer_idx: int = 0) -> go.Figure:
    """Grid of all attention heads for a specific layer."""
    if not result.attention_maps or layer_idx >= len(result.attention_maps):
        fig = go.Figure()
        fig.update_layout(title="No attention data")
        return fig

    layer_map = result.attention_maps[layer_idx]
    if "self_attn" not in layer_map:
        fig = go.Figure()
        fig.update_layout(title="Self-attention not captured")
        return fig

    attn = layer_map["self_attn"]  # [H, L, L]
    H = attn.shape[0]
    cols = min(4, H)
    rows = (H + cols - 1) // cols

    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"Head {i}" for i in range(H)])

    L = attn.shape[1]
    labels = ["[S]"] + [f"s{i+1}" for i in range(L - 1)]

    for h in range(H):
        r = h // cols + 1
        c = h % cols + 1
        fig.add_trace(
            go.Heatmap(
                z=attn[h].numpy(), x=labels, y=labels,
                colorscale="Viridis", showscale=(h == 0),
            ),
            row=r, col=c,
        )

    fig.update_layout(
        title=f"All Heads Self-Attention (Layer {layer_idx})",
        height=250 * rows, width=250 * cols,
        template="plotly_dark",
    )
    return fig


def create_cross_attention_plot(result: GenerationResult, layer_idx: int = 0) -> go.Figure:
    """Cross-attention heatmap over context bank tokens."""
    if not result.attention_maps or layer_idx >= len(result.attention_maps):
        fig = go.Figure()
        fig.update_layout(title="No attention data")
        return fig

    layer_map = result.attention_maps[layer_idx]
    if "cross_attn" not in layer_map:
        fig = go.Figure()
        fig.update_layout(title="Cross-attention not captured")
        return fig

    attn = layer_map["cross_attn"]  # [H, L, K]
    avg = attn.mean(dim=0).numpy()  # [L, K]
    l_chain, k_ctx = avg.shape
    y_labels = ["[START]"] + [f"step_{i+1}" for i in range(l_chain - 1)]
    if result.context_labels and len(result.context_labels) >= k_ctx:
        x_labels = result.context_labels[:k_ctx]
    else:
        x_labels = [f"ctx_{i}" for i in range(k_ctx)]

    fig = go.Figure(
        data=go.Heatmap(
            z=avg,
            x=x_labels,
            y=y_labels,
            colorscale="Viridis",
            colorbar=dict(title="Weight"),
        )
    )
    fig.update_layout(
        title=f"Cross-Attention to Context Bank (Layer {layer_idx}, avg heads)",
        xaxis_title="Context Token",
        yaxis_title="Chain Position",
        height=450,
        template="plotly_dark",
    )
    return fig


def create_step_metrics_plot(result: GenerationResult) -> go.Figure:
    """Per-step cosine similarity and critic energy."""
    steps = list(range(1, result.num_steps + 1))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Cosine to Target", "Critic Energy"],
    )

    if result.step_cos_to_target:
        fig.add_trace(
            go.Scatter(x=steps, y=result.step_cos_to_target,
                       mode="lines+markers", name="cos(step, target)",
                       marker=dict(color="lime")),
            row=1, col=1,
        )

    if result.step_critic_energies:
        fig.add_trace(
            go.Scatter(x=steps, y=result.step_critic_energies,
                       mode="lines+markers", name="E(query, step)",
                       marker=dict(color="orange")),
            row=1, col=2,
        )
        # Add target energy line
        if result.critic_energy_answer is not None:
            fig.add_hline(y=result.critic_energy_answer, line_dash="dash",
                          line_color="lime", annotation_text="E(target)",
                          row=1, col=2)

    fig.update_layout(
        height=400, template="plotly_dark",
        title="Per-Step Quality Metrics",
    )
    return fig


def create_energy_landscape_3d(result: GenerationResult) -> go.Figure:
    """3D energy surface around reasoning chain path."""
    if result.landscape_data is None or "error" in result.landscape_data:
        fig = go.Figure()
        fig.update_layout(title="Landscape not available (load critic first)")
        return fig

    ld = result.landscape_data
    xs = ld["xs"]
    ys = ld["ys"]
    zs = ld["energy_grid"]

    fig = go.Figure()

    # Energy surface
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        colorscale="Inferno",
        opacity=0.8,
        name="Energy Surface",
        colorbar=dict(title="Energy"),
    ))

    # Chain trajectory
    chain_x = ld["chain_x"]
    chain_y = ld["chain_y"]
    chain_e = ld["chain_energies"]

    fig.add_trace(go.Scatter3d(
        x=chain_x, y=chain_y, z=chain_e,
        mode="lines+markers",
        line=dict(color="lime", width=5),
        marker=dict(size=6, color="lime"),
        name="Chain Path",
    ))

    # Mark start and end
    fig.add_trace(go.Scatter3d(
        x=[chain_x[0]], y=[chain_y[0]], z=[chain_e[0]],
        mode="markers+text",
        marker=dict(size=10, color="cyan", symbol="diamond"),
        text=["START"], textposition="top center",
        name="Start",
    ))
    if len(chain_x) > 1:
        fig.add_trace(go.Scatter3d(
            x=[chain_x[-1]], y=[chain_y[-1]], z=[chain_e[-1]],
            mode="markers+text",
            marker=dict(size=10, color="red", symbol="diamond"),
            text=["ANSWER"], textposition="top center",
            name="Answer",
        ))

    fig.update_layout(
        title="Energy Landscape Around Chain Path",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="Energy",
        ),
        height=600, width=700,
        template="plotly_dark",
    )
    return fig


def create_norm_plot(result: GenerationResult) -> go.Figure:
    """Plot vector norms along chain."""
    steps = list(range(1, result.num_steps + 1))
    target_norm = 0.2051

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=result.step_norms,
        mode="lines+markers", name="‖v‖",
        marker=dict(color="cyan"),
    ))
    fig.add_hline(y=target_norm, line_dash="dash", line_color="yellow",
                  annotation_text=f"target={target_norm}")

    fig.update_layout(
        title="Vector Norms Along Chain",
        xaxis_title="Step", yaxis_title="‖v‖",
        height=350, template="plotly_dark",
    )
    return fig


# ── Metrics Formatting ───────────────────────────────────────────

def format_metrics_markdown(result: GenerationResult) -> str:
    """Format GenerationResult as readable Markdown."""
    lines = [
        f"### Generation Results ({result.mode.upper()})",
        f"- **Steps**: {result.num_steps} (requested {result.requested_steps}) | **Time**: {result.elapsed_ms:.0f} ms",
        "",
    ]
    if result.step_cap_applied and result.step_cap_reason:
        lines.append(f"- **Step cap applied**: `{result.step_cap_reason}`")
        lines.append("")
    if result.early_stop:
        lines.append(f"- **Early stop**: yes (step {result.early_stop_step})")
    else:
        lines.append("- **Early stop**: no")
    lines.append(f"- **Repeat resamples**: {result.repeat_resamples}")
    if result.diversity_cos_mean is not None:
        lines.append(f"- **Candidate diversity (pairwise cos mean)**: {result.diversity_cos_mean:.4f}")
    if result.context_labels:
        lines.append(f"- **Context bank**: {', '.join(result.context_labels)}")
    lines.append("")

    if result.step_cos_to_target:
        lines.append("#### Cosine to Target")
        for i, c in enumerate(result.step_cos_to_target):
            marker = " ← answer" if i == len(result.step_cos_to_target) - 1 else ""
            lines.append(f"- Step {i+1}: **{c:.4f}**{marker}")
        lines.append("")

    if result.step_critic_energies:
        lines.append("#### Critic Energy")
        for i, e in enumerate(result.step_critic_energies):
            lines.append(f"- Step {i+1}: E={e:.4f}")
        if result.critic_energy_answer is not None:
            lines.append(f"- **Target**: E={result.critic_energy_answer:.4f}")
        if result.critic_energy_query is not None:
            lines.append(f"- **Query**: E={result.critic_energy_query:.4f}")
        if result.critic_rank_acc is not None:
            lines.append(f"- **Rank acc vs random**: {result.critic_rank_acc:.2%}")
        lines.append("")

    if result.step_norms:
        lines.append("#### Norms")
        norms = result.step_norms
        lines.append(f"- Mean: {sum(norms)/len(norms):.4f}, Min: {min(norms):.4f}, Max: {max(norms):.4f}")
        lines.append("")

    if result.chain_texts:
        lines.append("#### Decoded Chain")
        for i, t in enumerate(result.chain_texts):
            label = "**Answer**" if i == len(result.chain_texts) - 1 else f"Step {i+1}"
            lines.append(f"- {label}: {t}")
        lines.append("")

    if result.rerank_candidates:
        lines.append("#### Reranking (Best-of-N)")
        for c in result.rerank_candidates:
            marker = " ✓" if c["idx"] == result.rerank_best_idx else ""
            lines.append(f"- Candidate {c['idx']}: E={c['energy']:.4f}, cos={c['cos']:.4f}{marker}")
        lines.append("")

    if result.response_text:
        lines.append(f"#### Response\n> {result.response_text}\n")

    if result.input_text:
        lines.append(f"#### Input Text\n> {result.input_text}\n")
    if result.target_text:
        lines.append(f"#### Target Text\n> {result.target_text}\n")

    return "\n".join(lines)


def assemble_response(chain_texts: list[str]) -> str:
    """Assemble chain steps into a clean response text.

    Deduplicates near-identical steps and concatenates the unique reasoning
    steps followed by the final answer.  This is the "chat mode" view: instead
    of showing every chain step, collapse repetitions and return a readable
    multi-sentence response.
    """
    if not chain_texts:
        return ""

    # Normalize whitespace.
    texts = [t.strip() for t in chain_texts if t.strip()]
    if not texts:
        return ""

    # Deduplicate consecutive near-identical texts.
    unique: list[str] = [texts[0]]
    for t in texts[1:]:
        prev = unique[-1].lower().rstrip(".!?,;:")
        curr = t.lower().rstrip(".!?,;:")
        # Skip if identical, substring, or only differs by punctuation.
        if curr == prev:
            continue
        if curr in prev or prev in curr:
            # Keep the longer version.
            if len(t) > len(unique[-1]):
                unique[-1] = t
            continue
        unique.append(t)

    return " ".join(unique)


def export_metrics_json(result: GenerationResult) -> str:
    """Export metrics as JSON string."""
    data = {
        "mode": result.mode,
        "num_steps": result.num_steps,
        "requested_steps": result.requested_steps,
        "step_cap_applied": result.step_cap_applied,
        "step_cap_reason": result.step_cap_reason,
        "early_stop": result.early_stop,
        "early_stop_step": result.early_stop_step,
        "repeat_resamples": result.repeat_resamples,
        "diversity_cos_mean": result.diversity_cos_mean,
        "context_labels": result.context_labels,
        "elapsed_ms": result.elapsed_ms,
        "step_cos_to_target": result.step_cos_to_target,
        "step_norms": result.step_norms,
        "step_critic_energies": result.step_critic_energies,
        "critic_energy_answer": result.critic_energy_answer,
        "critic_energy_query": result.critic_energy_query,
        "critic_rank_acc": result.critic_rank_acc,
        "chain_texts": result.chain_texts,
        "response_text": result.response_text,
        "input_text": result.input_text,
        "target_text": result.target_text,
    }
    if result.rerank_candidates:
        data["rerank_candidates"] = result.rerank_candidates
        data["rerank_best_idx"] = result.rerank_best_idx
    return json.dumps(data, indent=2, ensure_ascii=False)


def export_metrics_csv(result: GenerationResult) -> str:
    """Export per-step metrics as CSV."""
    lines = ["step,cos_to_target,norm,critic_energy"]
    for i in range(result.num_steps):
        cos = result.step_cos_to_target[i] if i < len(result.step_cos_to_target) else ""
        norm = result.step_norms[i] if i < len(result.step_norms) else ""
        energy = result.step_critic_energies[i] if i < len(result.step_critic_energies) else ""
        lines.append(f"{i+1},{cos},{norm},{energy}")
    return "\n".join(lines)


# ── High-Level Entry Points ──────────────────────────────────────

def _build_context_bank_from_steps(
    v_query: Tensor,
    v_steps: Tensor | None,
    max_slots: int,
) -> tuple[Tensor, list[str]]:
    """Build context memory bank [K, D] = [query, evidence slots...]."""
    max_slots = max(1, int(max_slots))
    labels = ["query"]
    if v_steps is None or not isinstance(v_steps, torch.Tensor) or v_steps.ndim != 2 or v_steps.shape[0] == 0:
        return v_query.unsqueeze(0), labels

    evidence_slots = max(0, max_slots - 1)
    if evidence_slots == 0:
        return v_query.unsqueeze(0), labels

    evidence = v_steps[:evidence_slots]
    labels.extend([f"ev_{i+1}" for i in range(evidence.shape[0])])
    bank = torch.cat([v_query.unsqueeze(0), evidence], dim=0)
    return bank, labels


def run_from_data(
    data_path: str,
    sample_idx: int = 0,
    num_steps: int = 1,
    num_candidates: int = 1,
    beam_width: int = 1,
    temperature: float = 1.0,
    noise_std: float = 0.01,
    grid_size: int = 30,
) -> GenerationResult:
    """
    Run generation from a data file (hotpotqa_sonar.pt).

    Returns GenerationResult with all diagnostics.
    """
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    split = data.get("val", data.get("train", data))
    if isinstance(split, list):
        sample = split[min(sample_idx, len(split) - 1)]
    else:
        raise ValueError(f"Unknown data format: {type(split)}")

    v_query = sample["v_question"]
    v_answer = sample["v_answer"]
    v_steps = sample.get("v_steps", None)
    context_slots = _state.generator_context_bank_size or 4
    v_context_bank, context_labels = _build_context_bank_from_steps(v_query, v_steps, context_slots)
    if isinstance(v_steps, torch.Tensor) and v_steps.ndim == 2 and v_steps.shape[0] > 0:
        v_context = v_steps.mean(dim=0)
    else:
        v_context = v_query

    result = run_generation(
        v_query,
        num_steps=num_steps,
        v_target=v_answer,
        num_candidates=num_candidates,
        beam_width=beam_width,
        temperature=temperature,
        noise_std=noise_std,
        v_context=v_context,
        v_context_bank=v_context_bank,
        context_labels=context_labels,
    )

    # Compute 3D landscape
    if _state.critic is not None and result.v_chain is not None:
        result.landscape_data = compute_chain_landscape(
            v_query, result.v_chain, v_context=v_context, grid_size=grid_size,
        )

    # Critic analysis
    if _state.critic is not None and result.v_chain is not None:
        critic_info = analyze_critic_on_chain(
            v_query, result.v_chain, v_target=v_answer, v_context=v_context
        )
        result.critic_rank_acc = critic_info.get("rank_acc")

    return result


def run_from_text(
    text: str,
    num_steps: int = 1,
    num_candidates: int = 1,
    beam_width: int = 1,
    temperature: float = 1.0,
    noise_std: float = 0.01,
    grid_size: int = 30,
) -> GenerationResult:
    """
    Run generation from text input (requires SONAR).

    Returns GenerationResult with decoded chain texts.
    """
    if _state.sonar is None:
        raise RuntimeError("SONAR not loaded")
    if _state.generator is None:
        raise RuntimeError("ChainGenerator not loaded")

    device = torch.device(_state.device)

    # Encode query
    v_query = _state.sonar.encode([text]).squeeze(0).cpu()  # [D]
    v_context_bank = v_query.unsqueeze(0)
    context_labels = ["query"]

    result = run_generation(
        v_query,
        num_steps=num_steps,
        num_candidates=num_candidates,
        beam_width=beam_width,
        temperature=temperature,
        noise_std=noise_std,
        v_context=v_query,
        v_context_bank=v_context_bank,
        context_labels=context_labels,
    )
    result.input_text = text

    # Decode chain steps to text
    if result.v_chain is not None:
        chain_for_decode = result.v_chain.to(device)
        result.chain_texts = _state.sonar.decode_safe(chain_for_decode)
        result.response_text = assemble_response(result.chain_texts)

    # Compute 3D landscape
    if _state.critic is not None and result.v_chain is not None:
        result.landscape_data = compute_chain_landscape(
            v_query, result.v_chain, v_context=v_query, grid_size=grid_size,
        )

    return result
