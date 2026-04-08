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


# ── Data Classes ─────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """Full diagnostics from one ChainGenerator run."""

    # Mode
    mode: str = "system1"  # "system1" or "system2"
    num_steps: int = 1

    # Input / Output
    v_query: Tensor | None = None
    v_chain: Tensor | None = None       # [N, D] generated chain
    v_target: Tensor | None = None      # [D] ground-truth answer (if known)

    # Text
    input_text: str | None = None
    chain_texts: list[str] = field(default_factory=list)  # decoded chain steps
    target_text: str | None = None

    # Per-step metrics
    step_cos_to_target: list[float] = field(default_factory=list)
    step_norms: list[float] = field(default_factory=list)
    step_critic_energies: list[float] = field(default_factory=list)

    # Attention maps: list of dicts per layer
    # Each dict: {"self_attn": [H, L, L], "cross_attn": [H, L, 1]}
    attention_maps: list[dict[str, Tensor]] = field(default_factory=list)

    # Critic analysis
    critic_rank_acc: float | None = None
    critic_energy_answer: float | None = None
    critic_energy_query: float | None = None

    # 3D landscape data
    landscape_data: dict | None = None  # {xs, ys, zs, chain_points}

    # Reranking
    rerank_candidates: list[dict] | None = None  # [{chain, cos, energy}]
    rerank_best_idx: int | None = None

    # Timing
    elapsed_ms: float = 0.0


@dataclass
class DiagState:
    """Module-level state for loaded models."""

    generator: nn.Module | None = None
    generator_cfg: Any = None
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
        _state.device = str(dev)

        epoch = ckpt.get("epoch", "?")
        best = ckpt.get("best_metric", "?")
        return (
            f"ChainGenerator loaded: {model.num_params:,} params\n"
            f"  Epoch: {epoch}, Best cos: {best}\n"
            f"  Layers: {cfg.n_layers}, Heads: {cfg.n_heads}, FFN: {cfg.dim_feedforward}\n"
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

def _extract_attention_maps(model: nn.Module, v_query: Tensor, chain_input: Tensor) -> list[dict]:
    """
    Extract self-attention and cross-attention maps from all decoder layers.

    Args:
        model: ChainGenerator
        v_query: [1, D]
        chain_input: [1, L, D] decoder input sequence

    Returns:
        List of dicts per layer: {"self_attn": [H, L, L], "cross_attn": [H, L, 1]}
    """
    attention_maps = []
    hooks = []

    def make_self_attn_hook(layer_idx):
        def hook(module, args, output):
            # CausalRoPESelfAttention.forward captures q, k after RoPE
            pass  # We use a different strategy — intercept SDPA
        return hook

    # Strategy: temporarily replace SDPA with a version that captures weights
    captured = {}

    def make_capture_hook(name):
        def hook(module, args, kwargs, output):
            # For SDPA-based attention, we need to compute weights manually
            # from Q, K that were already processed
            pass
        return hook

    # Simpler approach: run forward with hooks on the projection outputs
    for layer_idx, layer in enumerate(model.layers):
        layer_maps = {}

        # --- Self attention ---
        sa = layer.self_attn
        q_captured = {}

        def make_qk_hook(attn_module, storage, prefix):
            def q_hook(mod, inp, out):
                storage[f"{prefix}_out"] = out.detach()
            return q_hook

        hq = sa.q_proj.register_forward_hook(make_qk_hook(sa, q_captured, "q"))
        hk = sa.k_proj.register_forward_hook(make_qk_hook(sa, q_captured, "k"))
        hooks.extend([hq, hk])

        # --- Cross attention ---
        ca = layer.cross_attn
        cross_captured = {}
        hcq = ca.q_proj.register_forward_hook(make_qk_hook(ca, cross_captured, "q"))
        hck = ca.k_proj.register_forward_hook(make_qk_hook(ca, cross_captured, "k"))
        hooks.extend([hcq, hck])

        captured[layer_idx] = {"self": q_captured, "cross": cross_captured}

    # Run forward pass
    # v_query is [D], needs to be [1, 1, D] for cross-attention KV
    context = v_query.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
    x = chain_input
    with torch.no_grad():
        for layer in model.layers:
            x = layer(x, context)

    # Compute attention weights from captured Q, K
    for layer_idx in range(len(model.layers)):
        sa = model.layers[layer_idx].self_attn
        ca = model.layers[layer_idx].cross_attn
        sc = captured[layer_idx]["self"]
        cc = captured[layer_idx]["cross"]

        layer_map = {}

        # Self-attention weights
        if "q_out" in sc and "k_out" in sc:
            H = sa.n_heads
            Dh = sa.head_dim
            q = sc["q_out"].view(1, -1, H, Dh).transpose(1, 2)  # [1, H, L, Dh]
            k = sc["k_out"].view(1, -1, H, Dh).transpose(1, 2)

            # Apply RoPE
            L = q.shape[2]
            rope = sa._get_rope(L, q.device)
            from cebcm.models.chain_head import _apply_rope
            q = _apply_rope(q, rope)
            k = _apply_rope(k, rope)

            # Compute attention weights
            scale = Dh ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            # Apply causal mask
            causal = torch.triu(torch.ones(L, L, device=attn.device) * float("-inf"), diagonal=1)
            attn = attn + causal
            attn = F.softmax(attn, dim=-1)
            layer_map["self_attn"] = attn[0].cpu()  # [H, L, L]

        # Cross-attention weights
        if "q_out" in cc and "k_out" in cc:
            H = ca.n_heads
            Dh = ca.head_dim
            q = cc["q_out"].view(1, -1, H, Dh).transpose(1, 2)
            k = cc["k_out"].view(1, -1, H, Dh).transpose(1, 2)
            scale = Dh ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            layer_map["cross_attn"] = attn[0].cpu()  # [H, L, 1]

        attention_maps.append(layer_map)

    # Remove hooks
    for h in hooks:
        h.remove()

    return attention_maps


# ── Generation with Full Diagnostics ─────────────────────────────

@torch.no_grad()
def run_generation(
    v_query: Tensor,
    num_steps: int = 1,
    v_target: Tensor | None = None,
    num_candidates: int = 1,
) -> GenerationResult:
    """
    Run ChainGenerator with full diagnostics.

    Args:
        v_query:  [D] question embedding
        num_steps: chain length (1 = System 1, N = System 2)
        v_target: [D] optional ground-truth for metrics
        num_candidates: >1 enables best-of-N with critic reranking

    Returns:
        GenerationResult with all diagnostics populated
    """
    if _state.generator is None:
        raise RuntimeError("ChainGenerator not loaded. Load checkpoint first.")

    model = _state.generator
    device = torch.device(_state.device)
    t0 = time.time()

    v_q = v_query.unsqueeze(0).to(device)  # [1, D]

    # ── Generate chain(s) ──
    if num_candidates > 1 and _state.critic is not None:
        # Best-of-N reranking
        candidates = []
        for _ in range(num_candidates):
            chain = model.generate(v_q, num_steps=num_steps)  # [1, N, D]
            candidates.append(chain[0])  # [N, D]

        # Rank by critic energy on final step
        best_idx = 0
        best_energy = float("inf")
        cand_info = []
        for i, c in enumerate(candidates):
            v_ans = c[-1:]  # [1, D]
            e = _state.critic(v_q[0:1], v_ans).item()
            cos = F.cosine_similarity(v_ans, v_target.unsqueeze(0).to(device), dim=-1).item() if v_target is not None else 0.0
            cand_info.append({"idx": i, "energy": e, "cos": cos})
            if e < best_energy:
                best_energy = e
                best_idx = i

        v_chain = candidates[best_idx]
    else:
        v_chain = model.generate(v_q, num_steps=num_steps)[0]  # [N, D]
        cand_info = None
        best_idx = None

    result = GenerationResult(
        mode="system1" if num_steps == 1 else "system2",
        num_steps=num_steps,
        v_query=v_query.cpu(),
        v_chain=v_chain.cpu(),
        v_target=v_target.cpu() if v_target is not None else None,
    )

    # ── Per-step metrics ──
    for i in range(v_chain.shape[0]):
        result.step_norms.append(v_chain[i].norm().item())
        if v_target is not None:
            cos = F.cosine_similarity(
                v_chain[i:i+1].to(device),
                v_target.unsqueeze(0).to(device), dim=-1
            ).item()
            result.step_cos_to_target.append(cos)

    # ── Critic energy along chain ──
    if _state.critic is not None:
        for i in range(v_chain.shape[0]):
            v_step = v_chain[i:i+1].to(device)
            e = _state.critic(v_q[0:1], v_step).item()
            result.step_critic_energies.append(e)

        # Energy at query position and answer
        e_q = _state.critic(v_q[0:1], v_q[0:1]).item()
        result.critic_energy_query = e_q
        if v_target is not None:
            e_a = _state.critic(v_q[0:1], v_target.unsqueeze(0).to(device)).item()
            result.critic_energy_answer = e_a

    # ── Extract attention maps ──
    # Re-run forward to capture attention (teacher-forced with generated chain)
    start = model.start_token.expand(1, -1, -1)  # [1, 1, D]
    decoder_input = torch.cat([start, v_chain[:-1].unsqueeze(0).to(device)], dim=1)  # [1, N, D]
    result.attention_maps = _extract_attention_maps(model, v_q[0], decoder_input)

    # ── Reranking info ──
    if cand_info is not None:
        result.rerank_candidates = cand_info
        result.rerank_best_idx = best_idx

    result.elapsed_ms = (time.time() - t0) * 1000
    _state.last_result = result
    return result


# ── Critic Analysis on Chain ─────────────────────────────────────

@torch.no_grad()
def analyze_critic_on_chain(
    v_query: Tensor,
    v_chain: Tensor,
    v_target: Tensor | None = None,
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

    # Per-step energy
    per_step = []
    for i in range(v_chain.shape[0]):
        v = v_chain[i:i+1].to(device)
        e = critic(v_q, v).item()
        per_step.append(e)

    # Target energy
    e_target = None
    if v_target is not None:
        e_target = critic(v_q, v_target.unsqueeze(0).to(device)).item()

    # Random negative energies
    target_norm = v_chain.norm(dim=-1).mean().item()
    random_vecs = F.normalize(torch.randn(num_random_negatives, v_query.shape[-1]), dim=-1) * target_norm
    random_energies = []
    for i in range(num_random_negatives):
        v = random_vecs[i:i+1].to(device)
        e = critic(v_q, v).item()
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
        "energy_query": critic(v_q, v_q).item(),
    }


# ── 3D Energy Landscape Around Chain Path ────────────────────────

@torch.no_grad()
def compute_chain_landscape(
    v_query: Tensor,
    v_chain: Tensor,
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
            e = critic(v_q, v.unsqueeze(0)).item()
            energy_grid[i, j] = e

    # Chain step energies
    chain_energies = []
    for i in range(N):
        e = critic(v_q, chain[i:i+1]).item()
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
    """Bar chart of cross-attention weights to v_query per head."""
    if not result.attention_maps or layer_idx >= len(result.attention_maps):
        fig = go.Figure()
        fig.update_layout(title="No attention data")
        return fig

    layer_map = result.attention_maps[layer_idx]
    if "cross_attn" not in layer_map:
        fig = go.Figure()
        fig.update_layout(title="Cross-attention not captured")
        return fig

    attn = layer_map["cross_attn"]  # [H, L, 1]
    avg = attn.mean(dim=0).squeeze(-1).numpy()  # [L]
    L = len(avg)
    labels = ["[START]"] + [f"step_{i+1}" for i in range(L - 1)]

    fig = go.Figure(data=go.Bar(x=labels, y=avg, marker_color="cyan"))
    fig.update_layout(
        title=f"Cross-Attention to Query (Layer {layer_idx}, avg heads)",
        xaxis_title="Chain Position",
        yaxis_title="Attention Weight",
        height=350,
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
        f"- **Steps**: {result.num_steps} | **Time**: {result.elapsed_ms:.0f} ms",
        "",
    ]

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

    if result.input_text:
        lines.append(f"#### Input Text\n> {result.input_text}\n")
    if result.target_text:
        lines.append(f"#### Target Text\n> {result.target_text}\n")

    return "\n".join(lines)


def export_metrics_json(result: GenerationResult) -> str:
    """Export metrics as JSON string."""
    data = {
        "mode": result.mode,
        "num_steps": result.num_steps,
        "elapsed_ms": result.elapsed_ms,
        "step_cos_to_target": result.step_cos_to_target,
        "step_norms": result.step_norms,
        "step_critic_energies": result.step_critic_energies,
        "critic_energy_answer": result.critic_energy_answer,
        "critic_energy_query": result.critic_energy_query,
        "critic_rank_acc": result.critic_rank_acc,
        "chain_texts": result.chain_texts,
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

def run_from_data(
    data_path: str,
    sample_idx: int = 0,
    num_steps: int = 1,
    num_candidates: int = 1,
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

    result = run_generation(v_query, num_steps=num_steps,
                            v_target=v_answer, num_candidates=num_candidates)

    # Compute 3D landscape
    if _state.critic is not None and result.v_chain is not None:
        result.landscape_data = compute_chain_landscape(
            v_query, result.v_chain, grid_size=grid_size,
        )

    # Critic analysis
    if _state.critic is not None and result.v_chain is not None:
        critic_info = analyze_critic_on_chain(v_query, result.v_chain, v_answer)
        result.critic_rank_acc = critic_info.get("rank_acc")

    return result


def run_from_text(
    text: str,
    num_steps: int = 1,
    num_candidates: int = 1,
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

    result = run_generation(v_query, num_steps=num_steps,
                            num_candidates=num_candidates)
    result.input_text = text

    # Decode chain steps to text
    if result.v_chain is not None:
        chain_for_decode = result.v_chain.to(device)
        result.chain_texts = _state.sonar.decode_safe(chain_for_decode)

    # Compute 3D landscape
    if _state.critic is not None and result.v_chain is not None:
        result.landscape_data = compute_chain_landscape(
            v_query, result.v_chain, grid_size=grid_size,
        )

    return result
