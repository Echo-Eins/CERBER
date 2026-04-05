#!/usr/bin/env python3
"""
Stage 3 Inference Test: System 1 vs System 2 comparison.

Tests the full inference pipeline on SQuAD SONAR data:
  1. Loads trained Chain Head + Pairwise critic checkpoints
  2. Runs System 1 (Fast Shot) and System 2 (Deep Thinking) on test pairs
  3. Compares cosine similarity to ground truth answers
  4. Optionally decodes results via SONAR decoder for qualitative inspection

Usage:
  # Quick test with synthetic data (no checkpoints needed):
  python test_inference.py --synthetic

  # Full test with trained models:
  python test_inference.py \
    --chain-ckpt checkpoints/best_chain_head.pt \
    --pairwise-ckpt ../../experiments/03_Stage_1.5/checkpoints/best_critic.pt \
    --data ../../data/squad_sequences.pt

  # With SONAR decoding (requires sonar-space):
  python test_inference.py --synthetic --decode
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cebcm.data.sequence_loading import load_sonar_sequences
from cebcm.models.chain_head import ChainHeadConfig, EBTChainHead
from cebcm.inference.system_switching import (
    System1Config,
    System2Config,
    ThinkingResult,
    run_system1,
    run_system2,
)


# ─── Model Loading ────────────────────────────────────────────────────

def load_chain_head(ckpt_path: str, device: torch.device) -> EBTChainHead:
    """
    Load trained Chain Head from checkpoint.

    Handles both checkpoint formats:
      - Phase A: state under "model", config flat or under "config"
      - Phase B: state under "chain_head", config nested under "config" -> "chain_head"
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Resolve config dict (may be flat or nested)
    cfg_dict = ckpt.get("config", {})
    if "chain_head" in cfg_dict:
        # Phase B saves full config — chain head params are nested
        cfg_dict = cfg_dict["chain_head"]

    cfg = ChainHeadConfig(
        d_model=cfg_dict.get("d_model", 1024),
        n_heads=cfg_dict.get("n_heads", 8),
        n_layers=cfg_dict.get("n_layers", 2),
        dim_feedforward=cfg_dict.get("dim_feedforward", 2048),
        max_chain_len=cfg_dict.get("max_chain_len", 20),
        dropout=0.0,  # No dropout at inference
        energy_hidden=cfg_dict.get("energy_hidden", 512),
        temperature=cfg_dict.get("temperature", 0.07),
    )
    model = EBTChainHead(cfg).to(device)

    # Resolve state dict key (Phase A: "model", Phase B: "chain_head")
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    elif "chain_head" in ckpt:
        model.load_state_dict(ckpt["chain_head"])
    else:
        available = [k for k in ckpt.keys() if not k.startswith("_")]
        raise KeyError(f"No chain head state found. Available keys: {available}")

    model.eval()
    print(f"  Chain Head loaded: {model.num_params:,} params from {ckpt_path}")
    if "val_metrics" in ckpt:
        print(f"  Checkpoint metrics: {ckpt['val_metrics']}")
    return model


def load_pairwise(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load trained pairwise energy critic.

    Handles two checkpoint formats:
      1. Stage 1.5 radial_angular: keys critic1_state (Angular), critic2_state (Radial)
      2. SimpleEnergy: key model or model_state_dict
    """
    from cebcm.models.energy import SimpleEnergy
    from cebcm.models.energy_decomposed import AngularEnergyCritic

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    # Stage 1.5 radial_angular format
    if "critic1_state" in ckpt:
        norm_mode = cfg.get("angular_norm_mode", cfg.get("norm_mode", "none"))
        activation = cfg.get("angular_activation", cfg.get("activation", "silu"))
        dim = cfg.get("energy_dim", 1024)
        hidden = cfg.get("angular_hidden_dims", [2048, 1024, 512])
        clamp = cfg.get("angular_energy_output_clamp", None)
        model = AngularEnergyCritic(
            dim=dim,
            hidden_dims=hidden,
            norm_mode=norm_mode,
            activation=activation,
            energy_output_clamp=clamp,
        )
        model.load_state_dict(ckpt["critic1_state"])
        model = model.to(device)
        model.eval()
        print(f"  Pairwise (AngularEnergyCritic) loaded from {ckpt_path}")
        return model

    # SimpleEnergy format
    if "model" in ckpt or "model_state_dict" in ckpt:
        model = SimpleEnergy(
            dim=cfg.get("dim", cfg.get("energy_dim", 1024)),
            hidden_dims=cfg.get("hidden_dims", cfg.get("energy_hidden_dims", [2048, 1024, 512])),
            norm_mode=cfg.get("norm_mode", "none"),
            activation=cfg.get("activation", "silu"),
        )
        state_key = "model" if "model" in ckpt else "model_state_dict"
        model.load_state_dict(ckpt[state_key])
        model = model.to(device)
        model.eval()
        print(f"  Pairwise (SimpleEnergy) loaded from {ckpt_path}")
        return model

    available = list(ckpt.keys())
    raise KeyError(
        f"Cannot find pairwise state in checkpoint. "
        f"Expected 'critic1_state' or 'model'. Available keys: {available}"
    )


def make_synthetic_pairwise(device: torch.device) -> torch.nn.Module:
    """Create a simple synthetic energy function for testing without checkpoint."""

    class SyntheticEnergy(torch.nn.Module):
        """Cosine-distance energy: E = 1 - cos(q, c). Simple but functional."""
        def forward(self, v_query: Tensor, v_candidate: Tensor) -> Tensor:
            cos = F.cosine_similarity(v_query, v_candidate, dim=-1)
            return 1.0 - cos  # [B]

        def energy_and_grad(self, v_query: Tensor, v_candidate: Tensor):
            """Compute energy and gradient (required by Langevin dynamics)."""
            v_candidate = v_candidate.detach().requires_grad_(True)
            energy = self.forward(v_query, v_candidate)
            grad = torch.autograd.grad(energy.sum(), v_candidate)[0]
            return energy.detach(), grad.detach()

    model = SyntheticEnergy().to(device)
    print("  Using synthetic cosine-distance energy (no pairwise checkpoint)")
    return model


# ─── Data Loading ─────────────────────────────────────────────────────

def load_test_pairs(
    data_path: str,
    n_samples: int = 20,
    seed: int = 42,
) -> list[dict]:
    """
    Load test pairs from SONAR sequence data.

    Creates (query, init, target) triples:
      - query = first vector of a sequence (context)
      - target = last vector (ground truth answer)
      - init = noisy version of target (simulates IPP output)
    """
    sequences, _, source, meta = load_sonar_sequences(
        data_path,
        min_seq_len=1,
    )
    print(
        f"  Sequence payload: source={source}, "
        f"format={meta.get('input_format', 'unknown')}"
    )

    # Filter sequences long enough for meaningful pairs
    valid = [s for s in sequences if s.shape[0] >= 5]
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(valid), generator=gen)[:n_samples]

    pairs = []
    for idx in perm:
        seq = valid[idx.item()]
        v_query = seq[0]    # Context
        v_target = seq[-1]  # Ground truth
        # Noisy init: interpolate between random and target (simulates IPP)
        noise = torch.randn_like(v_target) * 0.05
        v_init = F.normalize(v_target + noise, dim=-1) * v_target.norm()
        pairs.append({
            "v_query": v_query,
            "v_target": v_target,
            "v_init": v_init,
            "seq_len": seq.shape[0],
        })

    print(f"  Loaded {len(pairs)} test pairs from {data_path}")
    return pairs


def make_synthetic_pairs(n_samples: int = 20, d: int = 1024, seed: int = 42) -> list[dict]:
    """
    Generate synthetic test pairs for smoke testing without real data.

    Creates pairs where target = normalized(query + small offset).
    Langevin should converge v_init → v_target.
    """
    torch.manual_seed(seed)
    target_norm = 0.2051

    pairs = []
    for _ in range(n_samples):
        v_query = F.normalize(torch.randn(d), dim=-1) * target_norm
        # Target is nearby on SONAR sphere
        offset = torch.randn(d) * 0.02
        v_target = F.normalize(v_query + offset, dim=-1) * target_norm
        # Init is farther away (simulates starting point)
        noise = torch.randn(d) * 0.08
        v_init = F.normalize(v_query + noise, dim=-1) * target_norm
        pairs.append({
            "v_query": v_query,
            "v_target": v_target,
            "v_init": v_init,
            "seq_len": 10,
        })

    print(f"  Generated {len(pairs)} synthetic test pairs")
    return pairs


# ─── Inference Runner ─────────────────────────────────────────────────

def run_test(
    pairs: list[dict],
    pairwise_fn: torch.nn.Module,
    chain_head: EBTChainHead | None,
    device: torch.device,
    langevin_kwargs: dict,
) -> dict:
    """
    Run System 1 and System 2 on all test pairs and collect metrics.

    Returns dict with per-sample and aggregate results.
    """
    sys1_cfg = System1Config(
        max_steps_choices=[10],  # Fixed for reproducible comparison
        cruise_ratio_choices=[0.5],
    )
    sys2_cfg = System2Config(
        max_steps_choices=[50],  # Fixed for comparison
        cruise_ratio_choices=[0.0],
        chain_eval_every=5,
        backtrack_patience=30,
        max_chain_len=20,
    )

    results = {
        "system1": [],
        "system2": [],
    }

    for i, pair in enumerate(pairs):
        v_query = pair["v_query"].unsqueeze(0).to(device)
        v_target = pair["v_target"].unsqueeze(0).to(device)
        v_init = pair["v_init"].unsqueeze(0).to(device)

        # Baseline: cosine of init to target
        cos_init = F.cosine_similarity(v_init, v_target, dim=-1).item()

        # ── System 1 ──
        t0 = time.time()
        result1 = run_system1(
            energy_fn=pairwise_fn,
            v_query=v_query,
            v_init=v_init,
            cfg=sys1_cfg,
            langevin_kwargs=langevin_kwargs,
            v_target=v_target,
        )
        t1 = time.time()

        cos_sys1 = F.cosine_similarity(result1.v_final, v_target, dim=-1).item()
        results["system1"].append({
            "idx": i,
            "cos_init": cos_init,
            "cos_final": cos_sys1,
            "delta": cos_sys1 - cos_init,
            "steps": result1.num_steps,
            "time_ms": (t1 - t0) * 1000,
            "final_energy": result1.energy_trajectory[-1] if result1.energy_trajectory else 0,
        })

        # ── System 2 ──
        if chain_head is not None:
            t0 = time.time()
            result2 = run_system2(
                pairwise_fn=pairwise_fn,
                chain_head=chain_head,
                v_query=v_query,
                v_init=v_init,
                cfg=sys2_cfg,
                langevin_kwargs=langevin_kwargs,
                v_target=v_target,
            )
            t2 = time.time()

            cos_sys2 = F.cosine_similarity(result2.v_final, v_target, dim=-1).item()
            results["system2"].append({
                "idx": i,
                "cos_init": cos_init,
                "cos_final": cos_sys2,
                "delta": cos_sys2 - cos_init,
                "steps": result2.num_steps,
                "time_ms": (t2 - t0) * 1000,
                "final_energy": result2.energy_trajectory[-1] if result2.energy_trajectory else 0,
                "chain_energies": result2.chain_energies,
                "backtracks": result2.backtrack_count,
            })

        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{len(pairs)} samples")

    return results


# ─── Reporting ────────────────────────────────────────────────────────

def print_report(results: dict):
    """Print aggregate comparison between System 1 and System 2."""
    print("\n" + "=" * 70)
    print("INFERENCE TEST RESULTS")
    print("=" * 70)

    for mode in ["system1", "system2"]:
        data = results.get(mode, [])
        if not data:
            continue

        cos_init = [d["cos_init"] for d in data]
        cos_final = [d["cos_final"] for d in data]
        deltas = [d["delta"] for d in data]
        times = [d["time_ms"] for d in data]

        n = len(data)
        label = "System 1 (Fast Shot)" if mode == "system1" else "System 2 (Deep Thinking)"

        print(f"\n{'─' * 40}")
        print(f"  {label} ({n} samples)")
        print(f"{'─' * 40}")
        print(f"  cos(init, target):  mean={sum(cos_init)/n:.4f}")
        print(f"  cos(final, target): mean={sum(cos_final)/n:.4f}")
        print(f"  Improvement (Δcos): mean={sum(deltas)/n:.4f}  "
              f"min={min(deltas):.4f}  max={max(deltas):.4f}")
        print(f"  Steps: {data[0]['steps']}")
        print(f"  Time: mean={sum(times)/n:.1f}ms  total={sum(times):.0f}ms")

        # Positive improvement rate
        improved = sum(1 for d in deltas if d > 0)
        print(f"  Improved: {improved}/{n} ({100*improved/n:.0f}%)")

        if mode == "system2":
            backtracks = [d.get("backtracks", 0) for d in data]
            print(f"  Backtracks: mean={sum(backtracks)/n:.1f}  max={max(backtracks)}")

    # Head-to-head comparison
    s1 = results.get("system1", [])
    s2 = results.get("system2", [])
    if s1 and s2:
        n = min(len(s1), len(s2))
        sys2_wins = sum(1 for i in range(n) if s2[i]["cos_final"] > s1[i]["cos_final"])
        print(f"\n{'─' * 40}")
        print(f"  HEAD-TO-HEAD: System 2 wins {sys2_wins}/{n} ({100*sys2_wins/n:.0f}%)")
        avg_diff = sum(s2[i]["cos_final"] - s1[i]["cos_final"] for i in range(n)) / n
        print(f"  Avg cos advantage (Sys2 - Sys1): {avg_diff:+.4f}")
        print(f"{'─' * 40}")


def decode_samples(
    results: dict,
    pairs: list[dict],
    device: torch.device,
    n_show: int = 5,
):
    """Decode sample vectors to text via SONAR for qualitative inspection."""
    try:
        from cebcm.models.sonar_wrapper import SONARWrapper
    except ImportError:
        print("\n  [SKIP] SONAR decoder not available (pip install sonar-space)")
        return

    print(f"\n{'=' * 70}")
    print("DECODED SAMPLES (via SONAR)")
    print("=" * 70)

    sonar = SONARWrapper(device=str(device))

    s1 = results.get("system1", [])
    s2 = results.get("system2", [])

    for i in range(min(n_show, len(s1))):
        pair = pairs[i]
        v_query = pair["v_query"].unsqueeze(0).to(device)
        v_target = pair["v_target"].unsqueeze(0).to(device)
        v_init = pair["v_init"].unsqueeze(0).to(device)

        # Collect vectors to decode
        vecs = [v_query, v_target, v_init]
        labels = ["Query (V₀)", "Target (GT)", "Init (IPP)"]

        # For System 1 result, reconstruct v_final
        if i < len(s1):
            # We need to re-run or store v_final; for simplicity, note in report
            labels.append("System 1 final")
        if i < len(s2):
            labels.append("System 2 final")

        # Decode available vectors
        all_vecs = torch.cat(vecs, dim=0)
        texts = sonar.decode_safe(all_vecs)

        print(f"\n  Sample {i}:")
        for label, text in zip(labels[:len(texts)], texts):
            print(f"    {label}: {text[:120]}")
        print(f"    cos(init→target)={s1[i]['cos_init']:.4f}  "
              f"cos(sys1→target)={s1[i]['cos_final']:.4f}", end="")
        if i < len(s2):
            print(f"  cos(sys2→target)={s2[i]['cos_final']:.4f}")
        else:
            print()


# ─── Chain Head Smoke Test ────────────────────────────────────────────

def test_chain_head_ranking(
    chain_head: EBTChainHead,
    device: torch.device,
    n_tests: int = 10,
    target_norm: float = 0.2051,
):
    """
    Smoke test: does the Chain Head assign lower energy to coherent chains
    vs shuffled chains?

    This is the most basic sanity check — if this fails, the model hasn't
    learned anything useful.
    """
    print(f"\n{'=' * 70}")
    print("CHAIN HEAD RANKING SMOKE TEST")
    print("=" * 70)

    correct = 0
    total_gap = 0.0

    for i in range(n_tests):
        # Create a coherent chain: smooth interpolation on SONAR sphere
        L = 10
        D = 1024
        base = F.normalize(torch.randn(D), dim=-1) * target_norm
        direction = F.normalize(torch.randn(D), dim=-1) * 0.01
        chain = []
        for t in range(L):
            v = base + direction * t
            v = F.normalize(v, dim=-1) * target_norm
            chain.append(v)

        positive = torch.stack(chain).unsqueeze(0).to(device)  # [1, L, D]

        # Shuffled negative
        perm = torch.randperm(L)
        negative = positive[:, perm, :]  # [1, L, D]

        with torch.no_grad():
            E_pos = chain_head(positive).item()
            E_neg = chain_head(negative).item()

        gap = E_neg - E_pos
        total_gap += gap
        if E_pos < E_neg:
            correct += 1

        if i < 3:
            print(f"  Test {i}: E_pos={E_pos:.4f}  E_neg={E_neg:.4f}  gap={gap:+.4f}  "
                  f"{'OK' if E_pos < E_neg else 'FAIL'}")

    acc = correct / n_tests
    avg_gap = total_gap / n_tests
    print(f"\n  Ranking accuracy: {correct}/{n_tests} ({100*acc:.0f}%)")
    print(f"  Average energy gap: {avg_gap:+.4f}")

    if acc < 0.7:
        print("  WARNING: Chain Head cannot reliably rank coherent vs shuffled chains!")
        print("  Re-training Phase A with hard negatives is strongly recommended.")
    elif acc < 0.9:
        print("  MODERATE: Some ranking ability, but not robust enough for System 2.")
    else:
        print("  GOOD: Chain Head reliably ranks chains.")

    return acc


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 3 Inference Test")
    parser.add_argument("--chain-ckpt", default=None,
                        help="Path to Chain Head checkpoint (.pt)")
    parser.add_argument("--pairwise-ckpt", default=None,
                        help="Path to pairwise critic checkpoint (.pt)")
    parser.add_argument("--data", default=None,
                        help="Path to SONAR sequences (.pt)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (no checkpoints needed)")
    parser.add_argument("--decode", action="store_true",
                        help="Decode samples via SONAR (requires sonar-space)")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Number of test pairs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed)
    print(f"Device: {device}")

    # ── Load or create models ──
    chain_head = None
    if args.chain_ckpt:
        chain_head = load_chain_head(args.chain_ckpt, device)
    elif not args.synthetic:
        # Try default path
        default_path = Path(__file__).parent / "checkpoints" / "best_chain_head.pt"
        if default_path.exists():
            chain_head = load_chain_head(str(default_path), device)
        else:
            print(f"  No Chain Head checkpoint at {default_path}")
            print("  System 2 will be skipped. Use --chain-ckpt or --synthetic")
    else:
        # Synthetic: create untrained Chain Head for smoke testing
        cfg = ChainHeadConfig(d_model=1024, n_heads=8, n_layers=2, dropout=0.0)
        chain_head = EBTChainHead(cfg).to(device)
        chain_head.eval()
        print("  Using untrained Chain Head (synthetic mode)")

    if args.pairwise_ckpt:
        pairwise_fn = load_pairwise(args.pairwise_ckpt, device)
    else:
        pairwise_fn = make_synthetic_pairwise(device)

    # ── Load or generate test data ──
    if args.synthetic or not args.data:
        pairs = make_synthetic_pairs(args.n_samples, seed=args.seed)
    else:
        pairs = load_test_pairs(args.data, args.n_samples, args.seed)

    # ── Chain Head smoke test ──
    if chain_head is not None:
        test_chain_head_ranking(chain_head, device)

    # ── Run inference comparison ──
    langevin_kwargs = {
        "lr": 0.01,
        "noise_scale": 0.005,
        "target_norm": 0.2051,
    }

    print(f"\n{'=' * 70}")
    print("RUNNING INFERENCE COMPARISON")
    print("=" * 70)

    results = run_test(pairs, pairwise_fn, chain_head, device, langevin_kwargs)
    print_report(results)

    # ── Decode samples ──
    if args.decode:
        decode_samples(results, pairs, device)

    # ── Save results ──
    out_path = Path(__file__).parent / "logs" / "inference_test_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert tensors for JSON serialization
    serializable = {}
    for mode, data in results.items():
        serializable[mode] = []
        for d in data:
            sd = {}
            for k, v in d.items():
                if isinstance(v, list):
                    sd[k] = [float(x) if isinstance(x, (int, float)) else x for x in v]
                else:
                    sd[k] = v
            serializable[mode].append(sd)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
