"""
Stage 2: Energy Matching — Train UnconditionalEnergy on SONAR embeddings.

Pipeline:
    SONAR encoder (frozen) → pre-encoded .pt → [UnconditionalEnergy] → EM training
                                                  ^^^TRAINS^^^

Three training modes:
    1. energy_matching  — Pure EM loss from scratch
    2. nce_warmstart_em — NCE warmstart (10 epochs) → EM fine-tune
    3. cosine_em        — Cosine direction EM (for 1-Lipschitz networks)

Mathematical basis:
    Energy Matching (Balcerak et al., NeurIPS 2025) trains E_θ(x) such that
    -∇_x E_θ(x_t) ≈ u_t, where u_t = x₁ - x₀ is the OT velocity field.
    x_t = (1-t)·x₀ + t·x₁ interpolates between prior (x₀) and data (x₁).

    This is SIMULATION-FREE: no Langevin chains during training.
    The energy landscape simultaneously encodes transport + equilibrium.

Usage:
    # NCE warmstart → Energy Matching (recommended):
    python experiments/02_energy_matching/train.py \\
        --data data/wikitext_sonar_10k.pt --mode nce_warmstart_em

    # Pure Energy Matching:
    python experiments/02_energy_matching/train.py \\
        --data data/wikitext_sonar_10k.pt --mode energy_matching

    # Cosine EM (for 1-Lipschitz with orthonorm):
    python experiments/02_energy_matching/train.py \\
        --data data/wikitext_sonar_10k.pt --mode cosine_em

    # Ablation: spectral norm instead of orthonorm:
    python experiments/02_energy_matching/train.py \\
        --data data/wikitext_sonar_10k.pt --norm spectral_norm

Spec reference: §10.5, tasks/todo.md
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.energy_matching import EnergyMatchingConfig
from cebcm.models.energy_unconditional import UnconditionalEnergy
from cebcm.training.energy_matching import (
    energy_matching_loss,
    energy_matching_cosine,
    energy_matching_weighted,
    generate_samples_ode,
)
from cebcm.training.negative_buffer import (
    NegativeBuffer,
    nce_loss,
    nce_loss_simple,
)
from cebcm.data.dataset import SONARVectorDataset


# ============================================================
# Training epoch functions
# ============================================================

def train_epoch_em(
    model: UnconditionalEnergy,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: EnergyMatchingConfig,
    device: torch.device,
    epoch: int,
    global_step: int,
    loss_fn: str = "cosine_em",
) -> tuple[dict, int]:
    """Train one epoch with Energy Matching loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, v_clean in enumerate(dataloader):
        v_clean = v_clean.to(device)

        # Warmup
        if global_step < config.warmup_steps:
            warmup_factor = max(global_step / config.warmup_steps, 1e-6)
            for pg in optimizer.param_groups:
                pg["lr"] = config.lr * warmup_factor

        # Select loss function
        if loss_fn == "energy_matching":
            loss = energy_matching_loss(
                energy_fn=model,
                x_data=v_clean,
                prior_std=config.prior_std,
                t_min=config.t_min,
                t_max=config.t_max,
            )
        elif loss_fn == "cosine_em":
            loss = energy_matching_cosine(
                energy_fn=model,
                x_data=v_clean,
                prior_std=config.prior_std,
                t_min=config.t_min,
                t_max=config.t_max,
                magnitude_weight=config.magnitude_weight,
            )
        elif loss_fn == "weighted_em":
            loss = energy_matching_weighted(
                energy_fn=model,
                x_data=v_clean,
                prior_std=config.prior_std,
                t_min=config.t_min,
                t_max=config.t_max,
                near_data_weight=config.near_data_weight,
            )
        else:
            raise ValueError(f"Unknown loss: {loss_fn}")

        optimizer.zero_grad()
        loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0, out=p.grad)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Skip non-finite OR outlier spikes (finite but huge losses that
        # push the model in wrong directions despite gradient clipping)
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm) or loss.item() > 100.0:
            optimizer.zero_grad(set_to_none=True)
            if batch_idx < 5 or (config.log_every > 0 and (batch_idx + 1) % config.log_every == 0):
                print(f"  [WARN] bad EM loss={loss.item():.4g} at batch {batch_idx + 1}; step skipped")
            global_step += 1
            continue

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        if config.log_every > 0 and (batch_idx + 1) % config.log_every == 0:
            e_scale = model.log_energy_scale.exp().item()
            print(
                f"  [{batch_idx + 1}/{len(dataloader)}] "
                f"loss={loss.item():.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f} "
                f"E_scale={e_scale:.2f}"
            )

    return {
        "loss": total_loss / max(num_batches, 1),
    }, global_step


def train_epoch_nce(
    model: UnconditionalEnergy,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: EnergyMatchingConfig,
    neg_buffer: NegativeBuffer,
    device: torch.device,
    epoch: int,
    global_step: int,
) -> tuple[dict, int]:
    """Train one epoch with NCE loss using negative buffer."""
    model.train()
    total_loss = 0.0
    total_e_data = 0.0
    total_e_noise = 0.0
    num_batches = 0

    for batch_idx, v_clean in enumerate(dataloader):
        v_clean = v_clean.to(device)
        B = v_clean.shape[0]

        # Warmup
        if global_step < config.warmup_steps:
            warmup_factor = max(global_step / config.warmup_steps, 1e-6)
            for pg in optimizer.param_groups:
                pg["lr"] = config.lr * warmup_factor

        # Sample negatives from buffer
        v_noise = neg_buffer.sample(B, device)

        # NCE loss
        if config.nce_loss_type == "full":
            loss = nce_loss(model, v_clean, v_noise, prior_std=config.prior_std)
        else:
            loss = nce_loss_simple(model, v_clean, v_noise)

        optimizer.zero_grad()
        loss.backward()

        # Sanitize gradients: replace inf/nan with 0 so a single
        # exploding element doesn't corrupt model parameters forever.
        for p in model.parameters():
            if p.grad is not None:
                torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0, out=p.grad)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Skip non-finite OR outlier spikes
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm) or loss.item() > 100.0:
            optimizer.zero_grad(set_to_none=True)
            if batch_idx < 5 or (config.log_every > 0 and (batch_idx + 1) % config.log_every == 0):
                print(f"  [WARN] bad NCE loss={loss.item():.4g} at batch {batch_idx + 1}; step skipped")
            global_step += 1
            continue

        optimizer.step()

        # Track energies
        with torch.no_grad():
            e_data = model(v_clean).mean().item()
            e_noise = model(v_noise).mean().item()

        total_loss += loss.item()
        total_e_data += e_data
        total_e_noise += e_noise
        num_batches += 1
        global_step += 1

        if config.log_every > 0 and (batch_idx + 1) % config.log_every == 0:
            print(
                f"  [{batch_idx + 1}/{len(dataloader)}] "
                f"NCE_loss={loss.item():.4f} "
                f"E_data={e_data:.4f} E_noise={e_noise:.4f} "
                f"gap={e_noise - e_data:.4f}"
            )

    # Refresh buffer after each epoch
    neg_buffer.refresh(model, device)

    return {
        "loss": total_loss / max(num_batches, 1),
        "e_data_mean": total_e_data / max(num_batches, 1),
        "e_noise_mean": total_e_noise / max(num_batches, 1),
        "energy_gap": (total_e_noise - total_e_data) / max(num_batches, 1),
    }, global_step


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_energy_matching(
    model: UnconditionalEnergy,
    dataset: SONARVectorDataset,
    config: EnergyMatchingConfig,
    device: torch.device,
    num_samples: int = 100,
) -> dict:
    """
    Evaluate the energy model:
    1. Energy statistics on real data
    2. Denoising quality via gradient following
    3. Sample quality (if enough epochs have passed)
    """
    model.eval()
    results = {}

    # --- 1. Energy statistics on real data ---
    indices = torch.randperm(len(dataset))[:num_samples]
    v_clean = torch.stack([dataset[i.item()] for i in indices]).to(device)

    energies = model(v_clean)
    results["energy_data_mean"] = energies.mean().item()
    results["energy_data_std"] = energies.std().item()

    # --- 2. Denoising quality (gradient following) ---
    for noise_scale in config.eval_noise_scales:
        norms = v_clean.norm(dim=-1, keepdim=True)
        v_noisy = v_clean + torch.randn_like(v_clean) * noise_scale * norms

        cos_before = F.cosine_similarity(v_clean, v_noisy, dim=-1)

        # Follow -∇E for 50 steps (no stochastic noise)
        v_current = v_noisy.clone()
        for _ in range(50):
            v_grad = v_current.detach().requires_grad_(True)
            with torch.enable_grad():
                energy = model(v_grad)
                grad = torch.autograd.grad(energy.sum(), v_grad)[0]
            v_current = v_current - 0.01 * grad
            if config.target_norm is not None:
                v_current = F.normalize(v_current, dim=-1) * config.target_norm

        cos_after = F.cosine_similarity(v_clean, v_current, dim=-1)

        results[f"noise_{noise_scale}"] = {
            "cos_before_mean": cos_before.mean().item(),
            "cos_after_mean": cos_after.mean().item(),
            "improvement": (cos_after - cos_before).mean().item(),
            "success_rate": (cos_after > cos_before).float().mean().item(),
        }

    # --- 3. Sample quality (generate and measure statistics) ---
    n_gen = min(config.eval_generate_samples, 200)
    samples = generate_samples_ode(
        energy_fn=model,
        num_samples=n_gen,
        dim=config.energy_dim,
        num_steps=config.eval_ode_steps,
        prior_std=config.prior_std,
        target_norm=config.target_norm,
        device=device,
    )

    # Measure sample statistics
    sample_norms = samples.norm(dim=-1)
    data_norms = v_clean.norm(dim=-1)

    # Pairwise cosine similarity within samples
    samples_norm = F.normalize(samples, dim=-1)
    pairwise_cos = (samples_norm @ samples_norm.T).fill_diagonal_(0)
    n = samples_norm.shape[0]

    # Pairwise cosine within data
    data_norm = F.normalize(v_clean, dim=-1)
    data_pairwise_cos = (data_norm @ data_norm.T).fill_diagonal_(0)

    results["samples"] = {
        "norm_mean": sample_norms.mean().item(),
        "norm_std": sample_norms.std().item(),
        "data_norm_mean": data_norms.mean().item(),
        "pairwise_cos_mean": pairwise_cos.sum().item() / (n * (n - 1)),
        "data_pairwise_cos_mean": data_pairwise_cos.sum().item() / (num_samples * (num_samples - 1)),
        "energy_mean": model(samples).mean().item(),
    }

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Energy Matching training")
    parser.add_argument("--data", type=str, required=True, help="Path to .pt dataset")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None,
                        choices=["energy_matching", "nce_warmstart_em", "cosine_em", "weighted_em"],
                        help="Training mode")
    parser.add_argument("--norm", type=str, default=None,
                        choices=["orthonorm", "spectral_norm", "none"])
    parser.add_argument("--activation", type=str, default=None,
                        choices=["groupsort", "lipschitz_spline", "relu"])
    parser.add_argument("--no_nce_warmstart", action="store_true",
                        help="Disable NCE warmstart even in nce_warmstart_em mode")
    parser.add_argument("--ed", type=str, default=None,
                        help="Eval denoising frequency: 'none' to disable, or integer N for every N epochs (default: config.eval_every_epoch)")
    args = parser.parse_args()

    config = EnergyMatchingConfig()
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.mode is not None:
        if args.mode == "nce_warmstart_em":
            config.nce_warmstart = True
            config.loss_type = "cosine_em"
        elif args.mode in ("energy_matching", "cosine_em", "weighted_em"):
            config.loss_type = args.mode
            config.nce_warmstart = False
        else:
            config.loss_type = args.mode
    if args.norm is not None:
        config.norm_mode = args.norm
    if args.activation is not None:
        config.activation = args.activation
    if args.no_nce_warmstart:
        config.nce_warmstart = False
    config.use_wandb = args.wandb

    # --ed flag: eval denoising frequency
    if args.ed is not None:
        if args.ed.lower() == "none":
            config.eval_every_epoch = 0  # 0 = disabled
        else:
            config.eval_every_epoch = int(args.ed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──
    print(f"Loading dataset from {args.data}...")
    full_dataset = SONARVectorDataset(args.data)
    print(f"  Total vectors: {len(full_dataset)}, dim: {full_dataset.embeddings.shape[1]}")

    n_test = min(config.num_test_sentences, len(full_dataset) // 10)
    n_train = len(full_dataset) - n_test
    train_dataset = full_dataset.subset(0, n_train)
    test_dataset = full_dataset.subset(n_train, len(full_dataset))
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )

    # ── Model ──
    model = UnconditionalEnergy(
        dim=config.energy_dim,
        hidden_dims=config.energy_hidden_dims,
        norm_mode=config.norm_mode,
        activation=config.activation,
        ortho_n_iters=config.ortho_n_iters,
        groupsort_size=config.groupsort_size,
        spline_num_knots=config.spline_num_knots,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"UnconditionalEnergy: {num_params:,} parameters")
    print(f"  Architecture: norm={config.norm_mode}, activation={config.activation}")
    print(f"  Hidden dims: {config.energy_hidden_dims}")
    print(f"  Loss: {config.loss_type}")
    print(f"  NCE warmstart: {config.nce_warmstart} ({config.nce_epochs} epochs)")
    print(f"  Prior: N(0, {config.prior_std:.5f}²I), target_norm={config.target_norm}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=config.lr * 0.1
    )

    start_epoch = 0
    global_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        print(f"  Resumed at epoch {start_epoch}")

    # ── Negative Buffer ──
    neg_buffer = NegativeBuffer(
        buffer_size=config.buffer_size,
        dim=config.energy_dim,
        init_std=config.prior_std,
        refresh_fraction=config.buffer_refresh_fraction,
        langevin_steps=config.buffer_langevin_steps,
        langevin_lr=config.buffer_langevin_lr,
        langevin_noise=config.buffer_langevin_noise,
        target_norm=config.target_norm,
    )
    # Seed buffer with noisy data (much better than random init)
    neg_buffer.seed_from_data(train_dataset.embeddings, noise_scale=config.buffer_seed_noise)
    print(f"  Negative buffer: {config.buffer_size} vectors, seeded from data")

    # ── WandB ──
    wandb_run = None
    if config.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=config.wandb_project,
            config={
                "lr": config.lr,
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "loss_type": config.loss_type,
                "norm_mode": config.norm_mode,
                "activation": config.activation,
                "nce_warmstart": config.nce_warmstart,
                "nce_epochs": config.nce_epochs,
                "prior_std": config.prior_std,
                "target_norm": config.target_norm,
                "num_params": num_params,
            },
        )

    # ── Output dirs ──
    output_dir = Path(config.output_dir)
    checkpoint_dir = Path(config.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──
    print(f"\n{'='*60}")
    print(f"Energy Matching Training — {config.num_epochs} epochs")
    if config.nce_warmstart:
        print(f"  Phase 1: NCE warmstart ({config.nce_epochs} epochs)")
        print(f"  Phase 2: {config.loss_type} ({config.num_epochs - config.nce_epochs} epochs)")
    else:
        print(f"  Loss: {config.loss_type}")
    print(f"  Batch size: {config.batch_size}, LR: {config.lr}")
    print(f"  t range: [{config.t_min}, {config.t_max}]")
    print(f"{'='*60}\n")

    all_metrics: list[dict] = []
    best_improvement = -float("inf")
    start_time = time.time()

    for epoch in range(start_epoch, config.num_epochs):
        epoch_start = time.time()

        # Determine training mode for this epoch
        in_nce_phase = config.nce_warmstart and epoch < config.nce_epochs

        if in_nce_phase:
            phase_str = f"NCE [{epoch + 1}/{config.nce_epochs}]"
            print(f"Epoch {epoch + 1}/{config.num_epochs} ({phase_str})")
            train_metrics, global_step = train_epoch_nce(
                model, train_loader, optimizer, config,
                neg_buffer, device, epoch, global_step,
            )
        else:
            em_epoch = epoch - config.nce_epochs if config.nce_warmstart else epoch
            phase_str = f"EM [{em_epoch + 1}]"
            print(f"Epoch {epoch + 1}/{config.num_epochs} ({phase_str})")
            train_metrics, global_step = train_epoch_em(
                model, train_loader, optimizer, config,
                device, epoch, global_step,
                loss_fn=config.loss_type,
            )

        # Step scheduler
        if global_step >= config.warmup_steps:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        metrics_str = " ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
        print(f"  {metrics_str} lr={optimizer.param_groups[0]['lr']:.6f} ({epoch_time:.1f}s)")

        # ── Evaluate ──
        eval_metrics = None
        if config.eval_every_epoch > 0 and (
            (epoch + 1) % config.eval_every_epoch == 0 or epoch == config.num_epochs - 1
        ):
            print("  Evaluating...")
            eval_metrics = evaluate_energy_matching(
                model, test_dataset, config, device,
                num_samples=config.eval_num_samples,
            )

            # Print denoising results
            for key, val in eval_metrics.items():
                if key.startswith("noise_"):
                    print(
                        f"    {key}: cos {val['cos_before_mean']:.4f} → "
                        f"{val['cos_after_mean']:.4f} "
                        f"(Δ={val['improvement']:+.4f}, "
                        f"success={val['success_rate']:.0%})"
                    )

            # Print sample quality
            if "samples" in eval_metrics:
                s = eval_metrics["samples"]
                print(
                    f"    samples: norm={s['norm_mean']:.4f}±{s['norm_std']:.4f} "
                    f"(data={s['data_norm_mean']:.4f}), "
                    f"pairwise_cos={s['pairwise_cos_mean']:.4f} "
                    f"(data={s['data_pairwise_cos_mean']:.4f})"
                )

            # Check best
            avg_improvement = sum(
                v["improvement"] for k, v in eval_metrics.items() if k.startswith("noise_")
            ) / max(1, sum(1 for k in eval_metrics if k.startswith("noise_")))

            if avg_improvement > best_improvement:
                best_improvement = avg_improvement
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "metrics": eval_metrics,
                        "config": {
                            "energy_dim": config.energy_dim,
                            "energy_hidden_dims": config.energy_hidden_dims,
                            "norm_mode": config.norm_mode,
                            "activation": config.activation,
                            "loss_type": config.loss_type,
                        },
                    },
                    checkpoint_dir / "best.pt",
                )
                print(f"    * New best (avg improvement: {avg_improvement:+.4f})")

        # Log
        epoch_record = {"epoch": epoch + 1, "train": train_metrics, "phase": "nce" if in_nce_phase else "em"}
        if eval_metrics is not None:
            epoch_record["eval"] = eval_metrics
        all_metrics.append(epoch_record)

        if wandb_run is not None:
            log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
            log_dict["train/lr"] = optimizer.param_groups[0]["lr"]
            log_dict["train/phase"] = 0 if in_nce_phase else 1
            if eval_metrics is not None:
                for key, val in eval_metrics.items():
                    if isinstance(val, dict):
                        for mk, mv in val.items():
                            log_dict[f"eval/{key}/{mk}"] = mv
                    else:
                        log_dict[f"eval/{key}"] = val
            wandb_run.log(log_dict, step=epoch + 1)

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                },
                checkpoint_dir / f"epoch_{epoch + 1:03d}.pt",
            )

    # ── Final evaluation ──
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")

    print("\nFinal evaluation...")
    model.eval()
    final_eval = evaluate_energy_matching(
        model, test_dataset, config, device,
        num_samples=min(200, len(test_dataset)),
    )

    for key, val in final_eval.items():
        if key.startswith("noise_"):
            print(
                f"  {key}: cos {val['cos_before_mean']:.4f} → "
                f"{val['cos_after_mean']:.4f} "
                f"(Δ={val['improvement']:+.4f}, "
                f"success={val['success_rate']:.0%})"
            )

    # Save final
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": config.num_epochs - 1,
            "global_step": global_step,
            "final_eval": final_eval,
            "config": {
                "energy_dim": config.energy_dim,
                "energy_hidden_dims": config.energy_hidden_dims,
                "norm_mode": config.norm_mode,
                "activation": config.activation,
                "loss_type": config.loss_type,
            },
        },
        checkpoint_dir / "final.pt",
    )

    # Save metrics
    summary = {
        "total_time_seconds": total_time,
        "num_epochs": config.num_epochs,
        "best_improvement": best_improvement,
        "final_eval": final_eval,
        "config": {
            "lr": config.lr,
            "batch_size": config.batch_size,
            "loss_type": config.loss_type,
            "norm_mode": config.norm_mode,
            "activation": config.activation,
            "nce_warmstart": config.nce_warmstart,
            "nce_epochs": config.nce_epochs,
            "prior_std": config.prior_std,
            "target_norm": config.target_norm,
        },
        "epoch_metrics": all_metrics,
    }
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved to {output_dir / 'training_metrics.json'}")

    # ── Kill criterion ──
    print(f"\n{'='*60}")
    print("KILL CRITERION CHECK")
    print(f"{'='*60}")
    all_pass = True
    for key, val in final_eval.items():
        if key.startswith("noise_"):
            passed = val["improvement"] > 0 and val["success_rate"] > 0.5
            status = "PASS" if passed else "FAIL"
            print(f"  {key}: improvement={val['improvement']:+.4f}, success={val['success_rate']:.0%} → {status}")
            if not passed:
                all_pass = False

    if all_pass:
        print("\n  VERDICT: Energy Matching PASSED — gradients improve denoising")
    else:
        print("\n  VERDICT: Energy Matching FAILED — review architecture or hyperparameters")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
