#!/usr/bin/env python3
"""
HotpotQA → SONAR dataset preparation pipeline.

Downloads HotpotQA from HuggingFace, extracts QA triplets with reasoning
chains (supporting facts), encodes all text through SONAR, and saves as
a .pt file ready for autoregressor training.

Output format:
    {
        "samples": [
            {
                "question": str,
                "answer": str,
                "reasoning_steps": [str, ...],
                "v_question": Tensor[1024],
                "v_answer": Tensor[1024],
                "v_steps": Tensor[N, 1024],
                "type": str,  # "bridge" or "comparison"
            },
            ...
        ],
        "meta": {
            "dataset": "hotpot_qa",
            "config": "distractor",
            "split": str,
            "n_samples": int,
            "sonar_dim": 1024,
        }
    }

Usage:
    # Full pipeline (requires SONAR + CUDA):
    py -3 scripts/prepare_hotpotqa.py --output data/hotpotqa_sonar.pt

    # Download + parse only (no SONAR, saves raw text):
    py -3 scripts/prepare_hotpotqa.py --output data/hotpotqa_raw.pt --skip-encode

    # Smaller subset for testing:
    py -3 scripts/prepare_hotpotqa.py --max-samples 500 --output data/hotpotqa_sonar_500.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def download_hotpotqa(
    split: str = "train",
    config: str = "distractor",
) -> list[dict]:
    """Download HotpotQA from HuggingFace datasets."""
    from datasets import load_dataset

    print(f"Downloading HotpotQA ({config}, {split})...")
    ds = load_dataset("hotpot_qa", config, split=split)
    print(f"  Downloaded {len(ds)} samples")
    return list(ds)


def extract_reasoning_chain(sample: dict) -> list[str]:
    """
    Extract ordered reasoning steps from HotpotQA supporting_facts.

    supporting_facts = {"title": [str, ...], "sent_id": [int, ...]}
    context = {"title": [str, ...], "sentences": [[str, ...], ...]}

    Returns list of supporting fact sentences in order.
    """
    sf_titles = sample["supporting_facts"]["title"]
    sf_sent_ids = sample["supporting_facts"]["sent_id"]

    # Build title → sentences lookup
    ctx_titles = sample["context"]["title"]
    ctx_sentences = sample["context"]["sentences"]
    title_to_sents: dict[str, list[str]] = {}
    for title, sents in zip(ctx_titles, ctx_sentences):
        title_to_sents[title] = sents

    # Extract supporting fact sentences in order
    steps = []
    for title, sent_id in zip(sf_titles, sf_sent_ids):
        if title in title_to_sents:
            sents = title_to_sents[title]
            if 0 <= sent_id < len(sents):
                sent = sents[sent_id].strip()
                if sent and sent not in steps:  # deduplicate
                    steps.append(sent)

    return steps


def parse_samples(
    raw_data: list[dict],
    max_samples: int | None = None,
    min_steps: int = 2,
    max_answer_len: int = 200,
    min_answer_len: int = 1,
) -> list[dict]:
    """
    Parse HotpotQA samples into clean QA triplets with reasoning chains.

    Filters out samples with too few reasoning steps or degenerate answers.
    """
    samples = []
    skipped = {"short_answer": 0, "long_answer": 0, "few_steps": 0}

    for item in raw_data:
        answer = item["answer"].strip()
        if len(answer) < min_answer_len:
            skipped["short_answer"] += 1
            continue
        if len(answer) > max_answer_len:
            skipped["long_answer"] += 1
            continue

        steps = extract_reasoning_chain(item)
        if len(steps) < min_steps:
            skipped["few_steps"] += 1
            continue

        samples.append({
            "question": item["question"].strip(),
            "answer": answer,
            "reasoning_steps": steps,
            "type": item.get("type", "unknown"),
        })

        if max_samples is not None and len(samples) >= max_samples:
            break

    print(f"  Parsed {len(samples)} valid samples")
    print(f"  Skipped: {skipped}")
    return samples


def encode_samples_sonar(
    samples: list[dict],
    device: str = "cuda",
    batch_size: int = 64,
) -> list[dict]:
    """
    Encode all text fields through SONAR encoder.

    Adds v_question, v_answer, v_steps tensors to each sample.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cebcm.models.sonar_wrapper import SONARWrapper

    sonar = SONARWrapper(device=device)
    print(f"  SONAR encoder loaded on {device}")

    # Collect all texts for batched encoding
    all_questions = [s["question"] for s in samples]
    all_answers = [s["answer"] for s in samples]

    # Flatten all reasoning steps with index tracking
    step_texts: list[str] = []
    step_offsets: list[tuple[int, int]] = []  # (start, end) per sample
    for s in samples:
        start = len(step_texts)
        step_texts.extend(s["reasoning_steps"])
        step_offsets.append((start, len(step_texts)))

    print(f"  Encoding {len(all_questions)} questions...")
    t0 = time.time()
    v_questions = sonar.encode_batched(all_questions, batch_size=batch_size)
    print(f"    Done in {time.time() - t0:.1f}s")

    print(f"  Encoding {len(all_answers)} answers...")
    t0 = time.time()
    v_answers = sonar.encode_batched(all_answers, batch_size=batch_size)
    print(f"    Done in {time.time() - t0:.1f}s")

    print(f"  Encoding {len(step_texts)} reasoning steps...")
    t0 = time.time()
    v_all_steps = sonar.encode_batched(step_texts, batch_size=batch_size)
    print(f"    Done in {time.time() - t0:.1f}s")

    # Assign vectors to samples
    encoded = []
    for i, s in enumerate(samples):
        start, end = step_offsets[i]
        encoded.append({
            **s,
            "v_question": v_questions[i].cpu(),
            "v_answer": v_answers[i].cpu(),
            "v_steps": v_all_steps[start:end].cpu(),
        })

    # Stats
    norms_q = v_questions.norm(dim=-1)
    norms_a = v_answers.norm(dim=-1)
    print(f"  Question norms: mean={norms_q.mean():.4f}, std={norms_q.std():.4f}")
    print(f"  Answer norms:   mean={norms_a.mean():.4f}, std={norms_a.std():.4f}")

    cos_qa = torch.nn.functional.cosine_similarity(v_questions, v_answers, dim=-1)
    print(f"  cos(question, answer): mean={cos_qa.mean():.4f}, std={cos_qa.std():.4f}")

    return encoded


def main():
    parser = argparse.ArgumentParser(description="Prepare HotpotQA for CEBM training")
    parser.add_argument("--output", default="data/hotpotqa_sonar.pt",
                        help="Output .pt file path")
    parser.add_argument("--split", default="train",
                        help="HuggingFace split: train or validation")
    parser.add_argument("--config", default="distractor",
                        help="HotpotQA config: distractor or fullwiki")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (for testing)")
    parser.add_argument("--min-steps", type=int, default=2,
                        help="Minimum reasoning steps per sample")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="SONAR encoding batch size")
    parser.add_argument("--device", default=None,
                        help="Device for SONAR (auto-detect if not specified)")
    parser.add_argument("--skip-encode", action="store_true",
                        help="Skip SONAR encoding (save raw text only)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of data for validation split")
    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        if not args.skip_encode:
            print("WARNING: No CUDA available. SONAR encoding will be slow on CPU.")

    # Download
    raw = download_hotpotqa(split=args.split, config=args.config)

    # Parse
    samples = parse_samples(
        raw,
        max_samples=args.max_samples,
        min_steps=args.min_steps,
    )

    if not samples:
        print("ERROR: No valid samples after filtering!")
        sys.exit(1)

    # Print sample for sanity check
    s0 = samples[0]
    print(f"\n  Sample 0:")
    print(f"    Q: {s0['question'][:100]}")
    print(f"    A: {s0['answer'][:100]}")
    print(f"    Steps ({len(s0['reasoning_steps'])}):")
    for j, step in enumerate(s0["reasoning_steps"][:3]):
        print(f"      {j}: {step[:80]}...")
    print(f"    Type: {s0['type']}")

    # Encode through SONAR
    if not args.skip_encode:
        samples = encode_samples_sonar(samples, device=device, batch_size=args.batch_size)

    # Train/val split
    n = len(samples)
    n_val = max(1, int(n * args.val_split))
    n_train = n - n_val

    gen = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=gen).tolist()
    train_samples = [samples[i] for i in perm[:n_train]]
    val_samples = [samples[i] for i in perm[n_train:]]

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "train": train_samples,
        "val": val_samples,
        "meta": {
            "dataset": "hotpot_qa",
            "config": args.config,
            "split": args.split,
            "n_train": n_train,
            "n_val": n_val,
            "n_total": n,
            "sonar_dim": 1024,
            "encoded": not args.skip_encode,
            "min_steps": args.min_steps,
        },
    }

    torch.save(payload, str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved to {output_path} ({size_mb:.1f} MB)")
    print(f"  Train: {n_train} samples")
    print(f"  Val:   {n_val} samples")


if __name__ == "__main__":
    main()
