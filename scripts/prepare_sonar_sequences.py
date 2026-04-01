#!/usr/bin/env python3
"""
Prepare SONAR sequence datasets for Stage 2 autoregressor training.

Downloads datasets from HuggingFace, splits text into sentences,
encodes each sentence with SONAR, and saves as sequential .pt files.

Supported datasets:
  - SQuAD v2:       QA pairs → (context_sentences, question, answer) sequences
  - CNN/DailyMail:  Articles → paragraph-level sentence sequences
  - WikiText-103:   Articles → contiguous sentence sequences

Output format (.pt):
{
    "sequences": list[Tensor],  # variable-length [L_i, 1024]
    "texts": list[list[str]],   # original sentences per sequence
    "source": str,
    "metadata": {
        "num_sequences": int,
        "mean_length": float,
        "sonar_mean_norm": float,
        "encoding_batch_size": int,
    },
}

Usage:
    python scripts/prepare_sonar_sequences.py \
        --dataset squad \
        --output data/squad_sequences.pt \
        --max_sequences 10000 \
        --device cuda

    python scripts/prepare_sonar_sequences.py \
        --dataset cnn_dailymail \
        --output data/cnn_sequences.pt \
        --max_sequences 5000

    python scripts/prepare_sonar_sequences.py \
        --dataset wikitext \
        --output data/wikitext_sequences.pt \
        --max_sequences 8000
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm


# ============================================================
# Sentence splitting
# ============================================================

def split_sentences(text: str, min_len: int = 20, max_len: int = 500) -> list[str]:
    """
    Split text into sentences, filtering by length.

    Uses regex-based splitting on sentence-ending punctuation.
    Filters out headers, very short fragments, and overly long sentences.
    """
    # Clean up whitespace
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []

    # Split on sentence boundaries
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text)

    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) < min_len or len(s) > max_len:
            continue
        # Skip headers and section markers
        if s.startswith("=") or s.startswith("#"):
            continue
        # Must contain at least one alphabetic char
        if not any(c.isalpha() for c in s):
            continue
        sentences.append(s)

    return sentences


# ============================================================
# Dataset loaders
# ============================================================

def load_squad_sequences(
        max_sequences: int = 10000,
        min_context_sents: int = 3,
) -> list[list[str]]:
    """
    Load SQuAD v2 as sequences of (context_sentences + question + answer).

    Each sequence = [sent1, sent2, ..., question, answer]
    This teaches the model to process context → question → answer flows.
    """
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad_v2", split="train")

    sequences: list[list[str]] = []
    seen_contexts = set()

    for item in tqdm(ds, desc="Loading SQuAD"):
        if len(sequences) >= max_sequences:
            break

        context = item["context"].strip()
        question = item["question"].strip()
        answers = item["answers"]["text"]

        if not answers:  # unanswerable
            continue

        answer = answers[0].strip()
        if len(answer) < 5:
            continue

        # Deduplicate by context to avoid near-identical sequences
        ctx_hash = hash(context[:200])
        if ctx_hash in seen_contexts:
            continue
        seen_contexts.add(ctx_hash)

        # Split context into sentences
        ctx_sents = split_sentences(context, min_len=15, max_len=400)
        if len(ctx_sents) < min_context_sents:
            continue

        # Build sequence: context sentences + question + answer
        seq = ctx_sents + [question, answer]
        sequences.append(seq)

    print(f"  SQuAD: {len(sequences)} sequences loaded")
    return sequences


def load_cnn_sequences(
        max_sequences: int = 5000,
        min_sents: int = 5,
        max_sents: int = 40,
) -> list[list[str]]:
    """
    Load CNN/DailyMail articles as sentence sequences.

    Each sequence = article split into sentences.
    Provides longer narrative context for training.
    """
    from datasets import load_dataset

    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")

    sequences: list[list[str]] = []

    for item in tqdm(ds, desc="Loading CNN/DailyMail"):
        if len(sequences) >= max_sequences:
            break

        article = item["article"].strip()
        sents = split_sentences(article, min_len=20, max_len=400)

        if len(sents) < min_sents:
            continue

        # Truncate very long articles
        sents = sents[:max_sents]
        sequences.append(sents)

    print(f"  CNN/DailyMail: {len(sequences)} sequences loaded")
    return sequences


def load_wikitext_sequences(
        max_sequences: int = 8000,
        min_sents: int = 4,
        max_sents: int = 32,
        window_stride: int = 8,
) -> list[list[str]]:
    """
    Load WikiText-103 as sliding-window sentence sequences.

    Collects all sentences from the corpus, then creates overlapping
    windows of contiguous sentences. This provides diverse context
    patterns for surprise predictor training.
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    # Collect all sentences
    all_sentences: list[str] = []
    for item in ds:
        text = item["text"].strip()
        if not text or text.startswith("="):
            continue
        sents = split_sentences(text, min_len=20, max_len=400)
        all_sentences.extend(sents)

    print(f"  WikiText: {len(all_sentences)} total sentences")

    # Create sliding windows
    sequences: list[list[str]] = []
    for start in range(0, len(all_sentences) - min_sents, window_stride):
        if len(sequences) >= max_sequences:
            break
        end = min(start + max_sents, len(all_sentences))
        window = all_sentences[start:end]
        if len(window) >= min_sents:
            sequences.append(window)

    print(f"  WikiText: {len(sequences)} windowed sequences")
    return sequences


DATASET_LOADERS = {
    "squad": load_squad_sequences,
    "cnn_dailymail": load_cnn_sequences,
    "wikitext": load_wikitext_sequences,
}


# ============================================================
# SONAR encoding
# ============================================================

def encode_sequences(
        sequences: list[list[str]],
        device: str = "cuda",
        batch_size: int = 64,
        lang: str = "eng_Latn",
) -> list[torch.Tensor]:
    """
    Encode all sequences to SONAR vectors.

    Batches all sentences across all sequences for efficient encoding,
    then reassembles into per-sequence tensors.

    Returns:
        List of Tensors, each [L_i, 1024]
    """
    from cebcm.models.sonar_wrapper import SONARWrapper

    sonar = SONARWrapper(device=device)

    # Flatten all sentences with an index map
    flat_sentences: list[str] = []
    seq_boundaries: list[tuple[int, int]] = []  # (start, end) in flat list
    for seq in sequences:
        start = len(flat_sentences)
        flat_sentences.extend(seq)
        seq_boundaries.append((start, len(flat_sentences)))

    print(f"  Encoding {len(flat_sentences)} total sentences...")

    # Encode in batches
    all_embeddings: list[torch.Tensor] = []
    for i in tqdm(range(0, len(flat_sentences), batch_size), desc="SONAR encoding"):
        batch = flat_sentences[i: i + batch_size]
        emb = sonar.encode(batch, lang=lang)
        all_embeddings.append(emb.cpu())

    flat_embeddings = torch.cat(all_embeddings, dim=0)  # [total_sents, 1024]
    print(f"  Encoded: {flat_embeddings.shape}")

    # Reassemble into per-sequence tensors
    encoded_sequences: list[torch.Tensor] = []
    for start, end in seq_boundaries:
        encoded_sequences.append(flat_embeddings[start:end])

    return encoded_sequences


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare SONAR sequence datasets for Stage 2 training"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_LOADERS.keys()),
        help="Dataset to process",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .pt file path",
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=10000,
        help="Maximum number of sequences to process",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for SONAR encoding",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="SONAR encoding batch size",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="eng_Latn",
        help="Source language (FLORES-200 code)",
    )
    parser.add_argument(
        "--min_sents",
        type=int,
        default=3,
        help="Minimum sentences per sequence",
    )
    parser.add_argument(
        "--max_sents",
        type=int,
        default=40,
        help="Maximum sentences per sequence",
    )
    args = parser.parse_args()

    print(f"=== Preparing {args.dataset} sequences ===")

    # Load text sequences
    loader_kwargs: dict = {"max_sequences": args.max_sequences}
    if args.dataset == "squad":
        loader_kwargs["min_context_sents"] = args.min_sents
    elif args.dataset in ("cnn_dailymail", "wikitext"):
        loader_kwargs["min_sents"] = args.min_sents
        loader_kwargs["max_sents"] = args.max_sents

    sequences = DATASET_LOADERS[args.dataset](**loader_kwargs)

    if not sequences:
        print("ERROR: No sequences loaded!", file=sys.stderr)
        sys.exit(1)

    # Encode to SONAR
    encoded = encode_sequences(
        sequences,
        device=args.device,
        batch_size=args.batch_size,
        lang=args.lang,
    )

    # Compute statistics
    lengths = [s.shape[0] for s in encoded]
    norms = [s.norm(dim=-1).mean().item() for s in encoded]

    metadata = {
        "num_sequences": len(encoded),
        "mean_length": sum(lengths) / len(lengths),
        "max_length": max(lengths),
        "min_length": min(lengths),
        "sonar_mean_norm": sum(norms) / len(norms),
        "encoding_batch_size": args.batch_size,
        "lang": args.lang,
    }

    print(f"\n  Sequences: {metadata['num_sequences']}")
    print(f"  Mean length: {metadata['mean_length']:.1f}")
    print(f"  Length range: [{metadata['min_length']}, {metadata['max_length']}]")
    print(f"  Mean SONAR norm: {metadata['sonar_mean_norm']:.4f}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "sequences": encoded,
            "texts": sequences,
            "source": args.dataset,
            "metadata": metadata,
        },
        output_path,
    )
    print(f"\n  Saved to {output_path}")
    print(f"  File size: {output_path.stat().st_size / (1024 ** 2):.1f} MB")


if __name__ == "__main__":
    main()