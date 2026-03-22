"""
Encode text dataset into SONAR embeddings for offline training.

Usage:
    python -m cebcm.data.encode_dataset \
        --num_sentences 10000 \
        --output data/wikitext_sonar_10k.pt

Produces a .pt file with:
    {"embeddings": Tensor[N, 1024], "texts": list[str], "norms": Tensor[N]}
"""

import argparse
import re
from pathlib import Path

import torch
from tqdm import tqdm


def load_wikitext_sentences(
    num_sentences: int,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    min_length: int = 30,
    max_length: int = 300,
) -> list[str]:
    """
    Load and filter sentences from WikiText-103.

    Filters out:
    - Section headers (lines starting with = or empty)
    - Too short sentences (< min_length chars)
    - Too long sentences (> max_length chars)

    Returns:
        List of clean sentences.
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, split="train")

    sentences: list[str] = []
    for item in dataset:
        text = item["text"].strip()
        if not text or text.startswith("="):
            continue
        # Split into sentences on period/exclamation/question
        for sent in re.split(r"(?<=[.!?])\s+", text):
            sent = sent.strip()
            if min_length <= len(sent) <= max_length:
                sentences.append(sent)
                if len(sentences) >= num_sentences:
                    return sentences

    return sentences


def encode_sentences(
    sentences: list[str],
    device: str = "cuda",
    batch_size: int = 64,
    lang: str = "eng_Latn",
) -> torch.Tensor:
    """
    Encode sentences to SONAR embeddings.

    Args:
        sentences: List of text strings.
        device: CUDA or CPU.
        batch_size: Encoding batch size.
        lang: Language code.

    Returns:
        Tensor [N, 1024]
    """
    from cebcm.models.sonar_wrapper import SONARWrapper

    sonar = SONARWrapper(device=device)
    all_embeddings = []

    for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding"):
        batch = sentences[i : i + batch_size]
        emb = sonar.encode(batch, lang=lang)
        all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Encode WikiText to SONAR vectors")
    parser.add_argument("--num_sentences", type=int, default=10000)
    parser.add_argument("--output", type=str, default="data/wikitext_sonar_10k.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--min_length", type=int, default=30)
    parser.add_argument("--max_length", type=int, default=300)
    args = parser.parse_args()

    print(f"Loading {args.num_sentences} sentences from {args.dataset_name}...")
    sentences = load_wikitext_sentences(
        num_sentences=args.num_sentences,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        min_length=args.min_length,
        max_length=args.max_length,
    )
    print(f"  Loaded {len(sentences)} sentences")

    print(f"Encoding to SONAR embeddings on {args.device}...")
    embeddings = encode_sentences(
        sentences, device=args.device, batch_size=args.batch_size
    )
    norms = embeddings.norm(dim=-1)
    print(f"  Shape: {embeddings.shape}")
    print(f"  Norms: mean={norms.mean():.4f}, std={norms.std():.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "embeddings": embeddings,
            "texts": sentences,
            "norms": norms,
        },
        output_path,
    )
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
