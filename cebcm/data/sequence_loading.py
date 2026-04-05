"""
Unified SONAR sequence loading with backward compatibility.

Supported payload formats:
  1) {"sequences": list[Tensor[L_i, D]], ...}
  2) {"vectors": Tensor[N, L, D], "lengths": Tensor[N], ...}
  3) {"embeddings": Tensor[N, D], "texts": list[str], ...}  (legacy)
  4) list[Tensor[L_i, D]]

Legacy `embeddings` format is converted to contiguous fixed-size windows so
Stage2/Stage3 sequence pipelines can consume old datasets directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def safe_torch_load(path: str | Path) -> Any:
    """
    Load a .pt payload with clearer diagnostics on common failures.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p.resolve()}")

    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - error path
        hint = ""
        try:
            with p.open("rb") as f:
                head = f.read(256).decode("utf-8", errors="ignore")
            if "git-lfs.github.com/spec/v1" in head:
                hint = (
                    "File looks like a Git LFS pointer, not real tensor data. "
                    "Run `git lfs pull` (if LFS is used) or regenerate dataset."
                )
        except Exception:
            pass

        msg = f"Failed to load dataset '{p.resolve()}': {exc}"
        if hint:
            msg = f"{msg}. {hint}"
        raise RuntimeError(msg) from exc


def _normalize_text_slice(entry: Any, length: int) -> list[str]:
    if entry is None or length <= 0:
        return []
    if isinstance(entry, str):
        return [entry][:length]
    if isinstance(entry, (list, tuple)):
        out: list[str] = []
        for item in entry[:length]:
            if isinstance(item, str):
                out.append(item)
        return out
    return []


def _append_sequence(
    out_sequences: list[Tensor],
    out_texts: list[list[str]],
    seq: Tensor,
    txt: Any,
    max_seq_len: int | None,
    min_seq_len: int,
) -> None:
    if not isinstance(seq, Tensor) or seq.dim() != 2:
        return
    if max_seq_len is not None:
        seq = seq[:max_seq_len]
    if seq.shape[0] < min_seq_len:
        return
    out_sequences.append(seq)
    out_texts.append(_normalize_text_slice(txt, int(seq.shape[0])))


def _from_legacy_embeddings(
    raw: dict[str, Any],
    max_seq_len: int | None,
    min_seq_len: int,
    legacy_window_stride: int | None,
) -> tuple[list[Tensor], list[list[str]], dict[str, Any]]:
    embeddings = raw.get("embeddings")
    if not isinstance(embeddings, Tensor):
        embeddings = torch.as_tensor(embeddings)
    if embeddings.dim() != 2:
        raise ValueError(
            f"Legacy payload has invalid embeddings shape {tuple(embeddings.shape)}; expected [N, D]."
        )

    n = int(embeddings.shape[0])
    window = int(max_seq_len) if max_seq_len is not None else n
    if window <= 0:
        raise ValueError(f"max_seq_len must be positive; got {window}")
    stride = int(legacy_window_stride) if legacy_window_stride is not None else window
    stride = max(1, stride)

    raw_texts = raw.get("texts", [])
    if not isinstance(raw_texts, (list, tuple)):
        raw_texts = []

    sequences: list[Tensor] = []
    texts: list[list[str]] = []
    for start in range(0, n, stride):
        end = min(start + window, n)
        if end - start < min_seq_len:
            continue
        sequences.append(embeddings[start:end])
        if raw_texts:
            texts.append(_normalize_text_slice(list(raw_texts[start:end]), end - start))
        else:
            texts.append([])

    meta = {
        "converted_from": "embeddings",
        "legacy_num_vectors": n,
        "legacy_window_len": window,
        "legacy_window_stride": stride,
        "legacy_num_sequences": len(sequences),
    }
    return sequences, texts, meta


def load_sonar_sequences(
    source: str | Path | Any,
    *,
    max_seq_len: int | None = None,
    min_seq_len: int = 1,
    legacy_window_stride: int | None = None,
) -> tuple[list[Tensor], list[list[str]], str, dict[str, Any]]:
    """
    Load SONAR sequences from path or raw payload.

    Returns:
        (sequences, texts, source_name, metadata)
    """
    raw = safe_torch_load(source) if isinstance(source, (str, Path)) else source

    sequences: list[Tensor] = []
    texts: list[list[str]] = []
    source_name = "unknown"
    metadata: dict[str, Any] = {}

    if isinstance(raw, dict):
        source_name = str(raw.get("source", "unknown"))
        metadata = dict(raw.get("metadata", {}))

        if "sequences" in raw:
            raw_seqs = raw["sequences"]
            raw_texts = raw.get("texts", [None] * len(raw_seqs))
            for idx, seq in enumerate(raw_seqs):
                txt = raw_texts[idx] if idx < len(raw_texts) else None
                _append_sequence(sequences, texts, seq, txt, max_seq_len, min_seq_len)
            metadata.setdefault("input_format", "sequences")

        elif "vectors" in raw:
            vectors = raw["vectors"]
            if not isinstance(vectors, Tensor):
                vectors = torch.as_tensor(vectors)
            if vectors.dim() != 3:
                raise ValueError(
                    f"Padded payload has invalid vectors shape {tuple(vectors.shape)}; expected [N, L, D]."
                )

            n, l_max, _ = vectors.shape
            lengths_raw = raw.get("lengths")
            if lengths_raw is None:
                lengths = torch.full((n,), l_max, dtype=torch.long)
            else:
                lengths = torch.as_tensor(lengths_raw, dtype=torch.long).view(-1)
                if lengths.numel() != n:
                    raise ValueError(
                        f"Lengths size mismatch: vectors has N={n}, lengths has {lengths.numel()}."
                    )

            raw_texts = raw.get("texts", [None] * n)
            for i in range(n):
                li = int(lengths[i].item())
                li = max(0, min(li, l_max))
                seq = vectors[i, :li]
                txt = raw_texts[i] if i < len(raw_texts) else None
                _append_sequence(sequences, texts, seq, txt, max_seq_len, min_seq_len)
            metadata.setdefault("input_format", "vectors")

        elif "embeddings" in raw:
            seqs, txts, legacy_meta = _from_legacy_embeddings(
                raw=raw,
                max_seq_len=max_seq_len,
                min_seq_len=min_seq_len,
                legacy_window_stride=legacy_window_stride,
            )
            sequences.extend(seqs)
            texts.extend(txts)
            source_name = str(raw.get("source", "legacy_embeddings"))
            metadata.update(legacy_meta)
            metadata.setdefault("input_format", "embeddings")

        else:
            raise KeyError(
                f"Unknown dataset payload keys: {list(raw.keys())}. "
                "Expected one of: 'sequences', 'vectors', 'embeddings'."
            )

    elif isinstance(raw, list):
        for seq in raw:
            _append_sequence(sequences, texts, seq, None, max_seq_len, min_seq_len)
        source_name = "list"
        metadata = {"input_format": "list"}

    else:
        raise TypeError(
            f"Unsupported dataset payload type: {type(raw)}. "
            "Expected dict or list."
        )

    if sequences and "d_model" not in metadata:
        metadata["d_model"] = int(sequences[0].shape[-1])
    metadata["num_sequences_loaded"] = len(sequences)
    metadata["min_seq_len_filter"] = int(min_seq_len)
    metadata["max_seq_len_trunc"] = None if max_seq_len is None else int(max_seq_len)

    return sequences, texts, source_name, metadata
