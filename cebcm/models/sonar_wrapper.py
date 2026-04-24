"""
SONAR Wrapper — unified interface for Meta's SONAR encoder/decoder.

SONAR (Sentence-level multimOdal and laNguage-Agnostic Representations):
- Encoder: text → 1024d vector (one vector per sentence)
- Decoder: 1024d vector → text
- 200+ languages supported

Installation:
    # 1. Install fairseq2 with matching PyTorch/CUDA version FIRST:
    pip install fairseq2 --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.6.0/cu124
    # 2. Then install sonar-space:
    pip install sonar-space
"""

import os
import re
import shutil

import torch
import torch.nn.functional as F
from torch import Tensor


# fairseq2 raises a message of the form
#   "Model checkpoint of the <asset> asset card is erroneous.  Make sure
#    that it is downloaded correctly and, if not, delete your cached
#    version at <path>."
# when an interrupted download leaves a partial "tmpXXXX" file in the
# asset cache.  The loader refuses to reuse that file on the next run.
# We detect this exact message, scrub the offending cache entry, and
# retry once so that fairseq2 re-downloads the checkpoint cleanly.
_FAIRSEQ2_CORRUPT_CACHE_RE = re.compile(
    r"delete your cached version at\s+(\S+)",
    flags=re.IGNORECASE,
)


def _cleanup_fairseq2_cache(error_message: str) -> str | None:
    """Remove a corrupt fairseq2 cache entry referenced by ``error_message``.

    Returns the path that was removed, or ``None`` if the message did not
    reference a cache path or the path could not be removed.
    """
    match = _FAIRSEQ2_CORRUPT_CACHE_RE.search(error_message)
    if not match:
        return None

    raw_path = match.group(1).rstrip(".,;:")
    if not raw_path:
        return None

    # The message points at the temp file inside the asset directory.
    # Remove the entire asset directory so every sibling (partial manifests,
    # lock files, etc.) is purged and fairseq2 re-downloads from scratch.
    if os.path.isfile(raw_path):
        target = os.path.dirname(raw_path) or raw_path
    else:
        target = raw_path

    if not target or not os.path.exists(target):
        # Path no longer exists — maybe a concurrent process already cleaned
        # it; treat as "best effort succeeded" so the caller retries.
        return target if target else None

    try:
        if os.path.isdir(target):
            shutil.rmtree(target)
        else:
            os.remove(target)
    except OSError:
        return None
    return target


class SONARWrapper:
    """
    Unified wrapper for SONAR encoder and decoder.

    Usage:
        sonar = SONARWrapper(device="cuda")
        V = sonar.encode(["Hello, world!"])          # [1, 1024]
        texts = sonar.decode(V)                       # ["Hello, world!"]
        V_batch = sonar.encode_batched(big_list, batch_size=64)  # [N, 1024]
    """

    def __init__(
        self,
        encoder_name: str = "text_sonar_basic_encoder",
        decoder_name: str = "text_sonar_basic_decoder",
        tokenizer_name: str = "text_sonar_basic_encoder",
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.tokenizer_name = tokenizer_name

        self._encoder = None
        self._decoder = None

    def _load_encoder(self):
        """Lazy-load encoder on first use with cache-corruption recovery."""
        if self._encoder is None:
            from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

            try:
                self._encoder = TextToEmbeddingModelPipeline(
                    encoder=self.encoder_name,
                    tokenizer=self.tokenizer_name,
                    device=self.device,
                )
            except Exception as exc:
                cleaned = _cleanup_fairseq2_cache(str(exc))
                if cleaned is None:
                    raise
                print(
                    f"[SONARWrapper] Corrupt fairseq2 cache detected; "
                    f"removed {cleaned} and retrying encoder download."
                )
                self._encoder = TextToEmbeddingModelPipeline(
                    encoder=self.encoder_name,
                    tokenizer=self.tokenizer_name,
                    device=self.device,
                )
        return self._encoder

    def _load_decoder(self):
        """Lazy-load decoder on first use with cache-corruption recovery."""
        if self._decoder is None:
            from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

            try:
                self._decoder = EmbeddingToTextModelPipeline(
                    decoder=self.decoder_name,
                    tokenizer=self.tokenizer_name,
                    device=self.device,
                )
            except Exception as exc:
                cleaned = _cleanup_fairseq2_cache(str(exc))
                if cleaned is None:
                    raise
                print(
                    f"[SONARWrapper] Corrupt fairseq2 cache detected; "
                    f"removed {cleaned} and retrying decoder download."
                )
                self._decoder = EmbeddingToTextModelPipeline(
                    decoder=self.decoder_name,
                    tokenizer=self.tokenizer_name,
                    device=self.device,
                )
        return self._decoder

    def encode(
        self,
        texts: list[str],
        lang: str = "eng_Latn",
    ) -> Tensor:
        """
        Encode texts to SONAR embeddings.

        Args:
            texts: List of sentences to encode.
            lang: Language code (FLORES-200 format).

        Returns:
            Tensor of shape [len(texts), 1024].
        """
        encoder = self._load_encoder()
        embeddings = encoder.predict(texts, source_lang=lang)
        return embeddings  # [N, 1024]

    def decode(
        self,
        vectors: Tensor,
        lang: str = "eng_Latn",
        max_seq_len: int = 128,
    ) -> list[str]:
        """
        Decode SONAR embeddings back to text.

        Args:
            vectors: Tensor of shape [N, 1024].
            lang: Target language code.
            max_seq_len: Maximum output sequence length.

        Returns:
            List of decoded strings.
        """
        decoder = self._load_decoder()
        vectors = vectors.to(self.device)
        texts = decoder.predict(vectors, target_lang=lang, max_seq_len=max_seq_len)
        return texts

    def decode_safe(
        self,
        vectors: Tensor,
        lang: str = "eng_Latn",
        max_seq_len: int = 128,
    ) -> list[str]:
        """
        Decode embeddings one-by-one with OOM protection.

        Noisy/corrupted embeddings can cause beam search to generate
        very long sequences, exhausting VRAM. This method decodes each
        vector individually and catches OOM errors gracefully.

        Returns:
            List of decoded strings (or "[OOM]"/"[ERROR]" on failure).
        """
        results = []
        for i in range(vectors.shape[0]):
            try:
                v = vectors[i : i + 1]
                texts = self.decode(v, lang=lang, max_seq_len=max_seq_len)
                results.append(texts[0])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                results.append("[OOM]")
            except Exception as e:
                results.append(f"[ERROR: {type(e).__name__}]")
        return results

    def encode_batched(
        self,
        texts: list[str],
        lang: str = "eng_Latn",
        batch_size: int = 64,
    ) -> Tensor:
        """
        Encode a large list of texts in batches.

        Args:
            texts: List of sentences.
            lang: Language code.
            batch_size: Number of sentences per batch.

        Returns:
            Tensor of shape [len(texts), 1024].
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self.encode(batch, lang=lang)
            all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)

    def roundtrip(
        self,
        texts: list[str],
        lang: str = "eng_Latn",
    ) -> tuple[list[str], Tensor]:
        """
        Encode then decode — check reconstruction quality.

        Returns:
            (decoded_texts, embeddings)
        """
        V = self.encode(texts, lang=lang)
        decoded = self.decode(V, lang=lang)
        return decoded, V

    @staticmethod
    def cosine_similarity(V1: Tensor, V2: Tensor) -> Tensor:
        """Compute cosine similarity between two tensors."""
        return F.cosine_similarity(V1, V2, dim=-1)

    def estimate_vram(self) -> dict[str, float]:
        """
        Estimate VRAM usage of loaded models.

        Returns:
            Dict with component VRAM in MB.
        """
        vram = {}
        if self._encoder is not None:
            vram["encoder"] = sum(
                p.numel() * p.element_size()
                for p in self._encoder.model.parameters()
            ) / (1024 ** 2)
        if self._decoder is not None:
            vram["decoder"] = sum(
                p.numel() * p.element_size()
                for p in self._decoder.model.parameters()
            ) / (1024 ** 2)
        return vram
