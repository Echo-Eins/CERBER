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

import torch
import torch.nn.functional as F
from torch import Tensor


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
        """Lazy-load encoder on first use."""
        if self._encoder is None:
            from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

            self._encoder = TextToEmbeddingModelPipeline(
                encoder=self.encoder_name,
                tokenizer=self.tokenizer_name,
                device=self.device,
            )
        return self._encoder

    def _load_decoder(self):
        """Lazy-load decoder on first use."""
        if self._decoder is None:
            from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

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
        max_seq_len: int = 512,
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
