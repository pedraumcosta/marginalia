"""Turning text into vectors, locally and only locally.

Embedding touches every chunk, so it is the stage with the widest exposure — which is
why ADR 0003 gives it no hosted option at all. Two implementations:

``sentence-transformers``
    The real one. A small model on CPU. Optional dependency, imported lazily, because
    pulling a deep-learning stack in to run ``--help`` would be rude.

``hashing``
    Deterministic, dependency-free, offline. Hashes tokens into buckets, so it captures
    lexical overlap and **nothing semantic** — "car" and "automobile" land nowhere near
    each other. It exists so the test suite and CI can exercise the whole pipeline
    without a model download, and it is honest about what it is: retrieval quality
    measured with it says something about the plumbing, not about retrieval.

Every embedder returns L2-normalised float32, so cosine similarity is a dot product and
the store needs no normalisation of its own.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+")
DEFAULT_HASHING_DIM = 512
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingError(Exception):
    pass


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into normalised vectors."""

    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    @property
    def signature(self) -> str:
        """Identity of this embedder's vector space.

        Stored with the index. Vectors from two different signatures are not comparable,
        so a changed signature means the dense index must be rebuilt rather than appended
        to — silently mixing spaces produces retrieval that is wrong but not obviously so.
        """
        ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero vector stays zero rather than becoming NaN; it simply matches nothing.
    np.divide(matrix, norms, out=matrix, where=norms > 0)
    return matrix


class HashingEmbedder:
    """Deterministic bag-of-tokens hashing. Lexical only, by construction."""

    def __init__(self, dim: int = DEFAULT_HASHING_DIM) -> None:
        if dim <= 0:
            raise EmbeddingError("hashing embedder dim must be positive")
        self._dim = int(dim)

    @property
    def name(self) -> str:
        return "hashing"

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def signature(self) -> str:
        return f"hashing:{self._dim}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in TOKEN.findall(text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "big")
                bucket = value % self._dim
                # Signed hashing keeps unrelated collisions from always reinforcing.
                sign = 1.0 if (value >> 63) & 1 else -1.0
                out[row, bucket] += sign
        return _normalise(out)


class SentenceTransformerEmbedder:
    """A local sentence-transformers model on CPU."""

    def __init__(self, model: str = DEFAULT_MODEL, batch_size: int = 32) -> None:
        self._model_name = model
        self._batch_size = max(1, int(batch_size))
        self._model = None  # loaded on first use

    @property
    def name(self) -> str:
        return "sentence-transformers"

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise EmbeddingError(
                    "sentence-transformers is not installed. Install it with "
                    "`pip install -e '.[embed]'`, or set embeddings.provider to "
                    "'hashing' for a dependency-free run."
                ) from exc
            self._model = SentenceTransformer(self._model_name, device="cpu")
        return self._model

    @property
    def dim(self) -> int:
        model = self._load()
        # sentence-transformers renamed this; support both so the dimension -- which the
        # store's signature depends on -- does not start raising on a library upgrade.
        getter = getattr(model, "get_embedding_dimension", None)
        if getter is None:
            getter = model.get_sentence_embedding_dimension
        return int(getter())

    @property
    def signature(self) -> str:
        return f"sentence-transformers:{self._model_name}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def make_embedder(config: dict | None = None) -> Embedder:
    """Build the embedder named by configuration.

    Defaults to the real model: the dependency-free one is a testing affordance, and
    making it the default would quietly degrade anyone who forgot to configure.
    """
    cfg = dict(config or {})
    provider = str(cfg.get("provider", "sentence-transformers")).strip().lower()

    if provider in ("hashing", "hash", "test"):
        return HashingEmbedder(dim=int(cfg.get("dim", DEFAULT_HASHING_DIM)))
    if provider in ("sentence-transformers", "sentence_transformers", "st", "local"):
        return SentenceTransformerEmbedder(
            model=str(cfg.get("model", DEFAULT_MODEL)),
            batch_size=int(cfg.get("batch_size", 32)),
        )
    raise EmbeddingError(
        f"unknown embeddings provider {provider!r}. Known: sentence-transformers, hashing."
    )
