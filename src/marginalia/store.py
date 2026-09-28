"""Vector storage behind a narrow protocol.

ADR 0002 chooses exact search: at 10^4 chunks an ANN index buys latency this corpus does
not need and costs recall that would make every measurement ambiguous between retriever
quality and index error. ADR 0004 keeps that reversible — the protocol below is the whole
surface a retriever uses, so pgvector or Qdrant becomes one more adapter rather than a
rewrite.

``NumpyVectorStore`` holds the matrix in memory and persists it as a ``.npy`` beside a
small JSON sidecar. At 10^4 chunks and 384 dimensions that is about 15 MB.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

VECTORS_FILE = "vectors.npy"
SIDECAR_FILE = "vectors.json"


class StoreError(Exception):
    pass


@runtime_checkable
class VectorStore(Protocol):
    """The narrowest interface retrieval needs."""

    def upsert(self, ids: Sequence[str], vectors: np.ndarray) -> None: ...

    def delete(self, ids: Sequence[str]) -> None: ...

    def search(
        self, query: np.ndarray, k: int, *, allow: Collection[str] | None = None
    ) -> list[tuple[str, float]]: ...

    def stats(self) -> dict[str, object]: ...


class NumpyVectorStore:
    """Exact cosine similarity over an in-memory matrix.

    Vectors are expected L2-normalised (every embedder in this project returns them that
    way), so similarity is a plain dot product.
    """

    def __init__(self, dim: int, signature: str = "") -> None:
        if dim <= 0:
            raise StoreError("dim must be positive")
        self._dim = int(dim)
        self.signature = signature
        self._ids: list[str] = []
        self._pos: dict[str, int] = {}
        self._matrix = np.zeros((0, self._dim), dtype=np.float32)

    # ---- introspection -------------------------------------------------------

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._ids)

    def stats(self) -> dict[str, object]:
        return {
            "backend": "numpy-exact",
            "count": len(self._ids),
            "dim": self._dim,
            "signature": self.signature,
            "bytes": int(self._matrix.nbytes),
        }

    # ---- writes --------------------------------------------------------------

    def upsert(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        if len(ids) != len(vectors):
            raise StoreError(f"got {len(ids)} ids for {len(vectors)} vectors")
        if not len(ids):
            return
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self._dim:
            raise StoreError(f"expected vectors of dim {self._dim}, got shape {arr.shape}")

        fresh_ids: list[str] = []
        fresh_rows: list[np.ndarray] = []
        for i, cid in enumerate(ids):
            existing = self._pos.get(cid)
            if existing is None:
                fresh_ids.append(cid)
                fresh_rows.append(arr[i])
            else:
                self._matrix[existing] = arr[i]

        if fresh_ids:
            block = np.vstack(fresh_rows).astype(np.float32, copy=False)
            self._matrix = block if not len(self._ids) else np.vstack([self._matrix, block])
            for cid in fresh_ids:
                self._pos[cid] = len(self._ids)
                self._ids.append(cid)

    def delete(self, ids: Sequence[str]) -> None:
        doomed = {cid for cid in ids if cid in self._pos}
        if not doomed:
            return
        keep = [i for i, cid in enumerate(self._ids) if cid not in doomed]
        self._matrix = self._matrix[keep] if keep else np.zeros((0, self._dim), np.float32)
        self._ids = [self._ids[i] for i in keep]
        self._pos = {cid: i for i, cid in enumerate(self._ids)}

    def clear(self) -> None:
        self._ids, self._pos = [], {}
        self._matrix = np.zeros((0, self._dim), dtype=np.float32)

    # ---- reads ---------------------------------------------------------------

    def search(
        self, query: np.ndarray, k: int, *, allow: Collection[str] | None = None
    ) -> list[tuple[str, float]]:
        if not len(self._ids) or k <= 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._dim:
            raise StoreError(f"query dim {q.shape[0]} != store dim {self._dim}")

        scores = self._matrix @ q

        if allow is not None:
            allowed = set(allow)
            if not allowed:
                return []
            mask = np.array([cid in allowed for cid in self._ids], dtype=bool)
            # -inf rather than dropping rows, so indices still line up with self._ids.
            scores = np.where(mask, scores, -np.inf)

        take = min(k, len(self._ids))
        # argpartition then sort the head: O(n) rather than sorting the whole corpus.
        idx = np.argpartition(-scores, take - 1)[:take]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        return [(self._ids[i], float(scores[i])) for i in idx if np.isfinite(scores[i])]

    # ---- persistence ---------------------------------------------------------

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / VECTORS_FILE, self._matrix)
        (directory / SIDECAR_FILE).write_text(
            json.dumps(
                {"dim": self._dim, "signature": self.signature, "ids": self._ids},
                indent=0,
            )
        )

    @classmethod
    def load(cls, directory: Path) -> NumpyVectorStore:
        sidecar = directory / SIDECAR_FILE
        vectors = directory / VECTORS_FILE
        if not sidecar.is_file() or not vectors.is_file():
            raise StoreError(f"no vector store in {directory}")
        meta = json.loads(sidecar.read_text())
        store = cls(dim=int(meta["dim"]), signature=str(meta.get("signature", "")))
        matrix = np.load(vectors)
        ids = [str(i) for i in meta.get("ids", [])]
        if len(ids) != len(matrix):
            raise StoreError(
                f"vector store is inconsistent: {len(ids)} ids for {len(matrix)} vectors"
            )
        store._matrix = np.asarray(matrix, dtype=np.float32)
        store._ids = ids
        store._pos = {cid: i for i, cid in enumerate(ids)}
        return store

    @classmethod
    def open_or_create(cls, directory: Path, dim: int, signature: str) -> NumpyVectorStore:
        """Load an existing store, or start a fresh one if absent or incompatible.

        A signature mismatch discards rather than migrates: vectors from two different
        embedders are not comparable, and mixing them yields retrieval that is wrong
        without looking broken.
        """
        try:
            store = cls.load(directory)
        except (StoreError, ValueError, KeyError, json.JSONDecodeError):
            return cls(dim=dim, signature=signature)
        if store.dim != dim or store.signature != signature:
            return cls(dim=dim, signature=signature)
        return store
