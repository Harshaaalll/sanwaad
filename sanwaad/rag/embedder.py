"""Local ONNX embeddings via fastembed.

Retrieval is the one part of this pipeline that runs on every turn of every
channel, so it is the one part that must not cost anything per call. A
quantised multilingual MiniLM on CPU answers in single-digit milliseconds and
handles Devanagari and Latin-script Hinglish in the same vector space, which
matters because our complaints arrive in both.
"""

from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

from ..config import EMBED_MODEL

_lock = threading.Lock()
_model = None


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                import warnings

                from fastembed import TextEmbedding

                # fastembed warns about a pooling change that does not affect
                # us: the index and the queries are embedded by the same build.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    _model = TextEmbedding(model_name=EMBED_MODEL)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Embed a batch, L2-normalised so cosine similarity is a plain dot product."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    vecs = np.array(list(_get_model().embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


@lru_cache(maxsize=2048)
def embed_one(text: str) -> tuple[float, ...]:
    """Cached single-text embedding. Queries repeat far more than you'd expect —
    'refund nahi aaya' is asked a hundred different ways but lands on a handful
    of normalised forms."""
    return tuple(embed([text])[0].tolist())
