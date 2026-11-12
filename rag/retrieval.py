"""Vector index over chunks: TF-IDF by default, sentence embeddings optionally.

Both backends produce L2-normalised vectors, so the score is cosine similarity
and the same `min_score` logic decides when nothing relevant was found.
"""
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float


_SUFFIXES = ("ations", "ation", "ings", "ing", "ers", "ied", "ies", "ed", "es", "er", "ly", "s")


def _stem(word: str) -> str:
    """Tiny suffix-stripper so 'refunds'/'refunded'/'refund' share one term."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix in ("ied", "ies") else "")
    return word


def _analyze(text: str) -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    words = [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower())
             if w not in ENGLISH_STOP_WORDS]
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]


class TfidfEmbedder:
    """Sparse lexical vectors; no model download, works fully offline."""
    name = "tfidf"
    default_min_score = 0.08

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(analyzer=_analyze, sublinear_tf=True)

    def fit(self, texts):
        self.vectorizer.fit(texts)

    def encode(self, texts) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray().astype(np.float32)


class SentenceEmbedder:
    """Dense semantic vectors (`pip install sentence-transformers`; downloads ~90 MB once)."""
    name = "embeddings"
    default_min_score = 0.30

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def fit(self, texts):
        pass

    def encode(self, texts) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True),
                          dtype=np.float32)


def make_embedder(kind: str = "tfidf"):
    return {"tfidf": TfidfEmbedder, "embeddings": SentenceEmbedder}[kind]()


class VectorIndex:
    def __init__(self, embedder=None, min_score: float | None = None):
        self.embedder = embedder or TfidfEmbedder()
        self.min_score = self.embedder.default_min_score if min_score is None else min_score
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)

    def build(self, chunks: list[Chunk]) -> "VectorIndex":
        if not chunks:
            raise ValueError("No text found in the uploaded documents.")
        self.chunks = list(chunks)
        texts = [c.text for c in self.chunks]
        self.embedder.fit(texts)
        self.vectors = _normalise(self.embedder.encode(texts))
        return self

    def search(self, query: str, k: int = 4) -> list[Hit]:
        """Top-k chunks by cosine similarity, dropping anything below `min_score`."""
        if not self.chunks:
            return []
        q = _normalise(self.embedder.encode([query]))[0]
        scores = self.vectors @ q
        order = np.argsort(scores)[::-1][:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order
                if scores[i] >= self.min_score]

    # Persistence: chunks as JSON + vectors as .npy (the embedder is rebuilt on load).
    def save(self, directory: Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        (directory / "chunks.json").write_text(json.dumps({
            "embedder": self.embedder.name, "min_score": self.min_score,
            "chunks": [asdict(c) for c in self.chunks]}))

    @classmethod
    def load(cls, directory: Path) -> "VectorIndex":
        meta = json.loads((Path(directory) / "chunks.json").read_text())
        index = cls(make_embedder(meta["embedder"]), meta["min_score"])
        index.chunks = [Chunk(**c) for c in meta["chunks"]]
        index.embedder.fit([c.text for c in index.chunks])  # TF-IDF vocabulary is cheap to refit
        index.vectors = np.load(Path(directory) / "vectors.npy")
        return index


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)
