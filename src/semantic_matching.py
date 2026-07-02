"""Semantic matching with precomputed embeddings and BM25."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.utils import DATA_PROCESSED, normalize_text

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS_FILE = DATA_PROCESSED / "candidate_embeddings.npy"
IDS_FILE = DATA_PROCESSED / "candidate_ids.json"
CAREER_CORPUS_FILE = DATA_PROCESSED / "career_corpus.json"
BM25_INDEX_FILE = DATA_PROCESSED / "bm25_index.pkl"
JD_EMBEDDING_FILE = DATA_PROCESSED / "jd_embedding.npy"


class SemanticMatcher:
    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self.model: SentenceTransformer | None = None
        self.embeddings: np.ndarray | None = None
        self.candidate_ids: list[str] = []
        self.id_to_idx: dict[str, int] = {}
        self.bm25: BM25Okapi | None = None
        self.career_corpus: list[str] = []
        self.jd_embedding: np.ndarray | None = None

    def _load_model(self) -> SentenceTransformer:
        if self.model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def embed_texts(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        model = self._load_model()
        return model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    def precompute(
        self,
        candidates: list[dict[str, Any]],
        jd_text: str,
        career_texts: list[str],
        canonical_texts: list[str],
    ) -> None:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        self.candidate_ids = [c["candidate_id"] for c in candidates]
        self.id_to_idx = {cid: i for i, cid in enumerate(self.candidate_ids)}

        logger.info("Embedding %d candidates...", len(canonical_texts))
        self.embeddings = self.embed_texts(canonical_texts)
        np.save(EMBEDDINGS_FILE, self.embeddings)

        with open(IDS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.candidate_ids, f)

        self.career_corpus = career_texts
        with open(CAREER_CORPUS_FILE, "w", encoding="utf-8") as f:
            json.dump(career_corpus := career_texts, f)

        tokenized = [normalize_text(t).split() for t in career_corpus]
        self.bm25 = BM25Okapi(tokenized)
        with open(BM25_INDEX_FILE, "wb") as f:
            pickle.dump({"tokenized": tokenized, "corpus": career_corpus}, f)

        self.jd_embedding = self.embed_texts([jd_text])[0]
        np.save(JD_EMBEDDING_FILE, self.jd_embedding)
        logger.info("Precompute complete.")

    def load_artifacts(self) -> None:
        if not EMBEDDINGS_FILE.exists():
            raise FileNotFoundError(
                f"Missing {EMBEDDINGS_FILE}. Run precompute first."
            )
        raw = np.load(EMBEDDINGS_FILE).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self.embeddings = raw / norms

        with open(IDS_FILE, encoding="utf-8") as f:
            self.candidate_ids = json.load(f)
        self.id_to_idx = {cid: i for i, cid in enumerate(self.candidate_ids)}

        jd_raw = np.load(JD_EMBEDDING_FILE).astype(np.float32)
        jd_norm = np.linalg.norm(jd_raw)
        self.jd_embedding = jd_raw / (jd_norm if jd_norm > 0 else 1.0)

        with open(BM25_INDEX_FILE, "rb") as f:
            data = pickle.load(f)
        self.career_corpus = data["corpus"]
        self.bm25 = BM25Okapi(data["tokenized"])

    def semantic_similarity(self, candidate_id: str) -> float:
        if self.embeddings is None or self.jd_embedding is None:
            raise RuntimeError("Artifacts not loaded")
        idx = self.id_to_idx[candidate_id]
        return float(np.dot(self.embeddings[idx], self.jd_embedding))

    def semantic_similarity_batch(self, indices: np.ndarray) -> np.ndarray:
        if self.embeddings is None or self.jd_embedding is None:
            raise RuntimeError("Artifacts not loaded")
        emb = np.clip(self.embeddings[indices].astype(np.float64), -1.0, 1.0)
        jd = np.clip(self.jd_embedding.astype(np.float64), -1.0, 1.0)
        scores = emb @ jd
        return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    def bm25_scores_batch(self, query_terms: list[str]) -> np.ndarray:
        if self.bm25 is None:
            raise RuntimeError("BM25 not loaded")
        scores = self.bm25.get_scores(query_terms)
        return np.clip(np.asarray(scores, dtype=np.float32) / 20.0, 0.0, 1.0)
