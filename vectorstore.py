"""
core/vectorstore.py — ChromaDB-backed vector store.

Two persistent collections:
  - cve_intel:         cached NVD/KEV/vendor-advisory text, keyed by CVE ID
  - uploaded_reports:  chunks from user-uploaded PDF/DOCX/TXT/CSV/XLSX/JSON files

Embeddings are computed locally with sentence-transformers (no embeddings API
required), matching the original project's approach.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import settings

logger = logging.getLogger("sentry.vectorstore")

_embedder: SentenceTransformer | None = None
_chroma_client = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.embed_model_name)
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = get_embedder().encode(texts, normalize_embeddings=True)
    return np.asarray(vectors, dtype="float32").tolist()


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_dir)
    return _chroma_client


def get_collection(name: str):
    client = get_chroma_client()
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


class VectorStore:
    """High-level operations used by the rest of the app."""

    def __init__(self):
        self.cve_col = get_collection(settings.cve_collection)
        self.report_col = get_collection(settings.report_collection)

    # ------------------------------------------------------------------ CVE cache
    def cve_cached(self, cve_id: str) -> dict[str, Any] | None:
        try:
            result = self.cve_col.get(ids=[cve_id.upper()], include=["documents", "metadatas"])
        except Exception:
            return None
        if not result or not result.get("ids"):
            return None
        meta = result["metadatas"][0] or {}
        if time.time() - float(meta.get("cached_at", 0)) > settings.cve_cache_ttl:
            return None  # stale
        return {"text": result["documents"][0], "metadata": meta}

    def upsert_cve(self, cve_id: str, text: str, metadata: dict[str, Any]):
        metadata = {**metadata, "cached_at": time.time(), "type": "cve"}
        vector = embed_texts([text])[0]
        self.cve_col.upsert(ids=[cve_id.upper()], documents=[text], metadatas=[metadata], embeddings=[vector])

    def all_cached_cves(self) -> list[dict[str, Any]]:
        try:
            result = self.cve_col.get(include=["documents", "metadatas"])
        except Exception:
            return []
        out = []
        for cid, doc, meta in zip(result.get("ids", []), result.get("documents", []), result.get("metadatas", [])):
            out.append({"cve_id": cid, "text": doc, **(meta or {})})
        return out

    # ------------------------------------------------------------- uploaded reports
    def add_report_chunks(self, chunks: list[str], source_label: str, extra_meta: dict[str, Any] | None = None) -> int:
        if not chunks:
            return 0
        ids = [f"{source_label}-{uuid.uuid4().hex[:8]}-{i}" for i in range(len(chunks))]
        metadatas = [{"source": source_label, "chunk_index": i, "uploaded_at": time.time(), **(extra_meta or {})} for i in range(len(chunks))]
        vectors = embed_texts(chunks)
        self.report_col.upsert(ids=ids, documents=chunks, metadatas=metadatas, embeddings=vectors)
        return len(chunks)

    def report_sources(self) -> list[str]:
        try:
            result = self.report_col.get(include=["metadatas"])
        except Exception:
            return []
        return sorted({m.get("source", "unknown") for m in result.get("metadatas", [])})

    def query(self, collection, query_text: str, k: int = 5) -> list[dict[str, Any]]:
        if collection.count() == 0:
            return []
        vector = embed_texts([query_text])[0]
        result = collection.query(query_embeddings=[vector], n_results=min(k, collection.count()), include=["documents", "metadatas", "distances"])
        out = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            similarity = max(0.0, 1.0 - dist)  # cosine distance -> similarity
            out.append({"text": doc, "metadata": meta, "similarity": similarity})
        return out

    def query_reports(self, query_text: str, k: int = 5) -> list[dict[str, Any]]:
        return self.query(self.report_col, query_text, k)

    def query_cves(self, query_text: str, k: int = 5) -> list[dict[str, Any]]:
        return self.query(self.cve_col, query_text, k)


_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
