"""
memory/vector_store.py
──────────────────────
Qdrant-based long-term semantic memory for company research.
Stores completed research so repeat queries skip web search entirely.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
from sentence_transformers import SentenceTransformer
from config.settings import settings
import hashlib, json, time
from typing import Optional

_client: Optional[QdrantClient] = None
_embedder: Optional[SentenceTransformer] = None
VECTOR_DIM = 384


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url)
        _ensure_collection()
    return _client


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _ensure_collection():
    client = QdrantClient(url=settings.qdrant_url)
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )


def _make_id(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


def store_research(company_name: str, research_data: dict) -> None:
    """Store research in Qdrant after a successful deep_researcher run."""
    client = _get_client()
    embedder = _get_embedder()
    text = json.dumps(research_data)
    vector = embedder.encode(text).tolist()
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=[PointStruct(
            id=_make_id(company_name.lower()),
            vector=vector,
            payload={
                "company_name": company_name.lower(),
                "research": research_data,
                "timestamp": time.time(),
            }
        )]
    )


def retrieve_research(company_name: str) -> Optional[dict]:
    """Cache check before web search. Returns None if not found or stale."""
    client = _get_client()
    embedder = _get_embedder()
    vector = embedder.encode(company_name).tolist()
    results = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=vector,
        query_filter=Filter(must=[FieldCondition(
            key="company_name",
            match=MatchValue(value=company_name.lower())
        )]),
        limit=1,
        score_threshold=0.85,
    )
    if results:
        payload = results[0].payload
        age_days = (time.time() - payload["timestamp"]) / 86400
        if age_days < settings.research_cache_days:
            return payload["research"]
    return None


def semantic_search(query: str, top_k: int = 3) -> list[dict]:
    """Semantic search across all stored research."""
    client = _get_client()
    embedder = _get_embedder()
    vector = embedder.encode(query).tolist()
    results = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=vector,
        limit=top_k,
        score_threshold=0.6,
    )
    return [r.payload for r in results]


def store_run_outcome(company_name: str, quality_score: float, got_response: bool = False):
    """
    Store outcome feedback for learning loop.
    got_response = True when user marks that they got an interview/response.
    Used to correlate critic scores with real-world outcomes over time.
    """
    client = _get_client()
    embedder = _get_embedder()
    key = f"outcome:{company_name.lower()}:{int(time.time())}"
    vector = embedder.encode(key).tolist()
    client.upsert(
        collection_name=f"{settings.qdrant_collection}_outcomes",
        points=[PointStruct(
            id=_make_id(key),
            vector=vector,
            payload={
                "company_name": company_name.lower(),
                "quality_score": quality_score,
                "got_response": got_response,
                "timestamp": time.time(),
            }
        )]
    )
