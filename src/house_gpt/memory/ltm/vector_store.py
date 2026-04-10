import os
import hashlib
from functools import lru_cache
from house_gpt.memory.models import Memory
from house_gpt.core.settings import settings
from house_gpt.core.logger import AppLogger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, PointStruct, Distance,
    Filter, FieldCondition, MatchValue,
    CreateFieldIndex, PayloadSchemaType
)
from langchain_openai import OpenAIEmbeddings
from typing import List, Optional

logger = AppLogger("VectorStore")

_qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
    timeout=60
)
_embedding_model = OpenAIEmbeddings(
    model=settings.EMBEDDING_MODEL_NAME,
    base_url=settings.OPENROUTER_URL,
    api_key=settings.OPENROUTER_API_KEY
)

_embedding_cache: dict[str, list[float]] = {}

def _get_embedding(text: str) -> list[float]:
    """Return cached embedding or fetch from OpenRouter once."""
    key = hashlib.md5(text.encode()).hexdigest()
    if key not in _embedding_cache:
        _embedding_cache[key] = _embedding_model.embed_query(text)
    return _embedding_cache[key]


class VectorStore:
    REQUIRED_ENV_VARS = ["QDRANT_URL", "QDRANT_API_KEY"]
    COLLECTION_NAME = "long_term_memory"
    SIMILARITY_THRESHOLD = 0.7

    def __init__(self, user_id: str) -> None:
        self._validate_env_vars()
        self.model = _embedding_model
        self.client = _qdrant_client
        self._user_id = user_id
        self._collection_ready = False

    def _validate_env_vars(self) -> None:
        missing = [v for v in self.REQUIRED_ENV_VARS if not os.getenv(v)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    def _ensure_collection(self) -> None:
        if self._collection_ready:
            return
        collections = self.client.get_collections().collections
        exists = any(c.name == self.COLLECTION_NAME for c in collections)
        if not exists:
            self._create_collection()
        self._collection_ready = True

    def _create_collection(self) -> None:
        vector_size = len(_get_embedding("init"))
        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        self.client.create_payload_index(
            collection_name=self.COLLECTION_NAME,
            field_name="user_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    def store_memory(self, text: str, metadata: dict) -> None:
        self._ensure_collection()

        embedding = _get_embedding(text)

        similar_memory = self._find_similar_by_vector(embedding, k=1)
        if similar_memory and similar_memory.id:
            metadata["id"] = similar_memory.id

        logger.info(f"Storing new memory: '{text}'")
        point = PointStruct(
            id=metadata.get("id"),
            vector=embedding,
            payload={"user_id": self._user_id, "text": text, **metadata},
        )
        self.client.upsert(collection_name=self.COLLECTION_NAME, points=[point])

    def find_similarity_memory(self, text: str) -> Optional[Memory]:
        results = self.search_memories(text, k=1)
        if results and results[0].score >= self.SIMILARITY_THRESHOLD:
            return results[0]
        return None

    def _find_similar_by_vector(self, embedding: list[float], k: int = 1) -> Optional[Memory]:
        results = self._search_by_vector(embedding, k=k)
        if results and results[0].score >= self.SIMILARITY_THRESHOLD:
            return results[0]
        return None

    def search_memories(self, query: str, k: int = 5) -> List[Memory]:
        self._ensure_collection()
        return self._search_by_vector(_get_embedding(query), k=k)

    def _search_by_vector(self, embedding: list[float], k: int = 5) -> List[Memory]:
        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=embedding,
            limit=k,
            query_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=self._user_id))]
            ),
        ).points
        return [
            Memory(
                text=hit.payload["text"],
                user_id=hit.payload["user_id"],
                metadata={k: v for k, v in hit.payload.items() if k not in ("text", "user_id")},
                score=hit.score,
            )
            for hit in results
        ]


@lru_cache
def get_vector_store(user_id: str) -> VectorStore:
    return VectorStore(user_id)