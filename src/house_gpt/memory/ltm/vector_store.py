import os
from house_gpt.memory.models import Memory
from house_gpt.core.settings import settings
from functools import lru_cache
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, PointStruct, Distance,
    Filter, FieldCondition, MatchValue,
    CreateFieldIndex, PayloadSchemaType
)

from langchain_ollama.embeddings import OllamaEmbeddings
from typing import List, Optional



_qdrant_client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=60)
_embedding_model = OllamaEmbeddings(model=settings.EMBEDDING_MODEL_NAME)

class VectorStore:
    REQUIRED_ENV_VARS = ["QDRANT_URL", "QDRANT_API_KEY"]
    COLLECTION_NAME = 'long_term_memory'
    SIMILARITY_THRESHOLD = 0.7

    def __init__(self, user_id: str) -> None:
        self._validate_env_vars()
        self.model = _embedding_model
        self.client = _qdrant_client
        self._user_id = user_id

    def _validate_env_vars(self) -> None:
        missings_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missings_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missings_vars)}")

    def _collection_exists(self) -> bool:
        collections = self.client.get_collections().collections
        return any(col.name == self.COLLECTION_NAME for col in collections)

    def _create_collection(self) -> None:
        vector_size = len(self.model.embed_query("Fadi Fares"))
        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

        self.client.create_payload_index(
            collection_name=self.COLLECTION_NAME,
            field_name="user_id",
            field_schema=PayloadSchemaType.KEYWORD
        )

    def find_similarity_memory(self, text: str) -> Optional[Memory]:
        results = self.search_memories(text, k=1)
        if results and results[0].score >= self.SIMILARITY_THRESHOLD:
            return results[0]
        return None

    def store_memory(self, text: str, metadata: dict) -> None:
        similar_memory = self.find_similarity_memory(text)
        if similar_memory and similar_memory.id:
            metadata["id"] = similar_memory.id

        
        self.logger.info(f"Storing new memory: '{analysis.formatted_memory}'", user_id=self.user_id)

        embedding = self.model.embed_query(text)
        point = PointStruct(
            id=metadata.get("id"),
            vector=embedding,
            payload={
                "user_id": self._user_id,
                "text": text,
                **metadata
            }
        )

        self.client.upsert(collection_name=self.COLLECTION_NAME, points=[point])

    def search_memories(self, query: str, k: int = 5) -> List[Memory]:
        query_embedding = self.model.embed_query(query)
        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_embedding,
            limit=k,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=self._user_id)
                    )
                ]
            )
        ).points

        return [
            Memory(
                text=hit.payload["text"],
                user_id=hit.payload["user_id"],
                metadata={k:v for k, v in hit.payload.items() if k not in ("text", "user_id")},
                score=hit.score
            )
            for hit in results
        ]

@lru_cache
def get_vector_store(user_id: str) -> VectorStore:
    return VectorStore(user_id)