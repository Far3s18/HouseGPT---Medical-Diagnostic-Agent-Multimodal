import os
from house_gpt.memory.models import RAG, Memory
from house_gpt.core.settings import settings
from functools import lru_cache
from qdrant_client import QdrantClient
from fastembed import SparseTextEmbedding
from langchain_ollama.embeddings import OllamaEmbeddings
from typing import List, Optional
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector


class MedicalBooksRAG:
    REQUIRED_ENV_VARS = ["QDRANT_URL", "QDRANT_API_KEY"]
    COLLECTION_NAME = "house_gpt_books"
    SIMILARITY_THRESHOLD = 0.75

    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"

    def __init__(self) -> None:
        self._validate_env_vars()
        self.dense_model = OllamaEmbeddings(model=settings.EMBEDDING_MODEL_NAME)
        self.sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1", cache_dir="/home/fa-res/.cache/fastembed")
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=60
        )

    def _validate_env_vars(self) -> None:
        missings_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missings_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missings_vars)}")

    def _embed_sparse(self, text: str) -> SparseVector:
        result = list(self.sparse_model.embed([text]))[0]
        return SparseVector(indices=result.indices.tolist(), values=result.values.tolist())

    def search_data(self, query: str, k: int = 5):
        dense_vec = self.dense_model.embed_query(query)
        sparse_vec = self._embed_sparse(query)

        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            prefetch=[
                Prefetch(query=dense_vec, using=self.DENSE_VECTOR_NAME, limit=k * 3),
                Prefetch(query=sparse_vec, using=self.SPARSE_VECTOR_NAME, limit=k * 3),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k,
            with_vectors=False,
        ).points

        return [
            RAG(
                text=hit.payload["text"],
                book_title=hit.payload["book_title"],
                metadata={k:v for k, v in hit.payload.items() if k not in ("text", "book_title")},
                score=hit.score
            )
            for hit in results if hit.score >= self.SIMILARITY_THRESHOLD
        ]

@lru_cache
def get_medical_rag() -> MedicalBooksRAG:
    return MedicalBooksRAG()