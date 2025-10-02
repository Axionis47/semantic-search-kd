"""Pydantic schemas for FastAPI service."""

from typing import List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Search request schema."""

    query: str = Field(..., description="Search query", min_length=1, max_length=1000)
    k: int = Field(10, description="Number of results to return", ge=1, le=100)
    rerank: bool = Field(False, description="Whether to rerank with teacher model")
    rerank_top_k: int = Field(
        50, description="Number of candidates to rerank", ge=1, le=200
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "k": 10,
                "rerank": False,
                "rerank_top_k": 50,
            }
        }


class SearchResult(BaseModel):
    """Single search result."""

    doc_id: str = Field(..., description="Document ID")
    text: str = Field(..., description="Document text")
    score: float = Field(..., description="Relevance score")
    rank: int = Field(..., description="Rank position (1-indexed)")


class SearchResponse(BaseModel):
    """Search response schema."""

    query: str = Field(..., description="Original query")
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    reranked: bool = Field(..., description="Whether results were reranked")
    latency_ms: float = Field(..., description="Query latency in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "results": [
                    {
                        "doc_id": "doc_123",
                        "text": "Machine learning is a subset of AI...",
                        "score": 0.95,
                        "rank": 1,
                    }
                ],
                "total_results": 10,
                "reranked": False,
                "latency_ms": 12.5,
            }
        }


class EncodeRequest(BaseModel):
    """Encode request schema."""

    texts: List[str] = Field(
        ..., description="List of texts to encode", min_length=1, max_length=100
    )
    normalize: bool = Field(True, description="Whether to L2-normalize embeddings")

    class Config:
        json_schema_extra = {
            "example": {
                "texts": ["What is machine learning?", "How does AI work?"],
                "normalize": True,
            }
        }


class EncodeResponse(BaseModel):
    """Encode response schema."""

    embeddings: List[List[float]] = Field(..., description="List of embeddings")
    dimension: int = Field(..., description="Embedding dimension")
    num_texts: int = Field(..., description="Number of texts encoded")
    latency_ms: float = Field(..., description="Encoding latency in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                "dimension": 384,
                "num_texts": 2,
                "latency_ms": 5.2,
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    index_loaded: bool = Field(..., description="Whether index is loaded")
    index_size: int = Field(..., description="Number of documents in index")
    version: str = Field(..., description="Service version")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "index_loaded": True,
                "index_size": 10000,
                "version": "1.0.0",
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid request",
                "detail": "Query must be between 1 and 1000 characters",
            }
        }

