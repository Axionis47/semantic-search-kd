"""Pytest fixtures and configuration for the test suite.

This module provides shared fixtures for testing, including:
- Mock models and embeddings
- Test data generators
- API test clients
- Configuration overrides
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.config import Settings


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with safe defaults."""
    return Settings(
        environment="development",
        debug=True,
        student={"model_name": "intfloat/e5-small-v2", "device": "cpu"},
        teacher={"model_name": "BAAI/bge-reranker-large", "device": "cpu"},
        service={
            "host": "127.0.0.1",
            "port": 8000,
            "cors": {"allow_origins": ["http://localhost:3000"]},
            "rate_limit": {"enabled": False},
            "auth": {"enabled": False},
        },
    )


@pytest.fixture
def production_settings() -> Settings:
    """Create production-like settings for validation tests."""
    return Settings(
        environment="production",
        debug=False,
        service={
            "cors": {"allow_origins": ["https://example.com"]},
            "rate_limit": {"enabled": True, "requests_per_minute": 100},
            "auth": {"enabled": True, "api_keys": ["test-key-123"]},
        },
    )


# =============================================================================
# Mock Model Fixtures
# =============================================================================


@pytest.fixture
def mock_embeddings() -> np.ndarray:
    """Generate mock embeddings for testing."""
    # 10 documents, 384 dimensions (like e5-small)
    np.random.seed(42)
    embeddings = np.random.randn(10, 384).astype(np.float32)
    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


@pytest.fixture
def mock_student_model(mock_embeddings: np.ndarray) -> MagicMock:
    """Create a mock student model."""
    model = MagicMock()
    model.embedding_dim = 384
    model.max_length = 512

    def mock_encode(texts, **kwargs):
        n = len(texts)
        np.random.seed(hash(str(texts)) % (2**32))
        emb = np.random.randn(n, 384).astype(np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb

    model.encode.side_effect = mock_encode
    model.encode_queries.side_effect = mock_encode
    model.encode_documents.side_effect = mock_encode

    return model


@pytest.fixture
def mock_teacher_model() -> MagicMock:
    """Create a mock teacher model."""
    model = MagicMock()

    def mock_score(pairs, **kwargs):
        # Return random scores in [-5, 5] range like a cross-encoder
        np.random.seed(hash(str(pairs)) % (2**32))
        return np.random.uniform(-5, 5, len(pairs)).tolist()

    model.score.side_effect = mock_score
    model.predict.side_effect = mock_score

    return model


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_queries() -> List[str]:
    """Sample queries for testing."""
    return [
        "what is machine learning?",
        "how does knowledge distillation work?",
        "explain neural networks",
        "what is semantic search?",
        "how to train a bi-encoder?",
    ]


@pytest.fixture
def sample_documents() -> List[str]:
    """Sample documents for testing."""
    return [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Knowledge distillation is a technique to transfer knowledge from a large model to a smaller one.",
        "Neural networks are computing systems inspired by biological neural networks in the brain.",
        "Semantic search uses natural language understanding to find relevant information.",
        "Bi-encoders encode queries and documents separately for efficient retrieval.",
        "Deep learning uses multiple layers of neural networks for complex pattern recognition.",
        "Transformers are a type of neural network architecture based on self-attention.",
        "BERT is a pre-trained language model that can be fine-tuned for various NLP tasks.",
        "Vector databases store embeddings for efficient similarity search.",
        "Cross-encoders process query-document pairs together for accurate relevance scoring.",
    ]


@pytest.fixture
def sample_qrels() -> Dict[str, Dict[str, int]]:
    """Sample relevance judgments (qrels) for evaluation."""
    return {
        "q1": {"doc1": 1, "doc2": 0, "doc3": 1},
        "q2": {"doc2": 1, "doc4": 1, "doc5": 0},
        "q3": {"doc3": 1, "doc6": 0, "doc7": 1},
    }


@pytest.fixture
def teacher_scores() -> List[float]:
    """Sample teacher model scores."""
    return [3.5, -1.2, 2.8, 0.5, -2.1, 4.2, 1.1, -0.8, 2.3, -1.5]


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_index_dir(temp_dir: Path, mock_embeddings: np.ndarray) -> Path:
    """Create a temporary directory with a mock FAISS index."""
    import faiss

    index_dir = temp_dir / "index"
    index_dir.mkdir()

    # Create simple flat index
    index = faiss.IndexFlatIP(384)
    index.add(mock_embeddings)

    # Save index
    faiss.write_index(index, str(index_dir / "index.faiss"))

    # Save doc IDs
    doc_ids = [f"doc_{i}" for i in range(len(mock_embeddings))]
    with open(index_dir / "doc_ids.json", "w") as f:
        json.dump(doc_ids, f)

    # Save doc texts
    doc_texts = {f"doc_{i}": f"Document {i} content" for i in range(len(mock_embeddings))}
    with open(index_dir / "texts.json", "w") as f:
        json.dump(doc_texts, f)

    return index_dir


@pytest.fixture
def temp_corpus_file(temp_dir: Path, sample_documents: List[str]) -> Path:
    """Create a temporary corpus file."""
    import pandas as pd

    corpus_file = temp_dir / "corpus.parquet"

    df = pd.DataFrame(
        {
            "chunk_id": [f"chunk_{i}" for i in range(len(sample_documents))],
            "text": sample_documents,
            "doc_id": [f"doc_{i}" for i in range(len(sample_documents))],
        }
    )
    df.to_parquet(corpus_file)

    return corpus_file


# =============================================================================
# API Test Client Fixtures
# =============================================================================


@pytest.fixture
def test_client(
    mock_student_model: MagicMock,
    test_settings: Settings,
) -> Generator[TestClient, None, None]:
    """Create a test client with mocked models."""
    with patch("src.serve.app.get_settings", return_value=test_settings):
        with patch("src.serve.app.StudentModel", return_value=mock_student_model):
            from src.serve.app import create_app

            app = create_app(settings=test_settings)
            with TestClient(app) as client:
                yield client


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def assert_valid_embedding():
    """Factory fixture for embedding validation."""

    def _assert(embedding: np.ndarray, expected_dim: int = 384) -> None:
        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 2
        assert embedding.shape[1] == expected_dim
        assert not np.isnan(embedding).any()
        assert not np.isinf(embedding).any()

    return _assert


@pytest.fixture
def assert_valid_scores():
    """Factory fixture for score validation."""

    def _assert(scores: List[float], expected_len: int) -> None:
        assert len(scores) == expected_len
        assert all(isinstance(s, (int, float)) for s in scores)
        assert not any(np.isnan(s) for s in scores)

    return _assert
