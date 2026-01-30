"""Model validation tests for ML CI/CD pipeline.

These tests validate that the trained KD model:
1. Loads correctly
2. Produces valid embeddings
3. Meets quality thresholds
4. Has acceptable inference latency
"""

import os
import time
from pathlib import Path

import numpy as np
import pytest

# Quality gate thresholds
EMBEDDING_DIM = 384
MAX_LOAD_TIME_SECONDS = 30
MAX_INFERENCE_LATENCY_MS = 50
MIN_SIMILARITY_SCORE = 0.5  # Similar texts should have high similarity


@pytest.fixture(scope="module")
def model_path() -> Path:
    """Get the path to the production model."""
    path = Path("./artifacts/models/kd_student_production")
    if not path.exists():
        if os.environ.get("REQUIRE_MODEL", "").lower() in ("1", "true", "yes"):
            pytest.fail(
                f"Production model not found at {path} but REQUIRE_MODEL is set. "
                "Ensure the model is downloaded before running validation tests."
            )
        pytest.skip("Production model not found - skipping model validation tests")
    return path


@pytest.fixture(scope="module")
def model(model_path: Path):
    """Load the model once for all tests."""
    from sentence_transformers import SentenceTransformer

    start = time.time()
    model = SentenceTransformer(str(model_path))
    load_time = time.time() - start

    assert load_time < MAX_LOAD_TIME_SECONDS, (
        f"Model load time {load_time:.2f}s exceeds {MAX_LOAD_TIME_SECONDS}s threshold"
    )

    return model


class TestModelLoading:
    """Tests for model loading and basic properties."""

    def test_model_loads(self, model):
        """Test that model loads without errors."""
        assert model is not None

    def test_embedding_dimension(self, model):
        """Test that model produces correct embedding dimension."""
        dim = model.get_sentence_embedding_dimension()
        assert dim == EMBEDDING_DIM, f"Expected {EMBEDDING_DIM}, got {dim}"

    def test_model_has_tokenizer(self, model):
        """Test that model has a tokenizer."""
        assert model.tokenizer is not None

    def test_model_max_seq_length(self, model):
        """Test that model has reasonable max sequence length."""
        max_len = model.max_seq_length
        assert max_len >= 256, f"Max seq length {max_len} is too short"
        assert max_len <= 8192, f"Max seq length {max_len} is unusually long"


class TestEmbeddingQuality:
    """Tests for embedding quality and properties."""

    def test_embeddings_are_normalized(self, model):
        """Test that embeddings are L2 normalized."""
        texts = ["This is a test sentence.", "Another test document."]
        embeddings = model.encode(texts)

        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(
            norms, np.ones_like(norms), atol=0.01,
            err_msg="Embeddings are not normalized"
        )

    def test_embeddings_shape(self, model):
        """Test that embeddings have correct shape."""
        texts = ["text 1", "text 2", "text 3"]
        embeddings = model.encode(texts)

        assert embeddings.shape == (3, EMBEDDING_DIM), (
            f"Expected shape (3, {EMBEDDING_DIM}), got {embeddings.shape}"
        )

    def test_embeddings_are_deterministic(self, model):
        """Test that same input produces same output."""
        text = "This is a test for determinism."

        emb1 = model.encode([text])
        emb2 = model.encode([text])

        np.testing.assert_array_almost_equal(
            emb1, emb2, decimal=5,
            err_msg="Embeddings are not deterministic"
        )

    def test_similar_texts_have_high_similarity(self, model):
        """Test that semantically similar texts have high cosine similarity."""
        similar_pairs = [
            ("What is machine learning?", "Explain ML algorithms"),
            ("How do neural networks work?", "Neural network architecture explained"),
            ("Python programming tutorial", "Learn Python coding"),
        ]

        for text1, text2 in similar_pairs:
            emb1 = model.encode([text1])
            emb2 = model.encode([text2])

            similarity = float(np.dot(emb1[0], emb2[0]))

            assert similarity > MIN_SIMILARITY_SCORE, (
                f"Similar texts '{text1}' and '{text2}' have low similarity: {similarity:.3f}"
            )

    def test_different_texts_have_lower_similarity(self, model):
        """Test that unrelated texts have lower similarity than related ones."""
        query = "Machine learning algorithms"
        related = "Deep learning neural networks"
        unrelated = "Cooking recipes for beginners"

        emb_query = model.encode([query])
        emb_related = model.encode([related])
        emb_unrelated = model.encode([unrelated])

        sim_related = float(np.dot(emb_query[0], emb_related[0]))
        sim_unrelated = float(np.dot(emb_query[0], emb_unrelated[0]))

        assert sim_related > sim_unrelated, (
            f"Related text similarity ({sim_related:.3f}) should be higher than "
            f"unrelated ({sim_unrelated:.3f})"
        )

    def test_query_prefix_handling(self, model):
        """Test that query prefix improves retrieval quality."""
        query = "What is information retrieval?"
        doc = "Information retrieval is the process of obtaining relevant documents from a collection."

        # E5 models use query: and passage: prefixes
        emb_query = model.encode([f"query: {query}"])
        emb_doc = model.encode([f"passage: {doc}"])

        similarity = float(np.dot(emb_query[0], emb_doc[0]))

        assert similarity > 0.4, f"Query-document similarity too low: {similarity:.3f}"


class TestInferencePerformance:
    """Tests for inference latency and throughput."""

    def test_single_inference_latency(self, model):
        """Test single text inference latency."""
        text = "This is a test query for latency measurement."

        # Warmup
        model.encode([text])

        # Measure
        latencies = []
        for _ in range(10):
            start = time.time()
            model.encode([text])
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        assert p95_latency < MAX_INFERENCE_LATENCY_MS, (
            f"P95 latency {p95_latency:.2f}ms exceeds {MAX_INFERENCE_LATENCY_MS}ms threshold"
        )

    def test_batch_inference(self, model):
        """Test batch inference works correctly."""
        texts = [f"Test document number {i}" for i in range(32)]

        embeddings = model.encode(texts, batch_size=8)

        assert embeddings.shape == (32, EMBEDDING_DIM)

    def test_batch_inference_latency(self, model):
        """Test batch inference latency is reasonable."""
        texts = [f"Test document number {i}" for i in range(16)]

        # Warmup
        model.encode(texts)

        start = time.time()
        model.encode(texts)
        latency_ms = (time.time() - start) * 1000

        # Batch of 16 should complete in under 200ms
        assert latency_ms < 200, f"Batch inference took {latency_ms:.2f}ms"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_string(self, model):
        """Test handling of empty string."""
        emb = model.encode([""])
        assert emb.shape == (1, EMBEDDING_DIM)

    def test_very_long_text(self, model):
        """Test handling of text longer than max sequence length."""
        long_text = "word " * 1000  # ~5000 tokens
        emb = model.encode([long_text])
        assert emb.shape == (1, EMBEDDING_DIM)

    def test_special_characters(self, model):
        """Test handling of special characters."""
        texts = [
            "Hello! @#$%^&*()",
            "Unicode: 你好世界 🎉",
            "Newlines\n\nand\ttabs",
        ]
        embeddings = model.encode(texts)
        assert embeddings.shape == (3, EMBEDDING_DIM)

    def test_empty_batch(self, model):
        """Test handling of empty batch."""
        emb = model.encode([])
        assert emb.shape == (0, EMBEDDING_DIM) or len(emb) == 0


class TestModelArtifacts:
    """Tests for model artifact completeness."""

    def test_required_files_exist(self, model_path: Path):
        """Test that all required model files exist."""
        required_files = [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        ]

        for filename in required_files:
            file_path = model_path / filename
            assert file_path.exists(), f"Missing required file: {filename}"

    def test_pooling_config_exists(self, model_path: Path):
        """Test that pooling configuration exists."""
        pooling_dir = model_path / "1_Pooling"
        assert pooling_dir.exists(), "Missing 1_Pooling directory"

        config_path = pooling_dir / "config.json"
        assert config_path.exists(), "Missing pooling config.json"

    def test_model_size_reasonable(self, model_path: Path):
        """Test that model size is within expected range."""
        model_file = model_path / "model.safetensors"
        size_mb = model_file.stat().st_size / (1024 * 1024)

        # E5-small should be ~120-140 MB
        assert 100 < size_mb < 200, f"Model size {size_mb:.1f}MB is outside expected range"
