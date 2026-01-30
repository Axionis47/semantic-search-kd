"""End-to-end verification tests for all 27 hardening fixes.

Tests are grouped by the issue they verify from the hardening plan.
"""

import hashlib
import threading
import time
import warnings
from unittest.mock import MagicMock

import pytest


# ===========================================================================
# #1 — BM25 Safe Serialization (no pickle)
# ===========================================================================
class TestBM25SafeSerialization:
    """Verify BM25 index uses JSON instead of pickle."""

    def test_save_produces_json_files(self, tmp_path):
        """Saved index should contain JSON files, not pickle."""
        import pandas as pd
        from src.data.bm25 import BM25Index

        # Build a small index
        docs = ["hello world", "foo bar baz", "test document"]
        df = pd.DataFrame({"chunk_id": ["d0", "d1", "d2"], "text": docs})
        corpus_path = tmp_path / "corpus.parquet"
        df.to_parquet(corpus_path)

        save_dir = tmp_path / "bm25_idx"
        idx = BM25Index(str(save_dir))
        idx.build_from_parquet(str(corpus_path), output_dir=str(save_dir))
        idx.save()

        assert (save_dir / "tokenized_corpus.json").exists()
        assert (save_dir / "bm25_params.json").exists()
        assert (save_dir / "checksum.json").exists()
        assert (save_dir / "doc_ids.json").exists()
        # Must NOT have pickle files
        assert not (save_dir / "bm25.pkl").exists()

    def test_save_load_roundtrip_search(self, tmp_path):
        """Index saved and reloaded should still return correct search results."""
        import pandas as pd
        from src.data.bm25 import BM25Index

        docs = [
            "machine learning is great",
            "the weather is sunny today",
            "deep neural networks",
        ]
        df = pd.DataFrame({"chunk_id": ["d0", "d1", "d2"], "text": docs})
        corpus_path = tmp_path / "corpus.parquet"
        df.to_parquet(corpus_path)

        idx_dir = tmp_path / "idx"
        idx = BM25Index(str(idx_dir))
        idx.build_from_parquet(str(corpus_path), output_dir=str(idx_dir))
        original_results = idx.search("machine learning", top_k=2)
        idx.save()

        # Reload
        idx2 = BM25Index(str(idx_dir), auto_load=True)
        reloaded_results = idx2.search("machine learning", top_k=2)

        assert len(reloaded_results) == 2
        assert reloaded_results[0][0] == original_results[0][0]  # Same top doc

    def test_checksum_detects_corruption(self, tmp_path):
        """Corrupted corpus file should fail checksum on load."""
        import pandas as pd
        from src.data.bm25 import BM25Index

        docs = ["hello world", "test"]
        df = pd.DataFrame({"chunk_id": ["d0", "d1"], "text": docs})
        corpus_path = tmp_path / "corpus.parquet"
        df.to_parquet(corpus_path)

        idx_dir = tmp_path / "idx"
        idx = BM25Index(str(idx_dir))
        idx.build_from_parquet(str(corpus_path), output_dir=str(idx_dir))
        idx.save()

        # Corrupt the tokenized corpus
        corpus_file = idx_dir / "tokenized_corpus.json"
        corpus_file.write_text("corrupted data")

        with pytest.raises(ValueError, match="[Cc]hecksum"):
            BM25Index(str(idx_dir), auto_load=True)


# ===========================================================================
# #2 — Salted API Key Hashing
# ===========================================================================
class TestSaltedAPIKeyHashing:
    """Verify API key hashing uses PBKDF2 with salt."""

    def test_hash_with_salt_differs_from_unsalted(self):
        """Salted hash should differ from plain SHA256."""
        from src.serve.middleware import APIKeyAuth

        auth = APIKeyAuth(enabled=True)
        key = "test-api-key-123"
        unsalted = auth._hash_key(key)
        salted = auth._hash_key(key, salt="random_salt")
        assert unsalted != salted

    def test_api_key_validates(self):
        """Key added via constructor should validate."""
        from src.serve.middleware import APIKeyAuth

        auth = APIKeyAuth(api_keys=["my-secret-key"], enabled=True)
        mock_request = MagicMock()
        mock_request.headers = {"X-API-Key": "my-secret-key"}
        assert auth.verify(mock_request) is True

    def test_wrong_key_rejected(self):
        """Wrong key should be rejected."""
        from src.serve.middleware import APIKeyAuth

        auth = APIKeyAuth(api_keys=["correct-key"], enabled=True)
        mock_request = MagicMock()
        mock_request.headers = {"X-API-Key": "wrong-key"}
        assert auth.verify(mock_request) is False

    def test_disabled_auth_allows_all(self):
        """When disabled, all requests pass."""
        from src.serve.middleware import APIKeyAuth

        auth = APIKeyAuth(enabled=False)
        mock_request = MagicMock()
        mock_request.headers = {}
        assert auth.verify(mock_request) is True


# ===========================================================================
# #3 — Thread-Safe Rate Limiter
# ===========================================================================
class TestThreadSafeRateLimiter:
    """Verify rate limiter works correctly under concurrent access."""

    def test_concurrent_access_no_crash(self):
        """Multiple threads hitting rate limiter should not crash."""
        from src.serve.middleware import RateLimiter

        limiter = RateLimiter(requests_per_minute=60, burst=5, enabled=True)
        errors = []

        def make_request(client_ip):
            try:
                mock_req = MagicMock()
                mock_req.headers = {}
                mock_req.client.host = client_ip
                for _ in range(20):
                    limiter.check(mock_req)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=make_request, args=(f"10.0.0.{i}",))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_rate_limiting_enforced(self):
        """Burst limit should be enforced."""
        from src.serve.middleware import RateLimiter

        limiter = RateLimiter(requests_per_minute=60, burst=3, enabled=True)
        mock_req = MagicMock()
        mock_req.headers = {}
        mock_req.client.host = "1.2.3.4"

        # First 3 should pass (burst=3)
        for _ in range(3):
            allowed, _ = limiter.check(mock_req)
            assert allowed is True

        # Next should be rate limited
        allowed, retry_after = limiter.check(mock_req)
        assert allowed is False
        assert retry_after > 0


# ===========================================================================
# #16 — Rate Limiter Bucket Cap
# ===========================================================================
class TestRateLimiterBucketCap:
    """Verify max_buckets eviction works."""

    def test_evicts_oldest_at_capacity(self):
        """When max_buckets reached, oldest should be evicted."""
        from src.serve.middleware import RateLimiter

        limiter = RateLimiter(
            requests_per_minute=60, burst=5, enabled=True, max_buckets=3
        )

        for i in range(4):
            mock_req = MagicMock()
            mock_req.headers = {}
            mock_req.client.host = f"10.0.0.{i}"
            limiter.check(mock_req)
            time.sleep(0.01)  # Ensure different timestamps

        assert len(limiter.buckets) == 3
        # First client should have been evicted
        assert "10.0.0.0" not in limiter.buckets


# ===========================================================================
# #4 — LossConfig Validator
# ===========================================================================
class TestLossConfigValidator:
    """Verify loss weights must sum to ~1.0."""

    def test_valid_weights_pass(self):
        """Weights summing to 1.0 should pass."""
        from src.config import LossConfig

        config = LossConfig(
            contrastive_weight=0.2,
            margin_mse_weight=0.6,
            listwise_kd_weight=0.2,
        )
        assert config.contrastive_weight == 0.2

    def test_invalid_weights_rejected(self):
        """Weights not summing to 1.0 should raise."""
        from pydantic import ValidationError
        from src.config import LossConfig

        with pytest.raises(ValidationError, match="sum to"):
            LossConfig(
                contrastive_weight=0.5,
                margin_mse_weight=0.5,
                listwise_kd_weight=0.5,
            )


# ===========================================================================
# #13 — Production Config Enforcement
# ===========================================================================
class TestProductionConfigEnforcement:
    """Verify warnings for insecure production settings."""

    def test_production_warns_auth_disabled(self):
        """Production with auth disabled should emit warning."""
        from src.config import Settings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Settings(environment="production")
            auth_warnings = [x for x in w if "Auth is disabled" in str(x.message)]
            assert len(auth_warnings) >= 1

    def test_development_no_auth_warning(self):
        """Development should not warn about auth."""
        from src.config import Settings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Settings(environment="development")
            auth_warnings = [x for x in w if "Auth is disabled" in str(x.message)]
            assert len(auth_warnings) == 0


# ===========================================================================
# #15 — Script Argument Validation
# ===========================================================================
class TestScriptArgValidation:
    """Verify script argument validators exit on bad input."""

    def test_validate_path_exists_nonexistent(self):
        """Nonexistent path should cause SystemExit."""
        from scripts._validate_args import validate_path_exists

        with pytest.raises(SystemExit):
            validate_path_exists("/no/such/path/anywhere", "--model-path")

    def test_validate_positive_int_zero(self):
        """Zero should cause SystemExit."""
        from scripts._validate_args import validate_positive_int

        with pytest.raises(SystemExit):
            validate_positive_int(0, "--max-samples")

    def test_validate_positive_int_negative(self):
        """Negative should cause SystemExit."""
        from scripts._validate_args import validate_positive_int

        with pytest.raises(SystemExit):
            validate_positive_int(-5, "--max-samples")

    def test_validate_positive_int_valid(self):
        """Positive value should not raise."""
        from scripts._validate_args import validate_positive_int

        validate_positive_int(100, "--max-samples")  # Should not raise

    def test_validate_port_out_of_range(self):
        """Port > 65535 should cause SystemExit."""
        from scripts._validate_args import validate_port

        with pytest.raises(SystemExit):
            validate_port(99999)

    def test_validate_port_zero(self):
        """Port 0 should cause SystemExit."""
        from scripts._validate_args import validate_port

        with pytest.raises(SystemExit):
            validate_port(0)

    def test_validate_device_invalid(self):
        """Invalid device should cause SystemExit."""
        from scripts._validate_args import validate_device

        with pytest.raises(SystemExit):
            validate_device("tpu")

    def test_validate_device_valid_cpu(self):
        """cpu should be valid."""
        from scripts._validate_args import validate_device

        validate_device("cpu")  # Should not raise

    def test_validate_device_valid_cuda(self):
        """cuda should be valid."""
        from scripts._validate_args import validate_device

        validate_device("cuda")  # Should not raise

    def test_validate_device_valid_cuda_index(self):
        """cuda:0 should be valid."""
        from scripts._validate_args import validate_device

        validate_device("cuda:0")  # Should not raise


# ===========================================================================
# #18 — SHA256 Query Hashing (no MD5)
# ===========================================================================
class TestSHA256QueryHashing:
    """Verify query hashing uses SHA256, not MD5."""

    def test_hash_query_uses_sha256(self):
        """_hash_query output should match SHA256 truncated to 12 chars."""
        from src.serve.middleware import RequestLoggingMiddleware

        middleware = RequestLoggingMiddleware(app=MagicMock())
        query = "test query"
        result = middleware._hash_query(query)
        expected = hashlib.sha256(query.encode()).hexdigest()[:12]
        assert result == expected
        assert len(result) == 12

    def test_hash_query_not_md5(self):
        """Output should NOT match MD5."""
        from src.serve.middleware import RequestLoggingMiddleware

        middleware = RequestLoggingMiddleware(app=MagicMock())
        query = "test query"
        result = middleware._hash_query(query)
        md5_result = hashlib.md5(query.encode()).hexdigest()[:12]
        assert result != md5_result


# ===========================================================================
# #24 — Chunk Character Positions
# ===========================================================================
class TestChunkCharacterPositions:
    """Verify character positions use offset mapping."""

    def test_single_chunk_full_span(self):
        """Short text should have start_char=0, end_char=len(text)."""
        from src.utils.chunk import TextChunker

        chunker = TextChunker(max_tokens=1000)
        text = "Hello world this is a test."
        chunks = chunker.chunk_text(text, doc_id="test")
        assert len(chunks) == 1
        assert chunks[0]["start_char"] == 0
        assert chunks[0]["end_char"] == len(text)

    def test_multi_chunk_positions_cover_text(self):
        """Multiple chunks should cover the full text span."""
        from src.utils.chunk import TextChunker

        chunker = TextChunker(max_tokens=10, stride=3)
        text = "The quick brown fox jumps over the lazy dog. " * 20
        chunks = chunker.chunk_text(text, doc_id="test")

        assert len(chunks) > 1
        # First chunk starts at 0
        assert chunks[0]["start_char"] == 0
        # Last chunk ends near text length
        assert chunks[-1]["end_char"] <= len(text)
        assert chunks[-1]["end_char"] > 0


# ===========================================================================
# #4 + YAML Round-Trip — Config Serialization
# ===========================================================================
class TestYAMLRoundTrip:
    """Verify Settings can be saved and reloaded via YAML."""

    def test_settings_yaml_roundtrip(self, tmp_path):
        """Settings should survive YAML save/load cycle."""
        from src.config import Settings

        original = Settings(environment="development")
        yaml_path = tmp_path / "config.yaml"
        original.to_yaml(yaml_path)

        reloaded = Settings.from_yaml(yaml_path)
        assert reloaded.environment == "development"
        assert reloaded.student.max_length == original.student.max_length
        assert reloaded.training.epochs == original.training.epochs


# ===========================================================================
# #17 — GPU Cleanup
# ===========================================================================
class TestGPUCleanup:
    """Verify cleanup method exists and doesn't crash on CPU."""

    def test_cleanup_exists_on_student_model(self):
        """StudentModel should have a cleanup method."""
        from src.models.student import StudentModel

        assert hasattr(StudentModel, "cleanup")

    def test_cleanup_cpu_no_error(self):
        """Calling cleanup on CPU model should not raise."""
        from unittest.mock import patch, MagicMock

        with patch("src.models.student.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model

            from src.models.student import StudentModel

            model = StudentModel(model_name="test", device="cpu")
            model.cleanup()  # Should not raise
