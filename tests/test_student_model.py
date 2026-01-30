"""Tests for StudentModel class."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestStudentModel:
    """Tests for StudentModel initialization and encoding."""

    @patch("src.models.student.SentenceTransformer")
    def test_default_device_cpu(self, mock_st):
        """Test default device selection falls back to CPU."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        with patch("src.models.student.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            from src.models.student import StudentModel

            model = StudentModel(model_name="test-model", device=None)
            assert model.device == "cpu"

    @patch("src.models.student.SentenceTransformer")
    def test_explicit_device(self, mock_st):
        """Test explicit device assignment."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        from src.models.student import StudentModel

        model = StudentModel(model_name="test-model", device="cpu")
        assert model.device == "cpu"

    @patch("src.models.student.SentenceTransformer")
    def test_encode_single_text(self, mock_st):
        """Test encoding a single text string."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.randn(1, 384).astype(np.float32)
        mock_st.return_value = mock_model

        from src.models.student import StudentModel

        model = StudentModel(model_name="test-model", device="cpu")
        result = model.encode("hello world")

        # Should wrap single string into list
        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args
        assert isinstance(call_args[0][0], list)

    @patch("src.models.student.SentenceTransformer")
    def test_encode_multiple_texts(self, mock_st):
        """Test encoding multiple texts."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.randn(3, 384).astype(np.float32)
        mock_st.return_value = mock_model

        from src.models.student import StudentModel

        model = StudentModel(model_name="test-model", device="cpu")
        texts = ["text1", "text2", "text3"]
        result = model.encode(texts)

        assert result.shape == (3, 384)

    @patch("src.models.student.SentenceTransformer")
    def test_encode_queries_adds_prefix_for_e5(self, mock_st):
        """Test that E5 models get query prefix."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.randn(1, 384).astype(np.float32)
        mock_st.return_value = mock_model

        from src.models.student import StudentModel

        model = StudentModel(model_name="intfloat/e5-small-v2", device="cpu")
        model.encode_queries("test query")

        call_args = mock_model.encode.call_args
        assert call_args[0][0] == ["query: test query"]

    @patch("src.models.student.SentenceTransformer")
    def test_encode_documents_adds_prefix_for_e5(self, mock_st):
        """Test that E5 models get passage prefix."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.randn(1, 384).astype(np.float32)
        mock_st.return_value = mock_model

        from src.models.student import StudentModel

        model = StudentModel(model_name="intfloat/e5-small-v2", device="cpu")
        model.encode_documents("test document")

        call_args = mock_model.encode.call_args
        assert call_args[0][0] == ["passage: test document"]

    @patch("src.models.student.SentenceTransformer")
    def test_compute_similarity_normalized(self, mock_st):
        """Test similarity computation with normalized embeddings."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        from src.models.student import StudentModel

        model = StudentModel(model_name="test-model", device="cpu")

        # Create normalized embeddings
        q = np.random.randn(2, 384).astype(np.float32)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        d = np.random.randn(3, 384).astype(np.float32)
        d = d / np.linalg.norm(d, axis=1, keepdims=True)

        sims = model.compute_similarity(q, d)
        assert sims.shape == (2, 3)
        # Cosine similarity should be in [-1, 1]
        assert np.all(sims >= -1.01) and np.all(sims <= 1.01)

    @patch("src.models.student.SentenceTransformer")
    def test_cleanup_cpu_noop(self, mock_st):
        """Test cleanup on CPU does nothing."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        from src.models.student import StudentModel

        model = StudentModel(model_name="test-model", device="cpu")
        # Should not raise
        model.cleanup()
