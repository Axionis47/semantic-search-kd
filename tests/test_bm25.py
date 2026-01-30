"""Tests for BM25 index functionality."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.bm25 import BM25Index, build_bm25_index


class TestBM25Index:
    """Tests for BM25Index class."""

    @pytest.fixture
    def sample_corpus(self, tmp_path: Path) -> Path:
        """Create a sample corpus parquet file."""
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with many layers.",
            "Natural language processing helps computers understand text.",
            "Computer vision enables machines to interpret images.",
            "Reinforcement learning trains agents through rewards.",
        ]

        df = pd.DataFrame({
            "chunk_id": [f"doc_{i}" for i in range(len(documents))],
            "text": documents,
        })

        corpus_file = tmp_path / "corpus.parquet"
        df.to_parquet(corpus_file)
        return corpus_file

    @pytest.fixture
    def index_with_data(self, sample_corpus: Path, tmp_path: Path) -> BM25Index:
        """Create a BM25 index with sample data."""
        output_dir = tmp_path / "bm25_index"
        index = BM25Index()
        index.build_from_parquet(sample_corpus, output_dir)
        return index

    def test_build_from_parquet(self, sample_corpus: Path, tmp_path: Path):
        """Test building index from parquet file."""
        output_dir = tmp_path / "bm25_index"
        index = BM25Index()
        index.build_from_parquet(sample_corpus, output_dir)

        assert index.bm25 is not None
        assert len(index.doc_ids) == 5
        assert len(index.tokenized_corpus) == 5

    def test_save_and_load(self, index_with_data: BM25Index, tmp_path: Path):
        """Test saving and loading index."""
        # Save
        save_dir = tmp_path / "saved_index"
        save_dir.mkdir()
        index_with_data.index_path = save_dir
        index_with_data.save()

        # Verify files exist (JSON-based serialization)
        assert (save_dir / "doc_ids.json").exists()
        assert (save_dir / "tokenized_corpus.json").exists()
        assert (save_dir / "bm25_params.json").exists()
        assert (save_dir / "checksum.json").exists()

        # Load into new instance
        new_index = BM25Index(str(save_dir), auto_load=True)

        assert new_index.bm25 is not None
        assert len(new_index.doc_ids) == len(index_with_data.doc_ids)

    def test_search_returns_results(self, index_with_data: BM25Index):
        """Test that search returns results."""
        results = index_with_data.search("machine learning", top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)  # (doc_id, score)

    def test_search_relevance(self, index_with_data: BM25Index):
        """Test that search returns relevant results."""
        results = index_with_data.search("neural networks deep learning", top_k=5)

        doc_ids = [r[0] for r in results]
        # "Deep learning uses neural networks" should rank high
        assert "doc_1" in doc_ids[:2]

    def test_search_scores_sorted(self, index_with_data: BM25Index):
        """Test that results are sorted by score descending."""
        results = index_with_data.search("artificial intelligence", top_k=5)

        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_top_k_limit(self, index_with_data: BM25Index):
        """Test that top_k limits results."""
        results = index_with_data.search("learning", top_k=2)
        assert len(results) == 2

    def test_batch_search(self, index_with_data: BM25Index):
        """Test batch search functionality."""
        queries = ["machine learning", "neural networks", "computer vision"]
        results = index_with_data.batch_search(queries, top_k=3)

        assert len(results) == 3
        assert all(len(r) == 3 for r in results)

    def test_get_doc_text(self, index_with_data: BM25Index):
        """Test retrieving document text."""
        text = index_with_data.get_doc_text("doc_0")

        assert "machine" in text.lower()
        assert "learning" in text.lower()

    def test_get_doc_text_not_found(self, index_with_data: BM25Index):
        """Test retrieving non-existent document."""
        text = index_with_data.get_doc_text("nonexistent")
        assert text == ""

    def test_search_without_index_raises(self):
        """Test that search without loaded index raises error."""
        index = BM25Index()

        with pytest.raises(ValueError, match="Index not loaded"):
            index.search("test query")

    def test_batch_search_without_index_raises(self):
        """Test that batch_search without loaded index raises error."""
        index = BM25Index()

        with pytest.raises(ValueError, match="Index not loaded"):
            index.batch_search(["test query"])

    def test_tokenization(self, index_with_data: BM25Index):
        """Test that tokenization works correctly."""
        tokens = index_with_data._tokenize("Hello World Test")

        assert tokens == ["hello", "world", "test"]

    def test_load_nonexistent_raises(self, tmp_path: Path):
        """Test that loading non-existent index raises error."""
        index = BM25Index(str(tmp_path / "nonexistent"))

        with pytest.raises(FileNotFoundError):
            index.load()

    def test_custom_field_names(self, tmp_path: Path):
        """Test building index with custom field names."""
        documents = ["Doc one content", "Doc two content"]

        df = pd.DataFrame({
            "id": ["id_0", "id_1"],
            "content": documents,
        })

        corpus_file = tmp_path / "corpus.parquet"
        df.to_parquet(corpus_file)

        output_dir = tmp_path / "index"
        index = BM25Index()
        index.build_from_parquet(
            corpus_file,
            output_dir,
            text_field="content",
            id_field="id",
        )

        assert index.doc_ids == ["id_0", "id_1"]


class TestBuildBM25Index:
    """Tests for build_bm25_index helper function."""

    def test_returns_index(self, tmp_path: Path):
        """Test that helper returns BM25Index instance."""
        documents = ["Test document one", "Test document two"]

        df = pd.DataFrame({
            "chunk_id": ["doc_0", "doc_1"],
            "text": documents,
        })

        corpus_file = tmp_path / "corpus.parquet"
        df.to_parquet(corpus_file)

        output_dir = tmp_path / "index"
        index = build_bm25_index(corpus_file, output_dir)

        assert isinstance(index, BM25Index)
        assert index.bm25 is not None
        assert len(index.doc_ids) == 2


class TestBM25SearchQuality:
    """Tests for BM25 search quality."""

    @pytest.fixture
    def search_quality_corpus(self, tmp_path: Path) -> Path:
        """Create corpus for search quality tests."""
        documents = [
            "The quick brown fox jumps over the lazy dog.",
            "A fast auburn fox leaps above a sleepy canine.",
            "Python is a programming language used for machine learning.",
            "JavaScript is used for web development.",
            "Machine learning models require training data.",
            "Deep learning is a subset of machine learning.",
            "Natural language processing uses machine learning techniques.",
        ]

        df = pd.DataFrame({
            "chunk_id": [f"doc_{i}" for i in range(len(documents))],
            "text": documents,
        })

        corpus_file = tmp_path / "corpus.parquet"
        df.to_parquet(corpus_file)
        return corpus_file

    @pytest.fixture
    def quality_index(self, search_quality_corpus: Path, tmp_path: Path) -> BM25Index:
        """Create index for quality tests."""
        output_dir = tmp_path / "quality_index"
        index = BM25Index()
        index.build_from_parquet(search_quality_corpus, output_dir)
        return index

    def test_exact_match_ranks_high(self, quality_index: BM25Index):
        """Test that exact matches rank highly."""
        results = quality_index.search("quick brown fox", top_k=3)

        # "The quick brown fox..." should be first
        assert results[0][0] == "doc_0"

    def test_semantic_similarity(self, quality_index: BM25Index):
        """Test that semantically similar docs are found."""
        results = quality_index.search("machine learning", top_k=5)

        doc_ids = [r[0] for r in results]
        # All ML-related docs should be in top results
        ml_docs = {"doc_2", "doc_4", "doc_5", "doc_6"}
        assert len(ml_docs.intersection(set(doc_ids))) >= 3

    def test_irrelevant_query_low_scores(self, quality_index: BM25Index):
        """Test that irrelevant queries get low scores."""
        results = quality_index.search("basketball sports game", top_k=3)

        # All scores should be relatively low (no relevant docs)
        scores = [r[1] for r in results]
        assert all(s < 1.0 for s in scores)
