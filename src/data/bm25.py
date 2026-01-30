"""BM25 index building and querying using rank-bm25."""

import hashlib
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from loguru import logger
from rank_bm25 import BM25Okapi
from tqdm import tqdm


class BM25Index:
    """BM25 index for retrieval using rank-bm25."""

    def __init__(self, index_path: str = None, auto_load: bool = False):
        """
        Initialize BM25 index.

        Args:
            index_path: Path to saved index directory (optional)
            auto_load: Whether to automatically load index if it exists
        """
        self.index_path = Path(index_path) if index_path else None
        self.bm25 = None
        self.doc_ids = []
        self.tokenized_corpus = []

        if auto_load and self.index_path and (self.index_path / "tokenized_corpus.json").exists():
            self.load()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (split on whitespace and lowercase)."""
        return text.lower().split()

    def build_from_parquet(
        self,
        corpus_file: Path,
        output_dir: Path,
        text_field: str = "text",
        id_field: str = "chunk_id",
    ) -> None:
        """
        Build BM25 index from Parquet corpus.

        Args:
            corpus_file: Path to Parquet file with corpus
            output_dir: Output directory for index
            text_field: Field name for text content
            id_field: Field name for document ID
        """
        logger.info(f"Building BM25 index from {corpus_file}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load corpus
        df = pd.read_parquet(corpus_file)
        logger.info(f"Loaded {len(df)} documents")

        # Tokenize corpus
        self.doc_ids = df[id_field].tolist()
        self.tokenized_corpus = []

        for text in tqdm(df[text_field], desc="Tokenizing corpus"):
            self.tokenized_corpus.append(self._tokenize(str(text)))

        # Build BM25 index
        logger.info("Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Save index
        self.index_path = output_dir
        self.save()

        logger.info(f"BM25 index built successfully at {output_dir}")

    def save(self) -> None:
        """Save BM25 index to disk using JSON (no pickle for security)."""
        if not self.index_path:
            raise ValueError("index_path not set")

        self.index_path.mkdir(parents=True, exist_ok=True)

        # Save doc IDs
        with open(self.index_path / "doc_ids.json", "w") as f:
            json.dump(self.doc_ids, f)

        # Save tokenized corpus as JSON (list of token lists)
        with open(self.index_path / "tokenized_corpus.json", "w") as f:
            json.dump(self.tokenized_corpus, f)

        # Save BM25 parameters needed to reconstruct the index
        bm25_params = {
            "k1": getattr(self.bm25, "k1", 1.5),
            "b": getattr(self.bm25, "b", 0.75),
            "epsilon": getattr(self.bm25, "epsilon", 0.25),
            "corpus_size": len(self.tokenized_corpus),
        }
        with open(self.index_path / "bm25_params.json", "w") as f:
            json.dump(bm25_params, f)

        # Write checksum for integrity verification
        checksum = self._compute_checksum()
        with open(self.index_path / "checksum.json", "w") as f:
            json.dump({"sha256": checksum}, f)

        logger.info(f"Saved BM25 index to {self.index_path}")

    def _compute_checksum(self) -> str:
        """Compute SHA256 checksum of index data for integrity verification."""
        h = hashlib.sha256()
        h.update(json.dumps(self.doc_ids, sort_keys=True).encode())
        h.update(json.dumps(self.tokenized_corpus, sort_keys=True).encode())
        return h.hexdigest()

    def load(self) -> None:
        """Load existing BM25 index from JSON files (safe deserialization)."""
        if not self.index_path or not self.index_path.exists():
            raise FileNotFoundError(f"Index not found: {self.index_path}")

        logger.info(f"Loading BM25 index from {self.index_path}")

        # Load doc IDs
        with open(self.index_path / "doc_ids.json", "r") as f:
            self.doc_ids = json.load(f)

        # Load tokenized corpus from JSON
        corpus_json = self.index_path / "tokenized_corpus.json"
        if corpus_json.exists():
            try:
                with open(corpus_json, "r") as f:
                    self.tokenized_corpus = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Checksum/integrity failure: tokenized_corpus.json is corrupted — {e}"
                ) from e
        else:
            raise FileNotFoundError(
                f"tokenized_corpus.json not found at {self.index_path}. "
                "If you have a legacy .pkl index, rebuild it with build_from_parquet()."
            )

        # Verify integrity if checksum exists
        checksum_file = self.index_path / "checksum.json"
        if checksum_file.exists():
            with open(checksum_file, "r") as f:
                expected = json.load(f)["sha256"]
            actual = self._compute_checksum()
            if actual != expected:
                raise ValueError(
                    f"Index integrity check failed: checksum mismatch. "
                    f"Expected {expected[:12]}..., got {actual[:12]}..."
                )

        # Reconstruct BM25 index from tokenized corpus
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        logger.info(f"Loaded index with {len(self.doc_ids)} documents")

    def search(
        self,
        query: str,
        top_k: int = 100,
    ) -> List[Tuple[str, float]]:
        """
        Search BM25 index.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of (doc_id, score) tuples sorted by score descending
        """
        if self.bm25 is None:
            raise ValueError("Index not loaded. Call load() or build_from_parquet() first.")

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # Return (doc_id, score) tuples
        results = [(self.doc_ids[i], float(scores[i])) for i in top_indices]

        return results

    def batch_search(
        self,
        queries: List[str],
        top_k: int = 100,
    ) -> List[List[Tuple[str, float]]]:
        """
        Batch search BM25 index.

        Args:
            queries: List of query texts
            top_k: Number of results per query

        Returns:
            List of result lists (one per query)
        """
        if self.bm25 is None:
            raise ValueError("Index not loaded. Call load() first.")

        all_results = []
        for query in tqdm(queries, desc="BM25 search"):
            results = self.search(query, top_k=top_k)
            all_results.append(results)

        return all_results

    def get_doc_text(self, doc_id: str) -> str:
        """
        Get document text by ID (reconstructed from tokens).

        Args:
            doc_id: Document ID

        Returns:
            Document text (reconstructed)
        """
        if self.bm25 is None:
            raise ValueError("Index not loaded. Call load() or build_from_parquet() first.")

        try:
            idx = self.doc_ids.index(doc_id)
            return " ".join(self.tokenized_corpus[idx])
        except ValueError:
            return ""


def build_bm25_index(
    corpus_file: Path,
    output_dir: Path,
    text_field: str = "text",
    id_field: str = "chunk_id",
) -> BM25Index:
    """
    Build BM25 index from corpus file.

    Args:
        corpus_file: Path to Parquet corpus file
        output_dir: Output directory for index
        text_field: Field name for text
        id_field: Field name for ID

    Returns:
        BM25Index instance
    """
    index = BM25Index(str(output_dir))
    index.build_from_parquet(corpus_file, output_dir, text_field, id_field)
    return index


if __name__ == "__main__":
    from src.utils.logging import setup_logging

    setup_logging(log_level="INFO")

    # Example: Build BM25 index for MS MARCO
    corpus_file = Path("./data/chunks/msmarco/train.parquet")
    output_dir = Path("./artifacts/indexes/bm25_msmarco")

    if corpus_file.exists():
        index = build_bm25_index(corpus_file, output_dir)

        # Test search
        results = index.search("what is knowledge distillation?", top_k=10)
        print("\n=== BM25 Search Results ===")
        for doc_id, score in results[:5]:
            print(f"{doc_id}: {score:.4f}")
            print(f"  {index.get_doc_text(doc_id)[:200]}...")
    else:
        logger.error(f"Corpus file not found: {corpus_file}")
        logger.info("Run 'make data-prepare' first")

