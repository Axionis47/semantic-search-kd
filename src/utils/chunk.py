"""Text chunking utilities for long documents."""

from typing import List, Tuple

import numpy as np
from transformers import AutoTokenizer


class TextChunker:
    """Chunk long texts into overlapping passages."""

    def __init__(
        self,
        tokenizer_name: str = "intfloat/e5-small-v2",
        max_tokens: int = 1000,
        stride: int = 160,
    ):
        """
        Initialize text chunker.

        Args:
            tokenizer_name: HuggingFace tokenizer name
            max_tokens: Maximum tokens per chunk
            stride: Overlap between chunks (in tokens)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_tokens = max_tokens
        self.stride = stride

    def chunk_text(self, text: str, doc_id: str = None) -> List[dict]:
        """
        Chunk a single text into overlapping passages.

        Args:
            text: Input text to chunk
            doc_id: Optional document ID

        Returns:
            List of chunk dictionaries with keys:
                - chunk_id: Unique chunk ID
                - doc_id: Document ID
                - text: Chunk text
                - tokens: Number of tokens
                - start_char: Start character position
                - end_char: End character position
        """
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= self.max_tokens:
            # No chunking needed
            return [
                {
                    "chunk_id": f"{doc_id}_0" if doc_id else "0",
                    "doc_id": doc_id,
                    "text": text,
                    "tokens": len(tokens),
                    "start_char": 0,
                    "end_char": len(text),
                }
            ]

        # Create overlapping chunks
        chunks = []
        chunk_idx = 0
        start_token = 0

        while start_token < len(tokens):
            end_token = min(start_token + self.max_tokens, len(tokens))

            # Decode chunk
            chunk_tokens = tokens[start_token:end_token]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            # Estimate character positions (approximate)
            start_char = int(start_token / len(tokens) * len(text))
            end_char = int(end_token / len(tokens) * len(text))

            chunks.append(
                {
                    "chunk_id": f"{doc_id}_{chunk_idx}" if doc_id else str(chunk_idx),
                    "doc_id": doc_id,
                    "text": chunk_text,
                    "tokens": len(chunk_tokens),
                    "start_char": start_char,
                    "end_char": end_char,
                }
            )

            chunk_idx += 1

            # Move to next chunk with stride
            if end_token >= len(tokens):
                break
            start_token += self.max_tokens - self.stride

        return chunks

    def chunk_batch(self, texts: List[str], doc_ids: List[str] = None) -> List[dict]:
        """
        Chunk a batch of texts.

        Args:
            texts: List of input texts
            doc_ids: Optional list of document IDs

        Returns:
            List of all chunks from all documents
        """
        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(texts))]

        all_chunks = []
        for text, doc_id in zip(texts, doc_ids):
            chunks = self.chunk_text(text, doc_id)
            all_chunks.extend(chunks)

        return all_chunks


def maxsim_aggregation(chunk_scores: List[Tuple[str, float]]) -> dict:
    """
    Aggregate chunk scores using MaxSim (max score per document).

    Args:
        chunk_scores: List of (chunk_id, score) tuples
            chunk_id format: "{doc_id}_{chunk_idx}"

    Returns:
        Dictionary mapping doc_id to max score
    """
    doc_scores = {}

    for chunk_id, score in chunk_scores:
        # Extract doc_id from chunk_id
        if "_" in chunk_id:
            doc_id = "_".join(chunk_id.split("_")[:-1])
        else:
            doc_id = chunk_id

        # Keep max score
        if doc_id not in doc_scores or score > doc_scores[doc_id]:
            doc_scores[doc_id] = score

    return doc_scores


def compute_text_overlap(text1: str, text2: str) -> float:
    """
    Compute text overlap ratio using character-level Jaccard similarity.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Overlap ratio (0-1)
    """
    # Normalize
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()

    if not text1 or not text2:
        return 0.0

    # Character-level n-grams (n=3)
    def get_ngrams(text: str, n: int = 3) -> set:
        return set(text[i : i + n] for i in range(len(text) - n + 1))

    ngrams1 = get_ngrams(text1)
    ngrams2 = get_ngrams(text2)

    if not ngrams1 or not ngrams2:
        return 0.0

    # Jaccard similarity
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)

    return intersection / union if union > 0 else 0.0

