"""Prepare data: JSONL → Parquet with chunking."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.data.registry import get_dataset_config, get_manifest_path
from src.utils.chunk import TextChunker


def prepare_msmarco_split(
    input_file: Path,
    output_dir: Path,
    chunker: TextChunker,
    split_name: str,
) -> Dict:
    """
    Prepare a single MS MARCO split: JSONL → chunked Parquet.

    Args:
        input_file: Input JSONL file
        output_dir: Output directory for Parquet files
        chunker: TextChunker instance
        split_name: Split name (train/validation/test)

    Returns:
        Metadata dictionary
    """
    logger.info(f"Preparing {split_name} split from {input_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    num_queries = 0
    num_passages = 0

    with open(input_file, "r") as f:
        for line in tqdm(f, desc=f"Processing {split_name}"):
            try:
                data = json.loads(line)
                query_id = data.get("query_id", data.get("query", ""))
                query_text = data.get("query", "")
                passages = data.get("passages", [])

                num_queries += 1

                # Handle nested passages structure (MS MARCO v2.1 format)
                if isinstance(passages, dict):
                    passage_texts = passages.get("passage_text", [])
                    is_selected_list = passages.get("is_selected", [])

                    # Process each passage
                    for idx, passage_text in enumerate(passage_texts):
                        is_selected = is_selected_list[idx] if idx < len(is_selected_list) else 0

                        if not passage_text:
                            continue

                        num_passages += 1

                        # Chunk passage
                        chunks = chunker.chunk_text(
                            passage_text, doc_id=f"q{query_id}_p{num_passages}"
                        )

                        for chunk in chunks:
                            all_chunks.append(
                                {
                                    "chunk_id": chunk["chunk_id"],
                                    "doc_id": chunk["doc_id"],
                                    "query_id": query_id,
                                    "query_text": query_text,
                                    "text": chunk["text"],
                                    "tokens": chunk["tokens"],
                                    "is_relevant": is_selected,
                                    "split": split_name,
                                    "updated_at": datetime.now().isoformat(),
                                }
                            )
                else:
                    # Handle list format (older MS MARCO versions)
                    for passage in passages:
                        passage_text = passage.get("passage_text", "")
                        is_selected = passage.get("is_selected", 0)

                        if not passage_text:
                            continue

                        num_passages += 1

                        # Chunk passage
                        chunks = chunker.chunk_text(
                            passage_text, doc_id=f"q{query_id}_p{num_passages}"
                        )

                        for chunk in chunks:
                            all_chunks.append(
                                {
                                    "chunk_id": chunk["chunk_id"],
                                    "doc_id": chunk["doc_id"],
                                    "query_id": query_id,
                                    "query_text": query_text,
                                    "text": chunk["text"],
                                    "tokens": chunk["tokens"],
                                    "is_relevant": is_selected,
                                    "split": split_name,
                                    "updated_at": datetime.now().isoformat(),
                                }
                            )

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line: {e}")
                continue

    # Convert to DataFrame and save as Parquet
    df = pd.DataFrame(all_chunks)
    output_file = output_dir / f"{split_name}.parquet"
    df.to_parquet(output_file, index=False, compression="snappy")

    logger.info(
        f"Saved {len(df)} chunks to {output_file} ({num_queries} queries, {num_passages} passages)"
    )

    return {
        "file": str(output_file),
        "num_chunks": len(df),
        "num_queries": num_queries,
        "num_passages": num_passages,
    }


def prepare_beir_corpus(
    corpus_file: Path,
    output_dir: Path,
    chunker: TextChunker,
) -> Dict:
    """
    Prepare BEIR corpus: JSONL → chunked Parquet.

    Args:
        corpus_file: Input corpus JSONL file
        output_dir: Output directory
        chunker: TextChunker instance

    Returns:
        Metadata dictionary
    """
    logger.info(f"Preparing BEIR corpus from {corpus_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []

    with open(corpus_file, "r") as f:
        for line in tqdm(f, desc="Processing corpus"):
            try:
                data = json.loads(line)
                doc_id = data.get("doc_id", "")
                title = data.get("title", "")
                text = data.get("text", "")

                # Combine title and text
                full_text = f"{title}\n{text}" if title else text

                if not full_text:
                    continue

                # Chunk document
                chunks = chunker.chunk_text(full_text, doc_id=doc_id)

                for chunk in chunks:
                    all_chunks.append(
                        {
                            "chunk_id": chunk["chunk_id"],
                            "doc_id": chunk["doc_id"],
                            "title": title,
                            "text": chunk["text"],
                            "tokens": chunk["tokens"],
                            "updated_at": datetime.now().isoformat(),
                        }
                    )

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line: {e}")
                continue

    # Save as Parquet
    df = pd.DataFrame(all_chunks)
    output_file = output_dir / "corpus.parquet"
    df.to_parquet(output_file, index=False, compression="snappy")

    logger.info(f"Saved {len(df)} chunks to {output_file}")

    return {
        "file": str(output_file),
        "num_chunks": len(df),
        "num_docs": len(set(df["doc_id"])),
    }


def prepare_dataset(dataset_name: str, max_tokens: int = 1000, stride: int = 160) -> Dict:
    """
    Prepare a dataset: JSONL → chunked Parquet.

    Args:
        dataset_name: Dataset name (e.g., 'msmarco', 'beir_fiqa')
        max_tokens: Maximum tokens per chunk
        stride: Overlap between chunks

    Returns:
        Preparation metadata
    """
    logger.info(f"Preparing dataset: {dataset_name}")

    config = get_dataset_config(dataset_name)
    manifest_path = get_manifest_path(dataset_name)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Initialize chunker
    chunker = TextChunker(max_tokens=max_tokens, stride=stride)

    output_dir = config["chunks_dir"]
    prep_manifest = {
        "dataset": dataset_name,
        "chunking": {"max_tokens": max_tokens, "stride": stride},
        "splits": {},
    }

    if dataset_name == "msmarco":
        # Prepare MS MARCO splits
        for split_name, split_info in manifest["splits"].items():
            input_file = Path(split_info["file"])
            split_meta = prepare_msmarco_split(input_file, output_dir, chunker, split_name)
            prep_manifest["splits"][split_name] = split_meta

    elif dataset_name.startswith("beir_"):
        # Prepare BEIR corpus
        if "test" in manifest["splits"]:
            corpus_file = Path(manifest["splits"]["test"]["corpus_file"])
            corpus_meta = prepare_beir_corpus(corpus_file, output_dir, chunker)
            prep_manifest["splits"]["corpus"] = corpus_meta

    # Save preparation manifest
    prep_manifest_file = output_dir / "_manifest.json"
    with open(prep_manifest_file, "w") as f:
        json.dump(prep_manifest, f, indent=2)

    logger.info(f"Preparation complete for {dataset_name}")
    logger.info(f"Manifest saved to {prep_manifest_file}")

    return prep_manifest


def prepare_all_datasets() -> Dict[str, Dict]:
    """
    Prepare all datasets.

    Returns:
        Dictionary mapping dataset name to preparation metadata
    """
    all_manifests = {}

    # Prepare MS MARCO
    all_manifests["msmarco"] = prepare_dataset("msmarco")

    # Prepare BEIR datasets
    beir_datasets = ["beir_fiqa", "beir_scifact", "beir_trec_covid"]
    for dataset_name in beir_datasets:
        try:
            all_manifests[dataset_name] = prepare_dataset(dataset_name)
        except Exception as e:
            logger.error(f"Failed to prepare {dataset_name}: {e}")
            continue

    logger.info(f"\n=== Preparation Summary ===")
    for dataset_name, manifest in all_manifests.items():
        logger.info(f"{dataset_name}: {len(manifest['splits'])} splits prepared")

    return all_manifests


if __name__ == "__main__":
    from src.utils.logging import setup_logging

    setup_logging(log_level="INFO")

    # Prepare all datasets
    manifests = prepare_all_datasets()

