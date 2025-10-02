#!/usr/bin/env python3
"""Build FAISS HNSW index from trained KD model."""

import argparse
from pathlib import Path

from loguru import logger

from src.index.build_index import FAISSIndexBuilder
from src.models.student import StudentModel
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data-path", type=str, required=True, help="Path to parquet data")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for index")
    parser.add_argument("--max-docs", type=int, default=None, help="Max documents to index (for testing)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hnsw-m", type=int, default=32, help="HNSW M parameter")
    parser.add_argument("--hnsw-ef-construction", type=int, default=200, help="HNSW efConstruction")
    args = parser.parse_args()
    
    setup_logging(log_level="INFO")
    
    logger.info("="*80)
    logger.info("FAISS HNSW Index Building")
    logger.info("="*80)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = StudentModel(args.model_path, device=args.device)
    
    # Build index
    logger.info(f"Building index from {args.data_path}")
    builder = FAISSIndexBuilder(
        embedding_dim=384,
        index_type="HNSW",
        metric="cosine"
    )
    
    index = builder.build_from_parquet(
        model=model,
        parquet_path=Path(args.data_path),
        batch_size=args.batch_size,
        max_docs=args.max_docs,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction
    )
    
    # Save index
    output_dir = Path(args.output_dir)
    builder.save(output_dir)
    
    logger.info("\n" + "="*80)
    logger.info("INDEX BUILD COMPLETE!")
    logger.info("="*80)
    logger.info(f"Index saved to: {output_dir}")
    logger.info(f"Total vectors: {index.ntotal}")
    logger.info("="*80)


if __name__ == "__main__":
    main()

