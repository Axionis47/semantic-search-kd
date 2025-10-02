#!/usr/bin/env python3
"""
End-to-end KD training pipeline.

Runs complete pipeline:
1. Fetch data (if needed)
2. Prepare data (chunking)
3. Build BM25 index
4. Load teacher and student models
5. Run 3-stage curriculum training
6. Evaluate and save results
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader

from src.data.bm25 import BM25Index
from src.data.fetch import fetch_msmarco
from src.data.prepare import prepare_msmarco_split
from src.kd.losses import CombinedKDLoss
from src.kd.train import KDDataset, KDTrainer, collate_fn
from src.mining.miners import build_mining_curriculum
from src.models.student import StudentModel
from src.models.teacher import TeacherModel
from src.utils.chunk import TextChunker
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="KD Training Pipeline")

    # Data args
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="Data directory"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Max samples to use (for testing)",
    )

    # Model args
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="BAAI/bge-reranker-large",
        help="Teacher model name",
    )
    parser.add_argument(
        "--student-model",
        type=str,
        default="intfloat/e5-small-v2",
        help="Student model name",
    )

    # Training args
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--stage", type=int, default=1, help="Mining stage (1, 2, or 3)")

    # Output args
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./artifacts/models/kd_student",
        help="Output directory",
    )

    # System args
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    parser.add_argument(
        "--gcs-output-dir",
        type=str,
        default=None,
        help="GCS directory to upload trained model (e.g., gs://bucket/path)",
    )

    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()

    # Setup
    setup_logging(log_level=args.log_level)
    set_seed(args.seed)

    logger.info("=" * 80)
    logger.info("KD TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Max samples: {args.max_samples}")
    logger.info(f"Teacher: {args.teacher_model}")
    logger.info(f"Student: {args.student_model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Mining stage: {args.stage}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 80)

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw" / "msmarco"
    chunks_dir = data_dir / "chunks" / "msmarco"
    index_dir = Path("./artifacts/indexes/bm25_msmarco")

    # Step 1: Fetch data (if needed)
    logger.info("\n[1/7] Fetching data...")
    if not raw_dir.exists() or not list(raw_dir.glob("*.jsonl")):
        logger.info("Data not found, fetching MS MARCO...")
        fetch_msmarco(raw_dir, max_samples=args.max_samples)
    else:
        logger.info(f"✓ Data already exists: {raw_dir}")

    # Step 2: Prepare data (chunking)
    logger.info("\n[2/7] Preparing data...")
    chunks_dir.mkdir(parents=True, exist_ok=True)

    train_chunks_file = chunks_dir / "train.parquet"
    if not train_chunks_file.exists():
        logger.info("Chunking training data...")
        chunker = TextChunker(max_tokens=512, stride=80)
        prepare_msmarco_split(
            raw_dir / "train.jsonl",
            chunks_dir,
            chunker,
            "train",
        )
    else:
        logger.info(f"✓ Chunks already exist: {train_chunks_file}")

    # Step 3: Build BM25 index
    logger.info("\n[3/7] Building BM25 index...")
    index_dir.mkdir(parents=True, exist_ok=True)

    bm25_index = BM25Index(str(index_dir))
    if not (index_dir / "bm25.pkl").exists():
        logger.info("Building BM25 index from corpus...")
        bm25_index.build_from_parquet(
            str(train_chunks_file),
            str(index_dir),
            text_field="text",
            id_field="chunk_id",
        )
    else:
        logger.info(f"✓ BM25 index already exists: {index_dir}")
        bm25_index.load()

    # Step 4: Load models
    logger.info("\n[4/7] Loading models...")
    logger.info(f"Loading teacher: {args.teacher_model}")
    teacher = TeacherModel(model_name=args.teacher_model, device=args.device)

    logger.info(f"Loading student: {args.student_model}")
    student = StudentModel(model_name=args.student_model, device=args.device)

    # Step 5: Prepare training data
    logger.info("\n[5/7] Preparing training data...")

    # Load chunks (corpus)
    df = pd.read_parquet(train_chunks_file)
    logger.info(f"Loaded {len(df)} chunks")

    # Build corpus texts mapping
    corpus_texts = dict(zip(df["chunk_id"], df["text"]))

    # Load actual MS MARCO queries and qrels
    logger.info("Loading MS MARCO queries and qrels...")
    import json

    train_jsonl = data_dir / "raw" / "msmarco" / "train.jsonl"

    queries = []
    query_ids = []
    positives = []  # List of positive passage indices for each query

    with open(train_jsonl, "r") as f:
        for i, line in enumerate(f):
            if args.max_samples and i >= args.max_samples:
                break

            data = json.loads(line)
            query_text = data["query"]
            query_id = data["query_id"]

            # Get positive passages (is_selected == 1)
            is_selected = data["passages"]["is_selected"]
            passage_texts = data["passages"]["passage_text"]

            # Find positive passage indices
            positive_indices = [idx for idx, selected in enumerate(is_selected) if selected == 1]

            if len(positive_indices) == 0:
                continue  # Skip queries with no positive passages

            queries.append(query_text)
            query_ids.append(query_id)

            # Create chunk IDs for positive passages
            # Format: query_id_passage_idx
            pos_chunk_ids = [f"{query_id}_passage_{idx}" for idx in positive_indices]
            positives.append(pos_chunk_ids)

            # Add positive passages to corpus
            for idx in positive_indices:
                chunk_id = f"{query_id}_passage_{idx}"
                corpus_texts[chunk_id] = passage_texts[idx]

            # Add all passages to corpus for mining
            for idx, passage_text in enumerate(passage_texts):
                chunk_id = f"{query_id}_passage_{idx}"
                if chunk_id not in corpus_texts:
                    corpus_texts[chunk_id] = passage_text

    logger.info(f"Loaded {len(queries)} queries with {sum(len(p) for p in positives)} positive passages")

    # Step 6: Mine hard negatives
    logger.info(f"\n[6/7] Mining hard negatives (Stage {args.stage})...")

    negatives, teacher_scores = build_mining_curriculum(
        queries=queries,
        positives=positives,
        bm25_index_path=str(index_dir),
        teacher_model=teacher,
        student_model=student,
        corpus_texts=corpus_texts,
        stage=args.stage,
    )

    logger.info(f"Mined {sum(len(n) for n in negatives)} total negatives")

    # Create dataset
    dataset = KDDataset(
        queries=queries,
        positives=positives,
        negatives=negatives,
        teacher_scores=teacher_scores,
        corpus_texts=corpus_texts,
    )

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    logger.info(f"Created dataset: {len(dataset)} samples, {len(dataloader)} batches")

    # Step 7: Train
    logger.info("\n[7/7] Training...")

    # Setup loss and optimizer
    loss_fn = CombinedKDLoss(
        margin_mse_weight=0.6,
        listwise_kd_weight=0.2,
        contrastive_weight=0.2,
        temperature_start=4.0,
        temperature_end=2.0,
    )

    optimizer = torch.optim.AdamW(student.model.parameters(), lr=args.lr)

    # Create trainer
    trainer = KDTrainer(
        student_model=student,
        teacher_model=teacher,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=args.device,
        output_dir=args.output_dir,
    )

    # Train
    trainer.train(
        train_dataloader=dataloader,
        epochs=args.epochs,
        patience=args.patience,
    )

    logger.info("\n" + "=" * 80)
    logger.info("✓ TRAINING COMPLETE!")
    logger.info(f"Best model saved to: {args.output_dir}/best_model")
    logger.info("=" * 80)

    # Upload to GCS if specified
    if args.gcs_output_dir:
        logger.info(f"\nUploading model to GCS: {args.gcs_output_dir}")
        import subprocess

        try:
            subprocess.run(
                ["gsutil", "-m", "cp", "-r", args.output_dir, args.gcs_output_dir],
                check=True,
            )
            logger.info(f"✓ Model uploaded to {args.gcs_output_dir}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to upload to GCS: {e}")


if __name__ == "__main__":
    main()

