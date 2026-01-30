#!/usr/bin/env python3
"""Simple evaluation script for before/after KD comparison."""

import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger

from src.models.student import StudentModel
from src.utils.logging import setup_logging
from src.utils.metrics import ndcg_at_k, mrr_at_k


def evaluate_model(model: StudentModel, queries: list, corpus: list, relevance_labels: list, k_values: list = [1, 5, 10]) -> dict:
    """Evaluate a model on retrieval task."""
    logger.info(f"Evaluating model on {len(queries)} queries...")
    
    # Encode queries and corpus
    query_embs = model.encode_queries(queries, show_progress=True)
    corpus_embs = model.encode_documents(corpus, show_progress=True)
    
    # Compute similarities
    similarities = np.matmul(query_embs, corpus_embs.T)
    
    # Compute metrics
    metrics = {}
    for k in k_values:
        ndcg_scores = []
        mrr_scores = []
        
        for i, (query_sims, labels) in enumerate(zip(similarities, relevance_labels)):
            # Get top-k indices
            top_k_indices = np.argsort(query_sims)[::-1][:k]
            top_k_labels = [labels[idx] if idx < len(labels) else 0 for idx in top_k_indices]
            
            # Compute metrics
            ndcg = ndcg_at_k(top_k_labels, k=k)
            mrr = mrr_at_k(top_k_labels, k=k)
            
            ndcg_scores.append(ndcg)
            mrr_scores.append(mrr)
        
        metrics[f"ndcg@{k}"] = float(np.mean(ndcg_scores))
        metrics[f"mrr@{k}"] = float(np.mean(mrr_scores))
    
    logger.info(f"Evaluation complete: nDCG@10={metrics.get('ndcg@10', 0):.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    setup_logging(log_level="INFO")

    # Validate arguments
    from scripts._validate_args import validate_path_exists, validate_positive_int, validate_device
    validate_path_exists(args.model_path, "--model-path")
    validate_path_exists(args.data_path, "--data-path")
    validate_positive_int(args.max_samples, "--max-samples")
    validate_device(args.device)

    # Load data from MS MARCO JSONL
    logger.info(f"Loading data from {args.data_path}")

    queries = []
    corpus = []
    corpus_dict = {}  # text -> index mapping
    # First pass: collect all unique passages into corpus, record per-query relevance
    query_passage_relevance = []  # list of dict{passage_text -> relevance_score}

    with open(args.data_path, "r") as f:
        for i, line in enumerate(f):
            if i >= args.max_samples:
                break

            data = json.loads(line)
            query_text = data["query"]
            is_selected = data["passages"]["is_selected"]
            passage_texts = data["passages"]["passage_text"]

            # Skip if no positive passages
            if sum(is_selected) == 0:
                continue

            queries.append(query_text)

            # Record relevance per passage for this query
            q_relevance = {}
            for passage_text, selected in zip(passage_texts, is_selected):
                # Add to corpus if not already there
                if passage_text not in corpus_dict:
                    corpus_dict[passage_text] = len(corpus)
                    corpus.append(passage_text)
                # Record relevance (last wins if duplicate passages)
                q_relevance[passage_text] = selected

            query_passage_relevance.append(q_relevance)

    # Second pass: build label arrays now that corpus is fully built
    relevance_labels = []
    corpus_size = len(corpus)
    for q_relevance in query_passage_relevance:
        query_labels = [0] * corpus_size
        for passage_text, score in q_relevance.items():
            query_labels[corpus_dict[passage_text]] = score
        relevance_labels.append(query_labels)

    logger.info(f"Loaded {len(queries)} queries, {len(corpus)} documents")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = StudentModel(args.model_path, device=args.device)
    
    # Evaluate
    results = evaluate_model(model, queries, corpus, relevance_labels)
    
    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"nDCG@10: {results['ndcg@10']:.4f}")
    logger.info(f"MRR@10: {results['mrr@10']:.4f}")


if __name__ == "__main__":
    main()

