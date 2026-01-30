#!/usr/bin/env python3
"""Evaluate production KD model and compare with baseline."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.models.student import StudentModel
from src.models.teacher import TeacherModel
from src.utils.logging import setup_logging
from src.utils.metrics import ndcg_at_k, mrr_at_k


def load_test_data(data_path: Path, max_samples: int = 1000) -> tuple:
    """Load test data from parquet."""
    logger.info(f"Loading test data from {data_path}")
    
    df = pd.read_parquet(data_path)
    
    # Group by query
    queries = []
    documents_list = []
    relevance_list = []
    
    for query_id, group in df.groupby("query_id"):
        query = group.iloc[0]["query_text"]
        docs = group["text"].tolist()
        relevance = group["is_relevant"].tolist()
        
        queries.append(query)
        documents_list.append(docs)
        relevance_list.append(relevance)
        
        if len(queries) >= max_samples:
            break
    
    logger.info(f"Loaded {len(queries)} queries")
    return queries, documents_list, relevance_list


def evaluate_model(
    model,
    queries: List[str],
    documents_list: List[List[str]],
    relevance_list: List[List[int]],
    k_values: List[int] = [1, 5, 10],
    model_name: str = "Model"
) -> Dict[str, float]:
    """Evaluate a model on retrieval task."""
    logger.info(f"Evaluating {model_name} on {len(queries)} queries...")
    
    all_metrics = {f"ndcg@{k}": [] for k in k_values}
    all_metrics.update({f"mrr@{k}": [] for k in k_values})
    
    for i, (query, docs, labels) in enumerate(zip(queries, documents_list, relevance_list)):
        if i % 50 == 0:
            logger.info(f"Processing query {i+1}/{len(queries)}")
        
        # Encode query and documents
        if isinstance(model, StudentModel):
            query_emb = model.encode_queries([query])[0]
            doc_embs = model.encode_documents(docs)
        elif isinstance(model, TeacherModel):
            # Teacher is cross-encoder, score directly
            scores = []
            for doc in docs:
                score = model.predict_score(query, doc)
                scores.append(score)
            scores = np.array(scores)
            
            # Compute metrics for teacher
            for k in k_values:
                top_k_indices = np.argsort(scores)[::-1][:k]
                top_k_labels = [labels[idx] if idx < len(labels) else 0 for idx in top_k_indices]
                
                ndcg = ndcg_at_k(top_k_labels, k=k)
                mrr = mrr_at_k(top_k_labels, k=k)
                
                all_metrics[f"ndcg@{k}"].append(ndcg)
                all_metrics[f"mrr@{k}"].append(mrr)
            continue
        else:
            # SentenceTransformer
            query_emb = model.encode([query], convert_to_numpy=True)[0]
            doc_embs = model.encode(docs, convert_to_numpy=True)
        
        # Compute similarities
        similarities = np.dot(doc_embs, query_emb)
        
        # Compute metrics
        for k in k_values:
            top_k_indices = np.argsort(similarities)[::-1][:k]
            top_k_labels = [labels[idx] if idx < len(labels) else 0 for idx in top_k_indices]
            
            ndcg = ndcg_at_k(top_k_labels, k=k)
            mrr = mrr_at_k(top_k_labels, k=k)
            
            all_metrics[f"ndcg@{k}"].append(ndcg)
            all_metrics[f"mrr@{k}"].append(mrr)
    
    # Average metrics
    final_metrics = {}
    for key, values in all_metrics.items():
        final_metrics[key] = float(np.mean(values))
    
    logger.info(f"{model_name} Results:")
    for key, value in sorted(final_metrics.items()):
        logger.info(f"  {key}: {value:.4f}")
    
    return final_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kd-model-path", type=str, required=True, help="Path to KD-trained model")
    parser.add_argument("--data-path", type=str, required=True, help="Path to test data (parquet)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=1000, help="Max queries to evaluate")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip-teacher", action="store_true", help="Skip teacher evaluation (slow)")
    args = parser.parse_args()
    
    setup_logging(log_level="INFO")

    # Validate arguments
    from scripts._validate_args import validate_path_exists, validate_positive_int, validate_device
    validate_path_exists(args.kd_model_path, "--kd-model-path")
    validate_path_exists(args.data_path, "--data-path")
    validate_positive_int(args.max_samples, "--max-samples")
    validate_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    queries, documents_list, relevance_list = load_test_data(
        Path(args.data_path),
        max_samples=args.max_samples
    )
    
    results = {}
    
    # 1. Evaluate Vanilla Student (baseline)
    logger.info("\n" + "="*80)
    logger.info("Evaluating Vanilla Student (Baseline)")
    logger.info("="*80)
    vanilla_model = SentenceTransformer("intfloat/e5-small-v2", device=args.device)
    vanilla_metrics = evaluate_model(
        vanilla_model,
        queries,
        documents_list,
        relevance_list,
        model_name="Vanilla Student"
    )
    results["vanilla_student"] = vanilla_metrics
    
    # 2. Evaluate KD Student
    logger.info("\n" + "="*80)
    logger.info("Evaluating KD-Trained Student")
    logger.info("="*80)
    kd_model = StudentModel(args.kd_model_path, device=args.device)
    kd_metrics = evaluate_model(
        kd_model,
        queries,
        documents_list,
        relevance_list,
        model_name="KD Student"
    )
    results["kd_student"] = kd_metrics
    
    # 3. Evaluate Teacher (optional, very slow)
    if not args.skip_teacher:
        logger.info("\n" + "="*80)
        logger.info("Evaluating Teacher (Cross-Encoder)")
        logger.info("="*80)
        teacher_model = TeacherModel(device=args.device)
        teacher_metrics = evaluate_model(
            teacher_model,
            queries,
            documents_list,
            relevance_list,
            model_name="Teacher"
        )
        results["teacher"] = teacher_metrics
    
    # Compute improvements
    logger.info("\n" + "="*80)
    logger.info("IMPROVEMENT ANALYSIS")
    logger.info("="*80)
    
    improvements = {}
    for metric in ["ndcg@1", "ndcg@5", "ndcg@10", "mrr@1", "mrr@5", "mrr@10"]:
        vanilla_val = vanilla_metrics[metric]
        kd_val = kd_metrics[metric]
        improvement = ((kd_val - vanilla_val) / vanilla_val) * 100 if vanilla_val > 0 else 0
        improvements[metric] = improvement
        
        logger.info(f"{metric}:")
        logger.info(f"  Vanilla: {vanilla_val:.4f}")
        logger.info(f"  KD:      {kd_val:.4f}")
        logger.info(f"  Improvement: {improvement:+.2f}%")
    
    results["improvements"] = improvements
    
    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✓ Results saved to {results_path}")
    
    # Create summary
    summary_path = output_dir / "EVALUATION_SUMMARY.md"
    with open(summary_path, "w") as f:
        f.write("# Production Model Evaluation Results\n\n")
        f.write(f"**Test Queries:** {len(queries)}\n\n")
        
        f.write("## Metrics Comparison\n\n")
        f.write("| Metric | Vanilla Student | KD Student | Improvement |\n")
        f.write("|--------|----------------|------------|-------------|\n")
        for metric in ["ndcg@1", "ndcg@5", "ndcg@10", "mrr@1", "mrr@5", "mrr@10"]:
            vanilla_val = vanilla_metrics[metric]
            kd_val = kd_metrics[metric]
            improvement = improvements[metric]
            f.write(f"| {metric} | {vanilla_val:.4f} | {kd_val:.4f} | {improvement:+.2f}% |\n")
        
        f.write("\n## Key Findings\n\n")
        avg_improvement = np.mean([improvements[f"ndcg@{k}"] for k in [1, 5, 10]])
        f.write(f"- **Average nDCG Improvement:** {avg_improvement:+.2f}%\n")
        f.write(f"- **Best Improvement:** {max(improvements.values()):+.2f}% ({max(improvements, key=improvements.get)})\n")
        f.write(f"- **Model Size:** 127 MB (intfloat/e5-small-v2 with KD)\n")
        f.write(f"- **Training:** 3 epochs, Stage 2 mining (BM25 → Teacher)\n")
    
    logger.info(f"✓ Summary saved to {summary_path}")
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE!")
    logger.info("="*80)


if __name__ == "__main__":
    main()

