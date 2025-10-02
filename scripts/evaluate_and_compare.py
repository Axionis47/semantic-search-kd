#!/usr/bin/env python3
"""Comprehensive evaluation and comparison: Vanilla Student vs KD Student vs Teacher."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from loguru import logger

from src.data.prepare import load_prepared_data
from src.kd.eval import KDEvaluator
from src.models.student import StudentModel
from src.models.teacher import TeacherModel
from src.utils.logging import setup_logging


def load_test_data(data_dir: Path, max_samples: int = 1000) -> tuple:
    """Load test data for evaluation."""
    logger.info(f"Loading test data from {data_dir}")
    
    test_path = data_dir / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    
    df = pd.read_parquet(test_path)
    
    if max_samples:
        df = df.head(max_samples)
    
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
    
    logger.info(f"Loaded {len(queries)} queries with {len(df)} documents")
    return queries, documents_list, relevance_list


def evaluate_model(
    model,
    model_name: str,
    queries: List[str],
    documents_list: List[List[str]],
    relevance_list: List[List[int]],
    is_teacher: bool = False,
) -> Dict:
    """Evaluate a single model."""
    logger.info(f"Evaluating {model_name}...")
    
    evaluator = KDEvaluator(
        student=None if is_teacher else model,
        teacher=model if is_teacher else None,
    )
    
    if is_teacher:
        # Teacher uses cross-encoder scoring
        results = evaluator.evaluate_retrieval(
            queries=queries,
            documents_list=documents_list,
            relevance_list=relevance_list,
            k_values=[1, 5, 10, 20],
            use_teacher=True,
        )
    else:
        # Student uses bi-encoder retrieval
        results = evaluator.evaluate_retrieval(
            queries=queries,
            documents_list=documents_list,
            relevance_list=relevance_list,
            k_values=[1, 5, 10, 20],
            use_teacher=False,
        )
    
    logger.info(f"✓ {model_name} evaluation complete")
    return results


def generate_comparison_report(
    vanilla_results: Dict,
    kd_results: Dict,
    teacher_results: Dict,
    output_path: Path,
):
    """Generate comprehensive comparison report."""
    logger.info("Generating comparison report...")
    
    report = []
    report.append("# Knowledge Distillation Evaluation Report")
    report.append("")
    report.append(f"**Generated:** {pd.Timestamp.now()}")
    report.append("")
    report.append("---")
    report.append("")
    
    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    
    # Compare nDCG@10
    vanilla_ndcg10 = vanilla_results.get("ndcg@10", 0)
    kd_ndcg10 = kd_results.get("ndcg@10", 0)
    teacher_ndcg10 = teacher_results.get("ndcg@10", 0)
    
    improvement = ((kd_ndcg10 - vanilla_ndcg10) / vanilla_ndcg10) * 100
    teacher_ratio = (kd_ndcg10 / teacher_ndcg10) * 100
    
    report.append(f"- **Vanilla Student nDCG@10:** {vanilla_ndcg10:.4f}")
    report.append(f"- **KD Student nDCG@10:** {kd_ndcg10:.4f}")
    report.append(f"- **Teacher nDCG@10:** {teacher_ndcg10:.4f}")
    report.append(f"- **Improvement:** {improvement:+.2f}%")
    report.append(f"- **Teacher Performance Ratio:** {teacher_ratio:.2f}%")
    report.append("")
    
    # Acceptance Gates
    report.append("## Acceptance Gates")
    report.append("")
    
    gate1_pass = kd_ndcg10 >= 0.95 * teacher_ndcg10
    gate1_status = "✅ PASS" if gate1_pass else "❌ FAIL"
    report.append(f"1. **nDCG@10(KD) ≥ 95% of Teacher:** {gate1_status}")
    report.append(f"   - Required: {0.95 * teacher_ndcg10:.4f}")
    report.append(f"   - Actual: {kd_ndcg10:.4f}")
    report.append("")
    
    # Detailed Metrics
    report.append("## Detailed Metrics")
    report.append("")
    
    # Create comparison table
    report.append("### Retrieval Performance")
    report.append("")
    report.append("| Metric | Vanilla Student | KD Student | Teacher | KD Improvement |")
    report.append("|--------|----------------|------------|---------|----------------|")
    
    for metric in ["ndcg@1", "ndcg@5", "ndcg@10", "ndcg@20", "mrr@10"]:
        vanilla_val = vanilla_results.get(metric, 0)
        kd_val = kd_results.get(metric, 0)
        teacher_val = teacher_results.get(metric, 0)
        improvement = ((kd_val - vanilla_val) / vanilla_val * 100) if vanilla_val > 0 else 0
        
        report.append(
            f"| {metric} | {vanilla_val:.4f} | {kd_val:.4f} | {teacher_val:.4f} | {improvement:+.2f}% |"
        )
    
    report.append("")
    
    # Model Comparison
    report.append("## Model Comparison")
    report.append("")
    report.append("### Model Characteristics")
    report.append("")
    report.append("| Model | Type | Parameters | Embedding Dim | Inference Speed |")
    report.append("|-------|------|------------|---------------|-----------------|")
    report.append("| Vanilla Student | Bi-encoder | 33M | 384 | Fast |")
    report.append("| KD Student | Bi-encoder | 33M | 384 | Fast |")
    report.append("| Teacher | Cross-encoder | 560M | N/A | Slow |")
    report.append("")
    
    # Conclusions
    report.append("## Conclusions")
    report.append("")
    
    if gate1_pass:
        report.append("✅ **Knowledge distillation was successful!**")
        report.append("")
        report.append("The KD student model achieves ≥95% of teacher performance while being:")
        report.append("- **17x smaller** (33M vs 560M parameters)")
        report.append("- **10-50x faster** (bi-encoder vs cross-encoder)")
        report.append("- **Suitable for production deployment**")
    else:
        report.append("⚠️ **Knowledge distillation needs improvement.**")
        report.append("")
        report.append("Recommendations:")
        report.append("- Increase training epochs")
        report.append("- Tune loss weights")
        report.append("- Use more training data")
        report.append("- Adjust temperature annealing schedule")
    
    report.append("")
    report.append("---")
    report.append("")
    report.append("**End of Report**")
    
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report))
    logger.info(f"✓ Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare models")
    parser.add_argument(
        "--vanilla-model",
        type=str,
        default="intfloat/e5-small-v2",
        help="Path to vanilla student model",
    )
    parser.add_argument(
        "--kd-model",
        type=str,
        required=True,
        help="Path to KD-trained student model",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="BAAI/bge-reranker-large",
        help="Path to teacher model",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/chunks/msmarco"),
        help="Directory containing test data",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum number of test samples",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/evaluation"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu/cuda)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    logger.info("=" * 80)
    logger.info("Knowledge Distillation Evaluation & Comparison")
    logger.info("=" * 80)
    logger.info("")
    
    # Load test data
    queries, documents_list, relevance_list = load_test_data(
        args.data_dir, args.max_samples
    )
    
    # Load models
    logger.info("Loading models...")
    vanilla_student = StudentModel(args.vanilla_model, device=args.device)
    kd_student = StudentModel(args.kd_model, device=args.device)
    teacher = TeacherModel(args.teacher_model, device=args.device)
    logger.info("✓ All models loaded")
    logger.info("")
    
    # Evaluate models
    vanilla_results = evaluate_model(
        vanilla_student,
        "Vanilla Student",
        queries,
        documents_list,
        relevance_list,
        is_teacher=False,
    )
    
    kd_results = evaluate_model(
        kd_student,
        "KD Student",
        queries,
        documents_list,
        relevance_list,
        is_teacher=False,
    )
    
    teacher_results = evaluate_model(
        teacher,
        "Teacher",
        queries,
        documents_list,
        relevance_list,
        is_teacher=True,
    )
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = args.output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "vanilla_student": vanilla_results,
                "kd_student": kd_results,
                "teacher": teacher_results,
            },
            f,
            indent=2,
        )
    logger.info(f"✓ Results saved to {results_path}")
    
    # Generate report
    report_path = args.output_dir / "KD_REPORT.md"
    generate_comparison_report(
        vanilla_results, kd_results, teacher_results, report_path
    )
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Evaluation Complete!")
    logger.info("=" * 80)
    logger.info(f"Report: {report_path}")
    logger.info(f"Results: {results_path}")


if __name__ == "__main__":
    main()

