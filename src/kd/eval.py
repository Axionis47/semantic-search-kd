"""Evaluation module for Knowledge Distillation."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.models.student import StudentModel
from src.models.teacher import TeacherModel
from src.utils.metrics import (
    expected_calibration_error,
    kendall_tau,
    mrr_at_k,
    ndcg_at_k,
)


class KDEvaluator:
    """Evaluator for Knowledge Distillation models."""

    def __init__(
        self,
        student: StudentModel,
        teacher: Optional[TeacherModel] = None,
        vanilla_student: Optional[StudentModel] = None,
    ):
        """
        Initialize evaluator.

        Args:
            student: Trained student model (after KD)
            teacher: Teacher model (for comparison)
            vanilla_student: Untrained student model (for comparison)
        """
        self.student = student
        self.teacher = teacher
        self.vanilla_student = vanilla_student

    def evaluate_retrieval(
        self,
        queries: List[str],
        corpus: List[str],
        relevance_labels: List[List[int]],
        k_values: List[int] = [1, 5, 10, 20],
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance.

        Args:
            queries: List of query strings
            corpus: List of document strings
            relevance_labels: List of relevance labels for each query
            k_values: List of k values for nDCG@k, MRR@k
            batch_size: Batch size for encoding

        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating retrieval on {len(queries)} queries")

        # Encode corpus once
        logger.info("Encoding corpus...")
        corpus_embs = self.student.encode_documents(
            corpus, batch_size=batch_size, show_progress=True
        )

        # Evaluate each query
        all_scores = []
        for query in tqdm(queries, desc="Evaluating queries"):
            query_emb = self.student.encode_queries([query])
            scores = self.student.compute_similarity(query_emb, corpus_embs)[0]
            all_scores.append(scores)

        # Compute metrics
        metrics = {}
        for k in k_values:
            ndcg_scores = []
            mrr_scores = []

            for scores, labels in zip(all_scores, relevance_labels):
                # Get top-k indices
                top_k_indices = np.argsort(scores)[::-1][:k]
                top_k_labels = [labels[i] if i < len(labels) else 0 for i in top_k_indices]

                # Compute nDCG@k
                ndcg = ndcg_at_k(top_k_labels, k=k)
                ndcg_scores.append(ndcg)

                # Compute MRR@k
                mrr = mrr_at_k(top_k_labels, k=k)
                mrr_scores.append(mrr)

            metrics[f"ndcg@{k}"] = np.mean(ndcg_scores)
            metrics[f"mrr@{k}"] = np.mean(mrr_scores)

        logger.info(f"Retrieval metrics: {metrics}")
        return metrics

    def evaluate_ranking_quality(
        self,
        queries: List[str],
        doc_lists: List[List[str]],
        teacher_scores: Optional[List[List[float]]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate ranking quality (correlation with teacher).

        Args:
            queries: List of query strings
            doc_lists: List of document lists for each query
            teacher_scores: Pre-computed teacher scores (optional)

        Returns:
            Dictionary of metrics (Kendall-τ, ECE)
        """
        logger.info(f"Evaluating ranking quality on {len(queries)} queries")

        student_scores_all = []
        teacher_scores_all = []

        for i, (query, docs) in enumerate(
            tqdm(zip(queries, doc_lists), total=len(queries), desc="Scoring")
        ):
            # Get student scores
            query_emb = self.student.encode_queries([query])
            doc_embs = self.student.encode_documents(docs)
            student_scores = self.student.compute_similarity(query_emb, doc_embs)[0]
            student_scores_all.append(student_scores)

            # Get teacher scores
            if teacher_scores is not None:
                teacher_scores_all.append(teacher_scores[i])
            elif self.teacher is not None:
                pairs = [[query, doc] for doc in docs]
                t_scores = self.teacher.score(pairs)
                teacher_scores_all.append(t_scores)

        metrics = {}

        # Compute Kendall-τ (ranking correlation)
        if teacher_scores_all:
            kendall_scores = []
            for s_scores, t_scores in zip(student_scores_all, teacher_scores_all):
                # Convert scores to rankings (indices)
                s_ranking = list(np.argsort(s_scores)[::-1])
                t_ranking = list(np.argsort(t_scores)[::-1])
                tau = kendall_tau(s_ranking, t_ranking)
                kendall_scores.append(tau)
            metrics["kendall_tau"] = np.mean(kendall_scores)

        # Compute ECE (calibration)
        if teacher_scores_all:
            # Flatten all scores
            student_flat = np.concatenate(student_scores_all)
            teacher_flat = np.concatenate(teacher_scores_all)

            # Normalize to [0, 1]
            student_norm = (student_flat - student_flat.min()) / (
                student_flat.max() - student_flat.min() + 1e-8
            )
            teacher_norm = (teacher_flat - teacher_flat.min()) / (
                teacher_flat.max() - teacher_flat.min() + 1e-8
            )

            # Convert to binary accuracy (1 if student agrees with teacher, 0 otherwise)
            accuracies = (student_norm > 0.5).astype(float) == (teacher_norm > 0.5).astype(float)
            ece = expected_calibration_error(student_norm, accuracies.astype(float), n_bins=10)
            metrics["ece"] = ece

        logger.info(f"Ranking quality metrics: {metrics}")
        return metrics

    def compare_models(
        self,
        queries: List[str],
        doc_lists: List[List[str]],
        relevance_labels: List[List[int]],
        k_values: List[int] = [1, 5, 10, 20],
    ) -> pd.DataFrame:
        """
        Compare student (KD), vanilla student, and teacher.

        Args:
            queries: List of query strings
            doc_lists: List of document lists for each query
            relevance_labels: List of relevance labels for each query
            k_values: List of k values for metrics

        Returns:
            DataFrame with comparison results
        """
        logger.info("Comparing models...")

        results = []

        # Evaluate student (KD)
        logger.info("Evaluating student (KD)...")
        student_metrics = self._evaluate_model(
            self.student, queries, doc_lists, relevance_labels, k_values
        )
        student_metrics["model"] = "Student (KD)"
        results.append(student_metrics)

        # Evaluate vanilla student
        if self.vanilla_student is not None:
            logger.info("Evaluating vanilla student...")
            vanilla_metrics = self._evaluate_model(
                self.vanilla_student, queries, doc_lists, relevance_labels, k_values
            )
            vanilla_metrics["model"] = "Student (Vanilla)"
            results.append(vanilla_metrics)

        # Evaluate teacher
        if self.teacher is not None:
            logger.info("Evaluating teacher...")
            teacher_metrics = self._evaluate_teacher(
                queries, doc_lists, relevance_labels, k_values
            )
            teacher_metrics["model"] = "Teacher"
            results.append(teacher_metrics)

        df = pd.DataFrame(results)
        logger.info(f"\n{df.to_string()}")
        return df

    def _evaluate_model(
        self,
        model: StudentModel,
        queries: List[str],
        doc_lists: List[List[str]],
        relevance_labels: List[List[int]],
        k_values: List[int],
    ) -> Dict[str, float]:
        """Evaluate a single model."""
        metrics = {}

        for k in k_values:
            ndcg_scores = []
            mrr_scores = []

            for query, docs, labels in zip(queries, doc_lists, relevance_labels):
                # Get scores
                query_emb = model.encode_queries([query])
                doc_embs = model.encode_documents(docs)
                scores = model.compute_similarity(query_emb, doc_embs)[0]

                # Get top-k
                top_k_indices = np.argsort(scores)[::-1][:k]
                top_k_labels = [labels[i] if i < len(labels) else 0 for i in top_k_indices]

                # Compute metrics
                ndcg = ndcg_at_k(top_k_labels, k=k)
                mrr = mrr_at_k(top_k_labels, k=k)

                ndcg_scores.append(ndcg)
                mrr_scores.append(mrr)

            metrics[f"ndcg@{k}"] = np.mean(ndcg_scores)
            metrics[f"mrr@{k}"] = np.mean(mrr_scores)

        return metrics

    def _evaluate_teacher(
        self,
        queries: List[str],
        doc_lists: List[List[str]],
        relevance_labels: List[List[int]],
        k_values: List[int],
    ) -> Dict[str, float]:
        """Evaluate teacher model."""
        metrics = {}

        for k in k_values:
            ndcg_scores = []
            mrr_scores = []

            for query, docs, labels in zip(queries, doc_lists, relevance_labels):
                # Get scores
                pairs = [[query, doc] for doc in docs]
                scores = self.teacher.score(pairs)

                # Get top-k
                top_k_indices = np.argsort(scores)[::-1][:k]
                top_k_labels = [labels[i] if i < len(labels) else 0 for i in top_k_indices]

                # Compute metrics
                ndcg = ndcg_at_k(top_k_labels, k=k)
                mrr = mrr_at_k(top_k_labels, k=k)

                ndcg_scores.append(ndcg)
                mrr_scores.append(mrr)

            metrics[f"ndcg@{k}"] = np.mean(ndcg_scores)
            metrics[f"mrr@{k}"] = np.mean(mrr_scores)

        return metrics

    def generate_report(
        self,
        metrics: Dict[str, float],
        output_path: Path,
        training_config: Optional[Dict] = None,
    ):
        """
        Generate evaluation report.

        Args:
            metrics: Dictionary of metrics
            output_path: Path to save report
            training_config: Training configuration (optional)
        """
        logger.info(f"Generating report: {output_path}")

        report = []
        report.append("# Knowledge Distillation Evaluation Report\n")
        report.append(f"**Generated:** {pd.Timestamp.now()}\n\n")

        if training_config:
            report.append("## Training Configuration\n")
            for key, value in training_config.items():
                report.append(f"- **{key}:** {value}\n")
            report.append("\n")

        report.append("## Metrics\n\n")
        for key, value in metrics.items():
            report.append(f"- **{key}:** {value:.4f}\n")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("".join(report))
        logger.info(f"Report saved: {output_path}")

