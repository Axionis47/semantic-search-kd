"""
Mining curriculum for hard negative sampling.

Implements 3-stage mining:
1. BM25 mining (lexical)
2. Teacher mining (semantic, cross-encoder)
3. ANCE mining (adversarial, student-based)
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.data.bm25 import BM25Index
from src.models.student import StudentModel
from src.models.teacher import TeacherModel


class BM25Miner:
    """
    Stage 1: BM25-based hard negative mining.

    Mines lexically similar but semantically irrelevant documents.
    """

    def __init__(self, index_path: str):
        """
        Initialize BM25 miner.

        Args:
            index_path: Path to BM25 index directory
        """
        self.index = BM25Index(index_path)
        self.index.load()
        logger.info(f"BM25Miner initialized with {len(self.index.doc_ids)} documents")

    def mine(
        self,
        queries: List[str],
        positives: List[List[str]],
        top_k: int = 100,
        exclude_positives: bool = True,
    ) -> List[List[str]]:
        """
        Mine hard negatives using BM25.

        Args:
            queries: List of query strings
            positives: List of positive doc IDs for each query
            top_k: Number of candidates to retrieve
            exclude_positives: Whether to exclude positive docs from results

        Returns:
            List of hard negative doc IDs for each query
        """
        logger.info(f"Mining hard negatives with BM25 for {len(queries)} queries")

        all_negatives = []
        for query, pos_ids in tqdm(
            zip(queries, positives), total=len(queries), desc="BM25 mining"
        ):
            # Retrieve top-k candidates
            results = self.index.search(query, top_k=top_k)

            # Filter out positives
            if exclude_positives:
                pos_set = set(pos_ids)
                negatives = [doc_id for doc_id, _ in results if doc_id not in pos_set]
            else:
                negatives = [doc_id for doc_id, _ in results]

            all_negatives.append(negatives)

        logger.info(f"BM25 mining complete: avg {sum(len(n) for n in all_negatives) / len(all_negatives):.1f} negatives/query")
        return all_negatives


class TeacherMiner:
    """
    Stage 2: Teacher-based hard negative mining.

    Uses cross-encoder to find semantically challenging negatives.
    """

    def __init__(
        self,
        teacher_model: TeacherModel,
        confidence_threshold: float = 0.6,
    ):
        """
        Initialize teacher miner.

        Args:
            teacher_model: Loaded teacher model
            confidence_threshold: Minimum confidence for hard negatives
        """
        self.teacher = teacher_model
        self.confidence_threshold = confidence_threshold
        logger.info(f"TeacherMiner initialized (threshold={confidence_threshold})")

    def mine(
        self,
        queries: List[str],
        candidates: List[List[str]],
        candidate_texts: Dict[str, str],
        top_k: int = 10,
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Mine hard negatives using teacher model.

        Args:
            queries: List of query strings
            candidates: List of candidate doc IDs for each query (from BM25)
            candidate_texts: Mapping from doc ID to text
            top_k: Number of hard negatives to keep

        Returns:
            Tuple of (hard_negative_ids, teacher_scores)
        """
        logger.info(f"Mining hard negatives with teacher for {len(queries)} queries")

        all_negatives = []
        all_scores = []

        for query, cand_ids in tqdm(
            zip(queries, candidates), total=len(queries), desc="Teacher mining"
        ):
            # Get candidate texts
            cand_texts = [candidate_texts.get(doc_id, "") for doc_id in cand_ids]

            # Score with teacher
            scores = self.teacher.score(
                [(query, text) for text in cand_texts], batch_size=32
            )

            # Sort by score (descending) and take top-k
            sorted_pairs = sorted(
                zip(cand_ids, scores), key=lambda x: x[1], reverse=True
            )

            # Filter by confidence threshold
            hard_negatives = []
            hard_scores = []
            for doc_id, score in sorted_pairs[:top_k]:
                confidence = self.teacher.get_confidence(score)
                if confidence >= self.confidence_threshold:
                    hard_negatives.append(doc_id)
                    hard_scores.append(score)

            all_negatives.append(hard_negatives)
            all_scores.append(hard_scores)

        avg_negatives = sum(len(n) for n in all_negatives) / len(all_negatives)
        logger.info(f"Teacher mining complete: avg {avg_negatives:.1f} hard negatives/query")
        return all_negatives, all_scores


class ANCEMiner:
    """
    Stage 3: ANCE (Adversarial Negative Contrastive Estimation) mining.

    Uses student model to find adversarial negatives that fool the student.
    """

    def __init__(
        self,
        student_model: StudentModel,
        margin: float = 0.1,
    ):
        """
        Initialize ANCE miner.

        Args:
            student_model: Current student model
            margin: Minimum margin between positive and negative scores
        """
        self.student = student_model
        self.margin = margin
        logger.info(f"ANCEMiner initialized (margin={margin})")

    def mine(
        self,
        queries: List[str],
        positives: List[List[str]],
        candidates: List[List[str]],
        candidate_texts: Dict[str, str],
        positive_texts: Dict[str, str],
        top_k: int = 5,
    ) -> List[List[str]]:
        """
        Mine adversarial negatives using student model.

        Args:
            queries: List of query strings
            positives: List of positive doc IDs for each query
            candidates: List of candidate doc IDs for each query
            candidate_texts: Mapping from doc ID to text
            positive_texts: Mapping from positive doc ID to text
            top_k: Number of adversarial negatives to keep

        Returns:
            List of adversarial negative doc IDs for each query
        """
        logger.info(f"Mining adversarial negatives with student for {len(queries)} queries")

        all_negatives = []

        for query, pos_ids, cand_ids in tqdm(
            zip(queries, positives, candidates),
            total=len(queries),
            desc="ANCE mining",
        ):
            # Encode query
            query_emb = self.student.encode_queries([query])[0]

            # Encode positives
            pos_texts = [positive_texts.get(doc_id, "") for doc_id in pos_ids]
            pos_embs = self.student.encode_documents(pos_texts)

            # Encode candidates
            cand_texts = [candidate_texts.get(doc_id, "") for doc_id in cand_ids]
            cand_embs = self.student.encode_documents(cand_texts)

            # Compute similarities
            pos_scores = self.student.compute_similarity(
                query_emb.reshape(1, -1), pos_embs
            )[0]
            cand_scores = self.student.compute_similarity(
                query_emb.reshape(1, -1), cand_embs
            )[0]

            # Find adversarial negatives: high student score but should be negative
            # (i.e., student is confused)
            max_pos_score = pos_scores.max() if len(pos_scores) > 0 else 0.0

            # Adversarial negatives are those with score close to positive
            adversarial = []
            for doc_id, score in zip(cand_ids, cand_scores):
                if score >= max_pos_score - self.margin:
                    adversarial.append((doc_id, score))

            # Sort by score (descending) and take top-k
            adversarial.sort(key=lambda x: x[1], reverse=True)
            hard_negatives = [doc_id for doc_id, _ in adversarial[:top_k]]

            all_negatives.append(hard_negatives)

        avg_negatives = sum(len(n) for n in all_negatives) / len(all_negatives)
        logger.info(f"ANCE mining complete: avg {avg_negatives:.1f} adversarial negatives/query")
        return all_negatives


def build_mining_curriculum(
    queries: List[str],
    positives: List[List[str]],
    bm25_index_path: str,
    teacher_model: TeacherModel,
    student_model: StudentModel,
    corpus_texts: Dict[str, str],
    stage: int = 1,
) -> Tuple[List[List[str]], List[List[float]]]:
    """
    Build mining curriculum for a given stage.

    Args:
        queries: List of query strings
        positives: List of positive doc IDs for each query
        bm25_index_path: Path to BM25 index
        teacher_model: Teacher model
        student_model: Student model
        corpus_texts: Mapping from doc ID to text
        stage: Mining stage (1=BM25, 2=Teacher, 3=ANCE)

    Returns:
        Tuple of (hard_negatives, teacher_scores)
    """
    logger.info(f"Building mining curriculum: Stage {stage}")

    if stage == 1:
        # Stage 1: BM25 mining
        miner = BM25Miner(bm25_index_path)
        negatives = miner.mine(queries, positives, top_k=100)
        scores = [[0.0] * len(n) for n in negatives]  # Placeholder scores
        return negatives, scores

    elif stage == 2:
        # Stage 2: BM25 + Teacher mining
        bm25_miner = BM25Miner(bm25_index_path)
        bm25_candidates = bm25_miner.mine(queries, positives, top_k=100)

        teacher_miner = TeacherMiner(teacher_model, confidence_threshold=0.6)
        negatives, scores = teacher_miner.mine(
            queries, bm25_candidates, corpus_texts, top_k=10
        )
        return negatives, scores

    elif stage == 3:
        # Stage 3: BM25 + Teacher + ANCE mining
        bm25_miner = BM25Miner(bm25_index_path)
        bm25_candidates = bm25_miner.mine(queries, positives, top_k=100)

        teacher_miner = TeacherMiner(teacher_model, confidence_threshold=0.6)
        teacher_negatives, teacher_scores = teacher_miner.mine(
            queries, bm25_candidates, corpus_texts, top_k=20
        )

        ance_miner = ANCEMiner(student_model, margin=0.1)
        ance_negatives = ance_miner.mine(
            queries,
            positives,
            teacher_negatives,
            corpus_texts,
            corpus_texts,
            top_k=5,
        )

        # Combine teacher and ANCE negatives
        combined_negatives = []
        combined_scores = []
        for t_negs, t_scores, a_negs in zip(
            teacher_negatives, teacher_scores, ance_negatives
        ):
            # Take top teacher negatives + ANCE negatives
            combined = list(set(t_negs[:5] + a_negs))
            combined_negatives.append(combined)
            # Placeholder scores for ANCE negatives
            combined_scores.append(t_scores[:5] + [0.0] * len(a_negs))

        return combined_negatives, combined_scores

    else:
        raise ValueError(f"Invalid stage: {stage}. Must be 1, 2, or 3.")

