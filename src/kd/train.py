"""
Knowledge Distillation training with 3-stage curriculum.

Implements:
- 3-stage mining curriculum (BM25 → Teacher → ANCE)
- Combined KD losses with temperature annealing
- Early stopping and checkpointing
- Comprehensive logging and metrics
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.kd.losses import CombinedKDLoss
from src.mining.miners import build_mining_curriculum
from src.models.student import StudentModel
from src.models.teacher import TeacherModel
from src.utils.seed import set_seed


class KDDataset(Dataset):
    """Dataset for knowledge distillation training."""

    def __init__(
        self,
        queries: List[str],
        positives: List[List[str]],
        negatives: List[List[str]],
        teacher_scores: List[List[float]],
        corpus_texts: Dict[str, str],
    ):
        """
        Initialize KD dataset.

        Args:
            queries: List of query strings
            positives: List of positive doc IDs for each query
            negatives: List of negative doc IDs for each query
            teacher_scores: Teacher scores for negatives
            corpus_texts: Mapping from doc ID to text
        """
        self.queries = queries
        self.positives = positives
        self.negatives = negatives
        self.teacher_scores = teacher_scores
        self.corpus_texts = corpus_texts

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        pos_ids = self.positives[idx]
        neg_ids = self.negatives[idx]
        t_scores = self.teacher_scores[idx]

        # Get texts
        pos_texts = [self.corpus_texts.get(doc_id, "") for doc_id in pos_ids]
        neg_texts = [self.corpus_texts.get(doc_id, "") for doc_id in neg_ids]

        # Combine: [positive, negatives]
        doc_texts = pos_texts + neg_texts
        doc_ids = pos_ids + neg_ids

        # Teacher scores: [1.0 for positives, actual scores for negatives]
        teacher_scores = [1.0] * len(pos_ids) + t_scores

        return {
            "query": query,
            "doc_texts": doc_texts,
            "doc_ids": doc_ids,
            "teacher_scores": teacher_scores,
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        "queries": [item["query"] for item in batch],
        "doc_texts": [item["doc_texts"] for item in batch],
        "doc_ids": [item["doc_ids"] for item in batch],
        "teacher_scores": [item["teacher_scores"] for item in batch],
    }


class KDTrainer:
    """Knowledge Distillation trainer with curriculum learning."""

    def __init__(
        self,
        student_model: StudentModel,
        teacher_model: TeacherModel,
        loss_fn: CombinedKDLoss,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        output_dir: str = "./artifacts/models/kd_student",
    ):
        """
        Initialize KD trainer.

        Args:
            student_model: Student model to train
            teacher_model: Teacher model (frozen)
            loss_fn: Combined KD loss function
            optimizer: Optimizer for student
            device: Device to train on
            output_dir: Directory to save checkpoints
        """
        self.student = student_model
        self.teacher = teacher_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Move student model to device and set to training mode
        self.student.model.to(device)
        self.student.model.train()  # Enable training mode for gradients

        # Teacher (CrossEncoder) is already on the correct device from initialization
        # No need to move it explicitly

        self.best_loss = float("inf")
        self.patience_counter = 0

        logger.info(f"KDTrainer initialized (device={device})")

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        total_epochs: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            total_epochs: Total number of epochs

        Returns:
            Dictionary of metrics
        """
        self.student.model.train()
        total_loss = 0.0
        total_margin_mse = 0.0
        total_listwise_kd = 0.0
        total_contrastive = 0.0

        # Update temperature based on progress
        progress = epoch / total_epochs
        self.loss_fn.update_temperature(progress)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}")
        for batch_idx, batch in enumerate(pbar):
            queries = batch["queries"]
            doc_texts_batch = batch["doc_texts"]
            teacher_scores_batch = batch["teacher_scores"]

            batch_loss = 0.0
            batch_margin_mse = 0.0
            batch_listwise_kd = 0.0
            batch_contrastive = 0.0

            # Process each query in batch
            for query, doc_texts, teacher_scores in zip(
                queries, doc_texts_batch, teacher_scores_batch
            ):
                # Encode query (keeps gradients for backprop)
                query_emb = self.student.encode_with_gradients([query], normalize=True)

                # Encode documents (keeps gradients for backprop)
                doc_embs = self.student.encode_with_gradients(doc_texts, normalize=True)

                # Compute student scores (cosine similarity)
                # Embeddings are already normalized, so just dot product
                student_scores = torch.matmul(query_emb, doc_embs.T)[0]

                # Convert teacher scores to tensor
                teacher_scores_tensor = torch.tensor(
                    teacher_scores, dtype=torch.float32, device=self.device
                )

                # Compute loss
                loss_dict = self.loss_fn(
                    student_scores.unsqueeze(0), teacher_scores_tensor.unsqueeze(0)
                )

                batch_loss += loss_dict["loss"]
                batch_margin_mse += loss_dict["margin_mse"]
                batch_listwise_kd += loss_dict["listwise_kd"]
                batch_contrastive += loss_dict["contrastive"]

            # Average over batch
            batch_loss = batch_loss / len(queries)

            # Backward pass
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += batch_loss.item()
            total_margin_mse += batch_margin_mse / len(queries)
            total_listwise_kd += batch_listwise_kd / len(queries)
            total_contrastive += batch_contrastive / len(queries)

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{batch_loss.item():.4f}",
                    "temp": f"{self.loss_fn.current_temperature:.2f}",
                }
            )

        # Average metrics
        num_batches = len(dataloader)
        metrics = {
            "loss": total_loss / num_batches,
            "margin_mse": total_margin_mse / num_batches,
            "listwise_kd": total_listwise_kd / num_batches,
            "contrastive": total_contrastive / num_batches,
            "temperature": self.loss_fn.current_temperature,
        }

        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"

        self.student.model.save(str(checkpoint_path))

        # Save metrics
        metrics_path = self.output_dir / f"metrics_epoch_{epoch}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        epochs: int = 3,
        patience: int = 2,
    ):
        """
        Train student model with KD.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            epochs: Number of epochs
            patience: Early stopping patience
        """
        logger.info(f"Starting KD training: {epochs} epochs, patience={patience}")

        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_dataloader, epoch, epochs)

            logger.info(
                f"Epoch {epoch}/{epochs} - "
                f"Loss: {train_metrics['loss']:.4f}, "
                f"Margin-MSE: {train_metrics['margin_mse']:.4f}, "
                f"Listwise KD: {train_metrics['listwise_kd']:.4f}, "
                f"Contrastive: {train_metrics['contrastive']:.4f}, "
                f"Temp: {train_metrics['temperature']:.2f}"
            )

            # Save checkpoint
            self.save_checkpoint(epoch, train_metrics)

            # Early stopping
            if train_metrics["loss"] < self.best_loss:
                self.best_loss = train_metrics["loss"]
                self.patience_counter = 0

                # Save best model
                best_path = self.output_dir / "best_model"
                self.student.model.save(str(best_path))
                logger.info(f"✓ New best model saved: {best_path}")
            else:
                self.patience_counter += 1
                logger.info(
                    f"No improvement ({self.patience_counter}/{patience})"
                )

                if self.patience_counter >= patience:
                    logger.info("Early stopping triggered!")
                    break

        logger.info("Training complete!")


if __name__ == "__main__":
    # Test training setup
    from src.utils.logging import setup_logging

    setup_logging(log_level="INFO")

    logger.info("Testing KD training setup...")

    # Dummy data
    queries = ["What is machine learning?", "How does Python work?"]
    positives = [["doc1"], ["doc2"]]
    negatives = [["doc3", "doc4"], ["doc5", "doc6"]]
    teacher_scores = [[0.8, 0.6], [0.7, 0.5]]
    corpus_texts = {
        "doc1": "Machine learning is a subset of AI",
        "doc2": "Python is a programming language",
        "doc3": "Cats are animals",
        "doc4": "The sky is blue",
        "doc5": "Cars have wheels",
        "doc6": "Water is wet",
    }

    # Create dataset
    dataset = KDDataset(queries, positives, negatives, teacher_scores, corpus_texts)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    logger.info(f"✓ Dataset created: {len(dataset)} samples")
    logger.info(f"✓ DataLoader created: {len(dataloader)} batches")

    logger.info("KD training setup test complete!")

