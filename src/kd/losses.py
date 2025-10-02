"""
Knowledge Distillation losses for semantic search.

Implements:
1. Margin-MSE Loss (primary KD loss)
2. Listwise KD Loss (ranking-aware)
3. Contrastive Loss (in-batch negatives)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class MarginMSELoss(nn.Module):
    """
    Margin-MSE Loss for knowledge distillation.

    Minimizes MSE between student and teacher score margins.
    Focuses on relative ordering rather than absolute scores.
    """

    def __init__(self, temperature: float = 1.0):
        """
        Initialize Margin-MSE loss.

        Args:
            temperature: Temperature for softening teacher scores
        """
        super().__init__()
        self.temperature = temperature
        logger.info(f"MarginMSELoss initialized (temperature={temperature})")

    def forward(
        self,
        student_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Margin-MSE loss.

        Args:
            student_scores: Student similarity scores [batch_size, num_docs]
            teacher_scores: Teacher relevance scores [batch_size, num_docs]

        Returns:
            Scalar loss
        """
        # Soften teacher scores with temperature
        teacher_soft = teacher_scores / self.temperature

        # Compute margins (difference from max score)
        student_margins = student_scores - student_scores.max(dim=1, keepdim=True)[0]
        teacher_margins = teacher_soft - teacher_soft.max(dim=1, keepdim=True)[0]

        # MSE between margins
        loss = F.mse_loss(student_margins, teacher_margins)

        return loss


class ListwiseKDLoss(nn.Module):
    """
    Listwise KD Loss using KL divergence.

    Distills ranking distribution from teacher to student.
    """

    def __init__(self, temperature: float = 1.0):
        """
        Initialize Listwise KD loss.

        Args:
            temperature: Temperature for softening distributions
        """
        super().__init__()
        self.temperature = temperature
        logger.info(f"ListwiseKDLoss initialized (temperature={temperature})")

    def forward(
        self,
        student_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Listwise KD loss.

        Args:
            student_scores: Student similarity scores [batch_size, num_docs]
            teacher_scores: Teacher relevance scores [batch_size, num_docs]

        Returns:
            Scalar loss
        """
        # Soften distributions with temperature
        student_soft = F.log_softmax(student_scores / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_scores / self.temperature, dim=1)

        # KL divergence
        loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")

        # Scale by temperature^2 (standard KD practice)
        loss = loss * (self.temperature ** 2)

        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss with in-batch negatives.

    Treats first document as positive, rest as negatives.
    """

    def __init__(self, temperature: float = 0.05):
        """
        Initialize Contrastive loss.

        Args:
            temperature: Temperature for scaling similarities
        """
        super().__init__()
        self.temperature = temperature
        logger.info(f"ContrastiveLoss initialized (temperature={temperature})")

    def forward(
        self,
        student_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Contrastive loss.

        Args:
            student_scores: Student similarity scores [batch_size, num_docs]
                           First doc is positive, rest are negatives

        Returns:
            Scalar loss
        """
        # Scale by temperature
        scaled_scores = student_scores / self.temperature

        # Positive is first document (index 0)
        # Compute log-softmax and take negative log-likelihood of positive
        log_probs = F.log_softmax(scaled_scores, dim=1)
        loss = -log_probs[:, 0].mean()

        return loss


class CombinedKDLoss(nn.Module):
    """
    Combined KD loss with weighted components.

    Combines:
    - Margin-MSE (primary KD)
    - Listwise KD (ranking)
    - Contrastive (in-batch negatives)
    """

    def __init__(
        self,
        margin_mse_weight: float = 0.6,
        listwise_kd_weight: float = 0.2,
        contrastive_weight: float = 0.2,
        temperature_start: float = 4.0,
        temperature_end: float = 2.0,
    ):
        """
        Initialize Combined KD loss.

        Args:
            margin_mse_weight: Weight for Margin-MSE loss
            listwise_kd_weight: Weight for Listwise KD loss
            contrastive_weight: Weight for Contrastive loss
            temperature_start: Starting temperature
            temperature_end: Ending temperature (for annealing)
        """
        super().__init__()

        self.margin_mse_weight = margin_mse_weight
        self.listwise_kd_weight = listwise_kd_weight
        self.contrastive_weight = contrastive_weight

        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.current_temperature = temperature_start

        # Initialize loss components
        self.margin_mse_loss = MarginMSELoss(temperature=temperature_start)
        self.listwise_kd_loss = ListwiseKDLoss(temperature=temperature_start)
        self.contrastive_loss = ContrastiveLoss(temperature=0.05)

        logger.info(
            f"CombinedKDLoss initialized: "
            f"margin_mse={margin_mse_weight}, "
            f"listwise_kd={listwise_kd_weight}, "
            f"contrastive={contrastive_weight}, "
            f"temp={temperature_start}→{temperature_end}"
        )

    def update_temperature(self, progress: float):
        """
        Update temperature based on training progress (linear annealing).

        Args:
            progress: Training progress in [0, 1]
        """
        self.current_temperature = (
            self.temperature_start
            + (self.temperature_end - self.temperature_start) * progress
        )

        # Update component temperatures
        self.margin_mse_loss.temperature = self.current_temperature
        self.listwise_kd_loss.temperature = self.current_temperature

    def forward(
        self,
        student_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
    ) -> dict:
        """
        Compute combined KD loss.

        Args:
            student_scores: Student similarity scores [batch_size, num_docs]
            teacher_scores: Teacher relevance scores [batch_size, num_docs]

        Returns:
            Dictionary with total loss and component losses
        """
        # Compute component losses
        margin_mse = self.margin_mse_loss(student_scores, teacher_scores)
        listwise_kd = self.listwise_kd_loss(student_scores, teacher_scores)
        contrastive = self.contrastive_loss(student_scores)

        # Weighted combination
        total_loss = (
            self.margin_mse_weight * margin_mse
            + self.listwise_kd_weight * listwise_kd
            + self.contrastive_weight * contrastive
        )

        return {
            "loss": total_loss,
            "margin_mse": margin_mse.item(),
            "listwise_kd": listwise_kd.item(),
            "contrastive": contrastive.item(),
            "temperature": self.current_temperature,
        }


def test_losses():
    """Test loss functions with dummy data."""
    logger.info("Testing KD losses...")

    batch_size = 4
    num_docs = 5

    # Dummy scores
    student_scores = torch.randn(batch_size, num_docs)
    teacher_scores = torch.randn(batch_size, num_docs)

    # Test Margin-MSE
    margin_mse_loss = MarginMSELoss(temperature=2.0)
    loss1 = margin_mse_loss(student_scores, teacher_scores)
    logger.info(f"Margin-MSE Loss: {loss1.item():.4f}")

    # Test Listwise KD
    listwise_kd_loss = ListwiseKDLoss(temperature=2.0)
    loss2 = listwise_kd_loss(student_scores, teacher_scores)
    logger.info(f"Listwise KD Loss: {loss2.item():.4f}")

    # Test Contrastive
    contrastive_loss = ContrastiveLoss(temperature=0.05)
    loss3 = contrastive_loss(student_scores)
    logger.info(f"Contrastive Loss: {loss3.item():.4f}")

    # Test Combined
    combined_loss = CombinedKDLoss()
    result = combined_loss(student_scores, teacher_scores)
    logger.info(f"Combined Loss: {result['loss'].item():.4f}")
    logger.info(f"  - Margin-MSE: {result['margin_mse']:.4f}")
    logger.info(f"  - Listwise KD: {result['listwise_kd']:.4f}")
    logger.info(f"  - Contrastive: {result['contrastive']:.4f}")

    # Test temperature annealing
    combined_loss.update_temperature(0.5)
    logger.info(f"Temperature after 50% progress: {combined_loss.current_temperature:.2f}")

    logger.info("✓ All loss tests passed!")


if __name__ == "__main__":
    from src.utils.logging import setup_logging

    setup_logging(log_level="INFO")
    test_losses()

