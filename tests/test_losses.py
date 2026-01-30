"""Tests for knowledge distillation loss functions."""

import pytest
import torch

from src.kd.losses import (
    CombinedKDLoss,
    ContrastiveLoss,
    ListwiseKDLoss,
    MarginMSELoss,
)


class TestMarginMSELoss:
    """Tests for MarginMSELoss."""

    @pytest.fixture
    def loss_fn(self) -> MarginMSELoss:
        """Create MarginMSE loss instance."""
        return MarginMSELoss(temperature=2.0)

    @pytest.fixture
    def sample_scores(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Create sample student and teacher scores."""
        batch_size, num_docs = 4, 5
        student = torch.randn(batch_size, num_docs)
        teacher = torch.randn(batch_size, num_docs)
        return student, teacher

    def test_output_is_scalar(self, loss_fn: MarginMSELoss, sample_scores):
        """Test that loss output is a scalar."""
        student, teacher = sample_scores
        loss = loss_fn(student, teacher)

        assert loss.ndim == 0
        assert loss.shape == torch.Size([])

    def test_loss_is_non_negative(self, loss_fn: MarginMSELoss, sample_scores):
        """Test that MSE loss is always non-negative."""
        student, teacher = sample_scores
        loss = loss_fn(student, teacher)

        assert loss >= 0

    def test_identical_scores_give_zero_loss(self, loss_fn: MarginMSELoss):
        """Test that identical student and teacher scores give zero loss."""
        scores = torch.randn(4, 5)
        loss = loss_fn(scores, scores * loss_fn.temperature)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_gradient_flow(self, loss_fn: MarginMSELoss):
        """Test that gradients flow through the loss."""
        student = torch.randn(4, 5, requires_grad=True)
        teacher = torch.randn(4, 5)

        loss = loss_fn(student, teacher)
        loss.backward()

        assert student.grad is not None
        assert not torch.isnan(student.grad).any()

    def test_temperature_scaling(self):
        """Test that temperature affects the loss."""
        student = torch.randn(4, 5)
        teacher = torch.randn(4, 5)

        loss_low_temp = MarginMSELoss(temperature=1.0)(student, teacher)
        loss_high_temp = MarginMSELoss(temperature=4.0)(student, teacher)

        # Different temperatures should give different losses
        assert loss_low_temp.item() != loss_high_temp.item()

    def test_batch_independence(self, loss_fn: MarginMSELoss):
        """Test that loss handles different batch sizes."""
        for batch_size in [1, 4, 16]:
            student = torch.randn(batch_size, 5)
            teacher = torch.randn(batch_size, 5)
            loss = loss_fn(student, teacher)

            assert not torch.isnan(loss)
            assert loss >= 0


class TestListwiseKDLoss:
    """Tests for ListwiseKDLoss."""

    @pytest.fixture
    def loss_fn(self) -> ListwiseKDLoss:
        """Create ListwiseKD loss instance."""
        return ListwiseKDLoss(temperature=2.0)

    def test_output_is_scalar(self, loss_fn: ListwiseKDLoss):
        """Test that loss output is a scalar."""
        student = torch.randn(4, 5)
        teacher = torch.randn(4, 5)
        loss = loss_fn(student, teacher)

        assert loss.ndim == 0

    def test_loss_is_non_negative(self, loss_fn: ListwiseKDLoss):
        """Test that KL divergence is non-negative."""
        student = torch.randn(4, 5)
        teacher = torch.randn(4, 5)
        loss = loss_fn(student, teacher)

        assert loss >= 0

    def test_identical_distributions_give_zero_loss(self, loss_fn: ListwiseKDLoss):
        """Test that identical distributions give zero KL divergence."""
        scores = torch.randn(4, 5)
        loss = loss_fn(scores, scores)

        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_gradient_flow(self, loss_fn: ListwiseKDLoss):
        """Test that gradients flow through the loss."""
        student = torch.randn(4, 5, requires_grad=True)
        teacher = torch.randn(4, 5)

        loss = loss_fn(student, teacher)
        loss.backward()

        assert student.grad is not None
        assert not torch.isnan(student.grad).any()

    def test_temperature_squared_scaling(self):
        """Test that loss is scaled by temperature^2."""
        student = torch.randn(4, 5)
        teacher = torch.randn(4, 5)

        loss_t1 = ListwiseKDLoss(temperature=1.0)(student, teacher)
        loss_t2 = ListwiseKDLoss(temperature=2.0)(student, teacher)

        # With T=2, the T^2 scaling should make loss larger for same inputs
        # (ignoring the softening effect)
        assert loss_t1.item() != loss_t2.item()


class TestContrastiveLoss:
    """Tests for ContrastiveLoss."""

    @pytest.fixture
    def loss_fn(self) -> ContrastiveLoss:
        """Create Contrastive loss instance."""
        return ContrastiveLoss(temperature=0.05)

    def test_output_is_scalar(self, loss_fn: ContrastiveLoss):
        """Test that loss output is a scalar."""
        scores = torch.randn(4, 5)
        loss = loss_fn(scores)

        assert loss.ndim == 0

    def test_loss_is_non_negative(self, loss_fn: ContrastiveLoss):
        """Test that negative log likelihood is non-negative."""
        scores = torch.randn(4, 5)
        loss = loss_fn(scores)

        assert loss >= 0

    def test_perfect_separation_gives_low_loss(self, loss_fn: ContrastiveLoss):
        """Test that perfect separation gives low loss."""
        # First doc (positive) has much higher score
        scores = torch.tensor([
            [10.0, -10.0, -10.0, -10.0, -10.0],
            [10.0, -10.0, -10.0, -10.0, -10.0],
        ])
        loss = loss_fn(scores)

        # Should be close to 0 when positive is clearly dominant
        assert loss.item() < 0.1

    def test_no_separation_gives_high_loss(self, loss_fn: ContrastiveLoss):
        """Test that no separation gives higher loss."""
        # All docs have same score
        scores = torch.zeros(4, 5)
        loss = loss_fn(scores)

        # Should be around log(5) = 1.61 when uniform
        expected = torch.log(torch.tensor(5.0))
        assert loss.item() == pytest.approx(expected.item(), rel=0.1)

    def test_gradient_flow(self, loss_fn: ContrastiveLoss):
        """Test that gradients flow through the loss."""
        scores = torch.randn(4, 5, requires_grad=True)

        loss = loss_fn(scores)
        loss.backward()

        assert scores.grad is not None
        assert not torch.isnan(scores.grad).any()

    def test_temperature_effect(self):
        """Test that temperature affects sharpness."""
        scores = torch.randn(4, 5)

        loss_low_temp = ContrastiveLoss(temperature=0.01)(scores)
        loss_high_temp = ContrastiveLoss(temperature=1.0)(scores)

        # Different temperatures should give different losses
        assert loss_low_temp.item() != loss_high_temp.item()


class TestCombinedKDLoss:
    """Tests for CombinedKDLoss."""

    @pytest.fixture
    def loss_fn(self) -> CombinedKDLoss:
        """Create Combined KD loss instance."""
        return CombinedKDLoss(
            margin_mse_weight=0.6,
            listwise_kd_weight=0.2,
            contrastive_weight=0.2,
            temperature_start=4.0,
            temperature_end=2.0,
        )

    @pytest.fixture
    def sample_scores(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Create sample scores."""
        batch_size, num_docs = 4, 5
        student = torch.randn(batch_size, num_docs)
        teacher = torch.randn(batch_size, num_docs)
        return student, teacher

    def test_output_is_dict(self, loss_fn: CombinedKDLoss, sample_scores):
        """Test that output is a dictionary with expected keys."""
        student, teacher = sample_scores
        result = loss_fn(student, teacher)

        assert isinstance(result, dict)
        assert "loss" in result
        assert "margin_mse" in result
        assert "listwise_kd" in result
        assert "contrastive" in result
        assert "temperature" in result

    def test_total_loss_is_weighted_sum(self, loss_fn: CombinedKDLoss, sample_scores):
        """Test that total loss is weighted sum of components."""
        student, teacher = sample_scores
        result = loss_fn(student, teacher)

        expected = (
            loss_fn.margin_mse_weight * result["margin_mse"]
            + loss_fn.listwise_kd_weight * result["listwise_kd"]
            + loss_fn.contrastive_weight * result["contrastive"]
        )

        assert result["loss"].item() == pytest.approx(expected, rel=1e-5)

    def test_gradient_flow(self, loss_fn: CombinedKDLoss):
        """Test that gradients flow through combined loss."""
        student = torch.randn(4, 5, requires_grad=True)
        teacher = torch.randn(4, 5)

        result = loss_fn(student, teacher)
        result["loss"].backward()

        assert student.grad is not None
        assert not torch.isnan(student.grad).any()

    def test_temperature_annealing(self, loss_fn: CombinedKDLoss):
        """Test temperature annealing during training."""
        # Initial temperature
        assert loss_fn.current_temperature == 4.0

        # After 50% progress
        loss_fn.update_temperature(0.5)
        assert loss_fn.current_temperature == 3.0

        # After 100% progress
        loss_fn.update_temperature(1.0)
        assert loss_fn.current_temperature == 2.0

    def test_temperature_updates_components(self, loss_fn: CombinedKDLoss):
        """Test that temperature update affects component losses."""
        loss_fn.update_temperature(0.5)

        assert loss_fn.margin_mse_loss.temperature == loss_fn.current_temperature
        assert loss_fn.listwise_kd_loss.temperature == loss_fn.current_temperature

    def test_custom_weights(self):
        """Test custom weight configuration."""
        loss_fn = CombinedKDLoss(
            margin_mse_weight=0.8,
            listwise_kd_weight=0.1,
            contrastive_weight=0.1,
        )

        assert loss_fn.margin_mse_weight == 0.8
        assert loss_fn.listwise_kd_weight == 0.1
        assert loss_fn.contrastive_weight == 0.1

    def test_zero_weights(self):
        """Test that zero weights disable loss components."""
        loss_fn = CombinedKDLoss(
            margin_mse_weight=1.0,
            listwise_kd_weight=0.0,
            contrastive_weight=0.0,
        )

        student = torch.randn(4, 5)
        teacher = torch.randn(4, 5)

        result = loss_fn(student, teacher)

        # Only margin_mse should contribute
        assert result["loss"].item() == pytest.approx(result["margin_mse"], rel=1e-5)


class TestLossNumericalStability:
    """Tests for numerical stability of losses."""

    def test_margin_mse_with_large_scores(self):
        """Test MarginMSE with large score values."""
        loss_fn = MarginMSELoss(temperature=2.0)
        student = torch.randn(4, 5) * 100
        teacher = torch.randn(4, 5) * 100

        loss = loss_fn(student, teacher)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_listwise_with_large_scores(self):
        """Test ListwiseKD with large score values."""
        loss_fn = ListwiseKDLoss(temperature=2.0)
        student = torch.randn(4, 5) * 10
        teacher = torch.randn(4, 5) * 10

        loss = loss_fn(student, teacher)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_contrastive_with_small_temperature(self):
        """Test Contrastive with very small temperature."""
        loss_fn = ContrastiveLoss(temperature=0.01)
        scores = torch.randn(4, 5)

        loss = loss_fn(scores)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_combined_with_edge_cases(self):
        """Test CombinedKDLoss with edge cases."""
        loss_fn = CombinedKDLoss()

        # Single sample
        student = torch.randn(1, 5)
        teacher = torch.randn(1, 5)
        result = loss_fn(student, teacher)
        assert not torch.isnan(result["loss"])

        # Many documents
        student = torch.randn(4, 100)
        teacher = torch.randn(4, 100)
        result = loss_fn(student, teacher)
        assert not torch.isnan(result["loss"])
