"""Tests for mining module interfaces."""

import numpy as np
import pytest


class TestMiningConfig:
    """Test mining configuration validation."""

    def test_mining_config_defaults(self):
        """Test MiningConfig default values."""
        from src.config import MiningConfig

        config = MiningConfig()
        assert config.bm25_top_k == 100
        assert config.teacher_top_k == 50
        assert config.ance_enabled is True
        assert config.negatives_per_query == 7

    def test_mining_config_bounds(self):
        """Test MiningConfig boundary validation."""
        from src.config import MiningConfig

        with pytest.raises(Exception):
            MiningConfig(bm25_top_k=5)  # Below minimum of 10

        with pytest.raises(Exception):
            MiningConfig(negatives_per_query=0)  # Below minimum of 1

    def test_mining_config_valid_custom(self):
        """Test MiningConfig with valid custom values."""
        from src.config import MiningConfig

        config = MiningConfig(
            bm25_top_k=200,
            teacher_top_k=100,
            negatives_per_query=10,
        )
        assert config.bm25_top_k == 200
        assert config.teacher_top_k == 100
        assert config.negatives_per_query == 10
