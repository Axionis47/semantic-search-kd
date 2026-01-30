"""Tests for configuration management."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import (
    CORSConfig,
    FAISSConfig,
    LossConfig,
    MiningConfig,
    RateLimitConfig,
    SearchConfig,
    ServiceConfig,
    Settings,
    StudentModelConfig,
    TeacherModelConfig,
    TrainingConfig,
    get_settings,
)


class TestStudentModelConfig:
    """Tests for StudentModelConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = StudentModelConfig()
        assert config.model_name == "./artifacts/models/kd_student_production"
        assert config.max_length == 512
        assert config.embedding_dim == 384
        assert config.normalize_embeddings is True
        assert config.device == "cpu"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = StudentModelConfig(
            model_name="custom/model",
            max_length=256,
            embedding_dim=768,
            device="cuda",
        )
        assert config.model_name == "custom/model"
        assert config.max_length == 256
        assert config.embedding_dim == 768
        assert config.device == "cuda"

    def test_invalid_max_length(self):
        """Test validation of max_length bounds."""
        with pytest.raises(ValueError):
            StudentModelConfig(max_length=10)  # Too small

    def test_invalid_device_pattern(self):
        """Test validation of device pattern."""
        with pytest.raises(ValueError):
            StudentModelConfig(device="invalid")


class TestTeacherModelConfig:
    """Tests for TeacherModelConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TeacherModelConfig()
        assert config.model_name == "BAAI/bge-reranker-large"
        assert config.batch_size == 32

    def test_cuda_device_with_index(self):
        """Test cuda device with index."""
        config = TeacherModelConfig(device="cuda:0")
        assert config.device == "cuda:0"


class TestLossConfig:
    """Tests for LossConfig."""

    def test_default_weights(self):
        """Test that default loss weights are set correctly."""
        config = LossConfig()
        assert config.contrastive_weight == 0.2
        assert config.margin_mse_weight == 0.6
        assert config.listwise_kd_weight == 0.2

    def test_weights_sum_validation(self):
        """Test that weights can be configured."""
        config = LossConfig(
            contrastive_weight=0.3,
            margin_mse_weight=0.5,
            listwise_kd_weight=0.2,
        )
        total = (
            config.contrastive_weight
            + config.margin_mse_weight
            + config.listwise_kd_weight
        )
        assert abs(total - 1.0) < 0.01

    def test_temperature_schedule(self):
        """Test temperature annealing configuration."""
        config = LossConfig()
        assert config.temperature_start == 4.0
        assert config.temperature_end == 2.0
        assert config.temperature_start > config.temperature_end


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_hyperparameters(self):
        """Test default training hyperparameters."""
        config = TrainingConfig()
        assert config.epochs == 3
        assert config.batch_size == 32
        assert config.learning_rate == 2e-5
        assert config.warmup_steps == 1000

    def test_early_stopping_config(self):
        """Test early stopping configuration."""
        config = TrainingConfig(
            early_stopping_patience=5,
            early_stopping_metric="mrr@10",
        )
        assert config.early_stopping_patience == 5
        assert config.early_stopping_metric == "mrr@10"


class TestMiningConfig:
    """Tests for MiningConfig."""

    def test_default_mining_settings(self):
        """Test default mining configuration."""
        config = MiningConfig()
        assert config.bm25_top_k == 100
        assert config.teacher_top_k == 50
        assert config.ance_enabled is True
        assert config.negatives_per_query == 7

    def test_invalid_top_k(self):
        """Test validation of top_k bounds."""
        with pytest.raises(ValueError):
            MiningConfig(bm25_top_k=5)  # Too small


class TestFAISSConfig:
    """Tests for FAISSConfig."""

    def test_default_hnsw_settings(self):
        """Test default HNSW configuration."""
        config = FAISSConfig()
        assert config.index_type == "HNSW"
        assert config.metric == "inner_product"
        assert config.hnsw_m == 32
        assert config.hnsw_ef_construction == 200

    def test_invalid_index_type(self):
        """Test validation of index type."""
        with pytest.raises(ValueError):
            FAISSConfig(index_type="invalid")


class TestCORSConfig:
    """Tests for CORSConfig."""

    def test_default_cors(self):
        """Test default CORS settings."""
        config = CORSConfig()
        assert "http://localhost:3000" in config.allow_origins
        assert config.allow_credentials is False

    def test_wildcard_warning(self):
        """Test warning for wildcard origins."""
        with pytest.warns(UserWarning, match="insecure"):
            CORSConfig(allow_origins=["*"])


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""

    def test_default_rate_limit(self):
        """Test default rate limit settings."""
        config = RateLimitConfig()
        assert config.enabled is True
        assert config.requests_per_minute == 100
        assert config.burst == 20


class TestServiceConfig:
    """Tests for ServiceConfig."""

    def test_default_service_config(self):
        """Test default service configuration."""
        config = ServiceConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.workers == 1

    def test_invalid_log_level(self):
        """Test validation of log level."""
        with pytest.raises(ValueError):
            ServiceConfig(log_level="invalid")


class TestSearchConfig:
    """Tests for SearchConfig."""

    def test_default_search_config(self):
        """Test default search configuration."""
        config = SearchConfig()
        assert config.default_top_k == 10
        assert config.max_top_k == 100
        assert config.rerank_enabled is True


class TestSettings:
    """Tests for main Settings class."""

    def test_default_settings(self):
        """Test default settings initialization."""
        settings = Settings()
        assert settings.environment == "development"
        assert settings.debug is False
        assert isinstance(settings.student, StudentModelConfig)
        assert isinstance(settings.teacher, TeacherModelConfig)
        assert isinstance(settings.training, TrainingConfig)

    def test_from_yaml(self, tmp_path: Path):
        """Test loading settings from YAML file."""
        config_dict = {
            "environment": "staging",
            "debug": True,
            "student": {
                "model_name": "custom/student",
                "max_length": 256,
            },
        }

        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f)

        settings = Settings.from_yaml(yaml_path)
        assert settings.environment == "staging"
        assert settings.debug is True
        assert settings.student.model_name == "custom/student"
        assert settings.student.max_length == 256

    def test_to_yaml(self, tmp_path: Path):
        """Test saving settings to YAML file."""
        settings = Settings(environment="production")
        yaml_path = tmp_path / "output.yaml"

        settings.to_yaml(yaml_path)

        assert yaml_path.exists()
        with open(yaml_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded["environment"] == "production"

    def test_is_production(self):
        """Test production environment detection."""
        dev_settings = Settings(environment="development")
        prod_settings = Settings(environment="production")

        assert dev_settings.is_production() is False
        assert prod_settings.is_production() is True

    def test_validate_for_production_issues(self):
        """Test production validation catches issues."""
        # Settings with common production issues
        settings = Settings(
            environment="production",
            debug=True,
            service={
                "cors": {"allow_origins": ["*"]},
                "auth": {"enabled": False},
                "rate_limit": {"enabled": False},
                "reload": True,
            },
        )

        issues = settings.validate_for_production()

        assert len(issues) >= 4
        assert any("CORS" in issue for issue in issues)
        assert any("Auth" in issue for issue in issues)
        assert any("RateLimit" in issue for issue in issues)
        assert any("Debug" in issue for issue in issues)

    def test_validate_for_production_no_issues(self):
        """Test production validation passes for good config."""
        settings = Settings(
            environment="production",
            debug=False,
            service={
                "cors": {"allow_origins": ["https://example.com"]},
                "auth": {"enabled": True, "api_keys": ["key"]},
                "rate_limit": {"enabled": True},
                "reload": False,
            },
        )

        issues = settings.validate_for_production()
        assert len(issues) == 0

    def test_environment_variable_override(self, monkeypatch):
        """Test that environment variables override defaults."""
        monkeypatch.setenv("SEMANTIC_KD_ENVIRONMENT", "staging")
        monkeypatch.setenv("SEMANTIC_KD_DEBUG", "true")

        # Clear the cache to get fresh settings
        get_settings.cache_clear()

        settings = Settings()
        assert settings.environment == "staging"
        assert settings.debug is True

        # Clean up
        get_settings.cache_clear()

    def test_nested_env_override(self, monkeypatch):
        """Test nested environment variable override."""
        monkeypatch.setenv("SEMANTIC_KD_SERVICE__PORT", "9000")

        settings = Settings()
        assert settings.service.port == 9000

    def test_config_file_not_found(self):
        """Test error when config file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            Settings.from_yaml(Path("/nonexistent/config.yaml"))


class TestGetSettings:
    """Tests for get_settings function."""

    def test_caching(self):
        """Test that get_settings returns cached instance."""
        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_config_path_env(self, tmp_path: Path, monkeypatch):
        """Test loading from SEMANTIC_KD_CONFIG_PATH."""
        config_dict = {"environment": "staging", "debug": True}

        yaml_path = tmp_path / "test_config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f)

        monkeypatch.setenv("SEMANTIC_KD_CONFIG_PATH", str(yaml_path))
        get_settings.cache_clear()

        settings = get_settings()
        assert settings.environment == "staging"

        # Clean up
        get_settings.cache_clear()
        monkeypatch.delenv("SEMANTIC_KD_CONFIG_PATH")
