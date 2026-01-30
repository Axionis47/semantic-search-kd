"""Centralized configuration management using Pydantic Settings.

This module provides type-safe configuration loading from YAML files and
environment variables with validation.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# Model Configurations
# =============================================================================


class StudentModelConfig(BaseModel):
    """Configuration for the student bi-encoder model."""

    model_name: str = Field(
        default="./artifacts/models/kd_student_production",
        description="HuggingFace model name or local path to KD-trained model",
    )
    max_length: int = Field(default=512, ge=32, le=8192)
    embedding_dim: int = Field(default=384, ge=64, le=4096)
    normalize_embeddings: bool = Field(default=True)
    device: str = Field(default="cpu", pattern="^(cpu|cuda(:\\d+)?)$")


class TeacherModelConfig(BaseModel):
    """Configuration for the teacher cross-encoder model."""

    model_name: str = Field(
        default="BAAI/bge-reranker-large",
        description="HuggingFace model name or local path",
    )
    max_length: int = Field(default=512, ge=32, le=8192)
    device: str = Field(default="cpu", pattern="^(cpu|cuda(:\\d+)?)$")
    batch_size: int = Field(default=32, ge=1, le=512)


# =============================================================================
# Training Configurations
# =============================================================================


class LossConfig(BaseModel):
    """Configuration for knowledge distillation loss functions."""

    contrastive_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    margin_mse_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    listwise_kd_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    contrastive_temperature: float = Field(default=0.05, gt=0.0, le=1.0)
    temperature_start: float = Field(default=4.0, gt=0.0)
    temperature_end: float = Field(default=2.0, gt=0.0)

    @model_validator(mode="after")
    def weights_should_sum_to_one(self) -> "LossConfig":
        """Validate that loss weights approximately sum to 1."""
        total = self.contrastive_weight + self.margin_mse_weight + self.listwise_kd_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Loss weights must sum to ~1.0, got {total:.3f} "
                f"(contrastive={self.contrastive_weight}, "
                f"margin_mse={self.margin_mse_weight}, "
                f"listwise_kd={self.listwise_kd_weight})"
            )
        return self


class TrainingConfig(BaseModel):
    """Configuration for the training pipeline."""

    # Training hyperparameters
    epochs: int = Field(default=3, ge=1, le=100)
    batch_size: int = Field(default=32, ge=1, le=512)
    gradient_accumulation_steps: int = Field(default=2, ge=1)
    learning_rate: float = Field(default=2e-5, gt=0.0, le=1.0)
    warmup_steps: int = Field(default=1000, ge=0)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)

    # Mixed precision
    fp16: bool = Field(default=True)

    # Early stopping
    early_stopping_patience: int = Field(default=2, ge=1)
    early_stopping_metric: str = Field(default="ndcg@10")

    # Checkpointing
    save_steps: int = Field(default=500, ge=1)
    eval_steps: int = Field(default=500, ge=1)
    logging_steps: int = Field(default=100, ge=1)

    # Loss configuration
    loss: LossConfig = Field(default_factory=LossConfig)


class MiningConfig(BaseModel):
    """Configuration for hard negative mining."""

    # BM25 mining
    bm25_top_k: int = Field(default=100, ge=10, le=1000)

    # Teacher mining
    teacher_top_k: int = Field(default=50, ge=5, le=500)

    # ANCE mining
    ance_enabled: bool = Field(default=True)
    ance_warmup_steps: int = Field(default=1000, ge=0)

    # Negatives per query
    negatives_per_query: int = Field(default=7, ge=1, le=50)


# =============================================================================
# Index Configurations
# =============================================================================


class FAISSConfig(BaseModel):
    """Configuration for FAISS index."""

    index_type: str = Field(default="HNSW", pattern="^(Flat|IVF|HNSW|PQ)$")
    metric: str = Field(default="inner_product", pattern="^(l2|inner_product)$")

    # HNSW parameters
    hnsw_m: int = Field(default=32, ge=8, le=128)
    hnsw_ef_construction: int = Field(default=200, ge=50, le=500)
    hnsw_ef_search: int = Field(default=64, ge=16, le=256)

    # IVF parameters (if using IVF)
    ivf_nlist: int = Field(default=100, ge=1, le=10000)
    ivf_nprobe: int = Field(default=10, ge=1, le=1000)


# =============================================================================
# Service Configurations
# =============================================================================


class CORSConfig(BaseModel):
    """Configuration for CORS middleware."""

    enabled: bool = Field(default=True)
    allow_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    allow_methods: List[str] = Field(default_factory=lambda: ["GET", "POST"])
    allow_headers: List[str] = Field(default_factory=lambda: ["*"])
    allow_credentials: bool = Field(default=False)

    @field_validator("allow_origins")
    @classmethod
    def validate_origins(cls, v: List[str]) -> List[str]:
        """Warn if using wildcard in production."""
        if "*" in v:
            import warnings

            warnings.warn(
                "CORS allow_origins contains '*'. This is insecure for production.",
                UserWarning,
                stacklevel=2,
            )
        return v


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting."""

    enabled: bool = Field(default=True)
    requests_per_minute: int = Field(default=100, ge=1, le=10000)
    burst: int = Field(default=20, ge=1, le=100)


class AuthConfig(BaseModel):
    """Configuration for API authentication."""

    enabled: bool = Field(default=False)
    api_key_header: str = Field(default="X-API-Key")
    api_keys: List[str] = Field(default_factory=list)

    @field_validator("api_keys")
    @classmethod
    def validate_api_keys(cls, v: List[str], info: Any) -> List[str]:
        """Validate API keys if auth is enabled."""
        return v


class MonitoringConfig(BaseModel):
    """Configuration for observability."""

    prometheus_enabled: bool = Field(default=True)
    prometheus_port: int = Field(default=9090, ge=1024, le=65535)
    prometheus_path: str = Field(default="/metrics")

    opentelemetry_enabled: bool = Field(default=False)
    opentelemetry_endpoint: str = Field(default="http://localhost:4317")
    service_name: str = Field(default="semantic-kd")

    log_queries: bool = Field(default=False)
    log_latencies: bool = Field(default=True)


class ServiceConfig(BaseModel):
    """Configuration for the FastAPI service."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080, ge=1024, le=65535)
    workers: int = Field(default=1, ge=1, le=32)
    reload: bool = Field(default=False)
    log_level: str = Field(default="info", pattern="^(debug|info|warning|error|critical)$")

    cors: CORSConfig = Field(default_factory=CORSConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)


class SearchConfig(BaseModel):
    """Configuration for search behavior."""

    default_top_k: int = Field(default=10, ge=1, le=1000)
    max_top_k: int = Field(default=100, ge=1, le=10000)

    # Reranking
    rerank_enabled: bool = Field(default=True)
    rerank_top_k: int = Field(default=50, ge=1, le=500)
    rerank_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    rerank_timeout_ms: int = Field(default=5000, ge=100, le=30000)


# =============================================================================
# Data Configurations
# =============================================================================


class DataConfig(BaseModel):
    """Configuration for data processing."""

    # Paths
    raw_data_dir: Path = Field(default=Path("./data/raw"))
    chunks_dir: Path = Field(default=Path("./data/chunks"))
    artifacts_dir: Path = Field(default=Path("./artifacts"))

    # Chunking
    max_chunk_tokens: int = Field(default=512, ge=64, le=8192)
    chunk_stride: int = Field(default=80, ge=0, le=256)

    # Dataset
    dataset_name: str = Field(default="ms_marco")
    dataset_version: str = Field(default="v2.1")


# =============================================================================
# Main Settings Class
# =============================================================================


class Settings(BaseSettings):
    """Main application settings.

    Settings are loaded from (in order of precedence):
    1. Environment variables
    2. YAML configuration file
    3. Default values

    Environment variables are prefixed with SEMANTIC_KD_.
    Nested settings use double underscore (e.g., SEMANTIC_KD_SERVICE__PORT).
    """

    model_config = SettingsConfigDict(
        env_prefix="SEMANTIC_KD_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    # Environment
    environment: str = Field(
        default="development",
        pattern="^(development|staging|production)$",
    )
    debug: bool = Field(default=False)

    # Component configurations
    student: StudentModelConfig = Field(default_factory=StudentModelConfig)
    teacher: TeacherModelConfig = Field(default_factory=TeacherModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    mining: MiningConfig = Field(default_factory=MiningConfig)
    faiss: FAISSConfig = Field(default_factory=FAISSConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    @model_validator(mode="after")
    def enforce_production_settings(self) -> "Settings":
        """Enforce security settings in production."""
        if self.environment == "production":
            if not self.service.auth.enabled:
                import warnings

                warnings.warn(
                    "Auth is disabled in production. Set SEMANTIC_KD_SERVICE__AUTH__ENABLED=true",
                    UserWarning,
                    stacklevel=2,
                )
            if "*" in self.service.cors.allow_origins:
                import warnings

                warnings.warn(
                    "CORS allow_origins='*' in production. Restrict to specific origins.",
                    UserWarning,
                    stacklevel=2,
                )
        return self

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Settings":
        """Load settings from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            Settings instance with values from YAML file.
        """
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_yaml(self, yaml_path: Path) -> None:
        """Save settings to a YAML file.

        Args:
            yaml_path: Path to save YAML configuration.
        """
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(
                self.model_dump(mode="json"),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    def validate_for_production(self) -> List[str]:
        """Validate settings for production readiness.

        Returns:
            List of validation warnings/errors.
        """
        issues = []

        # Check CORS
        if "*" in self.service.cors.allow_origins:
            issues.append("CORS: allow_origins contains '*' - restrict to specific origins")

        # Check auth
        if not self.service.auth.enabled:
            issues.append("Auth: API authentication is disabled")

        # Check rate limiting
        if not self.service.rate_limit.enabled:
            issues.append("RateLimit: Rate limiting is disabled")

        # Check debug mode
        if self.debug:
            issues.append("Debug: Debug mode is enabled")

        # Check reload
        if self.service.reload:
            issues.append("Service: Auto-reload is enabled (development only)")

        return issues


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance.

    Loads from environment variable SEMANTIC_KD_CONFIG_PATH if set,
    otherwise uses defaults.

    Returns:
        Cached Settings instance.
    """
    config_path = os.environ.get("SEMANTIC_KD_CONFIG_PATH")

    if config_path:
        return Settings.from_yaml(Path(config_path))

    return Settings()


def load_yaml_config(yaml_path: str) -> Dict[str, Any]:
    """Load raw YAML configuration as dictionary.

    Args:
        yaml_path: Path to YAML file.

    Returns:
        Configuration dictionary.
    """
    with open(yaml_path) as f:
        return yaml.safe_load(f)
