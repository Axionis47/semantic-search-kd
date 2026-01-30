"""Custom exceptions for the semantic search application.

This module defines a hierarchy of exceptions for better error handling
and debugging throughout the application.
"""

from typing import Any, Dict, Optional


class SemanticKDError(Exception):
    """Base exception for all semantic-kd errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message.
            error_code: Machine-readable error code for API responses.
            details: Additional context for debugging.
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "SEMANTIC_KD_ERROR"
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
        }


# =============================================================================
# Model Errors
# =============================================================================


class ModelError(SemanticKDError):
    """Base exception for model-related errors."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if model_name:
            details["model_name"] = model_name
        super().__init__(message, details=details, **kwargs)


class ModelLoadError(ModelError):
    """Raised when a model fails to load."""

    def __init__(self, message: str, model_name: Optional[str] = None) -> None:
        super().__init__(
            message,
            model_name=model_name,
            error_code="MODEL_LOAD_ERROR",
        )


class ModelNotLoadedError(ModelError):
    """Raised when trying to use a model that hasn't been loaded."""

    def __init__(self, model_type: str = "model") -> None:
        super().__init__(
            f"{model_type.capitalize()} is not loaded. Call load() first.",
            error_code="MODEL_NOT_LOADED",
            details={"model_type": model_type},
        )


class EncodingError(ModelError):
    """Raised when text encoding fails."""

    def __init__(
        self,
        message: str,
        texts: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            message,
            model_name=model_name,
            error_code="ENCODING_ERROR",
            details={"num_texts": texts} if texts else {},
        )


# =============================================================================
# Index Errors
# =============================================================================


class IndexError(SemanticKDError):
    """Base exception for index-related errors."""

    pass


class IndexNotFoundError(IndexError):
    """Raised when an index file is not found."""

    def __init__(self, index_path: str) -> None:
        super().__init__(
            f"Index not found at: {index_path}",
            error_code="INDEX_NOT_FOUND",
            details={"index_path": index_path},
        )


class IndexNotBuiltError(IndexError):
    """Raised when trying to use an index that hasn't been built."""

    def __init__(self) -> None:
        super().__init__(
            "Index has not been built. Call build() first.",
            error_code="INDEX_NOT_BUILT",
        )


class IndexBuildError(IndexError):
    """Raised when index building fails."""

    def __init__(self, message: str, documents_processed: int = 0) -> None:
        super().__init__(
            message,
            error_code="INDEX_BUILD_ERROR",
            details={"documents_processed": documents_processed},
        )


# =============================================================================
# Data Errors
# =============================================================================


class DataError(SemanticKDError):
    """Base exception for data-related errors."""

    pass


class DataNotFoundError(DataError):
    """Raised when required data files are not found."""

    def __init__(self, data_path: str, data_type: str = "data") -> None:
        super().__init__(
            f"{data_type.capitalize()} not found at: {data_path}",
            error_code="DATA_NOT_FOUND",
            details={"path": data_path, "type": data_type},
        )


class DataValidationError(DataError):
    """Raised when data fails validation."""

    def __init__(self, message: str, field: Optional[str] = None) -> None:
        super().__init__(
            message,
            error_code="DATA_VALIDATION_ERROR",
            details={"field": field} if field else {},
        )


class DataParsingError(DataError):
    """Raised when data parsing fails."""

    def __init__(
        self,
        message: str,
        line_number: Optional[int] = None,
        file_path: Optional[str] = None,
    ) -> None:
        details = {}
        if line_number:
            details["line_number"] = line_number
        if file_path:
            details["file_path"] = file_path
        super().__init__(
            message,
            error_code="DATA_PARSING_ERROR",
            details=details,
        )


# =============================================================================
# Training Errors
# =============================================================================


class TrainingError(SemanticKDError):
    """Base exception for training-related errors."""

    pass


class TrainingConfigError(TrainingError):
    """Raised when training configuration is invalid."""

    def __init__(self, message: str, config_key: Optional[str] = None) -> None:
        super().__init__(
            message,
            error_code="TRAINING_CONFIG_ERROR",
            details={"config_key": config_key} if config_key else {},
        )


class CheckpointError(TrainingError):
    """Raised when checkpoint save/load fails."""

    def __init__(self, message: str, checkpoint_path: Optional[str] = None) -> None:
        super().__init__(
            message,
            error_code="CHECKPOINT_ERROR",
            details={"checkpoint_path": checkpoint_path} if checkpoint_path else {},
        )


class EarlyStoppingError(TrainingError):
    """Raised when early stopping is triggered unexpectedly."""

    def __init__(self, metric: str, best_value: float, patience: int) -> None:
        super().__init__(
            f"Early stopping triggered: {metric} did not improve for {patience} epochs",
            error_code="EARLY_STOPPING",
            details={
                "metric": metric,
                "best_value": best_value,
                "patience": patience,
            },
        )


# =============================================================================
# Search/API Errors
# =============================================================================


class SearchError(SemanticKDError):
    """Base exception for search-related errors."""

    pass


class InvalidQueryError(SearchError):
    """Raised when a search query is invalid."""

    def __init__(self, message: str, query: Optional[str] = None) -> None:
        details = {}
        if query:
            # Truncate for privacy
            details["query_preview"] = query[:50] + "..." if len(query) > 50 else query
        super().__init__(
            message,
            error_code="INVALID_QUERY",
            details=details,
        )


class SearchTimeoutError(SearchError):
    """Raised when a search operation times out."""

    def __init__(self, timeout_ms: int, operation: str = "search") -> None:
        super().__init__(
            f"{operation.capitalize()} timed out after {timeout_ms}ms",
            error_code="SEARCH_TIMEOUT",
            details={"timeout_ms": timeout_ms, "operation": operation},
        )


class RerankingError(SearchError):
    """Raised when reranking fails."""

    def __init__(self, message: str, num_documents: int = 0) -> None:
        super().__init__(
            message,
            error_code="RERANKING_ERROR",
            details={"num_documents": num_documents},
        )


# =============================================================================
# Authentication/Authorization Errors
# =============================================================================


class AuthError(SemanticKDError):
    """Base exception for authentication errors."""

    pass


class InvalidAPIKeyError(AuthError):
    """Raised when an API key is invalid."""

    def __init__(self) -> None:
        super().__init__(
            "Invalid or missing API key",
            error_code="INVALID_API_KEY",
        )


class RateLimitExceededError(AuthError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        limit: int,
        window_seconds: int,
        retry_after: Optional[int] = None,
    ) -> None:
        super().__init__(
            f"Rate limit exceeded: {limit} requests per {window_seconds} seconds",
            error_code="RATE_LIMIT_EXCEEDED",
            details={
                "limit": limit,
                "window_seconds": window_seconds,
                "retry_after": retry_after,
            },
        )


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigError(SemanticKDError):
    """Base exception for configuration errors."""

    pass


class ConfigNotFoundError(ConfigError):
    """Raised when a configuration file is not found."""

    def __init__(self, config_path: str) -> None:
        super().__init__(
            f"Configuration file not found: {config_path}",
            error_code="CONFIG_NOT_FOUND",
            details={"config_path": config_path},
        )


class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""

    def __init__(self, message: str, field: Optional[str] = None) -> None:
        super().__init__(
            message,
            error_code="CONFIG_VALIDATION_ERROR",
            details={"field": field} if field else {},
        )
