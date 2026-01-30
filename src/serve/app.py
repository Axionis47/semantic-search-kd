"""FastAPI application for semantic search service.

This module provides a production-ready API for semantic search with:
- Knowledge distillation-trained bi-encoder for fast retrieval
- Optional cross-encoder reranking for higher accuracy
- Rate limiting and API key authentication
- Prometheus metrics and structured logging
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src.config import Settings, get_settings
from src.index.build_index import FAISSIndexBuilder
from src.models.student import StudentModel
from src.models.teacher import TeacherModel
from src.serve.middleware import (
    APIKeyMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
)
from src.serve.schemas import (
    EncodeRequest,
    EncodeResponse,
    ErrorResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

# Version
VERSION = "1.1.0"


# =============================================================================
# Application State
# =============================================================================


class AppState:
    """Application state container for models and indexes."""

    def __init__(self) -> None:
        self.student: Optional[StudentModel] = None
        self.teacher: Optional[TeacherModel] = None
        self.index_builder: Optional[FAISSIndexBuilder] = None
        self.doc_ids: Optional[List[str]] = None
        self.doc_texts: Optional[Dict[str, str]] = None
        self.settings: Optional[Settings] = None
        self.ready: bool = False

    def is_ready(self) -> bool:
        """Check if the service is ready to handle requests."""
        return self.ready and self.student is not None


app_state = AppState()


# =============================================================================
# Lifespan Management
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    # Startup
    logger.info("Starting semantic search service...")

    settings = get_settings()
    app_state.settings = settings

    # Load student model
    student_path = settings.student.model_name
    logger.info(f"Loading student model: {student_path}")
    try:
        app_state.student = StudentModel(
            model_name=student_path,
            device=settings.student.device,
        )
        logger.info("Student model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load student model: {e}")
        raise

    # Load teacher model if configured
    if settings.search.rerank_enabled:
        teacher_path = settings.teacher.model_name
        logger.info(f"Loading teacher model: {teacher_path}")
        try:
            app_state.teacher = TeacherModel(
                model_name=teacher_path,
                device=settings.teacher.device,
            )
            logger.info("Teacher model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load teacher model (reranking disabled): {e}")

    app_state.ready = True
    logger.info("Service ready!")

    yield

    # Shutdown
    logger.info("Shutting down semantic search service...")
    app_state.ready = False


# =============================================================================
# App Factory
# =============================================================================


def create_app(
    student_model_path: Optional[str] = None,
    teacher_model_path: Optional[str] = None,
    index_dir: Optional[Path] = None,
    device: str = "cpu",
    settings: Optional[Settings] = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        student_model_path: Path to student model (overrides config).
        teacher_model_path: Path to teacher model (overrides config).
        index_dir: Directory containing FAISS index.
        device: Device to use (cpu/cuda).
        settings: Settings instance (uses default if not provided).

    Returns:
        Configured FastAPI application.
    """
    if settings is None:
        settings = get_settings()

    # Override settings if paths provided
    if student_model_path:
        settings.student.model_name = student_model_path
    if device:
        settings.student.device = device
        settings.teacher.device = device

    app = FastAPI(
        title="Semantic Search API",
        description="Production-grade semantic search with knowledge distillation",
        version=VERSION,
        lifespan=lifespan,
    )

    # Configure CORS
    cors_config = settings.service.cors
    if cors_config.enabled:
        # Validate CORS for production
        if settings.is_production() and "*" in cors_config.allow_origins:
            logger.warning(
                "CORS allows all origins in production - this is a security risk!"
            )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_config.allow_origins,
            allow_credentials=cors_config.allow_credentials,
            allow_methods=cors_config.allow_methods,
            allow_headers=cors_config.allow_headers,
        )

    # Add security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # Add request logging
    app.add_middleware(
        RequestLoggingMiddleware,
        log_queries=settings.service.monitoring.log_queries,
        log_latencies=settings.service.monitoring.log_latencies,
    )

    # Add rate limiting
    rate_limit = settings.service.rate_limit
    if rate_limit.enabled:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=rate_limit.requests_per_minute,
            burst=rate_limit.burst,
            enabled=rate_limit.enabled,
        )

    # Add API key authentication
    auth = settings.service.auth
    if auth.enabled:
        app.add_middleware(
            APIKeyMiddleware,
            api_keys=auth.api_keys,
            header_name=auth.api_key_header,
            enabled=auth.enabled,
        )

    # Register routes
    register_routes(app, settings)

    return app


# =============================================================================
# Route Registration
# =============================================================================


def register_routes(app: FastAPI, settings: Settings) -> None:
    """Register all API routes."""

    @app.get("/", response_model=Dict[str, Any])
    async def root() -> Dict[str, Any]:
        """Root endpoint with service information."""
        return {
            "service": "Semantic Search API",
            "version": VERSION,
            "status": "running" if app_state.is_ready() else "starting",
            "environment": settings.environment,
        }

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint for load balancers and orchestrators."""
        return HealthResponse(
            status="healthy" if app_state.is_ready() else "unhealthy",
            model_loaded=app_state.student is not None,
            index_loaded=app_state.index_builder is not None,
            index_size=len(app_state.doc_ids) if app_state.doc_ids else 0,
            version=VERSION,
        )

    @app.get("/ready")
    async def readiness() -> Dict[str, bool]:
        """Kubernetes readiness probe endpoint."""
        if not app_state.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready",
            )
        return {"ready": True}

    @app.get("/live")
    async def liveness() -> Dict[str, bool]:
        """Kubernetes liveness probe endpoint."""
        return {"alive": True}

    @app.post("/search", response_model=SearchResponse)
    async def search(request: SearchRequest) -> SearchResponse:
        """Search for semantically similar documents.

        Args:
            request: Search request with query and parameters.

        Returns:
            Search response with ranked results.

        Raises:
            HTTPException: If service is unavailable or search fails.
        """
        start_time = time.time()

        # Validate service state
        if app_state.student is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Student model not loaded",
            )

        if app_state.index_builder is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search index not loaded",
            )

        try:
            # Encode query
            query_emb = app_state.student.encode_queries([request.query])

            # Determine how many to retrieve
            k_retrieve = request.rerank_top_k if request.rerank else request.k

            # Search index
            distances, indices = app_state.index_builder.search(
                query_emb, k=k_retrieve
            )

            # Build results with document text
            results = []
            for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
                if idx < 0 or (app_state.doc_ids and idx >= len(app_state.doc_ids)):
                    continue

                doc_id = app_state.doc_ids[idx] if app_state.doc_ids else f"doc_{idx}"

                # Get document text if available
                doc_text = ""
                if app_state.doc_texts and doc_id in app_state.doc_texts:
                    doc_text = app_state.doc_texts[doc_id]

                results.append(
                    SearchResult(
                        doc_id=doc_id,
                        text=doc_text,
                        score=float(dist),
                        rank=rank,
                    )
                )

            # Optional reranking with teacher model
            reranked = False
            if request.rerank and app_state.teacher is not None and results:
                logger.debug(f"Reranking top {len(results)} results")

                # Score with teacher
                pairs = [[request.query, r.text] for r in results]
                teacher_scores = app_state.teacher.score(pairs)

                # Update scores
                for result, score in zip(results, teacher_scores):
                    result.score = float(score)

                # Re-sort by teacher scores
                results = sorted(results, key=lambda x: x.score, reverse=True)

                # Update ranks
                for i, result in enumerate(results, 1):
                    result.rank = i

                reranked = True

            # Limit to requested k
            results = results[: request.k]

            latency_ms = (time.time() - start_time) * 1000

            return SearchResponse(
                query=request.query,
                results=results,
                total_results=len(results),
                reranked=reranked,
                latency_ms=latency_ms,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Search failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Search failed: {str(e)}",
            )

    @app.post("/encode", response_model=EncodeResponse)
    async def encode(request: EncodeRequest) -> EncodeResponse:
        """Encode texts to dense vector embeddings.

        Args:
            request: Encode request with texts.

        Returns:
            Encode response with embeddings.

        Raises:
            HTTPException: If encoding fails.
        """
        start_time = time.time()

        if app_state.student is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded",
            )

        try:
            embeddings = app_state.student.encode(
                request.texts,
                convert_to_numpy=True,
                normalize=request.normalize,
            )

            latency_ms = (time.time() - start_time) * 1000

            return EncodeResponse(
                embeddings=embeddings.tolist(),
                dimension=embeddings.shape[1],
                num_texts=len(request.texts),
                latency_ms=latency_ms,
            )

        except Exception as e:
            logger.exception(f"Encoding failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Encoding failed: {str(e)}",
            )

    @app.post("/index/load")
    async def load_index(index_path: str) -> Dict[str, Any]:
        """Load a FAISS index from disk.

        Args:
            index_path: Path to the index directory.

        Returns:
            Status information about the loaded index.
        """
        try:
            index_dir = Path(index_path)
            if not index_dir.exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Index not found: {index_path}",
                )

            logger.info(f"Loading index from: {index_dir}")

            index_builder = FAISSIndexBuilder(
                embedding_dim=app_state.student.embedding_dim
            )
            index_builder.load(index_dir)

            app_state.index_builder = index_builder
            app_state.doc_ids = index_builder.doc_ids

            # Load document texts if available
            texts_path = index_dir / "texts.json"
            if texts_path.exists():
                import json

                with open(texts_path) as f:
                    app_state.doc_texts = json.load(f)
                logger.info(f"Loaded {len(app_state.doc_texts)} document texts")

            return {
                "status": "loaded",
                "index_path": str(index_dir),
                "num_documents": len(app_state.doc_ids),
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to load index: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load index: {str(e)}",
            )

    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        """Handle HTTP exceptions with consistent format."""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error=exc.detail).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle unexpected exceptions."""
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Internal server error",
                detail=str(exc) if not app_state.settings.is_production() else None,
            ).model_dump(),
        )


# =============================================================================
# Default App Instance
# =============================================================================


# Create default app for uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        app,
        host=settings.service.host,
        port=settings.service.port,
        log_level=settings.service.log_level,
    )
