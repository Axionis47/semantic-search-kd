"""FastAPI application for semantic search service."""

import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src.index.build_index import FAISSIndexBuilder
from src.models.student import StudentModel
from src.models.teacher import TeacherModel
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
VERSION = "1.0.0"

# Global state
app_state = {
    "student": None,
    "teacher": None,
    "index_builder": None,
    "documents": None,
    "doc_ids": None,
}


def create_app(
    student_model_path: str = "intfloat/e5-small-v2",
    teacher_model_path: Optional[str] = None,
    index_dir: Optional[Path] = None,
    device: str = "cpu",
) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        student_model_path: Path to student model
        teacher_model_path: Path to teacher model (optional, for reranking)
        index_dir: Directory containing FAISS index
        device: Device to use (cpu/cuda)

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="Semantic Search API",
        description="Production-grade semantic search with knowledge distillation",
        version=VERSION,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event():
        """Load models and index on startup."""
        logger.info("Starting up semantic search service...")

        # Load student model
        logger.info(f"Loading student model: {student_model_path}")
        app_state["student"] = StudentModel(student_model_path, device=device)
        logger.info("✓ Student model loaded")

        # Load teacher model (optional)
        if teacher_model_path:
            logger.info(f"Loading teacher model: {teacher_model_path}")
            app_state["teacher"] = TeacherModel(teacher_model_path, device=device)
            logger.info("✓ Teacher model loaded")

        # Load index
        if index_dir:
            logger.info(f"Loading index from: {index_dir}")
            index_builder = FAISSIndexBuilder(
                embedding_dim=app_state["student"].embedding_dim
            )
            index_builder.load(index_dir)
            app_state["index_builder"] = index_builder
            app_state["doc_ids"] = index_builder.doc_ids
            logger.info(f"✓ Index loaded: {len(app_state['doc_ids'])} documents")

        logger.info("✓ Service ready!")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        logger.info("Shutting down semantic search service...")

    @app.get("/", response_model=dict)
    async def root():
        """Root endpoint."""
        return {
            "service": "Semantic Search API",
            "version": VERSION,
            "status": "running",
        }

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            model_loaded=app_state["student"] is not None,
            index_loaded=app_state["index_builder"] is not None,
            index_size=len(app_state["doc_ids"]) if app_state["doc_ids"] else 0,
            version=VERSION,
        )

    @app.post("/search", response_model=SearchResponse)
    async def search(request: SearchRequest):
        """
        Search for similar documents.

        Args:
            request: Search request

        Returns:
            Search response with results
        """
        start_time = time.time()

        try:
            # Validate
            if app_state["student"] is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model not loaded",
                )

            if app_state["index_builder"] is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Index not loaded",
                )

            # Encode query
            query_emb = app_state["student"].encode_queries([request.query])

            # Search index
            k_retrieve = request.rerank_top_k if request.rerank else request.k
            distances, indices = app_state["index_builder"].search(
                query_emb, k=k_retrieve
            )

            # Get results
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(app_state["doc_ids"]):
                    results.append(
                        {
                            "doc_id": app_state["doc_ids"][idx],
                            "text": f"Document {idx}",  # TODO: Load actual text
                            "score": float(dist),
                            "rank": i + 1,
                        }
                    )

            # Rerank if requested
            reranked = False
            if request.rerank and app_state["teacher"] is not None:
                logger.info(f"Reranking top {len(results)} results")
                pairs = [[request.query, r["text"]] for r in results]
                teacher_scores = app_state["teacher"].score(pairs)

                # Re-sort by teacher scores
                for r, score in zip(results, teacher_scores):
                    r["score"] = float(score)

                results = sorted(results, key=lambda x: x["score"], reverse=True)

                # Update ranks
                for i, r in enumerate(results):
                    r["rank"] = i + 1

                reranked = True

            # Limit to k
            results = results[: request.k]

            # Compute latency
            latency_ms = (time.time() - start_time) * 1000

            return SearchResponse(
                query=request.query,
                results=[SearchResult(**r) for r in results],
                total_results=len(results),
                reranked=reranked,
                latency_ms=latency_ms,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

    @app.post("/encode", response_model=EncodeResponse)
    async def encode(request: EncodeRequest):
        """
        Encode texts to embeddings.

        Args:
            request: Encode request

        Returns:
            Encode response with embeddings
        """
        start_time = time.time()

        try:
            # Validate
            if app_state["student"] is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model not loaded",
                )

            # Encode
            embeddings = app_state["student"].encode(
                request.texts,
                convert_to_numpy=True,
                normalize=request.normalize,
            )

            # Compute latency
            latency_ms = (time.time() - start_time) * 1000

            return EncodeResponse(
                embeddings=embeddings.tolist(),
                dimension=embeddings.shape[1],
                num_texts=len(request.texts),
                latency_ms=latency_ms,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Encode error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error=exc.detail).dict(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        """Handle general exceptions."""
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Internal server error", detail=str(exc)
            ).dict(),
        )

    return app


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

