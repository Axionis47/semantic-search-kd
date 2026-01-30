#!/usr/bin/env python3
"""Start FastAPI service with trained KD model."""

import argparse
from pathlib import Path

import uvicorn
from loguru import logger

from src.serve.app import create_app
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    setup_logging(log_level="INFO")

    # Validate arguments
    from scripts._validate_args import validate_path_exists, validate_port, validate_device
    validate_path_exists(args.model_path, "--model-path")
    validate_port(args.port)
    validate_device(args.device)

    logger.info("="*80)
    logger.info("Starting Semantic Search Service")
    logger.info("="*80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Host: {args.host}:{args.port}")
    logger.info(f"Device: {args.device}")
    logger.info("="*80)
    
    # Create app
    app = create_app(
        student_model_path=args.model_path,
        device=args.device
    )
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()

