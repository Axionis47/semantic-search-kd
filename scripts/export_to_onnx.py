#!/usr/bin/env python3
"""Export trained KD model to ONNX with INT8 quantization."""

import argparse
from pathlib import Path

from loguru import logger

from src.models.export_onnx import export_student_model
from src.models.student import StudentModel
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip-quantize", action="store_true", help="Skip quantization")
    parser.add_argument("--skip-validate", action="store_true", help="Skip validation")
    args = parser.parse_args()
    
    setup_logging(log_level="INFO")
    
    logger.info("="*80)
    logger.info("ONNX Export & Quantization")
    logger.info("="*80)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    student = StudentModel(args.model_path, device=args.device)
    
    # Export
    output_dir = Path(args.output_dir)
    paths = export_student_model(
        student,
        output_dir=output_dir,
        quantize=not args.skip_quantize,
        validate=not args.skip_validate
    )
    
    logger.info("\n" + "="*80)
    logger.info("EXPORT COMPLETE!")
    logger.info("="*80)
    logger.info(f"ONNX model: {paths.get('onnx')}")
    if 'quantized' in paths:
        logger.info(f"Quantized model: {paths.get('quantized')}")
    logger.info("="*80)


if __name__ == "__main__":
    main()

