#!/usr/bin/env python3
"""Model registry for tracking and versioning trained models.

This script provides functionality to:
- Register new model versions
- List all registered models
- Promote models between stages (dev -> staging -> production)
- Retrieve model metadata
- Compare model versions

Usage:
    python scripts/model_registry.py register --model-path ./artifacts/models/kd_student_production
    python scripts/model_registry.py list
    python scripts/model_registry.py promote --version v1.0.0 --stage production
    python scripts/model_registry.py compare --version1 v1.0.0 --version2 v1.1.0
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Configuration
DEFAULT_REGISTRY_PATH = Path("./artifacts/model_registry")
GCS_REGISTRY_BUCKET = "gs://plotpointe-semantic-kd-models/registry"


def compute_model_hash(model_path: Path) -> str:
    """Compute SHA256 hash of model weights for integrity verification."""
    weights_file = model_path / "model.safetensors"
    if not weights_file.exists():
        weights_file = model_path / "pytorch_model.bin"

    if not weights_file.exists():
        raise FileNotFoundError(f"No model weights found in {model_path}")

    sha256_hash = hashlib.sha256()
    with open(weights_file, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()[:12]


def get_model_size_mb(model_path: Path) -> float:
    """Get total size of model directory in MB."""
    total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
    return total_size / (1024 * 1024)


def load_model_and_get_metrics(model_path: Path) -> Dict[str, Any]:
    """Load model and extract basic metrics."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        import time

        model = SentenceTransformer(str(model_path))

        # Get basic properties
        embedding_dim = model.get_sentence_embedding_dimension()
        max_seq_length = model.max_seq_length

        # Measure inference latency
        test_text = "This is a test sentence for latency measurement."
        model.encode([test_text])  # Warmup

        latencies = []
        for _ in range(10):
            start = time.time()
            model.encode([test_text])
            latencies.append((time.time() - start) * 1000)

        return {
            "embedding_dim": embedding_dim,
            "max_seq_length": max_seq_length,
            "inference_latency_ms": {
                "mean": float(np.mean(latencies)),
                "p95": float(np.percentile(latencies, 95)),
            },
        }
    except Exception as e:
        logger.warning(f"Could not load model for metrics: {e}")
        return {}


class ModelRegistry:
    """Local and remote model registry manager."""

    def __init__(self, local_path: Path = DEFAULT_REGISTRY_PATH):
        self.local_path = local_path
        self.local_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.local_path / "registry.json"
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from disk."""
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                self.registry = json.load(f)
        else:
            self.registry = {"models": {}, "latest": {}}

    def _save_registry(self) -> None:
        """Save registry to disk."""
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2, default=str)

    def register(
        self,
        model_path: Path,
        model_name: str,
        version: str,
        stage: str = "development",
        metrics: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        description: str = "",
    ) -> Dict[str, Any]:
        """Register a new model version.

        Args:
            model_path: Path to the model directory.
            model_name: Name of the model (e.g., 'kd-student').
            version: Semantic version (e.g., 'v1.0.0').
            stage: Deployment stage (development/staging/production).
            metrics: Evaluation metrics dictionary.
            training_config: Training configuration used.
            description: Human-readable description.

        Returns:
            Model card dictionary.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # Compute model hash for integrity
        model_hash = compute_model_hash(model_path)
        model_size = get_model_size_mb(model_path)

        # Get model metrics
        model_metrics = load_model_and_get_metrics(model_path)

        # Create model card
        model_card = {
            "model_name": model_name,
            "version": version,
            "stage": stage,
            "description": description,
            "model_hash": model_hash,
            "model_size_mb": round(model_size, 2),
            "model_path": str(model_path.absolute()),
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics or {},
            "model_metrics": model_metrics,
            "training_config": training_config or {},
            "tags": [],
        }

        # Add to registry
        if model_name not in self.registry["models"]:
            self.registry["models"][model_name] = {}

        self.registry["models"][model_name][version] = model_card

        # Update latest for this stage
        stage_key = f"{model_name}:{stage}"
        self.registry["latest"][stage_key] = version

        self._save_registry()

        logger.info(f"Registered {model_name}:{version} (stage={stage})")
        return model_card

    def list_models(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered models or versions of a specific model."""
        results = []

        if model_name:
            if model_name in self.registry["models"]:
                for version, card in self.registry["models"][model_name].items():
                    results.append(card)
        else:
            for name, versions in self.registry["models"].items():
                for version, card in versions.items():
                    results.append(card)

        return sorted(results, key=lambda x: x["registered_at"], reverse=True)

    def get_model(self, model_name: str, version: str) -> Optional[Dict[str, Any]]:
        """Get a specific model version."""
        return self.registry["models"].get(model_name, {}).get(version)

    def get_latest(self, model_name: str, stage: str = "production") -> Optional[str]:
        """Get the latest version for a model in a given stage."""
        return self.registry["latest"].get(f"{model_name}:{stage}")

    def promote(self, model_name: str, version: str, to_stage: str) -> Dict[str, Any]:
        """Promote a model version to a new stage.

        Args:
            model_name: Name of the model.
            version: Version to promote.
            to_stage: Target stage (staging/production).

        Returns:
            Updated model card.
        """
        model = self.get_model(model_name, version)
        if not model:
            raise ValueError(f"Model not found: {model_name}:{version}")

        old_stage = model["stage"]
        model["stage"] = to_stage
        model["promoted_at"] = datetime.now(timezone.utc).isoformat()
        model["promoted_from"] = old_stage

        # Update latest
        self.registry["latest"][f"{model_name}:{to_stage}"] = version
        self._save_registry()

        logger.info(f"Promoted {model_name}:{version} from {old_stage} to {to_stage}")
        return model

    def compare(
        self, model_name: str, version1: str, version2: str
    ) -> Dict[str, Any]:
        """Compare two model versions."""
        model1 = self.get_model(model_name, version1)
        model2 = self.get_model(model_name, version2)

        if not model1 or not model2:
            raise ValueError("One or both model versions not found")

        comparison = {
            "version1": version1,
            "version2": version2,
            "size_diff_mb": model2["model_size_mb"] - model1["model_size_mb"],
            "metrics_comparison": {},
        }

        # Compare metrics
        for key in set(model1.get("metrics", {}).keys()) | set(model2.get("metrics", {}).keys()):
            v1 = model1.get("metrics", {}).get(key)
            v2 = model2.get("metrics", {}).get(key)
            if v1 is not None and v2 is not None:
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    comparison["metrics_comparison"][key] = {
                        "v1": v1,
                        "v2": v2,
                        "diff": v2 - v1,
                        "pct_change": ((v2 - v1) / v1 * 100) if v1 != 0 else None,
                    }

        return comparison

    def sync_to_gcs(self) -> None:
        """Sync local registry to GCS."""
        try:
            subprocess.run(
                ["gsutil", "cp", str(self.registry_file), f"{GCS_REGISTRY_BUCKET}/registry.json"],
                check=True,
                capture_output=True,
            )
            logger.info(f"Registry synced to {GCS_REGISTRY_BUCKET}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to sync to GCS: {e.stderr.decode()}")
            raise

    def sync_from_gcs(self) -> None:
        """Sync registry from GCS to local."""
        try:
            subprocess.run(
                ["gsutil", "cp", f"{GCS_REGISTRY_BUCKET}/registry.json", str(self.registry_file)],
                check=True,
                capture_output=True,
            )
            self._load_registry()
            logger.info("Registry synced from GCS")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not sync from GCS: {e.stderr.decode()}")


def main():
    parser = argparse.ArgumentParser(description="Model Registry CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Register command
    register_parser = subparsers.add_parser("register", help="Register a new model")
    register_parser.add_argument("--model-path", type=str, required=True)
    register_parser.add_argument("--name", type=str, default="kd-student")
    register_parser.add_argument("--version", type=str, required=True)
    register_parser.add_argument("--stage", type=str, default="development")
    register_parser.add_argument("--description", type=str, default="")
    register_parser.add_argument("--metrics-file", type=str, help="JSON file with metrics")

    # List command
    list_parser = subparsers.add_parser("list", help="List registered models")
    list_parser.add_argument("--name", type=str, help="Filter by model name")

    # Get command
    get_parser = subparsers.add_parser("get", help="Get model details")
    get_parser.add_argument("--name", type=str, required=True)
    get_parser.add_argument("--version", type=str, required=True)

    # Promote command
    promote_parser = subparsers.add_parser("promote", help="Promote model to new stage")
    promote_parser.add_argument("--name", type=str, required=True)
    promote_parser.add_argument("--version", type=str, required=True)
    promote_parser.add_argument("--to-stage", type=str, required=True)

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two model versions")
    compare_parser.add_argument("--name", type=str, required=True)
    compare_parser.add_argument("--version1", type=str, required=True)
    compare_parser.add_argument("--version2", type=str, required=True)

    # Sync commands
    subparsers.add_parser("sync-to-gcs", help="Sync registry to GCS")
    subparsers.add_parser("sync-from-gcs", help="Sync registry from GCS")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    registry = ModelRegistry()

    if args.command == "register":
        metrics = None
        if args.metrics_file:
            with open(args.metrics_file) as f:
                metrics = json.load(f)

        card = registry.register(
            model_path=Path(args.model_path),
            model_name=args.name,
            version=args.version,
            stage=args.stage,
            metrics=metrics,
            description=args.description,
        )
        print(json.dumps(card, indent=2))

    elif args.command == "list":
        models = registry.list_models(args.name)
        for model in models:
            print(f"{model['model_name']}:{model['version']} ({model['stage']}) - {model['registered_at']}")

    elif args.command == "get":
        model = registry.get_model(args.name, args.version)
        if model:
            print(json.dumps(model, indent=2))
        else:
            print(f"Model not found: {args.name}:{args.version}")
            sys.exit(1)

    elif args.command == "promote":
        card = registry.promote(args.name, args.version, args.to_stage)
        print(json.dumps(card, indent=2))

    elif args.command == "compare":
        comparison = registry.compare(args.name, args.version1, args.version2)
        print(json.dumps(comparison, indent=2))

    elif args.command == "sync-to-gcs":
        registry.sync_to_gcs()

    elif args.command == "sync-from-gcs":
        registry.sync_from_gcs()


if __name__ == "__main__":
    main()
