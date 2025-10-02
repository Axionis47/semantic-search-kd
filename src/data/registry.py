"""Data registry for canonical paths and manifests."""

from pathlib import Path
from typing import Dict, Optional

# Base paths
DATA_DIR = Path("./data")
RAW_DIR = DATA_DIR / "raw"
CHUNKS_DIR = DATA_DIR / "chunks"
MANIFESTS_DIR = DATA_DIR / "manifests"

# Dataset configurations
DATASETS = {
    "msmarco": {
        "name": "ms_marco",
        "version": "v2.1",
        "hf_dataset": "ms_marco",
        "hf_config": "v2.1",
        "splits": ["train", "validation", "test"],
        "raw_dir": RAW_DIR / "msmarco",
        "chunks_dir": CHUNKS_DIR / "msmarco",
        "manifest": MANIFESTS_DIR / "msmarco_manifest.json",
    },
    "beir_fiqa": {
        "name": "fiqa",
        "version": "1.0.0",
        "beir_dataset": "fiqa",
        "splits": ["test"],
        "raw_dir": RAW_DIR / "beir" / "fiqa",
        "chunks_dir": CHUNKS_DIR / "beir" / "fiqa",
        "manifest": MANIFESTS_DIR / "beir_fiqa_manifest.json",
    },
    "beir_scifact": {
        "name": "scifact",
        "version": "1.0.0",
        "beir_dataset": "scifact",
        "splits": ["test"],
        "raw_dir": RAW_DIR / "beir" / "scifact",
        "chunks_dir": CHUNKS_DIR / "beir" / "scifact",
        "manifest": MANIFESTS_DIR / "beir_scifact_manifest.json",
    },
    "beir_trec_covid": {
        "name": "trec-covid",
        "version": "1.0.0",
        "beir_dataset": "trec-covid",
        "splits": ["test"],
        "raw_dir": RAW_DIR / "beir" / "trec-covid",
        "chunks_dir": CHUNKS_DIR / "beir" / "trec-covid",
        "manifest": MANIFESTS_DIR / "beir_trec_covid_manifest.json",
    },
}


def get_dataset_config(dataset_name: str) -> Dict:
    """
    Get configuration for a dataset.

    Args:
        dataset_name: Dataset name (e.g., 'msmarco', 'beir_fiqa')

    Returns:
        Dataset configuration dictionary

    Raises:
        ValueError: If dataset not found
    """
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Available: {list(DATASETS.keys())}"
        )
    return DATASETS[dataset_name]


def get_raw_path(dataset_name: str) -> Path:
    """Get raw data directory for a dataset."""
    config = get_dataset_config(dataset_name)
    return config["raw_dir"]


def get_chunks_path(dataset_name: str) -> Path:
    """Get chunks directory for a dataset."""
    config = get_dataset_config(dataset_name)
    return config["chunks_dir"]


def get_manifest_path(dataset_name: str) -> Path:
    """Get manifest file path for a dataset."""
    config = get_dataset_config(dataset_name)
    return config["manifest"]


def ensure_dirs() -> None:
    """Create all necessary directories."""
    DATA_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)
    CHUNKS_DIR.mkdir(exist_ok=True)
    MANIFESTS_DIR.mkdir(exist_ok=True)

    for dataset_config in DATASETS.values():
        dataset_config["raw_dir"].mkdir(parents=True, exist_ok=True)
        dataset_config["chunks_dir"].mkdir(parents=True, exist_ok=True)


def get_all_datasets() -> list:
    """Get list of all dataset names."""
    return list(DATASETS.keys())

