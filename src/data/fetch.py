"""Fetch MS MARCO and BEIR datasets."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from src.data.registry import DATASETS, ensure_dirs, get_dataset_config


def fetch_msmarco(output_dir: Path, max_samples: Optional[int] = None) -> Dict:
    """
    Fetch MS MARCO Passage Ranking dataset from HuggingFace.

    Args:
        output_dir: Directory to save raw data
        max_samples: Optional limit on number of samples (for testing)

    Returns:
        Manifest dictionary with file paths and metadata
    """
    logger.info("Fetching MS MARCO Passage Ranking dataset...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset from HuggingFace
    # Using ms_marco v2.1 configuration
    dataset = load_dataset("ms_marco", "v2.1")

    manifest = {
        "dataset": "msmarco",
        "version": "v2.1",
        "source": "huggingface:ms_marco",
        "splits": {},
    }

    # Save each split
    for split_name in ["train", "validation", "test"]:
        if split_name not in dataset:
            logger.warning(f"Split '{split_name}' not found in dataset")
            continue

        split_data = dataset[split_name]

        # Limit samples if specified
        if max_samples:
            split_data = split_data.select(range(min(max_samples, len(split_data))))

        # Save to JSONL
        output_file = output_dir / f"{split_name}.jsonl"
        logger.info(f"Saving {split_name} split to {output_file} ({len(split_data)} samples)")

        with open(output_file, "w") as f:
            for example in tqdm(split_data, desc=f"Writing {split_name}"):
                f.write(json.dumps(example) + "\n")

        manifest["splits"][split_name] = {
            "file": str(output_file),
            "num_samples": len(split_data),
        }

    logger.info(f"MS MARCO fetch complete: {len(manifest['splits'])} splits")
    return manifest


def fetch_beir_dataset(dataset_name: str, output_dir: Path) -> Dict:
    """
    Fetch a BEIR dataset (SKIPPED - BEIR not installed).

    Args:
        dataset_name: BEIR dataset name (e.g., 'fiqa', 'scifact')
        output_dir: Directory to save raw data

    Returns:
        Manifest dictionary
    """
    logger.warning(f"BEIR dataset fetching skipped (BEIR not installed): {dataset_name}")
    logger.info("To use BEIR datasets, install manually or use HuggingFace datasets")

    # Return empty manifest
    return {
        "dataset": f"beir_{dataset_name}",
        "version": "1.0.0",
        "source": f"beir:{dataset_name}",
        "splits": {},
        "skipped": True,
    }


def fetch_all_datasets(max_msmarco_samples: Optional[int] = None) -> Dict[str, Dict]:
    """
    Fetch all datasets (MS MARCO + BEIR).

    Args:
        max_msmarco_samples: Optional limit for MS MARCO (for testing)

    Returns:
        Dictionary mapping dataset name to manifest
    """
    ensure_dirs()

    manifests = {}

    # Fetch MS MARCO
    config = get_dataset_config("msmarco")
    manifest = fetch_msmarco(config["raw_dir"], max_samples=max_msmarco_samples)
    manifests["msmarco"] = manifest

    # Save manifest
    with open(config["manifest"], "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved manifest to {config['manifest']}")

    # Fetch BEIR datasets
    beir_datasets = ["fiqa", "scifact", "trec-covid"]
    for beir_name in beir_datasets:
        try:
            config_key = f"beir_{beir_name.replace('-', '_')}"
            config = get_dataset_config(config_key)
            manifest = fetch_beir_dataset(beir_name, config["raw_dir"])
            manifests[config_key] = manifest

            # Save manifest
            with open(config["manifest"], "w") as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"Saved manifest to {config['manifest']}")

        except Exception as e:
            logger.error(f"Failed to fetch {beir_name}: {e}")
            continue

    logger.info(f"All datasets fetched: {list(manifests.keys())}")
    return manifests


if __name__ == "__main__":
    from src.utils.logging import setup_logging

    setup_logging(log_level="INFO")

    # Fetch all datasets
    manifests = fetch_all_datasets()

    print("\n=== Fetch Summary ===")
    for dataset_name, manifest in manifests.items():
        print(f"\n{dataset_name}:")
        print(f"  Version: {manifest['version']}")
        print(f"  Splits: {list(manifest['splits'].keys())}")

