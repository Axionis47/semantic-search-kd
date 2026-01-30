"""Data integrity checks: hashes, counts, schema validation."""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set

from loguru import logger
from tqdm import tqdm

from src.data.registry import get_dataset_config, get_manifest_path


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        Hex digest of SHA256 hash
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def count_jsonl_lines(file_path: Path) -> int:
    """Count lines in a JSONL file."""
    with open(file_path, "r") as f:
        return sum(1 for _ in f)


def check_file_integrity(file_path: Path, expected_hash: str = None) -> Dict:
    """
    Check integrity of a single file.

    Args:
        file_path: Path to file
        expected_hash: Optional expected SHA256 hash

    Returns:
        Dictionary with check results
    """
    if not file_path.exists():
        return {"exists": False, "error": "File not found"}

    actual_hash = compute_file_hash(file_path)
    num_lines = count_jsonl_lines(file_path) if file_path.suffix == ".jsonl" else None

    result = {
        "exists": True,
        "hash": actual_hash,
        "num_lines": num_lines,
        "size_bytes": file_path.stat().st_size,
    }

    if expected_hash:
        result["hash_match"] = actual_hash == expected_hash

    return result


def check_no_duplicates(file_path: Path, id_field: str = "doc_id") -> Dict:
    """
    Check for duplicate IDs in a JSONL file.

    Args:
        file_path: Path to JSONL file
        id_field: Field name for ID

    Returns:
        Dictionary with duplicate check results
    """
    seen_ids: Set[str] = set()
    duplicates: List[str] = []

    with open(file_path, "r") as f:
        for line in tqdm(f, desc=f"Checking duplicates in {file_path.name}"):
            try:
                data = json.loads(line)
                doc_id = data.get(id_field)
                if doc_id:
                    if doc_id in seen_ids:
                        duplicates.append(doc_id)
                    seen_ids.add(doc_id)
            except json.JSONDecodeError:
                continue

    return {
        "total_ids": len(seen_ids),
        "num_duplicates": len(duplicates),
        "has_duplicates": len(duplicates) > 0,
        "duplicate_ids": duplicates[:10],  # First 10 duplicates
    }


def check_schema(file_path: Path, required_fields: List[str]) -> Dict:
    """
    Check that all records have required fields.

    Args:
        file_path: Path to JSONL file
        required_fields: List of required field names

    Returns:
        Dictionary with schema check results
    """
    missing_fields_count = {field: 0 for field in required_fields}
    total_records = 0

    with open(file_path, "r") as f:
        for line in tqdm(f, desc=f"Checking schema in {file_path.name}"):
            try:
                data = json.loads(line)
                total_records += 1
                for field in required_fields:
                    if field not in data or data[field] is None:
                        missing_fields_count[field] += 1
            except json.JSONDecodeError:
                continue

    return {
        "total_records": total_records,
        "missing_fields": {
            field: count for field, count in missing_fields_count.items() if count > 0
        },
        "schema_valid": all(count == 0 for count in missing_fields_count.values()),
    }


def check_msmarco_integrity(dataset_name: str = "msmarco") -> Dict:
    """
    Run all integrity checks for MS MARCO dataset.

    Args:
        dataset_name: Dataset name

    Returns:
        Dictionary with all check results
    """
    logger.info(f"Running integrity checks for {dataset_name}")

    config = get_dataset_config(dataset_name)
    manifest_path = get_manifest_path(dataset_name)

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return {"error": "Manifest not found"}

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    results = {
        "dataset": dataset_name,
        "checks": {},
        "passed": True,
    }

    # Check each split
    for split_name, split_info in manifest["splits"].items():
        file_path = Path(split_info["file"])
        logger.info(f"Checking {split_name} split: {file_path}")

        split_results = {}

        # File existence and hash
        file_check = check_file_integrity(file_path)
        split_results["file"] = file_check

        if not file_check["exists"]:
            results["passed"] = False
            results["checks"][split_name] = split_results
            continue

        # Row count
        expected_count = split_info.get("num_samples")
        actual_count = file_check["num_lines"]
        split_results["count_match"] = actual_count == expected_count
        if not split_results["count_match"]:
            logger.warning(
                f"Count mismatch in {split_name}: expected {expected_count}, got {actual_count}"
            )
            results["passed"] = False

        # No duplicates
        dup_check = check_no_duplicates(file_path, id_field="query_id")
        split_results["duplicates"] = dup_check
        if dup_check["has_duplicates"]:
            logger.warning(f"Found {dup_check['num_duplicates']} duplicates in {split_name}")
            results["passed"] = False

        # Schema validation
        schema_check = check_schema(file_path, required_fields=["query", "passages"])
        split_results["schema"] = schema_check
        if not schema_check["schema_valid"]:
            logger.warning(f"Schema validation failed for {split_name}")
            results["passed"] = False

        results["checks"][split_name] = split_results

    if results["passed"]:
        logger.info(f"✓ All integrity checks passed for {dataset_name}")
    else:
        logger.error(f"✗ Some integrity checks failed for {dataset_name}")

    return results


def check_all_datasets() -> Dict[str, Dict]:
    """
    Run integrity checks for all datasets.

    Returns:
        Dictionary mapping dataset name to check results
    """
    all_results = {}

    # Check MS MARCO
    all_results["msmarco"] = check_msmarco_integrity("msmarco")

    # Check BEIR datasets
    beir_datasets = ["beir_fiqa", "beir_scifact", "beir_trec_covid"]
    for dataset_name in beir_datasets:
        try:
            # Similar checks for BEIR (simplified)
            config = get_dataset_config(dataset_name)
            manifest_path = get_manifest_path(dataset_name)

            if not manifest_path.exists():
                logger.warning(f"Manifest not found for {dataset_name}, skipping")
                continue

            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            results = {
                "dataset": dataset_name,
                "checks": {},
                "passed": True,
            }

            # Check corpus file
            if "test" in manifest["splits"]:
                corpus_file = Path(manifest["splits"]["test"]["corpus_file"])
                if corpus_file.exists():
                    dup_check = check_no_duplicates(corpus_file, id_field="doc_id")
                    results["checks"]["corpus"] = {"duplicates": dup_check}
                    if dup_check["has_duplicates"]:
                        results["passed"] = False

            all_results[dataset_name] = results

        except (OSError, KeyError, ValueError) as e:
            logger.error(f"Failed to check {dataset_name}: {e}")
            all_results[dataset_name] = {"error": str(e), "passed": False}

    # Summary
    total = len(all_results)
    passed = sum(1 for r in all_results.values() if r.get("passed", False))
    logger.info(f"\n=== Integrity Check Summary ===")
    logger.info(f"Total datasets: {total}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {total - passed}")

    return all_results


if __name__ == "__main__":
    from src.utils.logging import setup_logging

    setup_logging(log_level="INFO")

    # Run all checks
    results = check_all_datasets()

    # Exit with error if any failed
    if not all(r.get("passed", False) for r in results.values()):
        exit(1)

