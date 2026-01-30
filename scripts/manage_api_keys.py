#!/usr/bin/env python3
"""API Key Management for Semantic Search Service.

This script manages API keys for the production service:
- Generate new API keys
- Store keys securely in GCP Secret Manager
- Rotate keys
- List active keys

Usage:
    python scripts/manage_api_keys.py generate --name "client-app-1"
    python scripts/manage_api_keys.py list
    python scripts/manage_api_keys.py revoke --key-id "abc123"
    python scripts/manage_api_keys.py rotate --key-id "abc123"
"""

import argparse
import hashlib
import json
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Configuration
SECRET_NAME = "semantic-kd-api-keys"
PROJECT_ID = "plotpointe"
KEYS_FILE = Path("./artifacts/api_keys/keys.json")


def generate_api_key(prefix: str = "sk") -> str:
    """Generate a secure API key.

    Format: sk_live_<32 random chars>
    """
    random_part = secrets.token_urlsafe(24)  # 32 chars
    return f"{prefix}_live_{random_part}"


def hash_api_key(key: str, salt: str = "") -> str:
    """Create PBKDF2-HMAC-SHA256 hash of API key for secure storage.

    Falls back to plain SHA256 if no salt provided (backward compat).
    """
    if salt:
        return hashlib.pbkdf2_hmac(
            "sha256", key.encode(), salt.encode(), 100_000
        ).hex()
    return hashlib.sha256(key.encode()).hexdigest()


def load_keys() -> Dict:
    """Load keys from local file."""
    if KEYS_FILE.exists():
        with open(KEYS_FILE) as f:
            return json.load(f)
    return {"keys": [], "revoked": []}


def save_keys(data: Dict) -> None:
    """Save keys to local file with restrictive permissions."""
    KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(KEYS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    # Restrict to owner-only read/write
    KEYS_FILE.chmod(0o600)


def sync_to_secret_manager(data: Dict) -> None:
    """Sync API key hashes and salts to GCP Secret Manager."""
    # Extract hashes and salts for the service to use
    active_entries = []
    for k in data["keys"]:
        if k.get("active", True):
            entry = {"hash": k["hash"]}
            if "salt" in k:
                entry["salt"] = k["salt"]
            active_entries.append(entry)

    # Include flat hash list for backward compat with existing middleware
    active_hashes = [e["hash"] for e in active_entries]
    secret_data = json.dumps({
        "api_key_hashes": active_hashes,
        "api_key_entries": active_entries,
    })

    try:
        # Check if secret exists
        result = subprocess.run(
            ["gcloud", "secrets", "describe", SECRET_NAME, "--project", PROJECT_ID],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            # Create secret
            subprocess.run(
                ["gcloud", "secrets", "create", SECRET_NAME, "--project", PROJECT_ID, "--replication-policy", "automatic"],
                check=True,
                capture_output=True
            )
            print(f"Created secret: {SECRET_NAME}")

        # Add new version
        subprocess.run(
            ["gcloud", "secrets", "versions", "add", SECRET_NAME, "--project", PROJECT_ID, "--data-file=-"],
            input=secret_data,
            text=True,
            check=True,
            capture_output=True
        )
        print(f"Synced {len(active_hashes)} API key hashes to Secret Manager")

    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not sync to Secret Manager: {e}")
        print("Keys saved locally only.")


def generate_key(name: str, description: str = "") -> Dict:
    """Generate a new API key with salted hash."""
    data = load_keys()

    # Generate key and salt
    api_key = generate_api_key()
    salt = secrets.token_hex(16)
    key_hash = hash_api_key(api_key, salt)
    key_id = secrets.token_hex(4)  # Short ID for reference

    key_record = {
        "id": key_id,
        "name": name,
        "description": description,
        "hash": key_hash,
        "salt": salt,
        "prefix": api_key[:12] + "...",  # Store prefix for identification
        "created_at": datetime.now(timezone.utc).isoformat(),
        "active": True,
    }

    data["keys"].append(key_record)
    save_keys(data)
    sync_to_secret_manager(data)

    return {
        "key_id": key_id,
        "api_key": api_key,  # Only shown once!
        "name": name,
        "created_at": key_record["created_at"],
    }


def list_keys() -> List[Dict]:
    """List all API keys (without the actual key values)."""
    data = load_keys()
    return [
        {
            "id": k["id"],
            "name": k["name"],
            "prefix": k["prefix"],
            "created_at": k["created_at"],
            "active": k.get("active", True),
        }
        for k in data["keys"]
    ]


def revoke_key(key_id: str) -> bool:
    """Revoke an API key."""
    data = load_keys()

    for key in data["keys"]:
        if key["id"] == key_id:
            key["active"] = False
            key["revoked_at"] = datetime.now(timezone.utc).isoformat()
            data["revoked"].append(key["hash"])
            save_keys(data)
            sync_to_secret_manager(data)
            return True

    return False


def rotate_key(key_id: str) -> Optional[Dict]:
    """Rotate an API key (revoke old, generate new with same name)."""
    data = load_keys()

    old_key = None
    for key in data["keys"]:
        if key["id"] == key_id:
            old_key = key
            break

    if not old_key:
        return None

    # Revoke old key
    revoke_key(key_id)

    # Generate new key with same name
    return generate_key(old_key["name"], f"Rotated from {key_id}")


def get_env_var_format() -> str:
    """Get API key hashes in environment variable format for Cloud Run."""
    data = load_keys()
    active_hashes = [k["hash"] for k in data["keys"] if k.get("active", True)]
    return json.dumps(active_hashes)


def main():
    parser = argparse.ArgumentParser(description="API Key Management")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate new API key")
    gen_parser.add_argument("--name", required=True, help="Name/identifier for the key")
    gen_parser.add_argument("--description", default="", help="Description")

    # List command
    subparsers.add_parser("list", help="List all API keys")

    # Revoke command
    revoke_parser = subparsers.add_parser("revoke", help="Revoke an API key")
    revoke_parser.add_argument("--key-id", required=True, help="Key ID to revoke")

    # Rotate command
    rotate_parser = subparsers.add_parser("rotate", help="Rotate an API key")
    rotate_parser.add_argument("--key-id", required=True, help="Key ID to rotate")

    # Export command
    subparsers.add_parser("export-env", help="Export hashes for Cloud Run env var")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "generate":
        result = generate_key(args.name, args.description)
        print("\n" + "=" * 60)
        print("NEW API KEY GENERATED")
        print("=" * 60)
        print(f"Key ID:    {result['key_id']}")
        print(f"Name:      {result['name']}")
        print(f"API Key:   {result['api_key']}")
        print("=" * 60)
        print("IMPORTANT: Save this API key securely!")
        print("It will NOT be shown again.")
        print("=" * 60 + "\n")

    elif args.command == "list":
        keys = list_keys()
        if not keys:
            print("No API keys found.")
        else:
            print(f"\n{'ID':<10} {'Name':<20} {'Prefix':<18} {'Active':<8} {'Created'}")
            print("-" * 80)
            for k in keys:
                status = "Yes" if k["active"] else "No"
                created = k["created_at"][:10]
                print(f"{k['id']:<10} {k['name']:<20} {k['prefix']:<18} {status:<8} {created}")

    elif args.command == "revoke":
        if revoke_key(args.key_id):
            print(f"Key {args.key_id} has been revoked.")
        else:
            print(f"Key {args.key_id} not found.")
            sys.exit(1)

    elif args.command == "rotate":
        result = rotate_key(args.key_id)
        if result:
            print("\n" + "=" * 60)
            print("API KEY ROTATED")
            print("=" * 60)
            print(f"Old Key ID: {args.key_id} (revoked)")
            print(f"New Key ID: {result['key_id']}")
            print(f"API Key:    {result['api_key']}")
            print("=" * 60)
            print("IMPORTANT: Update your applications with the new key!")
            print("=" * 60 + "\n")
        else:
            print(f"Key {args.key_id} not found.")
            sys.exit(1)

    elif args.command == "export-env":
        print(get_env_var_format())


if __name__ == "__main__":
    main()
