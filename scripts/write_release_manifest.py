"""Write an immutable manifest for one wheel and source distribution."""

from __future__ import annotations

import argparse
import email
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_identity(directory: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    """Return the version and identity for exactly one wheel and sdist."""
    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("release input must contain exactly one wheel and one source distribution")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError("release wheel has no unique METADATA file")
        version = email.message_from_bytes(archive.read(names[0])).get("Version")
    if not version:
        raise ValueError("release wheel has no version")
    return version, {
        "wheel": {
            "filename": wheels[0].name,
            "sha256": sha256(wheels[0]),
            "size": wheels[0].stat().st_size,
        },
        "sdist": {
            "filename": sdists[0].name,
            "sha256": sha256(sdists[0]),
            "size": sdists[0].stat().st_size,
        },
    }


def release_manifest(directory: Path, *, version: str, commit: str) -> dict[str, Any]:
    """Return a validated release manifest."""
    artifact_version, artifacts = artifact_identity(directory)
    if artifact_version != version:
        raise ValueError("artifact version differs from the requested release version")
    if not COMMIT_PATTERN.fullmatch(commit):
        raise ValueError("release commit is not a full lowercase SHA")
    return {
        "schema_version": 1,
        "distribution": "ml4t-diagnostic",
        "version": version,
        "commit": commit,
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = release_manifest(args.dist, version=args.version, commit=args.commit)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
