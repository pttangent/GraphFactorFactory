#!/usr/bin/env python3
"""Small, source-aware checkpoint helpers for partitioned P2 stages."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    stat = source.stat()
    return {
        "path": str(source.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def optional_file_fingerprint(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    source = Path(path)
    return file_fingerprint(source) if source.exists() else None


def small_file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(destination) + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def stage_checkpoint_valid(
    manifest_path: str | Path,
    *,
    stage: str,
    contract_version: str,
    inputs: dict[str, Any],
    config: dict[str, Any],
    output_path: str | Path,
) -> bool:
    payload = read_json(manifest_path)
    if not payload:
        return False
    if payload.get("stage") != stage:
        return False
    if payload.get("stage_contract_version") != contract_version:
        return False
    if payload.get("inputs") != inputs or payload.get("config") != config:
        return False
    status = payload.get("status")
    output = Path(output_path)
    if status == "complete":
        return int(payload.get("output_rows", 0)) > 0 and output.exists()
    if status == "empty":
        return int(payload.get("output_rows", 0)) == 0 and not output.exists()
    return False
