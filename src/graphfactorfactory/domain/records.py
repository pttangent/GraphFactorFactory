from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceFingerprint:
    paths: tuple[str, ...]
    total_bytes: int
    total_rows: int
    schema_sha256: str
    file_metadata_sha256: str
    factor_semantics_versions: tuple[str, ...]
    trade_semantics_versions: tuple[str, ...]


@dataclass(frozen=True)
class BuildResult:
    root: Path
    manifest_path: Path
    catalog_path: Path
    edge_rows: int
    node_feature_rows: int
    snapshot_rows: int
    label_rows: int
