from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p2_p0_canonical_direct import DatePart, process_date


def test_canonical_date_single_pass_writes_final_factors_and_resumes(tmp_path: Path):
    date = "2026-01-02"
    date_dir = tmp_path / "p0" / f"date={date}"
    date_dir.mkdir(parents=True)
    times = pd.to_datetime(
        ["2026-01-02T14:30:00Z"] * 4 + ["2026-01-02T14:35:00Z"] * 4,
        utc=True,
    )
    edges = pd.DataFrame(
        {
            "decision_time": times,
            "window_end": times,
            "layer_id": [1, 1, 2, 2] * 2,
            "lookback_minutes": [5, 5, 15, 15] * 2,
            "src_id": [1, 2, 1, 3] * 2,
            "dst_id": [2, 3, 3, 2] * 2,
            "weight": [0.2, 0.3, -0.2, 0.4] * 2,
        }
    )
    edge_path = date_dir / "edges.parquet"
    edges.to_parquet(edge_path, index=False, row_group_size=3)

    labels = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(
                [
                    "2026-01-02T14:30:00Z",
                    "2026-01-02T14:30:00Z",
                    "2026-01-02T14:30:00Z",
                    "2026-01-02T14:35:00Z",
                    "2026-01-02T14:35:00Z",
                    "2026-01-02T14:35:00Z",
                ],
                utc=True,
            ),
            "symbol_id": [1, 2, 3, 1, 2, 3],
            "label_entry_time": pd.to_datetime(
                ["2026-01-02T14:30:00Z"] * 3 + ["2026-01-02T14:35:00Z"] * 3,
                utc=True,
            ),
            "label_entry_time_5m": pd.to_datetime(
                ["2026-01-02T14:30:00Z"] * 3 + ["2026-01-02T14:35:00Z"] * 3,
                utc=True,
            ),
            "label_exit_time_5m": pd.to_datetime(
                ["2026-01-02T14:35:00Z"] * 3 + ["2026-01-02T14:40:00Z"] * 3,
                utc=True,
            ),
            "label_5m": [0.01, 0.02, -0.01, 0.03, -0.02, 0.01],
        }
    )
    labels.to_parquet(date_dir / "labels.parquet", index=False)

    out_root = tmp_path / "out"
    result = process_date(
        DatePart(date, edge_path),
        str(tmp_path / "p0"),
        str(out_root),
        ["5m"],
        "5m",
        None,
        None,
        {"node", "spillover", "graph"},
        False,
        None,
        2,
        0.0,
        1,
    )
    assert result["status"] == "complete"
    assert not (tmp_path / "p0_alpha_shards").exists()
    assert (out_root / "p0_node_features" / f"date={date}" / "layer_id=1" / "scale=5m" / "p0_node_features.parquet").exists()
    assert (out_root / "p0_edge_spillover" / f"date={date}" / "layer_id=2" / "scale=15m" / "p0_edge_spillover_features.parquet").exists()
    assert (out_root / "p0_graph_state" / f"date={date}" / "layer_id=1" / "scale=5m" / "p0_graph_state_features.parquet").exists()

    resumed = process_date(
        DatePart(date, edge_path),
        str(tmp_path / "p0"),
        str(out_root),
        ["5m"],
        "5m",
        None,
        None,
        {"node", "spillover", "graph"},
        True,
        None,
        2,
        0.0,
        1,
    )
    assert resumed["status"] == "skipped"
    manifest = json.loads(
        (out_root / "p0_direct_status" / f"date={date}" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["input_mode"] == "one_date_one_scan_no_physical_alpha_shards"


def test_full_runner_no_longer_materializes_month_alpha_shards():
    text = (SCRIPTS / "run_full_alpha_streaming_6m.py").read_text(encoding="utf-8")
    assert "LOCAL_P0_SHARDS" not in text
    assert "shard_p0_edges_by_layer_scale.py" not in text
    assert '"--layers", "3,6,8,9,11"' not in text
    assert '"--scales", "15m,30m"' not in text
    assert "archive_month_outputs(month)" in text
