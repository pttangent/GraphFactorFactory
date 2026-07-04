from __future__ import annotations

import pyarrow as pa

EDGE_SCHEMA = pa.schema([
    ("decision_time", pa.timestamp("ns", tz="UTC")),
    ("window_start", pa.timestamp("ns", tz="UTC")),
    ("window_end", pa.timestamp("ns", tz="UTC")),
    ("layer_id", pa.int16()),
    ("src_id", pa.int32()),
    ("dst_id", pa.int32()),
    ("weight", pa.float32()),
    ("src_rank", pa.int16()),
    ("dst_rank", pa.int16()),
    ("directed", pa.bool_()),
    ("lag_bars", pa.int16()),
    ("window_points", pa.int16()),
    ("vector_dimension", pa.int16()),
])
NODE_SCHEMA = pa.schema([
    ("decision_time", pa.timestamp("ns", tz="UTC")),
    ("layer_id", pa.int16()),
    ("symbol_id", pa.int32()),
    ("degree", pa.int16()),
    ("strength", pa.float32()),
    ("core_z", pa.float32()),
    ("neighbor_reversal", pa.float32()),
    ("neighbor_signed_flow", pa.float32()),
    ("layer_participation", pa.float32()),
])
SNAPSHOT_SCHEMA = pa.schema([
    ("decision_time", pa.timestamp("ns", tz="UTC")),
    ("window_start", pa.timestamp("ns", tz="UTC")),
    ("window_end", pa.timestamp("ns", tz="UTC")),
    ("layer_id", pa.int16()),
    ("universe_count", pa.int32()),
    ("active_nodes", pa.int32()),
    ("edge_count", pa.int32()),
    ("mean_degree", pa.float32()),
    ("mean_strength", pa.float32()),
    ("window_points", pa.int16()),
    ("vector_dimension", pa.int16()),
    ("lsh_bits", pa.int16()),
    ("used_columns", pa.string()),
    ("elapsed_ms_total_snapshot", pa.int32()),
])
