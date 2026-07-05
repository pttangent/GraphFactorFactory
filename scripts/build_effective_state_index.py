from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def build_index(edges_path: Path, output_path: Path) -> pd.DataFrame:
    parquet = pq.ParquetFile(edges_path)
    rows: list[dict] = []
    columns = ["decision_time", "layer_id", "src_id", "dst_id", "weight", "directed", "lag_bars"]
    for row_group in range(parquet.num_row_groups):
        table = parquet.read_row_group(row_group, columns=columns)
        timestamp = table["decision_time"][0].as_py()
        layer = table["layer_id"].to_numpy(zero_copy_only=False).astype(np.int16)
        src = table["src_id"].to_numpy(zero_copy_only=False).astype(np.int32)
        dst = table["dst_id"].to_numpy(zero_copy_only=False).astype(np.int32)
        weight = table["weight"].to_numpy(zero_copy_only=False).astype(np.float32)
        directed = table["directed"].to_numpy(zero_copy_only=False).astype(np.uint8)
        lag = table["lag_bars"].to_numpy(zero_copy_only=False).astype(np.int16)
        order = np.lexsort((lag, directed, weight.view(np.int32), dst, src, layer))
        digest = hashlib.blake2b(digest_size=16)
        for values in (layer[order], src[order], dst[order], weight[order], directed[order], lag[order]):
            digest.update(values.tobytes())
        rows.append({
            "row_group": row_group,
            "decision_time": timestamp,
            "edge_count": len(src),
            "graph_state_hash": digest.hexdigest(),
        })
    result = pd.DataFrame(rows).sort_values("decision_time").reset_index(drop=True)
    result["graph_state_changed"] = result["graph_state_hash"].ne(result["graph_state_hash"].shift())
    result["effective_state_index"] = result["graph_state_state_changed"].cumsum().astype(int) - 1
    result["carry_forward_count"] = result.groupby("effective_state_index").cumcount()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("edges", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_index(args.edges, args.output)


if __name__ == "__main__":
    main()
