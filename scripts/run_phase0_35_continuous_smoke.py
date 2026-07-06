from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from graphfactorfactory.application.graph import MultilayerGraphBuilder
from graphfactorfactory.domain.config import BuildConfig
from graphfactorfactory.domain.layers import LAYER_SCALES, MAX_LOOKBACK_MINUTES

FRAME = None
UNIVERSE = None
SYMBOLS = None
CONFIG = None
OUT_DIR = None


def _load_window(path: str, trade_date: str, universe: list[str]):
    start = pd.Timestamp(f"{trade_date} 13:30:00+00:00")
    end = pd.Timestamp(f"{trade_date} 14:29:00+00:00")
    columns = {"timestamp", "available_time", "symbol", "ret_1m", "log_ret_1m", "signed_dollar_flow"}
    for item in LAYER_SCALES:
        columns.update(item.layer.columns)
    table = pq.read_table(
        path,
        columns=sorted(columns),
        filters=[("timestamp", ">=", start.to_pydatetime()), ("timestamp", "<=", end.to_pydatetime()), ("available_time", "<=", end.to_pydatetime())],
    )
    frame = table.to_pandas()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["available_time"] = pd.to_datetime(frame["available_time"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    return frame[frame.symbol.isin(universe)].copy()


def _select_universe(path: str, trade_date: str, limit: int):
    start = pd.Timestamp(f"{trade_date} 13:30:00+00:00")
    end = pd.Timestamp(f"{trade_date} 14:29:00+00:00")
    table = pq.read_table(path, columns=["timestamp", "available_time", "symbol"], filters=[("timestamp", ">=", start.to_pydatetime()), ("timestamp", "<=", end.to_pydatetime()), ("available_time", "<=", end.to_pydatetime())])
    frame = table.to_pandas()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    counts = frame.groupby(frame.symbol.astype(str)).timestamp.nunique().sort_values(ascending=False)
    return counts[counts >= 55].index.astype(str).tolist()[:limit]


def _init_worker(path: str, trade_date: str, universe: list[str], out_dir: str):
    global FRAME, UNIVERSE, SYMBOLS, CONFIG, OUT_DIR
    UNIVERSE = universe
    FRAME = _load_window(path, trade_date, universe)
    SYMBOLS = pd.DataFrame({"symbol_id": range(len(universe)), "symbol": universe})
    CONFIG = BuildConfig(frequency="1min", graph_window_minutes=30, graph_step_minutes=1, minimum_window_points=3, store_labels=False)
    OUT_DIR = Path(out_dir)


def _run_minute(decision_iso: str):
    decision = pd.Timestamp(decision_iso)
    key = decision.strftime("%H%M")
    snapshot_path = OUT_DIR / f"minute_{key}_snapshots.parquet"
    edge_path = OUT_DIR / f"minute_{key}_edges.parquet"
    node_path = OUT_DIR / f"minute_{key}_nodes.parquet"
    done_path = OUT_DIR / f"minute_{key}.done.json"
    if done_path.exists() and snapshot_path.exists() and edge_path.exists() and node_path.exists():
        return {"decision_time": decision_iso, "status": "resumed"}

    started = time.perf_counter()
    window_start = decision - pd.Timedelta(minutes=MAX_LOOKBACK_MINUTES)
    window = FRAME[(FRAME.available_time <= decision) & (FRAME.timestamp <= decision) & (FRAME.timestamp > window_start)]
    builder = MultilayerGraphBuilder(CONFIG, SYMBOLS)
    products = builder.build_snapshot(window, decision)
    products.snapshots.to_parquet(snapshot_path, index=False)
    products.edges.to_parquet(edge_path, index=False)
    products.node_features.to_parquet(node_path, index=False)
    done_path.write_text(json.dumps({"decision_time": decision_iso, "snapshot_rows": len(products.snapshots), "edge_rows": len(products.edges), "node_rows": len(products.node_features), "elapsed_seconds": time.perf_counter() - started}, indent=2))
    return {"decision_time": decision_iso, "status": "computed"}


def run_day(path: str, trade_date: str, root: Path, workers: int, universe_limit: int):
    day_dir = root / trade_date
    day_dir.mkdir(parents=True, exist_ok=True)
    universe_file = day_dir / "universe.json"
    universe = json.loads(universe_file.read_text()) if universe_file.exists() else _select_universe(path, trade_date, universe_limit)
    universe_file.write_text(json.dumps(universe, indent=2))
    decisions = pd.date_range(f"{trade_date} 13:30:00+00:00", periods=60, freq="1min")
    pending = [d.isoformat() for d in decisions if not (day_dir / f"minute_{d.strftime('%H%M')}.done.json").exists()]
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(path, trade_date, universe, str(day_dir))) as pool:
            futures = {pool.submit(_run_minute, d): d for d in pending}
            for future in as_completed(futures):
                print(trade_date, future.result(), flush=True)

    snapshots = pd.concat([pd.read_parquet(p) for p in sorted(day_dir.glob("minute_*_snapshots.parquet"))], ignore_index=True)
    edges = pd.concat([pd.read_parquet(p) for p in sorted(day_dir.glob("minute_*_edges.parquet"))], ignore_index=True)
    snapshots.to_parquet(day_dir / "continuous_60m_snapshots.parquet", index=False)
    edges.to_parquet(day_dir / "continuous_60m_edges.parquet", index=False)
    report = {"trade_date": trade_date, "decision_minutes": 60, "universe_count": len(universe), "workers": workers, "checkpoint_minutes": len(list(day_dir.glob("minute_*.done.json"))), "snapshot_rows": len(snapshots), "edge_rows": len(edges), "elapsed_seconds_this_invocation": time.perf_counter() - started, "passed": len(list(day_dir.glob("minute_*.done.json"))) == 60}
    (day_dir / "continuous_60m_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, help="Folder containing date=YYYY-MM-DD/data.parquet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dates", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--universe-limit", type=int, default=600)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    reports = []
    for trade_date in args.dates:
        path = str(Path(args.input_root) / f"date={trade_date}" / "data.parquet")
        reports.append(run_day(path, trade_date, root, args.workers, args.universe_limit))
    aggregate = {"days": reports, "all_days_passed": all(r["passed"] for r in reports)}
    (root / "continuous_60m_aggregate_report.json").write_text(json.dumps(aggregate, indent=2))
    if not aggregate["all_days_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
