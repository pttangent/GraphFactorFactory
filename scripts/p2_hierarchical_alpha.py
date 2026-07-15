#!/usr/bin/env python3
"""Full-panel hierarchical P2 alpha lab.

Research chain:
    stock reversal -> theme reversal -> node relative-to-theme reversal
    -> core/periphery -> lifecycle -> cross-theme network increment

The worker grain is (date, layer, scale, level, time chunk). Work is
largest-first, bounded, source-fingerprinted, resumable, and memory-capped for
a 24-core/128GB Windows host. The production default discovers every common
P1 Layer-Scale partition for the requested dates. It performs no date,
snapshot, row-group, or layer sampling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ARROW_NUM_THREADS",
    "POLARS_MAX_THREADS",
):
    os.environ.setdefault(_name, "1")

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from p2_checkpoint import file_fingerprint, read_json, write_json_atomic
from p2_parallel_runtime import bounded_process_map

CONTRACT = "p2-hierarchical-alpha-v6-full"
HORIZONS = ("5m", "15m", "30m")
MIN_NODE = 200
MIN_THEME = 20
PLACEBO_SEEDS = (101, 202, 303)


def _sql_lit(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _checkpoint_path(root: Path, task: dict[str, Any]) -> Path:
    key = "|".join(
        str(task.get(k, ""))
        for k in ("kind", "date", "layer_id", "scale", "level", "chunk")
    )
    return root / f"{task['kind']}-{hashlib.sha256(key.encode()).hexdigest()[:20]}.json"


def _label_source(root: Path, date: str) -> Path:
    candidates = [
        root / f"{date}.parquet",
        root / f"date={date}" / "raw_1m.parquet",
        root / date / "raw_1m.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"raw_1m not found for {date} under {root}")


def _prepare_labels_one(
    date: str,
    source: Path,
    symbols: Path,
    output: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Build exact trailing/forward returns from raw one-minute closes.

    ``bar_end`` is the information-availability timestamp. Missing exact
    endpoints remain null instead of being forward-filled. The past window
    ends at the decision time and every future window starts at it, so past
    and future returns do not overlap.
    """
    manifest = output.with_suffix(".manifest.json")
    inputs = {
        "raw_1m": file_fingerprint(source),
        "symbols": file_fingerprint(symbols),
    }
    config = {
        "price_time": "bar_end",
        "price": "close",
        "past_minutes": 15,
        "future_minutes": [5, 15, 30],
        "endpoint_policy": "exact_no_fill",
    }
    old = read_json(manifest)
    if (
        not force
        and output.exists()
        and old
        and old.get("contract") == CONTRACT
        and old.get("inputs") == inputs
        and old.get("config") == config
    ):
        return {
            "date": date,
            "status": "reused",
            "path": str(output),
            "rows": old.get("rows", 0),
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".tmp")
    temporary.unlink(missing_ok=True)
    con = duckdb.connect()
    try:
        con.execute("PRAGMA threads=4")
        con.execute("PRAGMA memory_limit='3GB'")
        con.execute(
            f"""
            CREATE TEMP TABLE bars AS
            SELECT symbol,
                   CAST(bar_end AS TIMESTAMP) AS decision_time,
                   CAST(close AS DOUBLE) AS px
            FROM read_parquet({_sql_lit(source)})
            WHERE close > 0
            QUALIFY row_number() OVER (
                PARTITION BY symbol, bar_end ORDER BY sequence DESC
            ) = 1
            """
        )
        con.execute(
            f"CREATE TEMP VIEW symbol_map AS "
            f"SELECT symbol_id, symbol FROM read_parquet({_sql_lit(symbols)})"
        )
        target = str(temporary).replace("'", "''")
        con.execute(
            f"""
            COPY (
              SELECT s.symbol_id,
                     c.decision_time,
                     c.px / p.px - 1 AS past_ret_15m,
                     f5.px / c.px - 1 AS future_ret_5m,
                     f15.px / c.px - 1 AS future_ret_15m,
                     f30.px / c.px - 1 AS future_ret_30m
              FROM bars c
              JOIN symbol_map s USING (symbol)
              LEFT JOIN bars p
                ON p.symbol = c.symbol
               AND p.decision_time = c.decision_time - INTERVAL 15 MINUTE
              LEFT JOIN bars f5
                ON f5.symbol = c.symbol
               AND f5.decision_time = c.decision_time + INTERVAL 5 MINUTE
              LEFT JOIN bars f15
                ON f15.symbol = c.symbol
               AND f15.decision_time = c.decision_time + INTERVAL 15 MINUTE
              LEFT JOIN bars f30
                ON f30.symbol = c.symbol
               AND f30.decision_time = c.decision_time + INTERVAL 30 MINUTE
              ORDER BY c.decision_time, s.symbol_id
            ) TO '{target}' (
              FORMAT PARQUET,
              COMPRESSION ZSTD,
              ROW_GROUP_SIZE 250000
            )
            """
        )
    finally:
        con.close()
    os.replace(temporary, output)
    parquet = pq.ParquetFile(output)
    try:
        rows = parquet.metadata.num_rows
    finally:
        parquet.close()
    write_json_atomic(
        manifest,
        {
            "contract": CONTRACT,
            "status": "complete",
            "date": date,
            "inputs": inputs,
            "config": config,
            "rows": rows,
            "output": str(output),
        },
    )
    return {"date": date, "status": "computed", "path": str(output), "rows": rows}


def _parse_time_expr(column: str) -> str:
    return (
        f"strptime(regexp_extract({column}, '^ts=([^|]+)', 1), "
        "'%Y-%m-%dT%H%M%S_%f')"
    )


def _metric_sql(
    table: str,
    score: str,
    target: str,
    where: str,
    min_sample: int,
    signal: str,
    horizon: str,
    unit: str,
) -> str:
    """Generate per-snapshot rank IC and top-minus-bottom quintile spread."""
    return f"""
    WITH valid AS (
      SELECT decision_time,
             CAST(({score}) AS DOUBLE) AS score,
             CAST(({target}) AS DOUBLE) AS y
      FROM {table}
      WHERE ({where})
        AND isfinite(({score}))
        AND isfinite(({target}))
    ), ranked AS (
      SELECT *,
             percent_rank() OVER (
               PARTITION BY decision_time ORDER BY score
             ) AS score_rank,
             percent_rank() OVER (
               PARTITION BY decision_time ORDER BY y
             ) AS target_rank
      FROM valid
    )
    SELECT decision_time,
           '{signal}' AS signal,
           '{horizon}' AS horizon,
           '{unit}' AS unit,
           count(*) AS sample_count,
           corr(score_rank, target_rank) AS rank_ic,
           10000.0 * (
             avg(y) FILTER (WHERE score_rank >= 0.8)
             - avg(y) FILTER (WHERE score_rank <= 0.2)
           ) AS spread_bps
    FROM ranked
    GROUP BY decision_time
    HAVING count(*) >= {int(min_sample)}
    """


def _configure_duckdb(task: dict[str, Any]) -> tuple[duckdb.DuckDBPyConnection, Path]:
    con = duckdb.connect()
    con.execute("PRAGMA threads=1")
    memory_gb = max(0.5, float(task.get("worker_memory_gb", 2.0)))
    con.execute(f"PRAGMA memory_limit='{memory_gb:.2f}GB'")
    temporary = Path(task["temp_root"]) / f"worker-{os.getpid()}"
    temporary.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"PRAGMA temp_directory='{str(temporary).replace(chr(39), chr(39) * 2)}'"
    )
    return con, temporary


def _task_fingerprint(task: dict[str, Any]) -> dict[str, Any]:
    keys = ("membership", "labels", "relation", "temporal")
    return {key: file_fingerprint(task[key]) for key in keys if task.get(key)}


def _cached(
    task: dict[str, Any], checkpoint: Path
) -> tuple[dict[str, Any] | None, dict[str, Any], str]:
    old = read_json(checkpoint)
    inputs = _task_fingerprint(task)
    scope = _hash(
        {
            key: value
            for key, value in task.items()
            if key not in {"checkpoint", "temp_root", "worker_memory_gb"}
        }
    )
    if (
        old
        and old.get("contract") == CONTRACT
        and old.get("status") == "complete"
        and old.get("inputs") == inputs
        and old.get("scope") == scope
    ):
        return old, inputs, scope
    return None, inputs, scope


def _process_stock(task: dict[str, Any]) -> dict[str, Any]:
    checkpoint = Path(task["checkpoint"])
    old, inputs, scope = _cached(task, checkpoint)
    if old:
        return {
            "status": "reused",
            "checkpoint": str(checkpoint),
            "record_count": len(old.get("records", [])),
            "elapsed_sec": 0.0,
        }
    started = time.time()
    con, temporary = _configure_duckdb(task)
    try:
        con.execute(
            f"CREATE TEMP VIEW stock AS "
            f"SELECT * FROM read_parquet({_sql_lit(task['labels'])}) "
            f"WHERE decision_time >= CAST({_sql_lit(task['time_lo'])} AS TIMESTAMP) "
            f"AND decision_time < CAST({_sql_lit(task['time_hi'])} AS TIMESTAMP)"
        )
        frames = []
        for horizon in HORIZONS:
            frames.append(
                con.execute(
                    _metric_sql(
                        "stock",
                        "-past_ret_15m",
                        f"future_ret_{horizon}",
                        "past_ret_15m IS NOT NULL",
                        MIN_NODE,
                        "stock_reversal",
                        horizon,
                        "node_raw",
                    )
                ).fetchdf()
            )
    finally:
        con.close()
        shutil.rmtree(temporary, ignore_errors=True)
    records = pd.concat(frames, ignore_index=True).to_dict("records") if frames else []
    payload = {
        "contract": CONTRACT,
        "status": "complete",
        "inputs": inputs,
        "scope": scope,
        "records": records,
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_json_atomic(checkpoint, payload)
    return {
        "status": "computed",
        "checkpoint": str(checkpoint),
        "record_count": len(records),
        "elapsed_sec": payload["elapsed_sec"],
    }


def _process_partition(task: dict[str, Any]) -> dict[str, Any]:
    checkpoint = Path(task["checkpoint"])
    old, inputs, scope = _cached(task, checkpoint)
    if old:
        return {
            "status": "reused",
            "checkpoint": str(checkpoint),
            "record_count": len(old.get("records", [])),
            "elapsed_sec": 0.0,
        }

    started = time.time()
    con, temporary = _configure_duckdb(task)
    try:
        lo_token, hi_token = task["theme_lo"], task["theme_hi"]
        parsed_time = _parse_time_expr("theme_id")
        con.execute(
            f"""
            CREATE TEMP TABLE m AS
            SELECT {parsed_time} AS decision_time,
                   hash(theme_id) AS theme_key,
                   member_id,
                   CAST(rank_in_theme AS INTEGER) AS rank_in_theme,
                   CAST(core_score AS DOUBLE) AS core_score
            FROM read_parquet({_sql_lit(task['membership'])})
            WHERE level = {_sql_lit(task['level'])}
              AND theme_id >= {_sql_lit(lo_token)}
              AND theme_id < {_sql_lit(hi_token)}
            """
        )
        con.execute(
            f"""
            CREATE TEMP TABLE j AS
            SELECT m.*,
                   l.past_ret_15m,
                   l.future_ret_5m,
                   l.future_ret_15m,
                   l.future_ret_30m,
                   count(*) OVER (
                     PARTITION BY m.decision_time, m.theme_key
                   ) AS theme_size
            FROM m
            JOIN read_parquet({_sql_lit(task['labels'])}) l
              ON l.decision_time = m.decision_time
             AND l.symbol_id = m.member_id
            WHERE l.past_ret_15m IS NOT NULL
            """
        )
        con.execute(
            """
            CREATE TEMP TABLE theme AS
            SELECT decision_time,
                   theme_key,
                   count(*) AS member_count,
                   sum(past_ret_15m) AS past_sum,
                   count(past_ret_15m) AS past_count,
                   avg(past_ret_15m) AS theme_past,
                   sum(future_ret_5m) AS future_sum_5m,
                   count(future_ret_5m) AS future_count_5m,
                   avg(future_ret_5m) AS theme_future_5m,
                   sum(future_ret_15m) AS future_sum_15m,
                   count(future_ret_15m) AS future_count_15m,
                   avg(future_ret_15m) AS theme_future_15m,
                   sum(future_ret_30m) AS future_sum_30m,
                   count(future_ret_30m) AS future_count_30m,
                   avg(future_ret_30m) AS theme_future_30m,
                   avg(past_ret_15m) FILTER (
                     WHERE rank_in_theme <= greatest(1, ceil(theme_size * 0.2))
                   ) AS core_past,
                   avg(past_ret_15m) FILTER (
                     WHERE rank_in_theme > floor(theme_size * 0.8)
                   ) AS periphery_past,
                   avg(future_ret_5m) FILTER (
                     WHERE rank_in_theme <= greatest(1, ceil(theme_size * 0.2))
                   ) AS core_future_5m,
                   avg(future_ret_15m) FILTER (
                     WHERE rank_in_theme <= greatest(1, ceil(theme_size * 0.2))
                   ) AS core_future_15m,
                   avg(future_ret_30m) FILTER (
                     WHERE rank_in_theme <= greatest(1, ceil(theme_size * 0.2))
                   ) AS core_future_30m,
                   avg(future_ret_5m) FILTER (
                     WHERE rank_in_theme > floor(theme_size * 0.8)
                   ) AS periphery_future_5m,
                   avg(future_ret_15m) FILTER (
                     WHERE rank_in_theme > floor(theme_size * 0.8)
                   ) AS periphery_future_15m,
                   avg(future_ret_30m) FILTER (
                     WHERE rank_in_theme > floor(theme_size * 0.8)
                   ) AS periphery_future_30m
            FROM j
            GROUP BY decision_time, theme_key
            """
        )
        con.execute(
            """
            CREATE TEMP VIEW node AS
            SELECT j.*,
                   t.theme_past,
                   t.theme_future_5m,
                   t.theme_future_15m,
                   t.theme_future_30m,
                   CASE WHEN t.past_count > 1
                        THEN (t.past_sum - j.past_ret_15m) / (t.past_count - 1)
                   END AS loo_theme_past,
                   CASE WHEN j.future_ret_5m IS NOT NULL AND t.future_count_5m > 1
                        THEN (t.future_sum_5m - j.future_ret_5m) / (t.future_count_5m - 1)
                   END AS loo_theme_future_5m,
                   CASE WHEN j.future_ret_15m IS NOT NULL AND t.future_count_15m > 1
                        THEN (t.future_sum_15m - j.future_ret_15m) / (t.future_count_15m - 1)
                   END AS loo_theme_future_15m,
                   CASE WHEN j.future_ret_30m IS NOT NULL AND t.future_count_30m > 1
                        THEN (t.future_sum_30m - j.future_ret_30m) / (t.future_count_30m - 1)
                   END AS loo_theme_future_30m
            FROM j
            JOIN theme t USING (decision_time, theme_key)
            """
        )

        frames: list[pd.DataFrame] = []
        for horizon in HORIZONS:
            frames.extend(
                [
                    con.execute(
                        _metric_sql(
                            "node",
                            "-past_ret_15m",
                            f"future_ret_{horizon}",
                            f"future_ret_{horizon} IS NOT NULL",
                            MIN_NODE,
                            "stock_reversal_matched",
                            horizon,
                            "node_matched",
                        )
                    ).fetchdf(),
                    con.execute(
                        _metric_sql(
                            "node",
                            "-loo_theme_past",
                            f"future_ret_{horizon}",
                            f"theme_size >= 3 AND loo_theme_past IS NOT NULL "
                            f"AND future_ret_{horizon} IS NOT NULL",
                            MIN_NODE,
                            "theme_context_reversal_node",
                            horizon,
                            "node_theme_context_loo",
                        )
                    ).fetchdf(),
                    con.execute(
                        _metric_sql(
                            "theme",
                            "-theme_past",
                            f"theme_future_{horizon}",
                            "member_count >= 3",
                            MIN_THEME,
                            "theme_reversal",
                            horizon,
                            "theme_equal",
                        )
                    ).fetchdf(),
                    con.execute(
                        _metric_sql(
                            "node",
                            "-(past_ret_15m - loo_theme_past)",
                            f"future_ret_{horizon} - loo_theme_future_{horizon}",
                            f"theme_size >= 3 AND loo_theme_past IS NOT NULL "
                            f"AND loo_theme_future_{horizon} IS NOT NULL",
                            MIN_NODE,
                            "relative_to_theme_reversal",
                            horizon,
                            "node_theme_neutral_loo",
                        )
                    ).fetchdf(),
                    con.execute(
                        _metric_sql(
                            "node",
                            "-(past_ret_15m - loo_theme_past)",
                            f"future_ret_{horizon} - loo_theme_future_{horizon}",
                            "rank_in_theme <= greatest(1, ceil(theme_size * 0.2)) "
                            f"AND loo_theme_past IS NOT NULL "
                            f"AND loo_theme_future_{horizon} IS NOT NULL",
                            50,
                            "relative_reversal_core",
                            horizon,
                            "core_node_neutral_loo",
                        )
                    ).fetchdf(),
                    con.execute(
                        _metric_sql(
                            "node",
                            "-(past_ret_15m - loo_theme_past)",
                            f"future_ret_{horizon} - loo_theme_future_{horizon}",
                            "rank_in_theme > floor(theme_size * 0.8) "
                            f"AND loo_theme_past IS NOT NULL "
                            f"AND loo_theme_future_{horizon} IS NOT NULL",
                            50,
                            "relative_reversal_periphery",
                            horizon,
                            "periphery_node_neutral_loo",
                        )
                    ).fetchdf(),
                    con.execute(
                        _metric_sql(
                            "theme",
                            "-(core_past - periphery_past)",
                            f"core_future_{horizon} - periphery_future_{horizon}",
                            "core_past IS NOT NULL AND periphery_past IS NOT NULL",
                            MIN_THEME,
                            "core_periphery_gap_reversal",
                            horizon,
                            "theme_role_gap",
                        )
                    ).fetchdf(),
                ]
            )

        for seed in PLACEBO_SEEDS:
            role_table = f"random_role_{seed}"
            role_theme = f"random_role_theme_{seed}"
            random_base = f"random_node_base_{seed}"
            random_theme = f"random_theme_{seed}"
            random_node = f"random_node_{seed}"
            con.execute(
                f"""
                CREATE TEMP TABLE {role_table} AS
                SELECT j.*,
                       row_number() OVER (
                         PARTITION BY decision_time, theme_key
                         ORDER BY hash(member_id, theme_key, {seed})
                       ) AS placebo_rank
                FROM j
                """
            )
            con.execute(
                f"""
                CREATE TEMP TABLE {role_theme} AS
                SELECT decision_time,
                       theme_key,
                       count(*) AS member_count,
                       avg(past_ret_15m) FILTER (
                         WHERE placebo_rank <= greatest(1, ceil(theme_size * 0.2))
                       ) AS core_past,
                       avg(past_ret_15m) FILTER (
                         WHERE placebo_rank > floor(theme_size * 0.8)
                       ) AS periphery_past,
                       avg(future_ret_5m) FILTER (
                         WHERE placebo_rank <= greatest(1, ceil(theme_size * 0.2))
                       ) AS core_future_5m,
                       avg(future_ret_15m) FILTER (
                         WHERE placebo_rank <= greatest(1, ceil(theme_size * 0.2))
                       ) AS core_future_15m,
                       avg(future_ret_30m) FILTER (
                         WHERE placebo_rank <= greatest(1, ceil(theme_size * 0.2))
                       ) AS core_future_30m,
                       avg(future_ret_5m) FILTER (
                         WHERE placebo_rank > floor(theme_size * 0.8)
                       ) AS periphery_future_5m,
                       avg(future_ret_15m) FILTER (
                         WHERE placebo_rank > floor(theme_size * 0.8)
                       ) AS periphery_future_15m,
                       avg(future_ret_30m) FILTER (
                         WHERE placebo_rank > floor(theme_size * 0.8)
                       ) AS periphery_future_30m
                FROM {role_table}
                GROUP BY decision_time, theme_key
                """
            )
            con.execute(
                f"""
                CREATE TEMP TABLE {random_base} AS
                WITH slots AS (
                  SELECT decision_time,
                         theme_key,
                         row_number() OVER (
                           PARTITION BY decision_time
                           ORDER BY theme_key, rank_in_theme, member_id
                         ) AS allocation_rank
                  FROM j
                ), shuffled AS (
                  SELECT decision_time,
                         member_id,
                         past_ret_15m,
                         future_ret_5m,
                         future_ret_15m,
                         future_ret_30m,
                         row_number() OVER (
                           PARTITION BY decision_time
                           ORDER BY hash(member_id, decision_time, {seed})
                         ) AS allocation_rank
                  FROM j
                )
                SELECT s.decision_time,
                       s.theme_key,
                       x.* EXCLUDE (decision_time, allocation_rank)
                FROM slots s
                JOIN shuffled x USING (decision_time, allocation_rank)
                """
            )
            con.execute(
                f"""
                CREATE TEMP TABLE {random_theme} AS
                SELECT decision_time,
                       theme_key,
                       count(*) AS member_count,
                       sum(past_ret_15m) AS past_sum,
                       count(past_ret_15m) AS past_count,
                       avg(past_ret_15m) AS theme_past,
                       sum(future_ret_5m) AS future_sum_5m,
                       count(future_ret_5m) AS future_count_5m,
                       avg(future_ret_5m) AS theme_future_5m,
                       sum(future_ret_15m) AS future_sum_15m,
                       count(future_ret_15m) AS future_count_15m,
                       avg(future_ret_15m) AS theme_future_15m,
                       sum(future_ret_30m) AS future_sum_30m,
                       count(future_ret_30m) AS future_count_30m,
                       avg(future_ret_30m) AS theme_future_30m
                FROM {random_base}
                GROUP BY decision_time, theme_key
                """
            )
            con.execute(
                f"""
                CREATE TEMP VIEW {random_node} AS
                SELECT b.*,
                       CASE WHEN t.past_count > 1
                            THEN (t.past_sum - b.past_ret_15m) / (t.past_count - 1)
                       END AS loo_theme_past,
                       CASE WHEN b.future_ret_5m IS NOT NULL AND t.future_count_5m > 1
                            THEN (t.future_sum_5m - b.future_ret_5m) / (t.future_count_5m - 1)
                       END AS loo_theme_future_5m,
                       CASE WHEN b.future_ret_15m IS NOT NULL AND t.future_count_15m > 1
                            THEN (t.future_sum_15m - b.future_ret_15m) / (t.future_count_15m - 1)
                       END AS loo_theme_future_15m,
                       CASE WHEN b.future_ret_30m IS NOT NULL AND t.future_count_30m > 1
                            THEN (t.future_sum_30m - b.future_ret_30m) / (t.future_count_30m - 1)
                       END AS loo_theme_future_30m,
                       t.member_count
                FROM {random_base} b
                JOIN {random_theme} t USING (decision_time, theme_key)
                """
            )
            for horizon in HORIZONS:
                frames.extend(
                    [
                        con.execute(
                            _metric_sql(
                                random_theme,
                                "-theme_past",
                                f"theme_future_{horizon}",
                                "member_count >= 3",
                                MIN_THEME,
                                f"random_group_reversal_s{seed}",
                                horizon,
                                "random_equal_size_group",
                            )
                        ).fetchdf(),
                        con.execute(
                            _metric_sql(
                                random_node,
                                "-loo_theme_past",
                                f"future_ret_{horizon}",
                                "member_count >= 3 AND loo_theme_past IS NOT NULL "
                                f"AND future_ret_{horizon} IS NOT NULL",
                                MIN_NODE,
                                f"random_theme_context_reversal_s{seed}",
                                horizon,
                                "node_random_context_loo",
                            )
                        ).fetchdf(),
                        con.execute(
                            _metric_sql(
                                random_node,
                                "-(past_ret_15m - loo_theme_past)",
                                f"future_ret_{horizon} - loo_theme_future_{horizon}",
                                "member_count >= 3 AND loo_theme_past IS NOT NULL "
                                f"AND loo_theme_future_{horizon} IS NOT NULL",
                                MIN_NODE,
                                f"random_relative_reversal_s{seed}",
                                horizon,
                                "random_group_neutral_loo",
                            )
                        ).fetchdf(),
                        con.execute(
                            _metric_sql(
                                role_theme,
                                "-(core_past - periphery_past)",
                                f"core_future_{horizon} - periphery_future_{horizon}",
                                "core_past IS NOT NULL AND periphery_past IS NOT NULL",
                                MIN_THEME,
                                f"random_role_gap_reversal_s{seed}",
                                horizon,
                                "theme_random_role_gap",
                            )
                        ).fetchdf(),
                    ]
                )

        if task.get("temporal"):
            con.execute(
                f"""
                CREATE TEMP TABLE life AS
                SELECT CAST(dst_time AS TIMESTAMP) AS decision_time,
                       hash(dst_theme_id) AS theme_key,
                       max(continuation_strength) AS continuation_strength,
                       max(jaccard) AS jaccard,
                       bool_or(hard_continue) AS hard_continue,
                       count(*) AS incoming_count
                FROM read_parquet({_sql_lit(task['temporal'])})
                WHERE level = {_sql_lit(task['level'])}
                  AND CAST(dst_time AS TIMESTAMP) >= CAST({_sql_lit(task['time_lo'])} AS TIMESTAMP)
                  AND CAST(dst_time AS TIMESTAMP) < CAST({_sql_lit(task['time_hi'])} AS TIMESTAMP)
                GROUP BY decision_time, theme_key
                """
            )
            con.execute(
                """
                CREATE TEMP VIEW theme_life AS
                SELECT t.*,
                       l.continuation_strength,
                       l.jaccard,
                       l.hard_continue,
                       l.incoming_count
                FROM theme t
                LEFT JOIN life l USING (decision_time, theme_key)
                """
            )
            for horizon in HORIZONS:
                frames.extend(
                    [
                        con.execute(
                            _metric_sql(
                                "theme_life",
                                "-theme_past",
                                f"theme_future_{horizon}",
                                "coalesce(hard_continue, false) "
                                "OR continuation_strength >= 0.5",
                                12,
                                "theme_reversal_stable",
                                horizon,
                                "stable_theme",
                            )
                        ).fetchdf(),
                        con.execute(
                            _metric_sql(
                                "theme_life",
                                "-theme_past",
                                f"theme_future_{horizon}",
                                "incoming_count IS NULL OR continuation_strength < 0.25",
                                12,
                                "theme_reversal_emerging",
                                horizon,
                                "emerging_theme",
                            )
                        ).fetchdf(),
                    ]
                )

        if task.get("relation"):
            relation_path = _sql_lit(task["relation"])
            time_lo = _sql_lit(task["time_lo"])
            time_hi = _sql_lit(task["time_hi"])
            level = _sql_lit(task["level"])
            con.execute(
                f"""
                CREATE TEMP TABLE rel AS
                WITH e AS (
                  SELECT decision_time,
                         hash(src_theme_id) AS src_key,
                         hash(dst_theme_id) AS dst_key,
                         relation_strength
                  FROM read_parquet({relation_path})
                  WHERE level = {level}
                    AND decision_time >= CAST({time_lo} AS TIMESTAMP)
                    AND decision_time < CAST({time_hi} AS TIMESTAMP)
                  UNION ALL
                  SELECT decision_time,
                         hash(dst_theme_id) AS src_key,
                         hash(src_theme_id) AS dst_key,
                         relation_strength
                  FROM read_parquet({relation_path})
                  WHERE level = {level}
                    AND decision_time >= CAST({time_lo} AS TIMESTAMP)
                    AND decision_time < CAST({time_hi} AS TIMESTAMP)
                ), eu AS (
                  SELECT decision_time,
                         src_key,
                         dst_key,
                         max(relation_strength) AS relation_strength
                  FROM e
                  WHERE src_key <> dst_key
                  GROUP BY decision_time, src_key, dst_key
                ), n AS (
                  SELECT eu.decision_time,
                         eu.dst_key AS theme_key,
                         sum(eu.relation_strength * s.theme_past)
                         / nullif(sum(abs(eu.relation_strength)), 0) AS neighbor_past
                  FROM eu
                  JOIN theme s
                    ON s.decision_time = eu.decision_time
                   AND s.theme_key = eu.src_key
                  GROUP BY eu.decision_time, eu.dst_key
                )
                SELECT n.*,
                       t.theme_past,
                       t.theme_future_5m,
                       t.theme_future_15m,
                       t.theme_future_30m
                FROM n
                JOIN theme t USING (decision_time, theme_key)
                """
            )
            con.execute(
                """
                CREATE TEMP VIEW rel_resid AS
                SELECT *,
                       neighbor_past
                       - avg(neighbor_past) OVER (PARTITION BY decision_time)
                       - coalesce(
                           regr_slope(neighbor_past, theme_past)
                           OVER (PARTITION BY decision_time),
                           0
                         ) * (
                           theme_past
                           - avg(theme_past) OVER (PARTITION BY decision_time)
                         ) AS neighbor_residual
                FROM rel
                """
            )
            for horizon in HORIZONS:
                frames.extend(
                    [
                        con.execute(
                            _metric_sql(
                                "rel_resid",
                                "-theme_past",
                                f"theme_future_{horizon}",
                                "neighbor_past IS NOT NULL",
                                MIN_THEME,
                                "theme_reversal_relation_sample",
                                horizon,
                                "relation_theme",
                            )
                        ).fetchdf(),
                        con.execute(
                            _metric_sql(
                                "rel_resid",
                                "neighbor_past - theme_past",
                                f"theme_future_{horizon}",
                                "neighbor_past IS NOT NULL",
                                MIN_THEME,
                                "network_gap",
                                horizon,
                                "relation_theme",
                            )
                        ).fetchdf(),
                        con.execute(
                            _metric_sql(
                                "rel_resid",
                                "neighbor_residual",
                                f"theme_future_{horizon}",
                                "neighbor_residual IS NOT NULL",
                                MIN_THEME,
                                "network_neighbor_residual",
                                horizon,
                                "relation_theme",
                            )
                        ).fetchdf(),
                    ]
                )
    finally:
        con.close()
        shutil.rmtree(temporary, ignore_errors=True)

    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    for key, value in (
        ("date", task["date"]),
        ("chunk", task["chunk"]),
        ("layer_id", task["layer_id"]),
        ("layer_name", task["layer_name"]),
        ("scale", task["scale"]),
        ("level", task["level"]),
    ):
        frame[key] = value
    records = frame.to_dict("records")
    payload = {
        "contract": CONTRACT,
        "status": "complete",
        "inputs": inputs,
        "scope": scope,
        "records": records,
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_json_atomic(checkpoint, payload)
    return {
        "status": "computed",
        "checkpoint": str(checkpoint),
        "record_count": len(records),
        "elapsed_sec": payload["elapsed_sec"],
    }


def _dispatch_task(task: dict[str, Any]) -> dict[str, Any]:
    return _process_stock(task) if task["kind"] == "stock" else _process_partition(task)


def _aggregate(snapshot: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [
        "date",
        "layer_id",
        "layer_name",
        "scale",
        "level",
        "signal",
        "horizon",
        "unit",
    ]
    daily = (
        snapshot.groupby(keys, dropna=False)
        .agg(
            snapshot_count=("rank_ic", "count"),
            sample_count=("sample_count", "sum"),
            mean_rank_ic=("rank_ic", "mean"),
            median_rank_ic=("rank_ic", "median"),
            std_rank_ic=("rank_ic", "std"),
            positive_ic_rate=("rank_ic", lambda series: (series > 0).mean()),
            mean_spread_bps=("spread_bps", "mean"),
            median_spread_bps=("spread_bps", "median"),
            positive_spread_rate=("spread_bps", lambda series: (series > 0).mean()),
        )
        .reset_index()
    )
    summary_keys = [
        "layer_id",
        "layer_name",
        "scale",
        "level",
        "signal",
        "horizon",
        "unit",
    ]
    summary = (
        daily.groupby(summary_keys, dropna=False)
        .agg(
            day_count=("date", "nunique"),
            snapshot_count=("snapshot_count", "sum"),
            sample_count=("sample_count", "sum"),
            mean_daily_ic=("mean_rank_ic", "mean"),
            median_daily_ic=("mean_rank_ic", "median"),
            min_daily_ic=("mean_rank_ic", "min"),
            max_daily_ic=("mean_rank_ic", "max"),
            daily_ic_direction_rate=("mean_rank_ic", lambda series: (series > 0).mean()),
            mean_daily_spread_bps=("mean_spread_bps", "mean"),
            median_daily_spread_bps=("mean_spread_bps", "median"),
            daily_spread_direction_rate=(
                "mean_spread_bps", lambda series: (series > 0).mean()
            ),
        )
        .reset_index()
    )
    return daily, summary


def _placebo_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    pairs = (
        ("theme_reversal", "random_group_reversal_s", "theme_vs_equal_size_random"),
        (
            "theme_context_reversal_node",
            "random_theme_context_reversal_s",
            "theme_context_vs_random_context",
        ),
        (
            "relative_to_theme_reversal",
            "random_relative_reversal_s",
            "relative_theme_vs_random_group",
        ),
        (
            "core_periphery_gap_reversal",
            "random_role_gap_reversal_s",
            "graph_roles_vs_random_roles",
        ),
    )
    key = ["layer_id", "layer_name", "scale", "level", "horizon"]
    outputs = []
    for actual, prefix, name in pairs:
        left = summary.loc[
            summary.signal.eq(actual),
            key
            + [
                "mean_daily_ic",
                "mean_daily_spread_bps",
                "daily_ic_direction_rate",
                "daily_spread_direction_rate",
            ],
        ].copy()
        left = left.rename(
            columns={column: f"actual_{column}" for column in left.columns if column not in key}
        )
        placebo = (
            summary.loc[summary.signal.str.startswith(prefix, na=False)]
            .groupby(key, dropna=False)
            .agg(
                placebo_seed_count=("signal", "nunique"),
                placebo_mean_daily_ic=("mean_daily_ic", "mean"),
                placebo_min_daily_ic=("mean_daily_ic", "min"),
                placebo_max_daily_ic=("mean_daily_ic", "max"),
                placebo_mean_daily_spread_bps=("mean_daily_spread_bps", "mean"),
                placebo_min_daily_spread_bps=("mean_daily_spread_bps", "min"),
                placebo_max_daily_spread_bps=("mean_daily_spread_bps", "max"),
            )
            .reset_index()
        )
        joined = left.merge(placebo, on=key, how="inner")
        if joined.empty:
            continue
        joined.insert(0, "comparison", name)
        joined["ic_increment_vs_placebo_mean"] = (
            joined["actual_mean_daily_ic"] - joined["placebo_mean_daily_ic"]
        )
        joined["spread_increment_vs_placebo_mean_bps"] = (
            joined["actual_mean_daily_spread_bps"]
            - joined["placebo_mean_daily_spread_bps"]
        )
        outputs.append(joined)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def _write_report(
    path: Path,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    stats: dict[str, Any],
) -> None:
    columns = [
        "layer_name",
        "scale",
        "level",
        "signal",
        "horizon",
        "unit",
        "day_count",
        "mean_daily_ic",
        "mean_daily_spread_bps",
        "daily_ic_direction_rate",
    ]
    core_signals = [
        "stock_reversal",
        "stock_reversal_matched",
        "theme_context_reversal_node",
        "theme_reversal",
        "relative_to_theme_reversal",
        "core_periphery_gap_reversal",
        "theme_reversal_stable",
        "theme_reversal_emerging",
        "network_neighbor_residual",
    ]
    core = summary.loc[summary.signal.isin(core_signals)].copy()
    core = core.sort_values(
        ["signal", "horizon", "mean_daily_ic"], ascending=[True, True, False]
    )
    lines = [
        "# 新 P2 三日全量分层 Alpha 报告",
        "",
        f"> 日期：{', '.join(stats['dates'])}。三日结果用于机制筛选，不是生产回测。",
        "",
        "## 运行审计",
        "",
        f"- 合约：`{CONTRACT}`",
        f"- 共同 Layer–Scale 面板：{stats['partition_panel_count']} 个。",
        f"- Level：{', '.join(stats['levels'])}。",
        f"- 任务数：{stats['tasks']}；worker：{stats['effective_workers']}/{stats['requested_workers']}。",
        f"- 全局 worker 内存预算：{stats['memory_budget_gb']} GB；每 worker DuckDB 上限：{stats['worker_memory_gb']:.2f} GB。",
        f"- `max_in_flight={stats['max_in_flight']}`；`tasks_per_child={stats['tasks_per_child']}`。",
        "- 无日期、快照、row-group、Layer 或 Scale 抽样。",
        "- 默认要求三个日期具有完全一致的 Layer–Scale 面板；不完整面板会失败。",
        "- 每 worker 强制 DuckDB/BLAS/Arrow 单线程，避免 24 进程内部再次过度并行。",
        "- 所有任务带输入指纹 checkpoint；中断后只重算变化或未完成任务。",
        "",
        "## 完整分区面板",
        "",
        pd.DataFrame(stats["partition_panel"]).to_markdown(index=False),
        "",
        "## 核心结果",
        "",
        core[columns].to_markdown(index=False, floatfmt=".6f") if not core.empty else "无有效指标。",
        "",
        "## 随机等规模／随机角色安慰剂",
        "",
        comparison.to_markdown(index=False, floatfmt=".6f")
        if not comparison.empty
        else "无有效安慰剂比较。",
        "",
        "## 解读规则",
        "",
        "1. `stock_reversal_matched` 是同一 Theme 成员宇宙内的传统单股反转基线。",
        "2. `theme_reversal` 必须优于等规模随机分组，才证明 P1 社区聚合有增量。",
        "3. `theme_context_reversal_node` 检查 Theme 信号能否映射回成员股票。",
        "4. `relative_to_theme_reversal` 使用 leave-one-out Theme，避免节点进入自己的基准。",
        "5. 核心／外围必须优于随机角色，才可解释为图结构价值。",
        "6. `network_neighbor_residual` 已控制目标 Theme 自身过去收益，才代表跨 Theme 残差增量。",
        "",
        "## 限制",
        "",
        "- 三个交易日足以否定明显机械机制，但不足以确认稳定 Alpha。",
        "- 全部结果为毛收益，不含点差、佣金、冲击、借券和容量约束。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _plan_workers(
    task_count: int, requested_workers: int, memory_budget_gb: float
) -> dict[str, Any]:
    requested = max(1, int(requested_workers))
    effective = min(requested, max(1, int(task_count)))
    per_worker = max(0.75, min(4.0, float(memory_budget_gb) / effective * 0.85))
    effective = min(
        effective,
        max(1, int(float(memory_budget_gb) // per_worker)),
    )
    return {
        "task_count": int(task_count),
        "requested_workers": requested,
        "effective_workers": effective,
        "memory_budget_gb": float(memory_budget_gb),
        "worker_memory_gb": float(per_worker),
    }


def _partition_key_from_path(path: Path) -> tuple[str, int, str] | None:
    date: str | None = None
    layer: int | None = None
    scale: str | None = None
    for part in path.parts:
        if part.startswith("date="):
            date = part.split("=", 1)[1]
        elif part.startswith("layer_id="):
            try:
                layer = int(part.split("=", 1)[1])
            except ValueError:
                return None
        elif part.startswith("scale="):
            scale = part.split("=", 1)[1]
    if date is None or layer is None or scale is None:
        return None
    return date, layer, scale


def _load_layer_names(path: str | None) -> dict[int, str]:
    if not path:
        return {}
    frame = pd.read_parquet(path, columns=["layer_id", "name"])
    return {
        int(row.layer_id): str(row.name)
        for row in frame.itertuples(index=False)
    }


def _discover_full_panel(
    root: Path,
    dates: list[str],
    layer_names: dict[int, str],
    *,
    allow_missing: bool,
) -> list[dict[str, Any]]:
    by_date: dict[str, dict[tuple[int, str], Path]] = {date: {} for date in dates}
    for path in root.rglob("theme_memberships.parquet"):
        parsed = _partition_key_from_path(path)
        if parsed is None:
            continue
        date, layer, scale = parsed
        if date in by_date:
            by_date[date][(layer, scale)] = path

    missing_dates = [date for date, partitions in by_date.items() if not partitions]
    if missing_dates:
        raise FileNotFoundError(
            f"no P1 memberships found for dates={missing_dates} under {root}"
        )

    sets = {date: set(partitions) for date, partitions in by_date.items()}
    common = set.intersection(*(sets[date] for date in dates))
    union = set.union(*(sets[date] for date in dates))
    if common != union and not allow_missing:
        details = {
            date: sorted(union - sets[date])
            for date in dates
            if union - sets[date]
        }
        raise RuntimeError(
            "incomplete three-day P1 panel; "
            f"missing partitions={details}; "
            "use --allow-missing-partitions only for debugging"
        )

    panel = sorted(common if not allow_missing else union, key=lambda value: (value[0], value[1]))
    rows: list[dict[str, Any]] = []
    for date in dates:
        for layer, scale in panel:
            membership = by_date[date].get((layer, scale))
            if membership is None:
                continue
            base = membership.parent
            relation = base / "theme_relation_edges.parquet"
            temporal = base / "temporal_theme_edges.parquet"
            rows.append(
                {
                    "date": date,
                    "layer_id": layer,
                    "layer_name": layer_names.get(layer, f"layer_{layer}"),
                    "scale": scale,
                    "membership": str(membership),
                    "relation": str(relation) if relation.exists() else None,
                    "temporal": str(temporal) if temporal.exists() else None,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-panel hierarchical P2 alpha validation")
    parser.add_argument("--p1-root", required=True)
    parser.add_argument(
        "--labels-root",
        required=True,
        help="root containing raw_1m parquet files; name retained for CLI compatibility",
    )
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--layers", help="layers.parquet used for layer names")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dates", default="2026-01-06,2026-01-07,2026-01-08")
    parser.add_argument(
        "--levels",
        default="B50",
        help="comma-separated P1 levels; B50 is the non-duplicated primary design",
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--memory-budget-gb",
        type=float,
        default=88.0,
        help="88GB leaves roughly 40GB for OS, parent, Arrow and cache on 128GB RAM",
    )
    parser.add_argument(
        "--tasks-per-child",
        type=int,
        default=4,
        help="recycle Pandas/Arrow workers after N tasks",
    )
    parser.add_argument("--max-in-flight", type=int, default=24)
    parser.add_argument(
        "--allow-missing-partitions",
        action="store_true",
        help="debug only; production default requires the same panel on every date",
    )
    parser.add_argument("--reset-checkpoints", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    temporary = output / "tmp"
    if args.reset_checkpoints:
        shutil.rmtree(checkpoints, ignore_errors=True)

    dates = [value.strip() for value in args.dates.split(",") if value.strip()]
    levels = [value.strip() for value in args.levels.split(",") if value.strip()]
    if not dates or not levels:
        raise ValueError("dates and levels must be non-empty")

    compact_root = output / "prepared_labels"
    prepared = []
    for date in dates:
        prepared.append(
            _prepare_labels_one(
                date,
                _label_source(Path(args.labels_root), date),
                Path(args.symbols),
                compact_root / f"date={date}" / "labels_compact.parquet",
            )
        )

    layer_names = _load_layer_names(args.layers)
    partitions = _discover_full_panel(
        Path(args.p1_root),
        dates,
        layer_names,
        allow_missing=args.allow_missing_partitions,
    )
    panel = sorted(
        {(part["layer_id"], part["scale"]) for part in partitions},
        key=lambda value: (value[0], value[1]),
    )
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for partition in partitions:
        by_date[partition["date"]].append(partition)

    tasks: list[dict[str, Any]] = []
    split = "17:45:00"
    for date in dates:
        labels = compact_root / f"date={date}" / "labels_compact.parquet"
        ranges = [
            (
                "A",
                f"{date} 00:00:00",
                f"{date} {split}",
                f"ts={date}T000000_0000",
                f"ts={date}T174500_0000",
            ),
            (
                "B",
                f"{date} {split}",
                f"{date} 23:59:59.999999",
                f"ts={date}T174500_0000",
                f"ts={date}T999999_9999",
            ),
        ]
        for chunk, time_lo, time_hi, _, _ in ranges:
            tasks.append(
                {
                    "kind": "stock",
                    "date": date,
                    "chunk": chunk,
                    "labels": str(labels),
                    "time_lo": time_lo,
                    "time_hi": time_hi,
                }
            )
        for partition in by_date[date]:
            for level in levels:
                for chunk, time_lo, time_hi, theme_lo, theme_hi in ranges:
                    task = {
                        **partition,
                        "kind": "partition",
                        "level": level,
                        "chunk": chunk,
                        "labels": str(labels),
                        "time_lo": time_lo,
                        "time_hi": time_hi,
                        "theme_lo": theme_lo,
                        "theme_hi": theme_hi,
                    }
                    tasks.append({key: value for key, value in task.items() if value is not None})

    for task in tasks:
        task["checkpoint"] = str(_checkpoint_path(checkpoints, task))
        task["temp_root"] = str(temporary)
    tasks.sort(
        key=lambda task: sum(
            Path(task[key]).stat().st_size
            for key in ("membership", "relation", "temporal")
            if task.get(key)
        ),
        reverse=True,
    )

    plan = _plan_workers(len(tasks), args.workers, args.memory_budget_gb)
    requested = plan["requested_workers"]
    effective = plan["effective_workers"]
    per_worker = plan["worker_memory_gb"]
    for task in tasks:
        task["worker_memory_gb"] = per_worker

    started = time.time()
    counts: dict[str, int] = defaultdict(int)
    if effective == 1:
        result_iterator = (_dispatch_task(task) for task in tasks)
    else:
        result_iterator = bounded_process_map(
            tasks,
            effective,
            _dispatch_task,
            max_in_flight=max(effective, args.max_in_flight),
            max_tasks_per_child=args.tasks_per_child,
        )
    for index, result in enumerate(result_iterator, 1):
        counts[result["status"]] += 1
        if index % 5 == 0 or index == len(tasks):
            print(
                f"[p2-hierarchical-full] {index}/{len(tasks)} "
                f"computed={counts['computed']} reused={counts['reused']}",
                flush=True,
            )

    records: list[dict[str, Any]] = []
    for task in tasks:
        payload = read_json(task["checkpoint"])
        if not payload or payload.get("status") != "complete":
            raise RuntimeError(f"missing or incomplete checkpoint: {task['checkpoint']}")
        records.extend(payload.get("records", []))
    snapshot = pd.DataFrame(records)
    if snapshot.empty:
        raise RuntimeError("no metrics generated")
    snapshot["decision_time"] = pd.to_datetime(
        snapshot["decision_time"], utc=True, errors="coerce"
    )
    defaults = (
        ("date", snapshot.decision_time.dt.strftime("%Y-%m-%d")),
        ("chunk", ""),
        ("layer_id", 0),
        ("layer_name", "stock_baseline"),
        ("scale", "15m"),
        ("level", "STOCK"),
    )
    for column, value in defaults:
        if column not in snapshot:
            snapshot[column] = value
        else:
            snapshot[column] = snapshot[column].fillna(value)

    daily, summary = _aggregate(snapshot)
    comparison = _placebo_comparison(summary)
    snapshot.to_parquet(output / "snapshot_metrics.parquet", index=False)
    daily.to_csv(output / "daily_alpha_summary.csv", index=False)
    summary.to_csv(output / "three_day_alpha_summary.csv", index=False)
    comparison.to_csv(output / "placebo_increment_comparison.csv", index=False)

    stats = {
        "contract": CONTRACT,
        "dates": dates,
        "levels": levels,
        "partition_panel_count": len(panel),
        "partition_panel": [
            {
                "layer_id": layer,
                "layer_name": layer_names.get(layer, f"layer_{layer}"),
                "scale": scale,
            }
            for layer, scale in panel
        ],
        "full_panel_required": not args.allow_missing_partitions,
        "no_sampling": True,
        "tasks": len(tasks),
        "requested_workers": requested,
        "effective_workers": effective,
        "memory_budget_gb": args.memory_budget_gb,
        "worker_memory_gb": per_worker,
        "max_in_flight": args.max_in_flight,
        "tasks_per_child": args.tasks_per_child,
        "computed": counts["computed"],
        "reused": counts["reused"],
        "elapsed_sec": round(time.time() - started, 3),
        "prepared_labels": prepared,
        "reference_24core_128gb_plan": _plan_workers(len(tasks), 24, 88.0),
    }
    write_json_atomic(output / "run_summary.json", stats)
    _write_report(
        output / "P2_HIERARCHICAL_ALPHA_3DAY_REPORT.md",
        summary,
        comparison,
        stats,
    )
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
