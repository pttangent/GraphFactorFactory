# P2 point-in-time contract

P2 exposes two separate contracts. They deliberately do not share a ranking or evaluation path.

## Intraday

- One feature row per `decision_time` and theme.
- Past returns become available only at their recorded `label_exit_time_*`.
- Minute targets require `label_entry_time_* > feature_time`.
- Cross-sectional normalization, ranking and top-bottom portfolios are computed inside each snapshot.
- P1 theme relations are undirected and expanded into symmetric neighbour-diffusion messages; theme-ID ordering is never interpreted as lead-lag direction.
- `late_confirmation_score_z` is causal and uses already-realized response alignment, not the session's final timestamp.
- Legacy `daily_*` score names remain as compatibility aliases in intraday output; `feature_contract=intraday_snapshot` is the authoritative semantic marker.

## Daily / interday

- Full-session aggregation is allowed only after the final snapshot of the session.
- Snapshot-local B50/B35 IDs are not treated as stable themes. P1 `temporal_theme_edges.parquet` is mandatory and is converted into within-session theme episodes.
- Only episodes still alive at the final session snapshot are eligible for an EOD signal.
- Pressure may aggregate over the full temporal episode, but the target basket and pre-response are taken from the episode's final snapshot; earlier snapshot labels are never averaged into the EOD target.
- Daily features may only be evaluated against next-open-executable labels such as `label_1d_open`.
- Close-start labels such as `label_1d` are rejected because their entry precedes EOD feature availability.
- The authoritative marker is `feature_contract=end_of_day_episode_next_open_execution`.

## Compatibility and cache safety

`scripts/p2_alpha_daily_features.py` remains the compatibility entrypoint and routes to `p2_alpha_pit_features.py`. Output manifests include `pit_contract_version=p2-pit-v2`; legacy outputs without this contract are not accepted by `--skip-existing`.

## Audit

```bash
python scripts/audit_p2_pit.py \
  --sample-root /path/to/sample \
  --rebuilt-root /path/to/rebuilt \
  --out reports/pit_audit.json

pytest -q tests/test_p2_pit_features.py
```

The checked intraday sample report is stored at `reports/pit_audit_sample_v2.json`. The supplied sample did not contain daily next-open labels or `temporal_theme_edges.parquet`, so the daily contract is validated by deterministic unit tests and enforced schema/runtime assertions rather than an empirical daily sample result.
