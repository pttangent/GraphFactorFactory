# P2 point-in-time contract

P2 now exposes two separate contracts.

## Intraday

- One feature row per `decision_time` and theme.
- Past returns become available only at their recorded `label_exit_time_*`.
- Minute targets require `label_entry_time_* > feature_time`.
- Cross-sectional normalization, ranking and top-bottom portfolios are computed inside each snapshot.
- P1 theme relations are treated as undirected and expanded into symmetric neighbour-diffusion messages; theme ID ordering is never interpreted as lead-lag direction.
- `late_confirmation_score_z` is causal and uses already-realized response alignment, not the session's final timestamp.

## Daily

- Full-session aggregation is allowed only after the final snapshot of the session.
- Daily features may only be evaluated against next-open-executable labels such as `label_1d_open`.
- Close-start labels such as `label_1d` are rejected for full-day factors because their entry precedes feature availability.

## Compatibility and cache safety

`scripts/p2_alpha_daily_features.py` remains the compatibility entrypoint and routes to `p2_alpha_pit_features.py`. Output manifests include `pit_contract_version=p2-pit-v2`; legacy outputs without this contract are not accepted by `--skip-existing`.

## Audit

```bash
python scripts/audit_p2_pit.py --sample-root /path/to/sample --rebuilt-root /path/to/rebuilt --out reports/pit_audit.json
pytest -q tests/test_p2_pit_features.py
```

The checked sample report is stored at `reports/pit_audit_sample_v2.json`.
