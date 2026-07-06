# Handoff: 39-day Intraday 7-arm Layer Analysis

## Repository and branch
- Repository: `pttangent/GraphFactorFactory`
- Branch: `codex/intraday-7arm-layer-analysis`

## Completed
- Recovered the `symbol_id -> symbol` mapping from `Smoke_Test_Output/graph_store/dimensions/symbols.parquet` (5,590 symbols).
- Downloaded all 39 Phase 1 trading-day `layer_communities.parquet` files for 2026-01-02 through 2026-02-27.
- Ran a 7-arm **intraday** lifecycle comparison by layer using a 5-minute evaluation grid and minimum community size 20.
- Generated full Layer × Arm metrics: confirmed paths, median life, S5/S15/S30/S60, revival rate, confirmed revival rate.
- Generated current arm recommendation per layer.

## Important result
- D9 is selected by the current structural score for 12/13 layers. `off_exchange` appears as D11 only because D9/D11/D13/D15 metrics are effectively tied and stable sorting selected D11.
- Do not conclude that D9 is final. D9 receives a mechanical persistence advantage from dormant/revival permission. The next agent must add matched controls and identity-drift penalties before recommending production parameters.

## Exact local/Drive inputs
- Google Drive folder: `[pt tangent]/US-Stock/Smoke_Test_Output/theme_discovery_phase1`
- Folder ID: `1QDlOFpuDpK8MoEmOtuV10VZFFZKrjMoW`
- Each trading-day folder contains `layer_communities.parquet`.
- Symbol mapping: `Smoke_Test_Output/graph_store/dimensions/symbols.parquet`.

## Reproduction
```bash
python scripts/run_intraday_7arm_5m.py \
  --input-root /path/to/phase1_input \
  --output-root /path/to/intraday_7arm_5m
```

The current script has constants at the top rather than CLI parsing. If running from this branch, either edit `ROOT`/`OUT` or first add argparse without changing arm semantics.

## Arm definitions
- A: entry containment 0.20; stay 0.20; no fingerprint; no gaps.
- B: entry containment 0.18 + fingerprint 0.10; stay 0.20; no gaps.
- C: B + assisted stay containment 0.10/fingerprint 0.16; weak gap 1.
- D9: C + dormant max 3; revival fingerprint 0.20; no breadth requirement.
- D11: D9 + breadth expansion 0.10.
- D13: revival fingerprint 0.22 + breadth 0.10.
- D15: revival fingerprint 0.30 + breadth 0.10 + three-state post-confirmation.

## Immediate next tasks
1. Validate D9 vs C/D11 with matched daily controls by layer and birth time.
2. Add false revival and path identity drift metrics.
3. Re-run shortlisted arms at 1-minute resolution.
4. Add SPY/QQQ/IWM residualized ReturnCorr as a new layer; do not overwrite raw ReturnCorr.
5. Join symbol metadata and compute sector/industry purity for raw vs residual ReturnCorr communities.
6. Add Phase 0 edge metrics: internal density, conductance, clustering, assortativity, connected components, giant-component share, degree distribution, and power-law vs lognormal/truncated-power-law tests.
7. Add 5/15/30/60m and T+1 return labels; structural persistence alone is not alpha.

## Quality warnings
- This run samples every fifth snapshot.
- All S5 values equal 1 because only confirmed paths are included; treat S15/S30/S60 as the informative survival metrics.
- Current revival confirmation equals revival for D9/D11/D13 because they have no post-confirmation requirement.
- D15 revival events may be unconfirmed due to its three-state post-confirmation rule.
- No null control and no multiple-testing correction are present.
- `return_corr` remains raw and may be dominated or fragmented by market beta and reciprocal top-k graph construction.

## Expected deliverable in the next chat
Produce a corrected table:
`Layer × Arm` with S15, S30, S60, matched-control excess, false-revival rate, core-retention, member-turnover, and recommended arm. Then explain whether layers genuinely need different arms or whether D9 only wins mechanically.
