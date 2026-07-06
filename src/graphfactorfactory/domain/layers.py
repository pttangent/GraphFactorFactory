from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerDefinition:
    layer_id: int
    name: str
    columns: tuple[str, ...]
    family: str
    directed: bool = False
    lag_bars: int = 0
    transform: str = "standard"
    lookbacks_minutes: tuple[int, ...] = (5, 15, 30)


@dataclass(frozen=True)
class LayerScaleDefinition:
    layer: LayerDefinition
    lookback_minutes: int
    scale_role: str
    decision_step_minutes: int
    minimum_points: int


_SCALE_ROLES = {5: "trigger", 15: "confirm", 30: "structural"}
_SCALE_STEPS = {5: 1, 15: 1, 30: 5}


LAYERS: tuple[LayerDefinition, ...] = (
    LayerDefinition(1, "return_corr_raw_1m", ("ret_1m",), "price", transform="return_corr_raw_1m", lookbacks_minutes=(15, 30)),
    LayerDefinition(2, "volume_expansion", ("volume_z_30m", "dollar_volume_z_30m"), "activity"),
    LayerDefinition(3, "trade_intensity", ("trade_count_z_30m", "avg_trade_size"), "activity"),
    LayerDefinition(4, "signed_flow", ("volume_ofi_proxy", "count_ofi_proxy", "signed_dollar_flow"), "trade_flow"),
    LayerDefinition(5, "large_trade_flow", ("large_trade_ofi_proxy", "large_trade_dollar_share", "large_trade_count"), "trade_flow", lookbacks_minutes=(30,)),
    LayerDefinition(6, "odd_lot_activity", ("odd_lot_trade_share", "odd_lot_volume_share"), "trade_flow"),
    LayerDefinition(7, "block_activity", ("block_trade_share", "block_volume_share"), "trade_flow", lookbacks_minutes=(30,)),
    LayerDefinition(8, "off_exchange", ("off_exchange_dollar_share", "off_exchange_trade_share"), "venue"),
    LayerDefinition(9, "venue_fragmentation", ("venue_fragmentation_proxy",), "venue"),
    LayerDefinition(10, "price_impact", ("price_impact_proxy", "liquidity_impact_proxy"), "liquidity"),
    LayerDefinition(11, "absorption", ("absorption_proxy", "flow_absorption_proxy"), "liquidity"),
    LayerDefinition(12, "flow_return_alignment", ("flow_return_alignment",), "interaction"),
    LayerDefinition(13, "report_latency", ("avg_report_lag_ns", "max_report_lag_ns", "correction_excluded_share"), "data_quality", lookbacks_minutes=(5,)),
    LayerDefinition(14, "return_corr_cross_sectional_1m", ("ret_1m",), "price", transform="return_corr_cross_sectional_1m", lookbacks_minutes=(15, 30)),
    LayerDefinition(15, "return_corr_cross_sectional_rolling_5m", ("log_ret_1m",), "price", transform="return_corr_cross_sectional_rolling_5m", lookbacks_minutes=(30,)),
)


def _minimum_points(layer: LayerDefinition, lookback_minutes: int) -> int:
    if layer.transform.startswith("return_corr_"):
        if layer.transform == "return_corr_cross_sectional_rolling_5m":
            return 20
        return 10 if lookback_minutes == 15 else 20
    if lookback_minutes == 5:
        return 3
    if lookback_minutes == 15:
        return 8
    return 12


def layer_scale_definitions(layers: tuple[LayerDefinition, ...] | None = None) -> tuple[LayerScaleDefinition, ...]:
    selected = tuple(layers or LAYERS)
    return tuple(
        LayerScaleDefinition(
            layer=layer,
            lookback_minutes=lookback,
            scale_role=_SCALE_ROLES[lookback],
            decision_step_minutes=_SCALE_STEPS[lookback],
            minimum_points=_minimum_points(layer, lookback),
        )
        for layer in selected
        for lookback in layer.lookbacks_minutes
    )


LAYER_SCALES = layer_scale_definitions()
MAX_LOOKBACK_MINUTES = max(item.lookback_minutes for item in LAYER_SCALES)

LAYER_BY_ID = {layer.layer_id: layer for layer in LAYERS}
LAYER_BY_NAME = {layer.name: layer for layer in LAYERS}
