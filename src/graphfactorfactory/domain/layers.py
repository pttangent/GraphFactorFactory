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


LAYERS: tuple[LayerDefinition, ...] = (
    LayerDefinition(1, "return_corr", ("ret_5m",), "price"),
    LayerDefinition(2, "volume_expansion", ("volume_z_30m", "dollar_volume_z_30m"), "activity"),
    LayerDefinition(3, "trade_intensity", ("trade_count_z_30m", "avg_trade_size"), "activity"),
    LayerDefinition(4, "signed_flow", ("volume_ofi_proxy", "count_ofi_proxy", "signed_dollar_flow"), "trade_flow"),
    LayerDefinition(5, "large_trade_flow", ("large_trade_ofi_proxy", "large_trade_dollar_share", "large_trade_count"), "trade_flow"),
    LayerDefinition(6, "odd_lot_activity", ("odd_lot_trade_share", "odd_lot_volume_share"), "trade_flow"),
    LayerDefinition(7, "block_activity", ("block_trade_share", "block_volume_share"), "trade_flow"),
    LayerDefinition(8, "off_exchange", ("off_exchange_dollar_share", "off_exchange_trade_share"), "venue"),
    LayerDefinition(9, "venue_fragmentation", ("venue_fragmentation_proxy",), "venue"),
    LayerDefinition(10, "price_impact", ("price_impact_proxy", "liquidity_impact_proxy"), "liquidity"),
    LayerDefinition(11, "absorption", ("absorption_proxy", "flow_absorption_proxy"), "liquidity"),
    LayerDefinition(12, "flow_return_alignment", ("flow_return_alignment",), "interaction"),
    LayerDefinition(13, "report_latency", ("avg_report_lag_ns", "max_report_lag_ns", "correction_excluded_share"), "data_quality"),
)

LAYER_BY_ID = {layer.layer_id: layer for layer in LAYERS}
LAYER_BY_NAME = {layer.name: layer for layer in LAYERS}
