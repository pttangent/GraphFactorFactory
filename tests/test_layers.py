from graphfactorfactory.domain.layers import LAYERS, LAYER_BY_NAME, LAYER_SCALES


def test_fifteen_layers_are_registered():
    assert len(LAYERS) == 15
    assert {layer.name for layer in LAYERS} >= {
        "return_corr_raw_1m",
        "return_corr_cross_sectional_1m",
        "return_corr_cross_sectional_rolling_5m",
        "signed_flow",
        "off_exchange",
        "absorption",
    }


def test_exactly_thirty_five_layer_scale_graphs_are_registered():
    assert len(LAYER_SCALES) == 35
    assert sum(item.lookback_minutes == 5 for item in LAYER_SCALES) == 10
    assert sum(item.lookback_minutes == 15 for item in LAYER_SCALES) == 11
    assert sum(item.lookback_minutes == 30 for item in LAYER_SCALES) == 14


def test_return_corr_layers_use_1m_nff_inputs():
    assert LAYER_BY_NAME["return_corr_raw_1m"].columns == ("ret_1m",)
    assert LAYER_BY_NAME["return_corr_cross_sectional_1m"].columns == ("ret_1m",)
    assert LAYER_BY_NAME["return_corr_cross_sectional_rolling_5m"].columns == ("log_ret_1m",)


def test_sparse_and_quality_layers_do_not_expand_blindly():
    assert LAYER_BY_NAME["large_trade_flow"].lookbacks_minutes == (30,)
    assert LAYER_BY_NAME["block_activity"].lookbacks_minutes == (30,)
    assert LAYER_BY_NAME["report_latency"].lookbacks_minutes == (5,)
