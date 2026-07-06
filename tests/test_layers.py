from graphfactorfactory.domain.layers import LAYERS, LAYER_BY_NAME


def test_fifteen_layers_are_registered():
    assert len(LAYERS) == 15
    assert {layer.name for layer in LAYERS} >= {
        "return_corr",
        "return_corr_market_residual",
        "return_corr_cross_sectional_residual",
        "signed_flow",
        "off_exchange",
        "absorption",
    }


def test_return_corr_layers_keep_distinct_transforms():
    assert LAYER_BY_NAME["return_corr"].transform == "return_corr_raw"
    assert LAYER_BY_NAME["return_corr_market_residual"].transform == "return_corr_market_residual"
    assert LAYER_BY_NAME["return_corr_cross_sectional_residual"].transform == "return_corr_cross_sectional_residual"
