from graphfactorfactory.domain.layers import LAYERS


def test_thirteen_layers_are_registered():
    assert len(LAYERS) == 13
    assert {layer.name for layer in LAYERS} >= {"return_corr", "signed_flow", "off_exchange", "absorption"}
