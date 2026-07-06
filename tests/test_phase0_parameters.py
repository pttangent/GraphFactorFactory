from graphfactorfactory.domain.config import BuildConfig


def test_graph_parameter_precedence():
    config = BuildConfig(
        top_k=8,
        degree_cap=6,
        minimum_similarity=0.10,
        graph_parameter_overrides={
            "family:trade_flow": {"minimum_similarity": 0.20},
            "layer:signed_flow": {"top_k": 6, "degree_cap": 4},
            "scale:signed_flow:30": {"minimum_similarity": 0.40},
        },
    )

    five = config.graph_parameters_for(
        layer_name="signed_flow",
        family="trade_flow",
        lookback_minutes=5,
    )
    thirty = config.graph_parameters_for(
        layer_name="signed_flow",
        family="trade_flow",
        lookback_minutes=30,
    )

    assert (five.top_k, five.degree_cap, five.minimum_similarity) == (6, 4, 0.20)
    assert (thirty.top_k, thirty.degree_cap, thirty.minimum_similarity) == (6, 4, 0.40)


def test_unrelated_layer_uses_family_then_base():
    config = BuildConfig(
        graph_parameter_overrides={"family:activity": {"minimum_similarity": 0.30}}
    )
    activity = config.graph_parameters_for(
        layer_name="volume_expansion",
        family="activity",
        lookback_minutes=15,
    )
    venue = config.graph_parameters_for(
        layer_name="off_exchange",
        family="venue",
        lookback_minutes=15,
    )
    assert activity.minimum_similarity == 0.30
    assert venue.minimum_similarity == config.minimum_similarity


def test_invalid_degree_cap_is_rejected():
    config = BuildConfig(
        graph_parameter_overrides={
            "scale:signed_flow:30": {"top_k": 4, "degree_cap": 6}
        }
    )
    try:
        config.graph_parameters_for(
            layer_name="signed_flow",
            family="trade_flow",
            lookback_minutes=30,
        )
    except ValueError as exc:
        assert "degree_cap" in str(exc)
    else:
        raise AssertionError("invalid parameters were accepted")
