from pathlib import Path


SOURCE = Path("scripts/p2_alpha_daily_features.py")


def _source_text() -> str:
    return SOURCE.read_text(encoding="utf-8")


def _function_block(src: str, start: str, end: str) -> str:
    return src[src.index(start):src.index(end)]


def test_relation_spillover_carries_destination_past_returns() -> None:
    """Underreaction must be built from destination theme past response, not future target returns."""
    src = _source_text()
    block = _function_block(src, "def relation_one", "def path_id")

    assert "dst_past_cols" in block
    assert "past_eq_" in block
    assert '"dst_" + c' in block
    assert "dst_theme_id" in block


def test_daily_underreaction_score_is_non_leaky() -> None:
    """daily_underreaction_score must not use target_*_mean_proxy in its signal formula."""
    src = _source_text()
    block = _function_block(src, "def daily_one", "def eval_daily")

    assert "dst_past_eq_" in block
    assert "target_pre_response_z" in block
    assert "expected_pressure_z" in block
    assert "underreaction_uses_future_target" in block
    assert "False" in block

    assignment_lines = [
        line.strip()
        for line in block.splitlines()
        if "daily_underreaction_score" in line and "=" in line
    ]
    assert assignment_lines, "daily_underreaction_score assignment missing"

    for line in assignment_lines:
        assert "target_" not in line or "target_pre_response_z" in line
        assert "mean_proxy" not in line
