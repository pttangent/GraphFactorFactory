from graphfactorfactory.themes.macro_theme import (
    MacroMatchConfig,
    MacroThemeMatcher,
    MacroThemePrototype,
)


def prototype(name, members, families, layers):
    return MacroThemePrototype(
        name,
        tuple(members),
        {member: 1.0 for member in members},
        {family: 1.0 for family in families},
        {layer: 1.0 for layer in layers},
        0.75,
        0.40,
        5,
    )


def test_common_mechanism_without_member_evidence_is_rejected():
    matcher = MacroThemeMatcher(MacroMatchConfig(threshold=0.45, member_evidence_gate=0.10))
    previous = prototype("p", [1, 2, 3], ["price", "flow"], ["return_corr", "signed_flow"])
    current = prototype("c", [100, 101, 102], ["price", "flow"], ["return_corr", "signed_flow"])
    assert matcher.match([current], [previous]) == []


def test_member_supported_macro_lineage_is_matched():
    matcher = MacroThemeMatcher()
    previous = prototype("p", [1, 2, 3, 4], ["price", "flow"], ["return_corr", "signed_flow"])
    current = prototype("c", [1, 2, 5, 6], ["price", "flow"], ["return_corr", "signed_flow"])
    matches = matcher.match([current], [previous])
    assert len(matches) == 1
    assert matches[0][3]["core_overlap"] > 0


def test_global_matching_does_not_reuse_macro_path():
    matcher = MacroThemeMatcher(MacroMatchConfig(threshold=0.40))
    previous = [prototype("p", [1, 2, 3, 4], ["price", "flow"], ["return_corr"])]
    current = [
        prototype("c1", [1, 2, 5], ["price", "flow"], ["return_corr"]),
        prototype("c2", [1, 3, 6], ["price", "flow"], ["return_corr"]),
    ]
    assert len(matcher.match(current, previous)) == 1
