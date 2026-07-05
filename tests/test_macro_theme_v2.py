from types import SimpleNamespace

from graphfactorfactory.themes.macro_theme_v2 import (
    StructuralMacroConfig,
    StructuralMacroMatcher,
)


def prototype(members, families, layers, *, persistence=0.75, cohesion=0.80, entropy=0.65):
    return SimpleNamespace(
        core_members=tuple(members),
        member_frequency={str(member): 1.0 for member in members},
        family_frequency={family: 1.0 for family in families},
        layer_frequency={layer: 1.0 for layer in layers},
        persistence=persistence,
        mean_consensus_score=0.45,
        cohesion=cohesion,
        core_ratio=0.60,
        member_entropy=entropy,
        log_size=1.8,
    )


def test_common_structure_without_member_evidence_is_rejected():
    matcher = StructuralMacroMatcher(StructuralMacroConfig(threshold=0.40, member_evidence_gate=0.10))
    previous = prototype([1, 2, 3], ["price", "flow"], ["return_corr", "signed_flow"])
    current = prototype([100, 101, 102], ["price", "flow"], ["return_corr", "signed_flow"])
    assert matcher.match([current], [previous]) == []


def test_structurally_consistent_member_supported_lineage_matches():
    matcher = StructuralMacroMatcher()
    previous = prototype([1, 2, 3, 4], ["price", "flow"], ["return_corr", "signed_flow"])
    current = prototype([1, 2, 5, 6], ["price", "flow"], ["return_corr", "signed_flow"])
    assert len(matcher.match([current], [previous])) == 1


def test_low_quality_pair_receives_stricter_threshold():
    matcher = StructuralMacroMatcher(StructuralMacroConfig(quality_gamma=0.20))
    high = prototype([1, 2, 3], ["price", "flow"], ["return_corr"], persistence=0.90, cohesion=0.90)
    low = prototype([1, 2, 3], ["price", "flow"], ["return_corr"], persistence=0.50, cohesion=0.50)
    _, high_threshold, _, _ = matcher.score(high, high)
    _, low_threshold, _, _ = matcher.score(low, low)
    assert low_threshold > high_threshold


def test_global_matching_does_not_reuse_previous_macro():
    matcher = StructuralMacroMatcher(StructuralMacroConfig(threshold=0.35))
    previous = [prototype([1, 2, 3, 4], ["price", "flow"], ["return_corr"])]
    current = [
        prototype([1, 2, 5], ["price", "flow"], ["return_corr"]),
        prototype([1, 3, 6], ["price", "flow"], ["return_corr"]),
    ]
    assert len(matcher.match(current, previous)) == 1
