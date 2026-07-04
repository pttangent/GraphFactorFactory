import pandas as pd

from graphfactorfactory.themes.models import ThemeCandidate
from graphfactorfactory.themes.stable_core_tracker import StableCoreTracker


def theme(instance, path, members, families=("price", "flow")):
    return ThemeCandidate(
        instance,
        path,
        pd.Timestamp("2026-01-07", tz="UTC"),
        tuple(members),
        ("return_corr", "signed_flow"),
        tuple(families),
        0.5,
        0.5,
        0.01,
    )


def test_birth_confirmation_and_dormant_revival():
    tracker = StableCoreTracker(threshold=0.45, history_frames=3, min_hits=2, grace_frames=1)
    first, records1 = tracker.assign(
        [theme("i1", "new-1", [1, 2, 3, 4])],
        [],
        {},
        timestamp=pd.Timestamp("2026-01-07 15:10", tz="UTC"),
        frame_minutes=1,
    )
    assert records1[0].status == "tentative"

    second, records2 = tracker.assign(
        [theme("i2", "new-2", [1, 2, 3, 5])],
        first,
        {records1[0].theme_instance_id: records1[0]},
        timestamp=pd.Timestamp("2026-01-07 15:11", tz="UTC"),
        frame_minutes=1,
    )
    assert records2[0].status == "active"
    assert records2[0].event_type == "continuation"
    assert second[0].theme_path_id == first[0].theme_path_id

    _, records3 = tracker.assign(
        [],
        second,
        {records2[0].theme_instance_id: records2[0]},
        timestamp=pd.Timestamp("2026-01-07 15:12", tz="UTC"),
        frame_minutes=1,
    )
    assert any(record.status == "dormant" for record in records3)

    revived, records4 = tracker.assign(
        [theme("i4", "new-4", [1, 2, 3, 6])],
        [],
        {},
        timestamp=pd.Timestamp("2026-01-07 15:13", tz="UTC"),
        frame_minutes=1,
    )
    assert records4[0].event_type == "revival"
    assert revived[0].theme_path_id == first[0].theme_path_id


def test_global_matching_does_not_reuse_path():
    tracker = StableCoreTracker(threshold=0.40)
    previous = [theme("p1", "path-1", [1, 2, 3, 4])]
    current = [
        theme("c1", "new-1", [1, 2, 3]),
        theme("c2", "new-2", [1, 2, 4]),
    ]
    assigned, _ = tracker.assign(
        current,
        previous,
        {},
        timestamp=pd.Timestamp("2026-01-07 15:11", tz="UTC"),
        frame_minutes=1,
    )
    inherited = [item for item in assigned if item.theme_path_id == "path-1"]
    assert len(inherited) == 1
