from __future__ import annotations

import pandas as pd

from scripts.repartition_feniks_50k_grouped import (
    effective_proposal_keys,
    grouped_split_assignment,
)


def test_effective_keys_capture_seed_shard_collisions() -> None:
    identifiers = pd.Series(
        [
            "train:26061701:1:42",
            "validation:26061702:0:42",
            "test:26061703:0:42",
        ]
    )
    keys = effective_proposal_keys(identifiers)
    assert keys.iloc[0] == keys.iloc[1]
    assert keys.iloc[1] != keys.iloc[2]


def test_group_assignment_never_splits_duplicate_proposals() -> None:
    keys = pd.Series(["a", "a", "b", "c", "c", "d", "e", "f"])
    assignment = grouped_split_assignment(
        keys,
        target_sizes={"train": 6, "validation": 1, "test": 1},
        seed=3,
    )
    assigned_rows = keys.map(assignment)
    assert assigned_rows[keys == "a"].nunique() == 1
    assert assigned_rows[keys == "c"].nunique() == 1
