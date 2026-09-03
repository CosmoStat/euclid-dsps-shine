from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.plot_feniks_epoch160_aggregated import (
    PARAMETERS,
    _select_panel_rows,
    _validate_dense_bank,
)


def _bank(rows: list[int], samples: int) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "row_index": np.repeat(rows, samples),
            "sample_id": np.tile(np.arange(samples), len(rows)),
        }
    )
    for index, parameter in enumerate(PARAMETERS):
        frame[parameter] = np.arange(len(frame), dtype=float) + index
    return frame


def test_dense_bank_requires_object_equal_joint_draws() -> None:
    frame = _bank([4, 9, 15], 4)
    _validate_dense_bank(
        frame,
        expected_rows={4, 9, 15},
        samples_per_object=4,
        label="test",
    )

    with pytest.raises(ValueError, match="not object-equal"):
        _validate_dense_bank(
            frame.iloc[:-1],
            expected_rows={4, 9, 15},
            samples_per_object=4,
            label="test",
        )


def test_individual_selection_spans_frozen_flux_quantiles() -> None:
    rows = list(range(16))

    assert _select_panel_rows(rows, 6) == [0, 3, 6, 9, 12, 15]
