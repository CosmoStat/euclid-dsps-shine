from __future__ import annotations

import numpy as np

from euclid_dsps.amortized.sc_asmc_reweight import rows_requiring_refresh


def test_final_repair_unions_low_ess_and_unresolved_rows() -> None:
    rows = rows_requiring_refresh(
        np.asarray([4, 8, 8]),
        np.asarray([3, 8, 11]),
        refresh_unresolved=True,
    )

    np.testing.assert_array_equal(rows, np.asarray([3, 4, 8, 11]))


def test_standard_reweight_does_not_add_unresolved_rows() -> None:
    rows = rows_requiring_refresh(
        np.asarray([4, 8]),
        np.asarray([3, 11]),
        refresh_unresolved=False,
    )

    np.testing.assert_array_equal(rows, np.asarray([4, 8]))


def test_final_repair_can_target_only_unresolved_rows() -> None:
    rows = rows_requiring_refresh(
        np.asarray([4, 8]),
        np.asarray([3, 11]),
        refresh_unresolved=True,
        refresh_low_ess=False,
    )

    np.testing.assert_array_equal(rows, np.asarray([3, 11]))
