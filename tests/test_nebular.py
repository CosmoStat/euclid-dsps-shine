from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from euclid_dsps.filters import FilterCurve
from euclid_dsps.nebular import (
    emline_inventory,
    fitted_redshift_modes,
    line_filter_crossings,
)


def _context() -> SimpleNamespace:
    return SimpleNamespace(
        ssp_emline_luminosity=np.asarray([[[1.0, 10.0], [2.0, 5.0]]]),
        ssp_emline_wave=np.asarray([5000.0, 6563.0]),
        ssp_emline_name=("OIII_5007", "Halpha"),
        filters={
            "r": FilterCurve(
                name="r",
                wave=np.linspace(5500.0, 7500.0, 10),
                transmission=np.ones(10),
                source="test",
            )
        },
        nebular_emission_mode="ssp_flux",
    )


def test_emline_inventory_ranks_lines_by_strength() -> None:
    inventory = emline_inventory(_context())

    assert inventory.loc[0, "line_name"] == "Halpha"
    assert inventory.loc[0, "strength_rank"] == 1


def test_line_filter_crossings_links_modes_to_filters() -> None:
    context = _context()
    inventory = emline_inventory(context)
    fits = pd.DataFrame({"fit_z_obs": [0.1, 0.1, 0.1, 0.5]})
    modes = fitted_redshift_modes(fits, bin_width=0.05, min_count=2)

    crossings = line_filter_crossings(context, inventory, modes, top_n_lines=2)

    assert not crossings.empty
    assert "r" in crossings["band"].tolist()
    assert set(crossings["line_name"]).intersection({"OIII_5007", "Halpha"})
