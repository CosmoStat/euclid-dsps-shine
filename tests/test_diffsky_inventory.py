from __future__ import annotations

from euclid_dsps.diffsky_data.inventory import rank_candidate_files
from euclid_dsps.diffsky_data.remote_listing import RemoteFile


def test_rank_candidate_files_prioritizes_diffsky_gals_hdf5() -> None:
    files = [
        RemoteFile("https://x/readme.md", "readme.md", 100, ".md", 0),
        RemoteFile(
            "https://x/lc.diffsky_gals.hdf5",
            "lc.diffsky_gals.hdf5",
            92 * 1024**2,
            ".hdf5",
            0,
        ),
    ]

    ranked = rank_candidate_files(files)

    assert ranked.iloc[0]["name"] == "lc.diffsky_gals.hdf5"
    assert "diffsky_gals" in ranked.iloc[0]["tags"]
