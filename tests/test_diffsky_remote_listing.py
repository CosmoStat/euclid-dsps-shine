from __future__ import annotations

from euclid_dsps.diffsky_data.remote_listing import list_remote_directory


class _Response:
    text = """
    <html><body><table>
    <tr><td><img></td><td><a href="file.diffsky_gals.hdf5">file.diffsky_gals.hdf5</a></td><td align="right">2026-01-01</td><td align="right"> 92M</td></tr>
    <tr><td><img></td><td><a href="meta.yaml">meta.yaml</a></td><td align="right">2026-01-01</td><td align="right">271 </td></tr>
    <tr><td><img></td><td><a href="../">Parent Directory</a></td><td></td><td align="right"> - </td></tr>
    </table></body></html>
    """

    def raise_for_status(self) -> None:
        return None


def test_list_remote_directory_parses_apache_listing(monkeypatch) -> None:
    monkeypatch.setattr(
        "euclid_dsps.diffsky_data.remote_listing.requests.get",
        lambda *args, **kwargs: _Response(),
    )

    files = list_remote_directory("https://example.test/data/")

    assert [item.name for item in files] == ["file.diffsky_gals.hdf5", "meta.yaml"]
    assert files[0].size_bytes == 92 * 1024**2
    assert files[0].extension == ".hdf5"
