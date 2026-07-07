from __future__ import annotations


def test_public_facades_import_without_native_dsps_import() -> None:
    import euclid_dsps.reporting as reporting
    import euclid_dsps.workflows as workflows

    assert callable(workflows.run_one)
    assert callable(workflows.fit_batch)
    assert callable(reporting.write_run_outputs)
