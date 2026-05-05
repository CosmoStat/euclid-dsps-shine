from __future__ import annotations


def test_public_facades_import_without_native_dsps_import() -> None:
    import euclid_dsps.pipeline as pipeline
    import euclid_dsps.reporting as reporting
    import euclid_dsps.reports as reports
    import euclid_dsps.workflows as workflows

    assert pipeline.run_one is workflows.run_one
    assert reports.write_run_outputs is reporting.write_run_outputs
