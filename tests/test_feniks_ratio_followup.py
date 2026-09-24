from pathlib import Path

import numpy as np

from scripts.feniks_ratio_followup import (
    _split_calibration_audit,
    apply_logit_offsets,
    fit_marginal_logit_offsets,
)


def test_marginal_logit_calibration_enforces_reference_ratio_identity():
    rng = np.random.default_rng(19)
    logits = rng.normal(size=(20_000, 4))
    logits[:, 0] += 2.0
    logc = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    frequencies = np.array([0.08, 0.17, 0.30, 0.45])

    offset, diagnostics = fit_marginal_logit_offsets(
        logc, frequencies, tolerance=1e-12, maximum_iterations=2000
    )
    calibrated = apply_logit_offsets(logc, offset)
    moment = np.exp(calibrated - np.log(frequencies)[None]).mean(axis=0)

    np.testing.assert_allclose(np.exp(calibrated).sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(moment, 1.0, atol=2e-5)
    assert diagnostics["calibration_ratio_moment_max_abs_error"] < 2e-5


def test_calibration_and_audit_split_is_disjoint_and_stratified():
    labels = np.repeat(np.arange(8), 20)
    test = np.arange(len(labels))

    calibration, audit = _split_calibration_audit(test, labels, seed=31)

    assert set(calibration).isdisjoint(audit)
    assert set(np.concatenate((calibration, audit))) == set(test)
    np.testing.assert_array_equal(np.bincount(labels[calibration]), np.full(8, 10))
    np.testing.assert_array_equal(np.bincount(labels[audit]), np.full(8, 10))


def test_followup_submission_reuses_banks_and_has_four_cells():
    repository = Path(__file__).parents[1]
    launcher = (repository / "scripts/submit_feniks_ratio_followup.sh").read_text()
    watcher = (repository / "scripts/watch_feniks_ratio_followup.sh").read_text()
    workflow = (repository / "scripts/feniks_ratio_followup.py").read_text()

    assert '--array="0-3%$CLASSIFIER_CONCURRENCY"' in launcher
    assert '--dependency="afterok:$CLASSIFIER_JOB"' in launcher
    assert "feniks_ratio_ladder prepare" not in launcher
    assert "simulation_banks_reused=True" in workflow
    assert "shared_target_split=True" in workflow
    assert "noiseless" in watcher
    assert "noisy" in watcher
