from __future__ import annotations

import numpy as np
import pandas as pd

from euclid_dsps.amortized.latent import latent_spec_from_config
from euclid_dsps.config import load_config
from euclid_dsps.prior_learning.inferred import load_inferred_theta_dataset


def test_load_inferred_theta_dataset_uses_config_latent_space(tmp_path) -> None:
    config = load_config(
        "configs/experiments/diffsky_hltds_joint_realnvp_kl_annealed_zscale005_tau2safe_h100.yaml"
    )
    spec = latent_spec_from_config(config)
    lower = np.asarray(spec.lower, dtype=float)
    upper = np.asarray(spec.upper, dtype=float)
    theta = lower + 0.5 * (upper - lower)
    frame = pd.DataFrame([theta], columns=spec.names)
    frame.insert(0, "object_id", [123])
    frame.insert(0, "row_index", [7])
    path = tmp_path / "map_estimates.parquet"
    frame.to_parquet(path, index=False)

    dataset = load_inferred_theta_dataset(config, input_paths=(path,))

    assert dataset.theta.shape == (1, len(spec.names))
    assert dataset.x.shape == (1, len(spec.names))
    assert dataset.source_rows.tolist() == [7]
    assert dataset.object_id.tolist() == [123]
    assert dataset.parameter_names == spec.names
