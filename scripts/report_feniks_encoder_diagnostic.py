#!/usr/bin/env python3
"""Print the high-signal outcome of the encoder-only exact diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summary = args.root / "encoder_diagnostic_summary.json"
    if not summary.exists():
        cohort = args.root / "cohort.parquet"
        done = len(list((args.root / "galaxies").glob("*/DONE")))
        print(f"exact_diagnostic=PENDING galaxies_done={done}")
        print(f"cohort_prepared={cohort.exists()}")
        print("NEXT_ACTION=WAIT_FOR_EXACT_DIAGNOSTIC")
        return
    payload = json.loads(summary.read_text())
    print(f"status={payload['status']}")
    print(f"scientific_diagnosis={payload['scientific_diagnosis']}")
    for domain, value in payload["domains"].items():
        raw = value["q_only_importance"]
        defensive = value["defensive_importance"]
        geometry = value["geometry"]
        agreement = value["agreement_with_nuts"]
        print(
            f"{domain}: q_IS={raw['status']} ESS={raw['median_raw_ess_fraction']:.6f} "
            f"bad_k={raw['fraction_pareto_k_gt_0p7']:.6f} "
            f"defensive_IS={defensive['status']} "
            f"ESS={defensive['median_raw_ess_fraction']:.6f} "
            f"bad_k={defensive['fraction_pareto_k_gt_0p7']:.6f} "
            f"q_width_ratio_max_median="
            f"{geometry['median_q_generalized_variance_ratio_max']:.4f} "
            f"q_W1_to_NUTS="
            f"{agreement['Encoder']['median_wasserstein_to_nuts_in_nuts_std']:.4f} "
            f"defensive_W1_to_NUTS="
            f"{agreement['Defensive + IS']['median_wasserstein_to_nuts_in_nuts_std']:.4f}"
        )
    print(f"NEXT_ACTION={payload['next_action']}")


if __name__ == "__main__":
    main()
