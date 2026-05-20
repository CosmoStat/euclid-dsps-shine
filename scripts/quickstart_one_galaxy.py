"""Run the one-galaxy smoke workflow from Python.

Usage:
    conda run -n shine python scripts/quickstart_one_galaxy.py
"""

from euclid_dsps.config import load_config
from euclid_dsps.workflows import fit_one, run_one


def main() -> None:
    config = load_config("configs/smoke_test.yaml")
    run_one(config, "outputs/runs/script_one")
    fit_one(config, "outputs/runs/script_fit_one")


if __name__ == "__main__":
    main()
