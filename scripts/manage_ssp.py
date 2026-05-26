#!/usr/bin/env python3
"""
Manage DSPS SSP models: list, download, and test.

Usage:
    python scripts/manage_ssp.py list
    python scripts/manage_ssp.py download fsps_v3.2_standard
    python scripts/manage_ssp.py test Data/ssp_data_fsps_v3.2_lgmet_age.h5
"""

import argparse
import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

try:
    import dsps
except ImportError:
    print(
        "Error: 'dsps' library not found. Please install it or run within the correct environment."
    )
    dsps = None

SSP_BASE_URL = "https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/"
DATA_DIR = Path("Data")

AVAILABLE_SSPS = {
    "fsps_v3.2_standard": {
        "filename": "ssp_data_fsps_v3.2_lgmet_age.h5",
        "description": "Standard FSPS v3.2 model with age and metallicity grids.",
    },
    "fsps_v3.2_sparse": {
        "filename": "ssp_data_fsps_v3.2_lgmet_age.sparse.h5",
        "description": "Sparse version of the standard FSPS v3.2 model.",
    },
    "fsps_v3.2_age": {
        "filename": "ssp_data_fsps_v3.2_age.h5",
        "description": "FSPS v3.2 model with age grid only (likely single metallicity).",
    },
    "fsps_v3.2_continuum": {
        "filename": "ssp_data_continuum_fsps_v3.2_lgmet_age.h5",
        "description": "Continuum-only FSPS v3.2 model.",
    },
    "fsps_v0.4.7_u-1.0": {
        "filename": "fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-1.0_logGasZ0.0.h5",
        "description": "Older FSPS model with ionization parameter logU=-1.0.",
    },
    "fsps_v0.4.7_u-2.0": {
        "filename": "fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5",
        "description": "Older FSPS model with ionization parameter logU=-2.0.",
    },
    "fsps_v0.4.7_u-3.0": {
        "filename": "fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-3.0_logGasZ0.0.h5",
        "description": "Older FSPS model with ionization parameter logU=-3.0.",
    },
    "fsps_v0.4.7_u-4.0": {
        "filename": "fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-4.0_logGasZ0.0.h5",
        "description": "Older FSPS model with ionization parameter logU=-4.0.",
    },
}


def ensure_data_dir():
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)


def download_ssp(name, overwrite=False):
    if name not in AVAILABLE_SSPS:
        print(f"Error: SSP '{name}' not found in available list.")
        return None

    ensure_data_dir()
    filename = AVAILABLE_SSPS[name]["filename"]
    url = SSP_BASE_URL + filename
    dest = DATA_DIR / filename

    if dest.exists() and not overwrite:
        print(f"File {dest} already exists. Use --overwrite to download again.")
    else:
        print(f"Downloading {url} to {dest}...")
        try:
            urlretrieve(url, dest)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading file: {e}")
            return None
    return dest


def test_ssp(path):
    if dsps is None:
        print("Cannot test: 'dsps' library is missing.")
        return False

    print(f"Testing SSP at {path}...")
    try:
        # dsps.load_ssp_templates returns a NamedTuple-like object
        ssp = dsps.load_ssp_templates(fn=str(path))
        print("SUCCESS: SSP loaded successfully!")
        print(
            f"  - Wavelength range: {ssp.ssp_wave.min():.2f} - {ssp.ssp_wave.max():.2f} Angstroms"
        )
        print(f"  - Wavelength bins:  {len(ssp.ssp_wave)}")

        if hasattr(ssp, "ssp_lgmet"):
            print(
                f"  - Metallicities:    {len(ssp.ssp_lgmet)} values (range: {ssp.ssp_lgmet.min():.2f} to {ssp.ssp_lgmet.max():.2f})"
            )

        print(
            f"  - Ages (log Gyr):   {len(ssp.ssp_lg_age_gyr)} values (range: {ssp.ssp_lg_age_gyr.min():.2f} to {ssp.ssp_lg_age_gyr.max():.2f})"
        )
        print(f"  - Flux matrix:      {ssp.ssp_flux.shape}")

        # Basic sanity check: flux should be non-negative
        if np.any(ssp.ssp_flux < 0):
            print("  - WARNING: Found negative flux values!")
        else:
            print("  - Data integrity:   OK (all fluxes >= 0)")

        return True
    except Exception as e:
        print(f"FAILURE: Error loading SSP: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Manage DSPS SSP models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # List command
    subparsers.add_parser("list", help="List available SSP models.")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download an SSP model.")
    download_parser.add_argument(
        "name",
        choices=list(AVAILABLE_SSPS.keys()) + ["all"],
        help="Name of the SSP to download or 'all'.",
    )
    download_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files."
    )
    download_parser.add_argument(
        "--no-test", action="store_true", help="Skip loading test after download."
    )

    # Test command
    test_parser = subparsers.add_parser(
        "test", help="Test an SSP model file by loading it."
    )
    test_parser.add_argument("path", type=str, help="Path to the SSP .h5 file.")

    args = parser.parse_args()

    if args.command == "list":
        print(f"{'Name':<25} | {'Filename':<60} | {'Description'}")
        print("-" * 120)
        for name, info in AVAILABLE_SSPS.items():
            print(f"{name:<25} | {info['filename']:<60} | {info['description']}")

    elif args.command == "download":
        names = AVAILABLE_SSPS.keys() if args.name == "all" else [args.name]
        for name in names:
            print(f"\n--- Processing {name} ---")
            path = download_ssp(name, overwrite=args.overwrite)
            if path and not args.no_test:
                test_ssp(path)

    elif args.command == "test":
        test_ssp(args.path)


if __name__ == "__main__":
    main()
