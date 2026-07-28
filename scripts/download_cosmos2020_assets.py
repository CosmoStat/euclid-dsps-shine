#!/usr/bin/env python3
"""Download Farmer v2.1, SVO filters, Pop-COSMOS, and public A24 summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from astropy.io.votable import parse_single_table

from euclid_dsps.cosmos2020 import (
    COSMOS_BANDS,
    ESO_TAP_URL,
    POPCOSMOS_COMMIT,
    POPCOSMOS_URL,
    ZENODO_RECORD,
    farmer_adql,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--skip-farmer", action="store_true")
    parser.add_argument("--skip-external-repo", action="store_true")
    parser.add_argument("--skip-zenodo", action="store_true")
    parser.add_argument("--timeout", type=int, default=7200)
    return parser.parse_args()


def _request(
    url: str,
    *,
    data: dict[str, str] | None = None,
    attempts: int = 5,
) -> bytes:
    encoded = None if data is None else urllib.parse.urlencode(data).encode("ascii")
    request = urllib.request.Request(url, data=encoded)
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            retryable = error.code == 429 or 500 <= error.code < 600
            if not retryable or attempt == attempts:
                raise
        except urllib.error.URLError:
            if attempt == attempts:
                raise
        delay = min(2 ** (attempt - 1), 30)
        print(
            f"[cosmos2020-download] transient error for {url}; "
            f"retry {attempt}/{attempts} in {delay}s",
            flush=True,
        )
        time.sleep(delay)
    raise AssertionError("unreachable")


def _tap_download(path: Path, max_rows: int | None, timeout: int) -> str:
    query = farmer_adql(max_rows)
    if max_rows is not None:
        payload = _request(
            f"{ESO_TAP_URL}/sync",
            data={
                "REQUEST": "doQuery",
                "LANG": "ADQL",
                "FORMAT": "fits",
                "MAXREC": str(max(int(max_rows), 1)),
                "QUERY": query,
            },
        )
        path.write_bytes(payload)
        return query

    opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())
    request = urllib.request.Request(
        f"{ESO_TAP_URL}/async",
        data=urllib.parse.urlencode(
            {
                "REQUEST": "doQuery",
                "LANG": "ADQL",
                "FORMAT": "fits",
                "MAXREC": "2000000",
                "QUERY": query,
            }
        ).encode("ascii"),
    )
    with opener.open(request, timeout=120) as response:
        job_url = response.geturl().rstrip("/")
    _request(f"{job_url}/phase", data={"PHASE": "RUN"})
    deadline = time.monotonic() + timeout
    while True:
        phase = _request(f"{job_url}/phase").decode("ascii").strip().upper()
        print(f"[cosmos2020-download] TAP phase={phase}", flush=True)
        if phase == "COMPLETED":
            break
        if phase in {"ERROR", "ABORTED"}:
            raise RuntimeError(f"ESO TAP job ended in phase {phase}: {job_url}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"ESO TAP job exceeded {timeout}s: {job_url}")
        time.sleep(15)
    path.write_bytes(_request(f"{job_url}/results/result"))
    return query


def _download_filters(output: Path) -> list[dict[str, str]]:
    rows = []
    output.mkdir(parents=True, exist_ok=True)
    for band in COSMOS_BANDS:
        url = (
            "https://svo2.cab.inta-csic.es/theory/fps3/fps.php?"
            + urllib.parse.urlencode({"ID": band.svo_id})
        )
        votable = output / f"{band.name}.vot"
        dat = output / f"{band.name}.dat"
        votable.write_bytes(_request(url))
        table = parse_single_table(votable).to_table()
        wave = np.asarray(table["Wavelength"], dtype=float)
        transmission = np.asarray(table["Transmission"], dtype=float)
        np.savetxt(dat, np.column_stack((wave, transmission)), fmt="%.10e")
        rows.append(
            {
                "band": band.name,
                "svo_id": band.svo_id,
                "url": url,
                "path": str(dat),
                "sha256": sha256_file(dat),
            }
        )
    return rows


def _clone_popcosmos(output: Path) -> dict[str, str]:
    if output.exists():
        head = subprocess.check_output(
            ["git", "-C", str(output), "rev-parse", "HEAD"], text=True
        ).strip()
        if head != POPCOSMOS_COMMIT:
            raise RuntimeError(f"Existing Pop-COSMOS checkout is {head}, expected pin")
    else:
        subprocess.run(["git", "clone", POPCOSMOS_URL, str(output)], check=True)
        subprocess.run(
            ["git", "-C", str(output), "checkout", "--detach", POPCOSMOS_COMMIT],
            check=True,
        )
    return {"url": POPCOSMOS_URL, "commit": POPCOSMOS_COMMIT, "path": str(output)}


def _download_zenodo(output: Path) -> list[dict[str, str]]:
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(
        _request(f"https://zenodo.org/api/records/{ZENODO_RECORD}").decode("utf-8")
    )
    rows = []
    wanted = {"README.txt", "summaries.txt"}
    for item in metadata["files"]:
        if item["key"] not in wanted:
            continue
        target = output / item["key"]
        algorithm, expected = item.get("checksum", ":").split(":", maxsplit=1)
        reuse = (
            target.is_file()
            and algorithm == "md5"
            and hashlib.md5(target.read_bytes()).hexdigest() == expected
        )
        if not reuse:
            target.write_bytes(_request(item["links"]["self"]))
        rows.append(
            {
                "record": ZENODO_RECORD,
                "name": item["key"],
                "path": str(target),
                "sha256": sha256_file(target),
            }
        )
    if {row["name"] for row in rows} != wanted:
        raise RuntimeError("Zenodo record did not expose README.txt and summaries.txt")
    return rows


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "status": "incomplete",
        "farmer": None,
        "filters": [],
        "popcosmos": None,
        "zenodo": [],
    }
    if not args.skip_farmer:
        farmer = args.out / (
            f"cosmos2020_farmer_v21_top{args.max_rows}.fits"
            if args.max_rows is not None
            else "cosmos2020_farmer_v21.fits"
        )
        query = _tap_download(farmer, args.max_rows, args.timeout)
        manifest["farmer"] = {
            "path": str(farmer),
            "sha256": sha256_file(farmer),
            "adql": query,
            "max_rows": args.max_rows,
        }
    manifest["filters"] = _download_filters(args.out / "filters")
    if not args.skip_external_repo:
        manifest["popcosmos"] = _clone_popcosmos(args.out / "external/pop-cosmos")
    if not args.skip_zenodo:
        manifest["zenodo"] = _download_zenodo(args.out / "zenodo")
    manifest["status"] = "complete"
    write_json(args.out / "download_manifest.json", manifest)
    shutil.copyfile(
        args.out / "download_manifest.json", args.out / "DOWNLOAD_COMPLETE.json"
    )
    print(f"[cosmos2020-download] complete -> {args.out}")


if __name__ == "__main__":
    main()
