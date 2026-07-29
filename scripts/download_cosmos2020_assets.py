#!/usr/bin/env python3
"""Download Farmer v2.1, SVO filters, Pop-COSMOS, and public A24 summaries."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import io
import json
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from astropy.io.votable import parse_single_table

from euclid_dsps.cosmos2020 import (
    COSMOS_BANDS,
    ESO_FARMER_V21_ID,
    ESO_FARMER_V21_SIZE,
    ESO_FARMER_V21_URL,
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
    parser.add_argument(
        "--filters-only",
        action="store_true",
        help="Refresh filter curves in an existing completed asset manifest.",
    )
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument(
        "--farmer-url",
        default=ESO_FARMER_V21_URL,
        help="Direct URL for the full Farmer v2.1 FITS product.",
    )
    parser.add_argument(
        "--tap-job-url",
        help="Resume an existing full-catalog ESO asynchronous TAP job.",
    )
    return parser.parse_args()


def _download_direct(
    path: Path,
    url: str,
    *,
    expected_size: int,
    attempts: int = 8,
) -> None:
    """Stream a public archive product with retry and HTTP range resume."""
    if path.is_file() and path.stat().st_size == expected_size:
        print(f"[cosmos2020-download] reusing complete Farmer FITS: {path}")
        return
    if path.exists():
        raise ValueError(
            f"Existing final Farmer file has size {path.stat().st_size}, "
            f"expected {expected_size}: {path}"
        )

    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_size:
            raise ValueError(
                f"Partial Farmer file exceeds expected size: {partial}"
            )
        request = urllib.request.Request(url)
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                status = response.getcode()
                if offset and status != 206:
                    print(
                        "[cosmos2020-download] server ignored range request; "
                        "restarting Farmer download",
                        flush=True,
                    )
                    offset = 0
                mode = "ab" if offset and status == 206 else "wb"
                downloaded = offset
                last_report = time.monotonic()
                with partial.open(mode) as stream:
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        stream.write(block)
                        downloaded += len(block)
                        if time.monotonic() - last_report >= 30:
                            fraction = 100.0 * downloaded / expected_size
                            print(
                                "[cosmos2020-download] Farmer "
                                f"{downloaded}/{expected_size} bytes "
                                f"({fraction:.1f}%)",
                                flush=True,
                            )
                            last_report = time.monotonic()
            actual_size = partial.stat().st_size
            if actual_size != expected_size:
                raise OSError(
                    f"incomplete Farmer download: {actual_size}/{expected_size} bytes"
                )
            partial.replace(path)
            print(
                f"[cosmos2020-download] Farmer direct download complete: {path}",
                flush=True,
            )
            return
        except (OSError, http.client.HTTPException) as error:
            if attempt == attempts:
                raise
            delay = min(2 ** (attempt - 1), 60)
            current = partial.stat().st_size if partial.exists() else 0
            print(
                "[cosmos2020-download] Farmer transfer interrupted "
                f"at {current}/{expected_size} bytes ({error}); "
                f"retry {attempt}/{attempts} in {delay}s",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


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


def _tap_download(
    path: Path,
    max_rows: int | None,
    timeout: int,
    *,
    tap_job_url: str | None = None,
) -> str:
    query = farmer_adql(max_rows)
    if max_rows is not None:
        if tap_job_url is not None:
            raise ValueError("--tap-job-url is only valid for a full async download")
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

    if tap_job_url is None:
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
    else:
        job_url = tap_job_url.rstrip("/")
        print(f"[cosmos2020-download] resuming TAP job={job_url}", flush=True)
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
        for attempt in range(1, 6):
            try:
                payload = _request(url)
                table = parse_single_table(io.BytesIO(payload)).to_table()
                wave = np.asarray(table["Wavelength"], dtype=float)
                transmission = np.asarray(table["Transmission"], dtype=float)
                if (
                    wave.ndim != 1
                    or transmission.shape != wave.shape
                    or len(wave) < 2
                    or not np.all(np.isfinite(wave))
                    or not np.all(np.isfinite(transmission))
                ):
                    raise ValueError(f"Invalid SVO filter payload for {band.svo_id}")

                with tempfile.NamedTemporaryFile(
                    dir=output, prefix=f".{band.name}.", suffix=".vot", delete=False
                ) as stream:
                    stream.write(payload)
                    temporary_votable = Path(stream.name)
                with tempfile.NamedTemporaryFile(
                    dir=output,
                    prefix=f".{band.name}.",
                    suffix=".dat",
                    mode="w",
                    delete=False,
                ) as stream:
                    np.savetxt(
                        stream,
                        np.column_stack((wave, transmission)),
                        fmt="%.10e",
                    )
                    temporary_dat = Path(stream.name)
                temporary_votable.replace(votable)
                temporary_dat.replace(dat)
                break
            except (OSError, ValueError) as error:
                if attempt == 5:
                    raise
                delay = min(2 ** (attempt - 1), 30)
                print(
                    f"[cosmos2020-download] invalid filter response for "
                    f"{band.svo_id} ({error}); retry {attempt}/5 in {delay}s",
                    flush=True,
                )
                time.sleep(delay)
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
    if args.filters_only:
        manifest_path = args.out / "download_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"--filters-only requires an existing manifest: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["filters"] = _download_filters(args.out / "filters")
        manifest["status"] = "complete"
        write_json(manifest_path, manifest)
        shutil.copyfile(manifest_path, args.out / "DOWNLOAD_COMPLETE.json")
        print(f"[cosmos2020-download] refreshed filters -> {args.out}")
        return
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
        if args.max_rows is None and args.tap_job_url is None:
            _download_direct(
                farmer,
                args.farmer_url,
                expected_size=ESO_FARMER_V21_SIZE,
            )
            manifest["farmer"] = {
                "path": str(farmer),
                "sha256": sha256_file(farmer),
                "method": "eso_phase3_direct",
                "archive_id": ESO_FARMER_V21_ID,
                "url": args.farmer_url,
                "size_bytes": farmer.stat().st_size,
                "max_rows": None,
            }
        else:
            query = _tap_download(
                farmer,
                args.max_rows,
                args.timeout,
                tap_job_url=args.tap_job_url,
            )
            manifest["farmer"] = {
                "path": str(farmer),
                "sha256": sha256_file(farmer),
                "method": "eso_tap",
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
