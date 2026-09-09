"""CPU-only concentration audit of recorded direct draws; never reweight a teacher."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_dsps.amortized.population_vem import sha256_file


def audit(root, out):
    if out.exists():
        raise FileExistsError("preserve existing audit directory")
    out.mkdir(parents=True)
    records, coordinates, sources = [], [], {}
    for path in sorted((root / "cases").rglob("direct_draws.npz")):
        summary_path = path.with_name("SUMMARY.json")
        summary = json.loads(summary_path.read_text())
        digest = sha256_file(path)
        if digest != summary["direct_draws_sha256"]:
            raise ValueError(f"draws changed: {path}")
        identity = str(path.parent.relative_to(root))
        sources[identity] = dict(
            draws_sha256=digest, summary_sha256=sha256_file(summary_path)
        )
        with np.load(path, allow_pickle=False) as bank:
            x = bank["x"][:, 0, :].astype(np.float64)
            parts = {
                k: bank[k][:, 0].astype(np.float64)
                for k in ("logq", "logprior", "loglike")
            }
        logw = parts["logprior"] + parts["loglike"] - parts["logq"]
        if not np.isfinite(logw).all() or not np.isfinite(x).all():
            records.append(dict(distribution=identity, status="NONFINITE_INPUT"))
            continue
        weights = np.exp(logw - logw.max())
        weights /= weights.sum()
        order = np.argsort(weights)[::-1]
        row = dict(
            distribution=identity,
            status="DESCRIPTIVE_ONLY",
            draws=len(weights),
            ess=1 / np.sum(weights**2),
            max_weight=weights.max(),
            top10_weight=weights[order[:10]].sum(),
            log_weight_range=np.ptp(logw),
        )
        for name, values in parts.items():
            row[name + "_mean"] = values.mean()
            row[name + "_weighted_mean"] = weights @ values
            row[name + "_top10_mean"] = values[order[:10]].mean()
        records.append(row)
        mean, std = x.mean(axis=0), x.std(axis=0)
        weighted = weights @ x
        centered = (x - mean) / np.where(std > 0, std, 1)
        cov = (centered * weights[:, None]).T @ centered - np.outer(
            weights @ centered, weights @ centered
        )
        pd.DataFrame(cov).to_csv(
            out / (identity.replace("/", "_") + "_weighted_cov.csv"), index=False
        )
        for j in range(x.shape[1]):
            coordinates.append(
                dict(
                    distribution=identity,
                    latent_coordinate=j,
                    mean=mean[j],
                    std=std[j],
                    weighted_mean=weighted[j],
                    weighted_shift_in_q_std=(weighted[j] - mean[j]) / std[j]
                    if std[j] > 0
                    else None,
                    top10_mean=x[order[:10], j].mean(),
                )
            )
    pd.DataFrame(records).to_csv(out / "concentration.csv", index=False)
    pd.DataFrame(coordinates).to_csv(out / "coordinates.csv", index=False)
    (out / "AUDIT.json").write_text(
        json.dumps(
            dict(
                status="SAVED_DRAW_AUDIT_COMPLETE",
                source_root=str(root.resolve()),
                sources=sources,
                distributions=len(records),
                decoder_calls=0,
                scientific_promotion=False,
                interpretation="Weighted moments describe this finite bank only; low ESS makes them unreliable posterior estimates. Coordinates are latent_x in config free-parameter order. Covariances use q-standardized coordinates. No truth, teacher, or checkpoint selection.",
            ),
            indent=2,
        )
    )
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    frame = audit(args.root, args.out)
    print("Distributions audited:", len(frame))
    if "max_weight" in frame:
        print(
            frame.sort_values("max_weight", ascending=False)
            .head(12)
            .to_string(index=False)
        )
    print("No decoder calls, no posterior promotion:", args.out)


if __name__ == "__main__":
    main()
