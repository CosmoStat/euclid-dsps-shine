# Catalogue provenance: what must change before the next run

Follow-up: the user confirmed 256/56/59 raw proposal files on Jean-Zay. The
[coherent parent runbook](feniks_coherent_parent_runbook.md) now supplies the
separate dataset-preparation workflow. The analysis below remains the evidence
that motivated it; historical data and receipts have not been overwritten.

## Decision

Do not restart the old population/posterior launcher unchanged. Reuse the
physical proposal pool if it still exists, but prepare a **new versioned 15D
photometric benchmark** with one frozen simulator, explicit population weights,
and no hidden photometric cuts before the intended r selection. This does not
require rerunning Diffsky halo/lightcone generation when the proposals survive.

The current catalogue remains useful as an intentionally mismatched historical
test. Reproducing its fluxes is not the same as showing which numerical
integrator is more accurate. Do not downgrade the production integrator merely
to improve agreement with old generated data.

## 1. The truth reference applies proposal weights twice

The native generator's `resample_weighted_proposals` draws final objects with
probability proportional to `galaxy_weight`. Its contract explicitly calls
the result an **unweighted final catalogue**. Layer construction then selects
photometric candidates and takes a uniform subsample, retaining the original
proposal weights as columns for provenance.

`scripts/feniks_avi_next_validation.py:build_selection_closure` chooses
`galaxy_weight` whenever present, and writes it as `population_weight`.
`scripts/report_feniks_forward_population.py` uses these truth weights by
default. There is no correction for their previous use in constructing the
sample. Reapplying them changes the target instead of recovering its parent.

For the actual synchronized catalogue, SHA256
`8a3552e17f59c22e3557a3873bb98f04e1945cbe223a711422d08d7d9ba8f94b`:

| Reference | Rows | Mean z, unweighted | Mean z, proposal-weighted again |
|---|---:|---:|---:|
| Full historical root catalogue | 50000 | 1.36539 | 1.25738 |
| Blind truth cohort | 7578 | 1.39276 | 1.28451 |

The copied `population_weight` equals the retained `galaxy_weight` exactly on
all 7578 matched identities. The proper empirical reference for this resampled
catalogue is equal row weights, not equal weights for the original proposal
pool. Raw proposals still require their sampling weights, applied **once**.
This distinction is catalogue-specific, not permission to discard weights
from every dataset in the repository.

Using the saved 16384 parent draws from
`avi_forward_population_r29_20260920_162915/report/learned_parent_15d.parquet`,
the following comparison requires no retraining or new simulation. Both
distances use the SAME unweighted blind-truth IQR as their denominator; they
are not a byte-for-byte reproduction of older resampled-report metrics.

| Physical coordinate | W1/IQR versus weighted-again truth | W1/IQR versus unweighted truth |
|---|---:|---:|
| Redshift | 0.24235 | 0.09228 |
| Log stellar mass | 0.33280 | 0.47922 |
| Log stellar metallicity | 0.59490 | 0.23945 |
| Dust Av | 0.31861 | 0.10117 |
| Dust slope | 0.40223 | 0.18624 |

The learned mean z is 1.47044. Correcting the reference removes part, not all,
of the apparent redshift discrepancy. Mass disagreement gets worse. This is
not a success certificate. Historical weighted-truth flow-capacity fits also
learned the declared weighted target: changing its scientific meaning does
not retroactively make their fit errors disappear.

## 2. Native generation and current inference use different flux contracts

The generator manifest records commit
`5a41c67e66d88bec61b4158b54562909cf340223`, `diffsky_basic` native SFHs and
legacy numerical settings. The spline postprocess projects SFHs to ten
contrasts. `build_feniks_spline15d_amortized_catalog.py` and the SBEB catalogue
join retain the old native fluxes; they do **not** rephotometer those spline
parameters. The word `exact` in a truth filename does not make its projected
parameters exact generators of those retained fluxes.

The inherited inference model subsequently switched to merged quadrature,
float64 MDF weights and float64 spline calculations. The precision audit
accidentally compared this current configuration with itself. A new read-only
local replay uses the same 128 stored objects in three distinct paths:

| Band | Native + legacy | Spline + legacy | Spline + current |
|---|---:|---:|---:|
| Roman F087 | 0.01780 | 0.09874 | 1.42602 |
| Roman F062 | 0.01430 | 0.10729 | 0.85278 |
| Roman F106 | 0.01794 | 0.14222 | 0.53993 |
| Roman F146 | 0.01940 | 0.17116 | 0.38955 |
| Euclid VIS | 0.00685 | 0.04519 | 0.32948 |

Values are per-band p95 absolute differences from stored noiseless fluxes,
divided by stored reported flux errors. Largest per-band p95: 0.01940 native,
0.17116 legacy spline and 1.42602 current spline. The legacy spline still has
object-level outliers, including a 1.739-sigma maximum in u. A p95 screen is
not an all-object closure test.

This isolates a substantial **numerical-contract change** contribution in
the problematic bands, in addition to SFH representation differences. It does
not isolate the three numerical flags individually or prove the old settings
are more accurate. No flow or classifier is involved in this replay.

Limitations: the replay uses current source with generator-compatible flags,
not an exactly recreated July environment. Local DSPS/JAX are 0.4.7/0.10.0,
versus manifest versions 0.4.8/0.10.2. Native replay residuals are small but
not bitwise zero. Both use diffstar 1.0.3 and diffmah 0.7.3. The local .venv
lacks those optional packages, so their pure-Python packages were loaded from
the existing shine environment for this measurement. This was CPU execution.

Artifacts:
`outputs/forward_population_results/avi_population_precision_20260925_105102/provenance_replay/`

- `native_vs_spline_replay.png`: the three distinct contracts, all 18 bands.
- `replay_metrics.csv`, `*_residuals.csv`: summary and per-object evidence.
- `provenance.json`: input hashes, versions, generator cuts and weight checks.
- `effective_configs.json`: exact resolved replay configurations.
- `FINAL.json`: execution complete, production readiness explicitly false.

## 3. The root catalogue was already selected before r < 29

The native manifest's `output_layers.inference_ready` is mirrored to root.
It requires at least five bands with true S/N > 5 and at least two magnitude
limit passes. An earlier support filter excludes clipped metallicities.
The later noisy r cut acts on this **already restricted catalogue**.

An r-only efficiency cannot undo earlier photometric cuts. Learning the
historical reference therefore does not demonstrate recovery of the full
photometrically unselected population. A configured physical support domain
must also be stated explicitly, including metallicity limits; it must not
silently stand in for the entire astrophysical population.

There is also a split caveat: the 50000 rows contain 41409 unique
`source_proposal_id` values. Splitting by reassigned `object_id` leaves
1659/7578 blind rows sharing an underlying proposal with the r-selected
training cohort. These are different noisy observations but not independent
latent truths. Group future splits by effective proposal identity; do not interpret
object-ID separation as fully independent truth-distribution validation.

Further source inspection confirms that even `source_proposal_id` separation
is insufficient: both the historical backend at `5a41c67` and the current
generator seed each realization with `source_seed + shard_index`. Original
split seeds 26061701/26061702/26061703 thus overlap. The earlier 1659 count is
not a complete effective-identity leakage count. The new preparation uses the
256 original train shards as a canonical pool (covering all effective seeds),
holds out entire realizations, and records `effective_seed:row` keys. It never
adds the 115 overlapping validation/test shards as extra population mass.

Direct readback of `all_50k.parquet` confirms the collision: 41409 original
IDs become 38243 effective keys. There are 2751 effective keys carrying multiple
original IDs (7506 catalogue rows); their 18 native truth coordinates agree
exactly. This is observed duplication, not only a possible random-seed concern.

## Minimal next work

1. **CPU, existing results:** rerender catalogue comparisons with explicit
   empirical row weights, preserving original reports. This fixes interpretation,
   not the learned model. Do not rerun the existing synthetic bootstrap campaign.
2. **One dataset preparation:** recover pre-photometric-cut proposal shards;
   apply proposal weights once, project SFHs to 15D, split by source identity,
   then generate noiseless fluxes, noise, errors and masks under one frozen
   current decoder. Save the unselected parent and its selected r<29 view.
   Recompute selected identities and efficiencies rather than copying them.
   Preserve the physical and SFH correlations of the parent proposals; do not
   replace truth with draws from the fitted 128-component family just to pass.
3. **Short correctness gate, same job chain:** check weight semantics,
   decoder/flux round trip, noise, selection, identities and absence of train/test
   proposal overlap. Validate the numerical decoder independently of its own
   generated data. Self-consistency is necessary, not physical correctness.
4. **One confirmation campaign:** fit the parent from observed features only;
   freeze it and train the supervised 15D posterior. Measure parent joint/marginal
   closure, observable predictions, coverage/ranks/bias/widths and conditional-SFH
   robustness on independent objects. Keeping SFH variable does not guarantee
   the fixed reference conditional matches this parent. Retain that model-error
   question rather than hiding it with a matched-family synthetic target.

What can be reused: proposal latents, physical assets/filters, mathematical
selection algebra, numerical optimizer and pipeline infrastructure. Existing
forward banks may be reused only if hashes of the full decoder/noise/mask/
feature/selection contract and reference prior agree. A new target catalogue
does not automatically invalidate compatible reference simulations. A changed
observation contract does invalidate affected flux features and alpha estimates.
The final posterior bank is tied to its learned parent and must not be treated
as draws from a newly fitted parent without a justified correction.

The old catalogue's proposal paths are recorded in its manifest but are absent
from this local mirror. Remote presence remains unverified. Do not invent a
regeneration command that assumes those large inputs exist. First check on
Jean-Zay (read-only, no sbatch or conda required):

```bash
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
DATA=Data/diffsky/synthetic/feniks_260617_dsps_closure_18band
for SPLIT in train validation test; do
  printf '%s: ' "$SPLIT"
  if [ -d "$DATA/proposals/$SPLIT" ]; then
    find "$DATA/proposals/$SPLIT" -maxdepth 1 -type f -name 'shard_*.parquet' | wc -l
  else
    printf 'proposal directory missing\n'
  fi
done
```

The manifest expects 256/56/59 shards. File counts are only an availability
screen, not a checksum/schema validation. If absent, locate an archived proposal
pool first; only then consider new Diffsky generation. `survey_like` and the
root 50k catalogue are both photometrically selected substitutes, not raw parents.

The original training/resubmission commands do not implement these corrected
contracts. Use the separate coherent-parent preparation workflow, not an
unchanged restart of the old campaign. No production training is submitted by
either this audit helper or the new dataset-preparation launcher.

## Reproduce the small replay

Use an environment with the existing project and its diffstar optional extra.
Give an unused output directory. From this local checkout:

```bash
RESULTS=outputs/forward_population_results
OLD="$RESULTS/avi_sbeb_benchmark_20260917_230455"
PREC="$RESULTS/avi_population_precision_20260925_105102"
DATA=Data/diffsky/synthetic/feniks_260617_dsps_closure_18band
JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu EUCLID_DSPS_REQUIRE_GPU=0 \
JAX_ENABLE_X64=true python -m scripts.audit_feniks_catalogue_provenance \
  --catalogue "$DATA/all_50k.parquet" --generator-manifest "$DATA/manifest.yaml" \
  --truth "$OLD/cohorts/r29/selection/true_parent.parquet" \
  --rows-csv "$PREC/decoder/residuals.csv" \
  --generator-config configs/diffsky_synthetic_feniks_260617_50k_survey_like_18band.yaml \
  --inference-config "$OLD/runtime/r29/source_config.yaml" \
  --out "$PREC/provenance_replay_$(date +%Y%m%d_%H%M%S)" --limit 128 --batch-size 8
```

This is a reproduction command, not an additional experiment required on the
critical path. The replay is already complete locally. Two focused helper tests,
Ruff and compilation passed. No cluster job was submitted.
