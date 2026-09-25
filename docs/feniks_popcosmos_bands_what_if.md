# LSST + Euclid + Roman + Pop-COSMOS: assessment only

No additional-band configuration or simulation is implemented here.

## Verified inventory

The target comparison is Thorp et al. (2024),
[arXiv:2406.19437, section III.1](https://arxiv.org/html/2406.19437v2#S3.SS1).
It uses 26 COSMOS2020 Farmer bands:

| Family | Count | Existing local curves |
|---|---:|---|
| CFHT u* | 1 | `u_megaprime_sagem.dat` |
| Subaru HSC grizy | 5 | `hsc_*.dat` |
| UltraVISTA YJHKs | 4 | `uvista_*_cosmos.dat` |
| Subaru intermediate bands | 12 | `ia*_cosmos.dat` |
| Subaru narrow bands | 2 | `NB711.SuprimeCam.dat`, `NB816.SuprimeCam.dat` |
| Spitzer IRAC 1/2 | 2 | `irac1_cosmos.dat`, `irac2_cosmos.dat` |

All 26 `.dat` files under `Data/cosmos2020/assets/filters/` were read with
`euclid_dsps.filters.load_filters` on 2026-09-25; arrays are finite and have
positive throughput. The order is recorded by `COSMOS_BANDS` in
`euclid_dsps/cosmos2020.py`. This inventory does not establish exact equality
to the release curves/calibration used in the publication.

With the present 18 filters, the union has 44 separate instrument bands. HSC r
and LSST r, for example, are not identical filters. Nearby broad bands can help
precision but do not necessarily resolve new degeneracies. The intermediate
and narrow bands offer new spectral discrimination; IRAC extends the wavelength
coverage. Gains at r<29 remain to be measured at the corresponding depths.

## What a depth-only experiment would establish

A limiting magnitude m5 fixes one reference flux scale. A simple Gaussian
noise model plus this depth would support an explicitly idealized survey
forecast. It does not reproduce the COSMOS measurement process or constitute
a matched Pop-COSMOS benchmark.

Before making that comparison, qualify curve units/conventions, spatial depth
variation, flux-dependent errors, masks, non-detections, PSF/deblending effects,
zero points/extinction corrections and relevant cross-band correlations. The
paper uses a robust Student-t2 likelihood with calibrated error-floor and
emission-line terms (section II.2.1); it is not just a Gaussian depth law.
Narrow bands especially expose emission-line/SPS model discrepancies, and IRAC
requires adequate SED wavelength coverage and a justified dust/AGN treatment.

The paper's principal selection is r<25 with positive flux/SNR in every band;
it also analyzes an IRAC-selected extension. Our noisy LSST r<29 selection is
a different scientific target. Do not silently import a positivity cut into
the r<29 experiment: recompute selection efficiencies whenever selection changes.

## Controlled comparison to consider later

One common parent realization and shared objects can support three fixed
observation sets: current 18 bands, matched COSMOS26, and the union of 44.
Preserve shared-band noise draws, explicit survey-specific error models,
independent train/validation/test partitions and a fixed selection for a
pure information-gain comparison. Posterior dimensions remain 15.

For real-catalogue inference, extra fluxes must be measured for the target
objects. Simulation-only extra bands describe a hypothetical survey. A 44-band
result alone cannot establish a better inference method than a 26-band
Pop-COSMOS result. Method superiority requires a matched data/selection test;
the union separately measures the advantage of additional observations.
