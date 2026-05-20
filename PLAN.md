# DSPS Plan

Living plan for the simplified Euclid + LSST DSPS workflow.

## Active Goal

Build a DSPS-like pipeline that fits SED parameters from Euclid + LSST
photometry, compares fitted quantities to FS2 truth/proxy columns, and keeps
workflow/docs small enough to audit.

## Active Public Workflow

Use one science config:

```bash
configs/fs2_phz1_science.yaml
```

Use three commands:

```bash
euclid-dsps fit --limit 100 --batch-size 50 --sed-samples 8 --out outputs/runs/science_fit
euclid-dsps posterior --index 0 --num-warmup 100 --num-samples 200 --out outputs/runs/posterior_one
euclid-dsps check --kind cosmos --limit 20 --out outputs/check/cosmos
```

## Current Science Contract

- Free parameters: `z_obs`, `log10_formed_mass_msun`, `sfh_t_peak`,
  `sfh_tau`, `log10_metallicity`.
- Not free in main config: `log10_sfr`, `dust_av`.
- Redshift: initialized from deterministic random flat draw, then fitted from
  photometry. No photo-z or PHZ interval prior.
- Dust: COSMOS proxy dust is row-injected in the main config.
- Metallicity truth: proxy only, because gas-phase O/H is not stellar
  metallicity.
- Emission-line columns: diagnostics only until a line model/assets exist.
- Priors: `weak_physical` for science, `flat_debug` for debugging,
  `popcosmos_like` reserved until exact mapping/units exist.

## Completed In Current Phase

- Public CLI simplified to `fit`, `posterior`, `check`.
- Old configs moved to `configs/legacy/`.
- PHZ interval prior support removed.
- Fast-grid/warmstart public config removed.
- Science config switched to random flat redshift initialization.
- Sample priors must match actual free parameters.
- Batch SED diagnostics now select worst-fit galaxies.
- Blank failed heatmaps no longer get written.
- Docs reduced to active workflow and science assumptions.
- Verification passed: `compileall`, full `pytest`, CLI CPU EDA smoke, and CLI
  CPU one-row/batch fit smoke with SED diagnostics.

## Remaining Work

- Validate exact POP-COSMOS parameter mapping and units before implementing
  `popcosmos_like`.
- Decide whether lognormal SFH is sufficient after comparing residual trends
  versus truth redshift, mass, SFR, and metallicity proxy.
- Add a small formal smoke dataset that exercises the full `fit` command
  without private FS2 data.
- Consider splitting `workflows/core.py` and `reporting/core.py` after science
  behavior stabilizes.
