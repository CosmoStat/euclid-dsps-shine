COSMOS SED Diagnostics
======================

Purpose
-------

COSMOS-template SEDs are pseudo-ground truth diagnostics. They are useful for
visual checks against the DSPS SED inferred from Euclid + LSST photometry, but
they are not wavelength-by-wavelength physical SPS truth.

Current Workflow
----------------

MAP fits write SED diagnostics directly:

.. code-block:: bash

   euclid-dsps fit \
     --config configs/fs2_phz1_science.yaml \
     --limit 100 \
     --batch-size 50 \
     --sed-samples 8 \
     --out outputs/runs/science_fit

For batch runs, the saved SED diagnostics are selected from the worst fits
rather than the first rows. Inspect ``sed_diagnostics_manifest.csv`` for the
row index, rank, and score.

Standalone checks remain available through ``check``:

.. code-block:: bash

   euclid-dsps check \
     --config configs/fs2_phz1_science.yaml \
     --kind cosmos \
     --limit 20 \
     --out outputs/check/cosmos

Interpretation
--------------

The science preset uses ``dust_model: cosmos_proxy_fixed``. COSMOS dust columns
are injected into DSPS so SED shape comparison is not mixed with a free dust
fit. Emission-line target sets stay out of the likelihood until a compatible
line model or SSP line assets exist.
