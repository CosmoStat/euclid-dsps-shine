COSMOS SED Diagnostics
======================

Purpose
-------

COSMOS-template SED reconstruction is a diagnostic comparison for fitted DSPS
SEDs. It is not a likelihood term and not physical SPS truth.

Commands
--------

Batch fits can save SED diagnostics:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/popcosmos_binned_compressed.yaml \
     fit --limit 100 \
     --batch-size 50 \
     --sed-samples 8 \
     --out outputs/runs/popcosmos_fit_100

Standalone COSMOS checks:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/popcosmos_binned_compressed.yaml \
     check --kind cosmos \
     --limit 20 \
     --out outputs/check/cosmos

Interpretation
--------------

The active likelihood fits observed LSST+Euclid fluxes. COSMOS templates,
component fractions, rest-frame ``*_abs`` columns, and dust proxy columns are
used only to inspect SED shape and normalization after the fact.
