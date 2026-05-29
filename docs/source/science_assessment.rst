Science Assessment
==================

Current Status
--------------

The current DSPS model is validated as an FSPS/Prospector-like broad-band
forward model for the active PopCosmos-like parameterization. The recommended
default config is:

.. code-block:: text

   configs/popcosmos_binned.yaml

It includes step SFH, Chabrier SSPs, Prospector/FSPS-like dust, raw FSPS/CLOUDY
gas, and the FSPS-native AGN component grid. The no-AGN config remains available
as an ablation/fallback, not as the main path.

Benchmark Result
----------------

Latest closure report:

.. code-block:: text

   outputs/report/popcosmos_binned_full_forward_fsps_closure_n500/report.md

Summary of the production broad-band levels:

.. list-table::
   :header-rows: 1

   * - Level
     - Median band p95 abs delta mag
     - Max band p95 abs delta mag
     - Verdict
   * - ``full_noagn``
     - ``0.0129``
     - ``0.0166``
     - Pass
   * - ``full_agn``
     - ``0.0123``
     - ``0.0371``
     - Pass

The configured target was median broad-band absolute delta magnitude below
``0.02`` and p95 absolute delta magnitude below ``0.05`` on bright finite rows.
The production full levels satisfy that target against the local
FSPS/Prospector reference.

What Is Validated
-----------------

The benchmark validates the integrated broad-band behavior of:

* Chabrier SSP normalization and metallicity conversion with ``z_sun=0.0142``;
* PopCosmos-like step SFH weights on the FSPS age grid;
* stored FSPS surviving stellar mass fractions;
* luminosity distance, redshift, and filter integration;
* FSPS Madau95-like IGM behavior;
* Prospector/FSPS-like dust for broad-band photometry;
* raw FSPS/CLOUDY gas grid in broad bands;
* FSPS-native AGN component grid with the current host attenuation and
  AGN/IGM ordering.

Remaining Scientific Caveats
----------------------------

This is not an official PopCosmos reproduction. It is a clean
FSPS/Prospector-like DSPS model.

Remaining caveats:

* ``emission_line_corrections: none`` means no PopCosmos learned line-by-line
  corrections are applied. The gas model is raw FSPS/CLOUDY.
* The Diffstar path is a comparison path and still depends on the reduced
  Diffstar/default-MAH setup.
* Very faint magnitude-space diagnostic rows can be non-finite. Inference uses
  flux-space likelihoods.
* The AGN component grid is large. Benchmarking uses lazy loading; production
  fitting loads the configured grid and should start with small batches.

Practical Recommendation
------------------------

Use the full AGN binned config for new broad-band experiments:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/popcosmos_binned.yaml \
     fit --limit 20 \
     --batch-size 2 \
     --out outputs/runs/dev_popcosmos_fullagn_batch20 \
     --sed-samples 4

Use no-AGN only for controlled tests:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/popcosmos_binned_noagn.yaml \
     fit --limit 20 \
     --batch-size 5 \
     --out outputs/runs/dev_popcosmos_noagn_batch20 \
     --sed-samples 4
