COSMOS Template SED Reconstruction
==================================

Purpose
-------

The Euclid Flagship / CosmoHub catalog does not contain true
wavelength-by-wavelength spectra. The ``cosmos-sed`` workflow reconstructs a
template-based pseudo-ground-truth SED from the catalog latent COSMOS columns:

* ``sed_cosmos_1`` and ``sed_cosmos_2`` select two templates from the local
  LePhare ``COSMOS_MOD.list`` order.
* ``ebv_cosmos_1`` and ``ebv_cosmos_2`` provide component-level color excess.
* ``ext_curve_cosmos_1`` and ``ext_curve_cosmos_2`` select component-level
  attenuation curves.
* ``frac_cosmos_1`` and ``frac_cosmos_2`` provide component weights.

This reconstruction is a diagnostic reference SED. It is not exact physical
SPS ground truth and should not be used as a training label without this caveat.

Template Resources
------------------

The preferred local source is the SciPIC value-added directory:

.. code-block:: text

   Data/value_added_data/galaxy_seds/
   Data/value_added_data/galaxy_extincts/
   Data/value_added_data/filters/

``galaxy_seds`` contains the 31 COSMOS templates from Ilbert et al. (2009):
template IDs 0--6 are ellipticals, 7--18 are S0/spiral templates, and 19--30
are starburst templates. The files are CSV tables of wavelength in Angstrom and
``F_lambda`` in ``erg/cm^2/s/Angstrom`` with arbitrary normalization. They are
template shapes, not object-specific spectra.

``galaxy_extincts`` contains one flat-Fnu no-extinction spectrum and four
attenuated flat-Fnu spectra. The workflow derives the attenuation law as:

.. math::

   k(\lambda) =
   \frac{\log_{10}(F_{noext}/F_{ext})}{0.4 \times 0.2}.

This matches the value-added data documentation and avoids depending on an
external LePhare cache for the COSMOS/SciPIC resources.

LePhare Fallback
----------------

If ``cosmos_sed.value_added_data_dir`` is not configured, the workflow reads
local LePhare auxiliary data downloaded with the script
``download_lephare_sed_gen.py``:

.. code-block:: text

   /home/<user>/.cache/lephare/data/sed/GAL/COSMOS_SED/COSMOS_MOD.list
   /home/<user>/.cache/lephare/data/ext/

``COSMOS_MOD.list`` has 31 entries in the local download. Catalog template IDs
``0`` through ``30`` map directly to this file order. The template files follow
the same wavelength and ``F_lambda`` convention. LePhare's manual states that
ASCII SED templates are written as ``lambda[A]`` and flux in ``erg/s/A/cm^2``
with increasing wavelength. LePhare converts other template formats into this
flux-per-wavelength form when building its binary SED libraries.

Reference:
  https://lephare.readthedocs.io/en/latest/detailed.html#build-the-rest-frame-templates-library

For this project, the absolute normalization of those COSMOS template files is
not treated as physical truth. The template shape is used first, then the whole
combined SED is scaled to the catalog rest-frame Euclid absolute fluxes. The
scaled product is therefore a COSMOS-template proxy SED, not a measured true
spectrum.

Extinction Curves
-----------------

The extinction-code mapping is configured, not hidden in code:

.. code-block:: yaml

   cosmos_sed:
     extinction:
       curves:
         0: none
         1: SMC_prevot
         2: SB_calzetti
         3: SB_calzetti_bump1
         4: SB_calzetti_bump2

LePhare documents these curves as ``k(lambda[A])`` versus ``lambda[A]``. The
attenuation applied to each component is:

.. math::

   F_{\lambda,att}(\lambda)
   =
   F_{\lambda}(\lambda)
   10^{-0.4\,E(B-V)\,k(\lambda)}.

If ``E(B-V)`` is missing, non-finite, or zero, no attenuation is applied. If
the curve code maps to ``none``, no attenuation is applied.

Component Combination
---------------------

For one row, the proxy SED is:

.. math::

   F_{\lambda,proxy}
   =
   f_1 F_{\lambda,1,att}
   +
   f_2 F_{\lambda,2,att}.

When ``frac_cosmos_1`` and ``frac_cosmos_2`` are present and their finite sum
is positive, the workflow normalizes them internally:

.. math::

   f_1 = frac_1/(frac_1 + frac_2), \quad
   f_2 = frac_2/(frac_1 + frac_2).

The current local parquet contains ``frac_cosmos_1`` and ``frac_cosmos_2``.
The default science config therefore requires them:

.. code-block:: yaml

   component_fraction_policy: strict

This is reported in ``cosmos_sed_validation.json`` and
``cosmos_sed_diagnostics.csv``. Older exports can still use
``component_fraction_policy: equal_if_missing`` as an explicit fallback.

Synthetic Photometry
--------------------

The template SED is normalized with Euclid rest-frame absolute flux columns:

.. code-block:: text

   euclid_vis_abs
   euclid_nisp_y_abs
   euclid_nisp_j_abs
   euclid_nisp_h_abs

These columns are interpreted as rest-frame flux density at 10 parsec in
``erg s^-1 cm^-2 Hz^-1``. The workflow integrates the unscaled
``F_lambda`` template through the Euclid passbands and computes a best-fit
scalar:

.. math::

   \alpha =
   \frac{\sum_i F_{\nu,model,i} F_{\nu,catalog,i}}
        {\sum_i F_{\nu,model,i}^2}.

The default filter convention is photon-counting AB-style:

.. math::

   \langle F_\nu \rangle =
   \frac{\int \lambda F_\lambda(\lambda) T(\lambda)\,d\lambda}
        {\int c T(\lambda)/\lambda\,d\lambda}.

``filter_response_kind: energy`` is also available if needed:

.. math::

   \langle F_\nu \rangle =
   \frac{\int F_\lambda(\lambda) T(\lambda)\,d\lambda}
        {\int c T(\lambda)/\lambda^2\,d\lambda}.

CLI
---

Small standalone run:

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1.yaml cosmos-sed \
     --limit 10 \
     --plot-samples 12 \
     --out outputs/runs/cosmos_sed_10

Compare against the current DSPS forward model:

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1.yaml cosmos-sed \
     --limit 10 \
     --compare-dsps \
     --out outputs/runs/cosmos_sed_dsps

Compare against fitted DSPS SEDs for a small sample:

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1.yaml cosmos-sed \
     --limit 3 \
     --fit-dsps \
     --out outputs/runs/cosmos_sed_fit_dsps

Compare against a chunked population MAP DSPS fit:

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1_10band.yaml cosmos-sed \
     --limit 64 \
     --batch-size 32 \
     --population-dsps \
     --plot-samples 16 \
     --out outputs/runs/cosmos_sed_population_dsps

Outputs
-------

Standalone outputs:

* ``cosmos_sed_validation.json``: missing columns, unique template IDs,
  extinction-code values, fraction diagnostics, and a ``value_added_data``
  resource report when the local SciPIC directory is configured.
* ``cosmos_sed_diagnostics.csv``: one row per galaxy with template IDs,
  extinction curves, E(B-V), fraction policy, ``alpha``, and Euclid absolute
  flux residuals.
* ``cosmos_seds.parquet``: long-form reconstructed SED sample with wavelength,
  scaled ``F_lambda``, 10 pc ``Fnu``, and ``Lnu``.
* ``cosmos_sed_example.csv`` and ``cosmos_sed_example.png``: first-row example.
* ``cosmos_sed_sample_set.png``: visual-inspection grid. In standalone mode it
  shows one COSMOS proxy SED per row. With DSPS comparison enabled, each row
  shows inferred DSPS versus COSMOS proxy SED on the left and
  ``log10(DSPS/COSMOS)`` residuals on the right. Wavelength is plotted in
  Angstrom and SEDs use ``Lnu`` to match the DSPS rest-SED comparison.
  ``color_kind`` is the HOD class (0=red sequence, 1=green valley,
  2=blue cloud); ``f1`` is the normalized first COSMOS component fraction.
* ``cosmos_template_pair_heatmap.png`` and
  ``cosmos_fraction_diagnostics.png``: template-pair and component-fraction
  population diagnostics.
* ``synthetic_vs_catalog_abs_flux.csv`` and
  ``synthetic_vs_catalog_abs_flux.png``: normalization reproducibility table and
  plot.

Branch 1 outputs when DSPS comparison is enabled:

* ``branch1_rest_sed_metrics.csv``: RMS log residual, median absolute log
  residual, DSPS-to-COSMOS scale, Euclid rest-color residuals, UV/optical/NIR
  log-slope residuals, and a simple D4000-like break proxy.
* ``branch1_rest_sed_comparison.csv``: common-grid SED comparison values.
* ``branch1_rest_sed_comparison_example.png``: one visual comparison.
* ``branch1_rest_color_residuals.png``: Euclid rest-color residual diagnostic.
* ``branch1_worst_sed_grid.png``: the 16 worst COSMOS-vs-DSPS SED overlays,
  selected by ``rms_log_sed_residual`` rather than by row order.
* ``branch1_rms_residual_heatmap.png``: median RMS log residual by
  ``z_true`` bin and ``color_kind``.
* ``branch1_rest_sed_metrics_by_group.csv`` and
  ``branch1_rest_sed_metrics.png``: grouped diagnostics by ``color_kind`` and
  redshift bin.
* ``branch1_population_validation.csv`` and
  ``branch1_population_validation.png``: population-level metrics versus
  ``color_kind``, redshift bin, apparent magnitude bin, SFR proxy, metallicity
  proxy, stellar-mass bin, COSMOS template pair, and dust-curve pair.

Branch 2 outputs when DSPS comparison is enabled:

* ``branch2_observed_photometry_metrics.csv``: DSPS observed-frame flux
  residuals against the configured target sets. The default is
  ``continuum_internal_dust`` only, because the current DSPS model is continuum
  plus dust and does not include nebular emission lines.
* ``branch2_observed_photometry_chi2.csv``: chi-square summary for target sets
  that include catalog uncertainty columns.
* ``branch2_observed_photometry_metrics_by_group.csv`` and
  ``branch2_observed_flux_residuals.png``: grouped observed-frame diagnostics.
  Relative residuals are clipped robustly for readability. For
  ``noisy_observation`` target sets, the plot uses clipped residuals in units
  of the catalog flux error instead of raw fractional residuals.
* ``branch2_population_validation.csv`` and
  ``branch2_population_validation.png``: population-level observed-photometry
  residual summaries using the same grouping axes as branch 1.

With ``--fit-dsps`` or ``--population-dsps``, the workflow also writes
``cosmos_dsps_fit_results.csv`` and ``cosmos_dsps_fit_trace.csv``. Population
mode additionally writes ``cosmos_dsps_population_hyperparameters.csv``.
Population mode also writes ``cosmos_dsps_population_report.json``. This file
states explicitly that the current population mode is chunk-regularized MAP,
not a learned population prior comparable to pop-cosmos.
The same run also writes the classic fit dashboard files from the photometric
likelihood:

* ``cosmos_dsps_likelihood_dashboard.png``;
* ``cosmos_dsps_likelihood_observed_vs_model.png``;
* ``cosmos_dsps_likelihood_residuals_by_band.png``;
* ``cosmos_dsps_likelihood_redshift_truth.png``;
* ``cosmos_dsps_likelihood_parameter_truth.png``.

Branch 2 target sets are:

.. list-table::
   :header-rows: 1

   * - Target set
     - Columns
   * - ``continuum_internal_dust``
     - all configured likelihood bands, e.g. LSST ``ugrizy`` and Euclid
       VIS/Y/J/H in the 10-band config
   * - ``emission_lines_internal_dust``
     - ``euclid_*_el_model3_ext``
   * - ``emission_lines_internal_dust_mw``
     - ``euclid_*_el_model3_ext_odonnell_ext``
   * - ``noisy_observation``
     - ``euclid_*_el_model3_ext_odonnell_ext_error_realization`` with
       ``euclid_*_el_model3_ext_odonnell_ext_error``

JAX Runtime Note
----------------

The DSPS comparison path uses JAX-vectorized chunks for forward SED generation,
MAP fits, and population MAP fits. The active backend is controlled before
JAX-heavy modules are imported:

.. code-block:: yaml

   runtime:
     jax_platforms: "cpu"
     disable_jax_plugin_autoload: true
     xla_python_client_preallocate: false
     require_gpu: false
     expected_gpu_name:

The default Euclid-only config is CPU-safe. The 10-band GPU config uses
``jax_platforms: "cuda"`` and plugin autoload enabled. Verify the local
``shine`` environment before large runs:

.. code-block:: bash

   /home/maxime/miniforge3/envs/shine/bin/python \
     scripts/check_jax_gpu.py \
     --require-nvidia \
     --hold-seconds 10

For production runs, force a hard failure when JAX does not expose the NVIDIA
GPU:

.. code-block:: bash

   export EUCLID_DSPS_JAX_PLATFORMS=cuda
   export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
   export EUCLID_DSPS_XLA_PYTHON_CLIENT_PREALLOCATE=false
   export EUCLID_DSPS_REQUIRE_GPU=1
   export EUCLID_DSPS_EXPECTED_GPU_NAME=NVIDIA
