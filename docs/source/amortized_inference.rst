Amortized Inference
===================

Purpose
-------

The active amortized workflow learns a photometric posterior approximation for
the controlled synthetic FENIKS/DSPS closure dataset. It combines:

* an encoder ``q_psi(x | flux, flux_err)``;
* the fixed DSPS decoder through ``euclid_dsps.parameter_vectors`` and
  ``euclid_dsps.model``;
* a RealNVP prior in bounded latent space;
* a Student-t flux likelihood.

The production config is:

.. code-block:: text

   configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml

FS2 is still available as a comparison config:

.. code-block:: text

   configs/amortized_fs2_realnvp.yaml

FENIKS Contract
---------------

``configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml`` uses the full
``diffsky_dsps_closure_full`` schema. The latent vector has 18 parameters:

.. code-block:: text

   z_obs
   log10_stellar_mass
   diffstar_lgmcrit
   diffstar_lgy_at_mcrit
   diffstar_indx_lo
   diffstar_indx_hi
   diffstar_lg_qt
   diffstar_qlglgdt
   diffstar_lg_drop
   diffstar_lg_rejuv
   diffmah_logm0
   diffmah_logtc
   diffmah_early_index
   diffmah_late_index
   diffmah_t_peak
   log10_stellar_metallicity
   dust_av
   dust_delta

The encoder input is the configured band vector:

.. code-block:: text

   [flux_1, ..., flux_B, fluxerr_1, ..., fluxerr_B]

For the 14-band FENIKS production sample, ``B=14`` and the feature dimension is
28. The 18-band survey-like FENIKS path uses the matching 18-band config and
feature dimensions.

Feature Normalization
---------------------

Fluxes are transformed as ``asinh(flux / flux_scale)`` and errors as
``log(fluxerr / err_scale + eps)``. The scales are learned from the training
catalog and written to ``feature_stats.json``. These normalized values are
encoder inputs only; DSPS always receives physical theta values and native
photometry units.

Prior Source
------------

The production config uses:

.. code-block:: yaml

   amortized:
     prior:
       source: supervised_checkpoint
       train_jointly: false

That checkpoint is produced by
``configs/prior_diffsky_synthetic_feniks_full_realnvp.yaml``. This keeps the
population prior learned from controlled FENIKS truth separate from the
photometric encoder training.

Run
---

Train:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml \
     amortized-train-diffsky \
     --out outputs/runs/amortized_diffsky_synthetic_feniks_full

Infer on the held-out split:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml \
     amortized-infer-diffsky \
     --checkpoint outputs/runs/amortized_diffsky_synthetic_feniks_full/checkpoints/best.eqx \
     --dataset Data/diffsky/synthetic/feniks_260617_dsps_closure/test.parquet \
     --out outputs/runs/amortized_diffsky_synthetic_feniks_full_test_infer

Finalize sharded inference:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml \
     amortized-finalize-inference \
     --out outputs/runs/amortized_diffsky_synthetic_feniks_full_test_infer

Outputs
-------

Training writes checkpoints, ``feature_stats.json``, loss histories, latent
diagnostics, and prior samples. Inference writes posterior samples, posterior
summaries, predictive flux residuals, redshift metrics, truth snapshots, and
collapse diagnostics.

Scientific Use
--------------

A low photometric residual is not a physical recovery claim. Use the held-out
FENIKS truths to inspect posterior calibration, parameter residuals, derived
quantities, and MAP/MCLMC comparisons before interpreting recovered galaxy
parameters.

Spline-15D Frozen-Prior Path
----------------------------

``configs/amortized_feniks_spline15d_18band_gpu.yaml`` replaces the 13 native
Diffstar/Diffmah SFH parameters with ten spline log-SFR contrasts. Together
with redshift, stellar mass, stellar metallicity, ``dust_av``, and
``dust_delta``, the encoder posterior has 15 dimensions.

The RealNVP checkpoint is loaded with its serialized mixed marginal transform:
positive log transforms for redshift and ``dust_av``, shifted-asinh transforms
for the other coordinates, and no whitening. The flow weights remain frozen;
only the photometric encoder is optimized. Every posterior sample follows this
fully differentiable path:

.. code-block:: text

   encoder x -> inverse prior normalization -> 15 physical parameters
             -> PCHIP spline SFH -> stellar-mass normalization
             -> DSPS age weights, MDF, dust and IGM -> 18-band photometry

The source Diffsky dataset is not regenerated. The helper below joins the
existing photometry with ``<split>_exact.parquet`` by ``object_id``:

.. code-block:: bash

   python scripts/build_feniks_spline15d_amortized_catalog.py \
     --source-dir Data/diffsky/synthetic/feniks_260617_dsps_closure_18band_grouped \
     --spline-dir Data/diffsky/synthetic/feniks_260617_spline15d_grouped_v3 \
     --out Data/diffsky/synthetic/feniks_260617_spline15d_grouped_v3/amortized

For Jean-Zay, ``scripts/feniks_spline15d_amortized_h100.slurm`` performs that
join if necessary, validates the completed prior checkpoint, and launches the
encoder training. Submit it with an ``afterok`` dependency on the active prior
continuation job so the selected prior checkpoint exists before training.
