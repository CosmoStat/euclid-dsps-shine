Diffsky Robust Population Prior Plan
====================================

Goal
----

This workflow separates four effects that were mixed in the first NN matrix:

* the numerical latent geometry used by the neural encoder;
* the DSPS likelihood optimum for the projected Diffsky truth;
* the effect of the currently learned RealNVP prior;
* the population prior learned post-hoc from MAP/MCLMC rather than directly
  from a collapsing amortized encoder.

The intended final method is empirical-Bayes-like:

1. infer per-galaxy parameters under a flat or very weak physical prior;
2. fit a RealNVP population prior to those MAP/MCLMC inferences;
3. rerun MAP/MCLMC under a tempered learned prior;
4. distill the neural encoder only after the prior target is stable.

Phase 1: Latent Geometry
------------------------

The old ``standardized_logit`` transform used
``fit.free_parameters.<name>.initial`` both as the encoder initialization and
as the center of the latent coordinate system.  A standard normal in latent
space was therefore not neutral in physical space.

New configs can decouple this with:

.. code-block:: yaml

   amortized:
     latent:
       center_source: midpoint
       centers:
         tau2: 1.0
       physical_scales:
         z_obs: 0.05
         tau2: 0.20

The validated low-z reference config is now
``configs/experiments/diffsky_hltds_joint_realnvp_kl_annealed_zscale005_tau2safe_h100.yaml``.
It keeps encoder initialization tied to ``fit.free_parameters.<name>.initial``
but changes the coordinate system used by the standard-normal latent reference:

* ``center_source: midpoint`` removes hidden physical constants from the
  neutral latent center;
* ``z_obs: 0.05`` makes one latent standard deviation correspond to a
  realistic low-z redshift scale instead of spreading mass to the bounds;
* ``centers.tau2: 1.0`` and ``physical_scales.tau2: 0.20`` avoid putting the
  reference mass close to the ``tau2=0`` dust boundary.

The geometry run from job ``1182520`` measured:

.. list-table::
   :header-rows: 1

   * - config
     - center
     - max near-bound fraction
     - parameters near bounds
   * - ``geometry_annealed_current``
     - ``fit_initial``
     - ``0.383735``
     - ``z_obs``, ``tau2``, ``tau1_over_tau2``
   * - ``geometry_annealed_zscale005``
     - midpoint
     - ``0.0``
     - none
   * - ``geometry_annealed_zscale003``
     - midpoint
     - ``0.0``
     - none
   * - ``geometry_annealed_zscale005_tau2safe``
     - midpoint plus ``tau2=1.0``
     - ``0.0``
     - none

This is not just a redshift issue.  In the old config, the reference
``x ~ N(0,1)`` put about ``31.8%`` of ``z_obs`` samples and ``38.4%`` of
``tau2`` samples inside the outer 5% of their physical bounds.  The dust
geometry can feed back into redshift through the usual dust/SFH/redshift
degeneracy.

Before launching a large training run, write the implicit-prior diagnostic:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/experiments/diffsky_hltds_joint_realnvp_kl_annealed_zscale005_tau2safe_h100.yaml \
     diffsky-latent-prior-geometry \
     --out outputs/runs/diffsky_robust_prior_diagnostics/geometry_tau2safe

Inspect:

* ``latent_prior_geometry.csv``
* ``latent_prior_geometry.png``
* ``parameters_near_bounds_5pct`` in ``latent_prior_geometry.json``

Phase 2: Projected Truth vs DSPS Optimum
----------------------------------------

The closure diagnostic now writes projected truth, NN posterior median, band
residuals, and fixed-nuisance redshift profiles before running any expensive
MAP.  Use ``--skip-map`` for the global diagnostic pass, then run MAP as a
separate sharded job.

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/experiments/diffsky_hltds_autoencoder_nokl_deterministic_h100.yaml \
     diffsky-closure-optimum-diagnostics \
     --checkpoint outputs/runs/diffsky_nn_matrix/nokl_det_balanced20k_seed42/checkpoints/best.eqx \
     --feature-stats outputs/runs/diffsky_nn_matrix/nokl_det_balanced20k_seed42/feature_stats.json \
     --nn-run outputs/runs/diffsky_nn_matrix/nokl_det_balanced20k_seed42_infer \
     --row-indices-file outputs/rowsets/diffsky_hltds_nokl_reference/balanced20k.txt \
     --out outputs/runs/diffsky_robust_prior_diagnostics/closure_optimum \
     --skip-map

Inspect:

* ``closure_optimum_summary.csv``
* ``delta_loglike_map_flat_minus_truth``
* ``z_true_vs_z_map_flat.png``
* ``loglike_truth_vs_map.png``
* ``closure_residuals_by_band_summary.csv``

Interpretation:

* if flat MAP stays near truth but NN goes high-z, the failure is amortization
  or learned-prior feedback;
* if flat MAP also prefers the upper redshift bound, the issue is in the DSPS
  likelihood, projected-truth contract, or photometric degeneracy;
* if projected truth is systematically low-likelihood, do not learn the
  population prior directly from the NN closure output.

Phase 3: Redshift Profiles
--------------------------

The closure command also writes fixed-nuisance redshift profiles for selected
objects:

* ``redshift_profile_samples.parquet``
* ``redshift_profile_summary.csv``
* ``redshift_profiles_fixed_nuisance.png``

These profiles answer whether the likelihood itself is monotonic toward
``z_obs`` upper bound when nuisance parameters are held fixed.

Phase 4: MAP Under Current Learned Prior
----------------------------------------

The old monolithic sweep is only appropriate for small throughput checks.  It
now writes per-batch shards and resumes from existing shards, but a full
``balanced20k`` sweep should still be launched with the sharded SLURM script
below.

For a smoke test:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/experiments/diffsky_hltds_joint_realnvp_kl_annealed_h100.yaml \
     diffsky-map-prior-sweep \
     --checkpoint outputs/runs/diffsky_nn_matrix/kl_annealed_balanced20k_seed42/checkpoints/best.eqx \
     --feature-stats outputs/runs/diffsky_nn_matrix/kl_annealed_balanced20k_seed42/feature_stats.json \
     --row-indices-file outputs/rowsets/diffsky_hltds_nokl_reference/balanced20k.txt \
     --out outputs/runs/diffsky_robust_prior_diagnostics/map_prior_sweep \
     --weights 0,0.1,1.0 \
     --limit 512 \
     --batch-size 256 \
     --n-starts 4 \
     --start-chunk-size 4 \
     --maxiter 120

For the full pass, shard over both rowset and prior weight:

.. code-block:: bash

   WEIGHTS=0,0.03,0.1,0.3,1.0 \
   SHARD_COUNT=8 \
   CONFIG=configs/experiments/diffsky_hltds_joint_realnvp_kl_annealed_h100.yaml \
   CHECKPOINT=outputs/runs/diffsky_nn_matrix/kl_annealed_balanced20k_seed42/checkpoints/best.eqx \
   FEATURE_STATS=outputs/runs/diffsky_nn_matrix/kl_annealed_balanced20k_seed42/feature_stats.json \
   ROWSET_FILE=outputs/rowsets/diffsky_hltds_nokl_reference/balanced20k.txt \
   OUT_ROOT=outputs/runs/diffsky_robust_prior_diagnostics/map_prior_sweep_sharded \
   sbatch --array=0-39 scripts/diffsky_map_prior_sweep_sharded_h100.slurm

Then finalize after all shards complete:

.. code-block:: bash

   OUT_ROOT=outputs/runs/diffsky_robust_prior_diagnostics/map_prior_sweep_sharded \
   sbatch scripts/diffsky_map_prior_sweep_finalize_h100.slurm

Inspect:

* ``map_prior_weight_sweep_summary.csv``
* ``z_bias_vs_prior_weight.png``
* ``z_upper_fraction_vs_prior_weight.png``

If redshift bias grows sharply with small ``prior_weight``, the current learned
prior is not a scientifically usable population prior.

The MAP prior term defaults to ``prior_density_space: x``.  This means the
RealNVP density is interpreted in the standardized latent coordinates used for
training and MCMC.  ``prior_density_space: theta`` is available for uncoupled
bounded-logit latents and subtracts the sigmoid-transform Jacobian; use it only
when the scientific question is explicitly a physical-space MAP.

Phase 5: Post-Hoc Inferred Prior
--------------------------------

The first robust prior should be trained from flat/weak-prior MAP or MCLMC
inferences:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/experiments/diffsky_hltds_joint_realnvp_kl_annealed_zscale005_tau2safe_h100.yaml \
     diffsky-train-inferred-prior \
     --input outputs/runs/diffsky_robust_prior_diagnostics/map_prior_sweep/prior_weight_0/map_estimates.parquet \
     --out outputs/runs/diffsky_inferred_prior_from_map_flat \
     --epochs 40 \
     --batch-size 512

Then use the resulting ``checkpoints/best.eqx`` as the next population-prior
candidate.  MCLMC samples can be added as extra ``--input`` values once the
calibration subset is available.

Jean Zay Launch Order
---------------------

Use the diagnostic array first.  It runs four geometry tasks, then closure,
then a small MAP-prior smoke.  The closure task uses ``--skip-map`` by default
so the fast truth/NN/redshift-profile diagnostics are not blocked by MAP.

.. code-block:: bash

   cd $WORK/euclid-dsps-shine

   CLOSURE_CHECKPOINT=outputs/runs/diffsky_nn_matrix/nokl_det_balanced20k_seed42/checkpoints/best.eqx \
   CLOSURE_FEATURE_STATS=outputs/runs/diffsky_nn_matrix/nokl_det_balanced20k_seed42/feature_stats.json \
   MAP_SWEEP_CHECKPOINT=outputs/runs/diffsky_nn_matrix/kl_annealed_balanced20k_seed42/checkpoints/best.eqx \
   MAP_SWEEP_FEATURE_STATS=outputs/runs/diffsky_nn_matrix/kl_annealed_balanced20k_seed42/feature_stats.json \
   NN_RUN=outputs/runs/diffsky_nn_matrix/nokl_det_balanced20k_seed42_infer \
   ROWSET_FILE=outputs/rowsets/diffsky_hltds_nokl_reference/balanced20k.txt \
   sbatch scripts/diffsky_robust_prior_diagnostics_h100.slurm

The full MAP sweep should use the sharded script, not the diagnostic array:

.. code-block:: bash

   WEIGHTS=0,0.03,0.1,0.3,1.0 \
   SHARD_COUNT=8 \
   CONFIG=configs/experiments/diffsky_hltds_joint_realnvp_kl_annealed_h100.yaml \
   CHECKPOINT=outputs/runs/diffsky_nn_matrix/kl_annealed_balanced20k_seed42/checkpoints/best.eqx \
   FEATURE_STATS=outputs/runs/diffsky_nn_matrix/kl_annealed_balanced20k_seed42/feature_stats.json \
   ROWSET_FILE=outputs/rowsets/diffsky_hltds_nokl_reference/balanced20k.txt \
   OUT_ROOT=outputs/runs/diffsky_robust_prior_diagnostics/map_prior_sweep_sharded \
   sbatch --array=0-39 scripts/diffsky_map_prior_sweep_sharded_h100.slurm

   OUT_ROOT=outputs/runs/diffsky_robust_prior_diagnostics/map_prior_sweep_sharded \
   sbatch --dependency=afterok:<MAP_SWEEP_JOBID> \
     scripts/diffsky_map_prior_sweep_finalize_h100.slurm

For a small MCLMC calibration subset, first create or reuse a rowset, then:

.. code-block:: bash

   ROWSET_FILE=outputs/rowsets/diffsky_hltds_nokl_reference/worst_100.txt \
   OUT_ROOT=outputs/runs/diffsky_flat_mclmc_calibration_worst100 \
   sbatch --array=0-3 scripts/diffsky_flat_mclmc_calibration_h100.slurm

Train the first post-hoc prior from flat MAP:

.. code-block:: bash

   INFERRED_INPUTS=outputs/runs/diffsky_robust_prior_diagnostics/map_prior_sweep/prior_weight_0/map_estimates.parquet \
   OUT_DIR=outputs/runs/diffsky_inferred_prior_from_map_flat \
   sbatch scripts/diffsky_inferred_prior_h100.slurm

Scaling Notes
-------------

For the first full ``balanced20k`` pass, MAP flat is cheaper and should be the
default source for the post-hoc prior.  MCLMC is better used as a calibration
subset first.  If the MCLMC subset has acceptable walltime and diagnostics, run
larger shards with ``--array=0-7`` and increase ``SHARD_COUNT=8``.

If a MAP job is cancelled, rerun the same command with the same ``OUT_ROOT``:
``diffsky-map-adam-prior`` skips existing ``map_estimates_shards/part_*.parquet``
files and only recomputes missing batches.
