:orphan:

Historical Diffsky NN Experiment Matrix
=======================================

.. warning::

   This page is a historical experiment record. It is not the production
   workflow. Use :doc:`production` for the current supported FENIKS/DSPS
   closure runbook and acceptance gates.

This runbook defines the post-initialization-fix Diffsky amortized NN
experiments. The goal is to separate reconstruction sanity checks from
learned-prior/KL experiments while evaluating every run on the same documented
``balanced20k`` rowset.

Run Order
---------

1. Build the diagnostic rowsets from the canonical no-KL reference run.
2. Launch the four-run NN matrix on ``balanced20k``.
3. Inspect training/inference diagnostics before interpreting learned priors.
4. Optionally run MAP/MCLMC reconstruction baselines on the worst rowsets.

Build Rowsets
-------------

The first job writes ``balanced20k.txt`` plus worst-object slices and a manifest:

.. code-block:: bash

   sbatch scripts/diffsky_nn_build_rowsets_h100.slurm

Default outputs:

.. code-block:: text

   outputs/rowsets/diffsky_hltds_nokl_reference/
     rowsets_manifest.json
     balanced20k.txt
     balanced20k_diagnostics.csv
     balanced20k_diagnostics.parquet
     worst_100.txt
     worst_500.txt
     worst_1000.txt

The balanced rowset is stratified by configured redshift bins and the
observable photometric-quality proxy
``median(obs_err_fnu_cgs / abs(obs_flux_fnu_cgs))``. It is a diagnostic training
selection, not a claim that the underlying population is uniform.

Launch NN Matrix
----------------

Submit the full four-job array after rowset generation:

.. code-block:: bash

   rowset_job=$(sbatch --parsable scripts/diffsky_nn_build_rowsets_h100.slurm)
   sbatch --dependency=afterok:${rowset_job} scripts/diffsky_nn_experiment_matrix_h100.slurm

The array tasks are:

.. list-table::
   :header-rows: 1
   :widths: 10 22 36 32

   * - Task
     - Name
     - Config
     - Purpose
   * - 0
     - ``nokl_det``
     - ``configs/experiments/diffsky_hltds_autoencoder_nokl_deterministic_h100.yaml``
     - Pure deterministic no-KL autoencoder reconstruction using only the encoder mean.
   * - 1
     - ``nokl_stoch``
     - ``configs/experiments/diffsky_hltds_autoencoder_nokl_h100.yaml``
     - Existing stochastic no-KL objective with corrected physical initialization.
   * - 2
     - ``kl_fixed``
     - ``configs/experiments/diffsky_hltds_joint_realnvp_kl_fixed_h100.yaml``
     - Fixed small-KL RealNVP run for testing immediate prior pressure.
   * - 3
     - ``kl_annealed``
     - ``configs/experiments/diffsky_hltds_joint_realnvp_kl_annealed_h100.yaml``
     - Annealed small-KL RealNVP run with entropy regularization and temperature annealing.

Default outputs:

.. code-block:: text

   outputs/runs/diffsky_nn_matrix/
     nokl_det_balanced20k_seed42/
     nokl_det_balanced20k_seed42_infer/
     nokl_stoch_balanced20k_seed42/
     nokl_stoch_balanced20k_seed42_infer/
     kl_fixed_balanced20k_seed42/
     kl_fixed_balanced20k_seed42_infer/
     kl_annealed_balanced20k_seed42/
     kl_annealed_balanced20k_seed42_infer/

Key Diagnostics
---------------

For every training run, inspect:

.. code-block:: text

   initial_theta_diagnostics.json
   training_summary.json
   training_log.csv
   training_epoch_summary.csv
   validation_redshift_bin_metrics.csv

For every inference run, inspect:

.. code-block:: text

   inference_summary.json
   posterior_summary.parquet
   posterior_predictive_residual_summary.parquet
   parameter_bound_diagnostics.csv
   feature_diagnostics.parquet

The first pass should answer these questions:

* Does ``z_obs`` still pile up near the upper bound ``0.35``?
* Do any other physical parameters pile up near bounds?
* Is deterministic no-KL at least competitive with stochastic no-KL in
  likelihood-space residuals?
* Do ``lsst_u`` and ``lsst_g`` remain the dominant bands?
* Are worst objects explained by SNR/error-over-flux diagnostics, or by genuine
  model/encoder failure?

Interpretation
--------------

``nokl_det`` is the clean reconstruction sanity check. It should be treated as
an autoencoder, not as a calibrated posterior.

``nokl_stoch`` keeps the old Gaussian-encoder machinery but no KL pressure. It
is useful only as an ablation against the previous implementation.

``kl_fixed`` tests whether even weak immediate KL/prior pressure stabilizes or
hurts reconstruction.

``kl_annealed`` is the main learned-prior candidate if the deterministic no-KL
run is healthy. It should not be interpreted physically unless its posterior
bound diagnostics, redshift-bin metrics, and per-band residuals are sane.

Useful Overrides
----------------

Run only one task:

.. code-block:: bash

   sbatch --array=0 scripts/diffsky_nn_experiment_matrix_h100.slurm

Use a different rowset:

.. code-block:: bash

   sbatch --export=ALL,ROWSET_FILE=outputs/rowsets/.../balanced20k.txt \
     scripts/diffsky_nn_experiment_matrix_h100.slurm

Change the root output directory:

.. code-block:: bash

   sbatch --export=ALL,RUN_ROOT=outputs/runs/diffsky_nn_matrix_v2 \
     scripts/diffsky_nn_experiment_matrix_h100.slurm

Smoke-test on a smaller rowset:

.. code-block:: bash

   sbatch --export=ALL,BALANCED_SIZE=2000,ROWSET_DIR=outputs/rowsets/diffsky_smoke2k \
     scripts/diffsky_nn_build_rowsets_h100.slurm
   sbatch --export=ALL,ROWSET_FILE=outputs/rowsets/diffsky_smoke2k/balanced_2000.txt \
     --array=0-1 scripts/diffsky_nn_experiment_matrix_h100.slurm
