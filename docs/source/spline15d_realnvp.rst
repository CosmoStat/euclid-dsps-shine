Spline-15D RealNVP Prior
========================

This is the active prior-learning path for the spline representation of the
FENIKS star-formation histories. It starts from an already generated Diffsky
dataset. Diffsky generation, spline projection, and prior training are three
separate runs.

15D truth contract
------------------

The learned vector contains exactly 15 coordinates:

* five DSPS-level coordinates: ``z_obs``, ``log10_stellar_mass``,
  ``log10_stellar_metallicity``, ``dust_av``, and ``dust_delta``;
* ten adjacent ``logSFR`` contrasts between eleven fixed spline nodes.

The node locations are fixed population-wide in normalized log cosmic time and
versioned in ``configs/feniks_spline15d_postprocess.yaml``. The ten contrasts
describe shape; stellar mass supplies the common SFH amplitude when age weights
are reconstructed. The projection also stores the eleven absolute native
``logSFR`` node values for scientific auditing, but these are not additional
RealNVP dimensions.

Separate post-processing
------------------------

Run ``scripts/build_feniks_spline15d_dataset.py`` only after the Diffsky
train/validation/test parquets exist. For every split it writes:

* ``<split>_exact.parquet``: the scientific 15D projection;
* ``<split>.parquet``: the continuous flow target, with exact-zero SFH
  contrasts uniformly dequantized within ``+-1e-4 dex``;
* ``<split>_spline_nodes.parquet``: object IDs, eleven node times, and eleven
  absolute native ``logSFR`` values;
* ``spline15d_contract.json``: node placement, column order, seeds, atom counts,
  source paths, and runtime provenance.

The object order and IDs are preserved. The command never generates Diffsky
galaxies and never reads an analysis artifact under ``outputs/``.

Asinh normalization
-------------------

The prior command fits one analytic transform per coordinate using the train
split only:

.. math::

   y_j = \frac{\lambda_j\,\operatorname{asinh}(x_j/\lambda_j)-c_j}{s_j}.

Each ``lambda`` is selected on a fixed log grid by marginal Gaussian quantile
RMSE. The center and scale are then frozen. The inverse is analytic, and the
same serialized transform is used for validation, test, checkpoint sampling,
and later inference. This is deliberately lower capacity than an empirical
quantile transform: only three scalars are stored per dimension and no
data-dependent knot table is learned.

RealNVP training and outputs
----------------------------

``scripts/train_feniks_spline15d_realnvp.py`` hard-rejects any flow type other
than ``realnvp``. The production configuration does not expose an RQ-spline NF.
The generic historical flow classes remain in the repository for checkpoint
compatibility, but they are not part of this pipeline.

The v2 run writes ``normalization.json``, a 15-row normalization parameter CSV,
the requested 15-by-2 before/after plot with Gaussian overlays and lambdas,
training/validation logs and history, strict ``best.eqx`` and ``last.eqx``
checkpoints, physical and normalized prior samples, held-out NLL values, and
truth-versus-prior marginal/correlation diagnostics. The truth/prior figure has
one physical-space and one normalized-space column per parameter. The best
checkpoint is reloaded from disk before final sampling, which tests the saved
contract.

Checkpoint selection is not based on NLL alone. At fixed epochs the command
generates validation samples and combines marginal KS, physical correlation,
base-space moments, normalized tails, and invalid physical samples. Test data
are never used for checkpoint or temperature selection. Exact truth hashes are
also compared against train, and validation/test NLL are reported separately on
the subset not exactly present in train.

The command scans an isotropic base temperature on validation after selecting
the checkpoint. It keeps both the unit-temperature output and a separately
named validation-calibrated output. ``test_baseline_comparison.csv`` compares
these against the independent-normal baseline in normalized space. Temperature
calibration is therefore explicit metadata, not an implicit change to the
asinh normalization or the RealNVP checkpoint.

The v2 architecture adds deterministic ``roll`` permutations between coupling
layers and supports a small base-moment penalty. The controlled ablation uses:

* ``a_control``: the v1 architecture with the new diagnostics;
* ``b_permutation``: permutation only;
* ``c_conservative``: permutation and tighter scale/shift clamps;
* ``d_regularized``: the conservative model plus base-moment regularization.

Jean-Zay launch
---------------

First run a small end-to-end smoke test in separate output directories:

.. code-block:: bash

   PROJ_SMOKE=$(sbatch --parsable \
     --export=ALL,SMOKE=1,OVERWRITE=1,SOURCE_DATASET_DIR=Data/diffsky/synthetic/feniks_260617_dsps_closure_18band,SPLINE_DATASET_DIR=Data/diffsky/synthetic/feniks_260617_spline15d_smoke \
     scripts/feniks_spline15d_project_h100.slurm)

   sbatch --dependency=afterok:${PROJ_SMOKE} \
     --export=ALL,SMOKE=1,SPLINE_DATASET_DIR=Data/diffsky/synthetic/feniks_260617_spline15d_smoke,OUT_DIR=outputs/runs/feniks_spline15d_realnvp_smoke \
     scripts/feniks_spline15d_realnvp_h100.slurm

The spline dataset is already present on Jean-Zay, so the v2 ablation can be
submitted directly. First run the four short smoke jobs:

.. code-block:: bash

   SMOKE_JOB=$(sbatch --parsable \
     --export=ALL,SMOKE=1,SPLINE_DATASET_DIR=Data/diffsky/synthetic/feniks_260617_spline15d,RUN_PREFIX=feniks_spline15d_realnvp_v2_smoke \
     scripts/feniks_spline15d_realnvp_ablation_h100.slurm)

   echo "smoke_array=${SMOKE_JOB}"

After inspecting the smoke jobs, submit the production ablation and rank runs
only after every array task succeeds:

.. code-block:: bash

   PRIOR_JOB=$(sbatch --parsable \
     --export=ALL,SPLINE_DATASET_DIR=Data/diffsky/synthetic/feniks_260617_spline15d,RUN_PREFIX=feniks_spline15d_realnvp_v2 \
     scripts/feniks_spline15d_realnvp_ablation_h100.slurm)

   echo "prior_array=${PRIOR_JOB}"

   # Run only once squeue no longer lists the array and all four runs completed.
   python scripts/compare_feniks_spline15d_realnvp_ablation.py \
     --runs-root outputs/runs \
     --pattern 'feniks_spline15d_realnvp_v2_[abcd]*' \
     --out outputs/reports/spline15d_realnvp_v2

Do not use ``OVERWRITE=1`` on an existing production spline directory unless
the replacement is intentional. The training job always refuses a non-empty
output directory.
