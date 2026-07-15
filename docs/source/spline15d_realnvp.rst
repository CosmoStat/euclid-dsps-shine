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

The run writes ``normalization.json``, a 15-row normalization parameter CSV,
the requested 15-by-2 before/after plot with Gaussian overlays and lambdas,
training/validation logs and history, strict ``best.eqx`` and ``last.eqx``
checkpoints, physical and normalized prior samples, held-out NLL values, and
truth-versus-prior marginal/correlation diagnostics. The best checkpoint is
reloaded from disk before final sampling, which tests the saved contract.

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

Then submit production projection and training:

.. code-block:: bash

   PROJ_JOB=$(sbatch --parsable \
     --export=ALL,OVERWRITE=1,SOURCE_DATASET_DIR=Data/diffsky/synthetic/feniks_260617_dsps_closure_18band,SPLINE_DATASET_DIR=Data/diffsky/synthetic/feniks_260617_spline15d \
     scripts/feniks_spline15d_project_h100.slurm)

   PRIOR_JOB=$(sbatch --parsable --dependency=afterok:${PROJ_JOB} \
     --export=ALL,SPLINE_DATASET_DIR=Data/diffsky/synthetic/feniks_260617_spline15d,OUT_DIR=outputs/runs/feniks_spline15d_realnvp_v1 \
     scripts/feniks_spline15d_realnvp_h100.slurm)

   echo "projection=${PROJ_JOB} prior=${PRIOR_JOB}"

Do not use ``OVERWRITE=1`` on an existing production spline directory unless
the replacement is intentional. The training job always refuses a non-empty
output directory.
