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

Leakage-free post-processing
----------------------------

The recovery pipeline never generates new Diffsky galaxies. It pools the
existing 50k rows, canonicalizes ``source_proposal_id`` using the effective
backend key ``source_seed + shard_index``, and assigns every complete proposal
group to exactly one split. ``grouped_split_audit.json`` records split sizes,
multiplicities, and zero cross-split group overlap. The v3 spline projector
refuses a source without this audit and embeds it in its own contract. It then
writes for every new split:

* ``<split>_exact.parquet``: the scientific 15D projection;
* ``<split>.parquet``: a backward-compatible projected target;
* ``<split>_spline_nodes.parquet``: object IDs, eleven node times, and eleven
  absolute native ``logSFR`` values;
* ``spline15d_contract.json``: node placement, column order, seeds, atom counts,
  source paths, and runtime provenance.

The original object IDs and split labels remain as audit columns in the grouped
dataset. New IDs are unique by split. Training aborts if an exact 15D truth row
still occurs in train and validation or train and test.

Asinh, atom dequantization, and whitening
-----------------------------------------

The prior command fits one analytic transform per coordinate using the train
split only:

.. math::

   y_j = \frac{\lambda_j\,\operatorname{asinh}(x_j/\lambda_j)-c_j}{s_j}.

Each ``lambda`` is selected on a fixed log grid by marginal Gaussian quantile
RMSE. The center and scale are then frozen. The inverse is analytic, and the
same serialized transform is used for validation, test, checkpoint sampling,
and later inference. This is deliberately lower capacity than an empirical
quantile transform: only three scalars are stored per dimension and no
data-dependent knot table is learned. Exact-zero SFH contrasts are then
dequantized uniformly within ``+-0.05`` in normalized asinh coordinates. This
turns the discrete atoms into explicit narrow continuous intervals. Sampling
applies the exact inverse rule and clips those intervals back to zero.

Finally, a train-fitted affine Cholesky transform whitens the full 15D
covariance. A standard normal in the resulting coordinates is therefore the
full-covariance Gaussian baseline in asinh space, not fifteen independent
physical marginals. The serialized inverse is:

``flow coordinates -> inverse whitening -> inverse asinh -> atom reclipping``.

RealNVP training and outputs
----------------------------

``scripts/train_feniks_spline15d_realnvp.py`` hard-rejects any flow type other
than ``realnvp``. The production configuration does not expose an RQ-spline NF.
The generic historical flow classes remain in the repository for checkpoint
compatibility, but they are not part of this pipeline.

The v3 run writes ``normalization.json``, a 15-row normalization parameter CSV,
the requested 15-by-2 before/after plot with Gaussian overlays and lambdas,
training/validation logs and history, strict ``best.eqx`` and ``last.eqx``
checkpoints, physical and normalized prior samples, held-out NLL values, and
truth-versus-prior marginal/correlation diagnostics. The truth/prior figure has
one physical-space and one normalized-space column per parameter. The best
checkpoint is reloaded from disk before final sampling, which tests the saved
contract.

The RealNVP is initialized exactly at the identity (``init_scale: 0``), so
epoch 0 is the affine Gaussian baseline. Checkpoint selection is not based on
NLL alone. At every epoch the command
generates validation samples and combines marginal KS, physical correlation,
base-space moments, sliced Wasserstein distance, clamp saturation, normalized
tails, and invalid physical samples. A trained checkpoint must beat epoch 0 by
a configured minimum score margin and
preserve its marginal KS and sliced Wasserstein within configured tolerances.
Early stopping is driven by this generative score. Test data are never used
until the checkpoint is frozen.

Production fixes the base temperature to one. The run still writes the legacy
temperature-named outputs for compatibility, but no calibration is performed.
``test_baseline_comparison.csv`` reports the epoch-0 affine Gaussian and the
selected unit-temperature RealNVP side by side.

Jean-Zay launch
---------------

One sequential job performs regrouping, spline projection, and training. A
small end-to-end smoke test uses separate output directories:

.. code-block:: bash

   sbatch --export=ALL,SMOKE=1,\
GROUPED_DATASET_DIR=Data/diffsky/synthetic/feniks_260617_grouped_v3_smoke,\
SPLINE_DATASET_DIR=Data/diffsky/synthetic/feniks_260617_spline15d_grouped_v3_smoke,\
OUT_DIR=outputs/runs/feniks_spline15d_realnvp_whitened_v3_smoke \
     scripts/feniks_spline15d_whitened_realnvp_h100.slurm

After the smoke job succeeds, launch the single production job:

.. code-block:: bash

   sbatch scripts/feniks_spline15d_whitened_realnvp_h100.slurm

All three output paths must be absent. This prevents accidental mixing of an
old grouped split, an old spline projection, and a new checkpoint.
