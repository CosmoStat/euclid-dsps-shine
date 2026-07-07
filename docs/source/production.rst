Production Runbook
==================

Scope
-----

The production path in this repository is the synthetic Diffsky/FENIKS
DSPS-closure workflow:

* generate independent train, validation, and test catalogues from weighted
  Diffsky/FENIKS proposal pools;
* recompute the final LSST+Roman fluxes with the local ``euclid_dsps`` forward
  model, not with Diffsky proposal photometry;
* validate exact same-parameter closure and the catalogue quality gates;
* train the supervised 18D RealNVP prior;
* train amortized inference with the supervised prior frozen;
* evaluate posterior calibration on the held-out test set.

The older HLTDS low-redshift PopCosmos path remains documented for debugging,
regression checks, and reference comparisons. It is not the production training
contract for an 18D physical closure model because the current projected HLTDS
truth table is not a direct ground-truth table for every fitted PopCosmos
dimension.

Production Contract
-------------------

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Item
     - Canonical value
     - Evidence to inspect
   * - Generation config
     - ``configs/diffsky_synthetic_feniks_260617_50k.yaml``
     - Normalized config in each run log and ``manifest.yaml``.
   * - Validation command
     - ``diffsky-validate-dsps-closure``
     - ``validation_report.json`` with ``gates.pass: true``.
   * - Dataset directory
     - ``Data/diffsky/synthetic/feniks_260617_dsps_closure/``
     - ``train.parquet``, ``validation.parquet``, ``test.parquet``,
       ``all_50k.parquet``.
   * - Split sizes
     - ``40000 / 5000 / 5000``
     - Manifest split counters and validation split checks.
   * - Truth schema
     - ``diffsky_dsps_closure_full``
     - ``schema.json`` and the 18-column truth check in validation.
   * - Photometry
     - 14 LSST+Roman ``fnu_cgs`` bands, true flux plus noisy flux plus
       positive flux errors.
     - ``flux_true_*``, ``flux_*``, ``fluxerr_*``, ``mask_*`` columns.
   * - Error model
     - ``m5_depth`` with ``sigma_sys_mag: 0.005``.
     - ``diagnostics/population/error_model_stats.csv`` and noise residual
       validation.
   * - Scientific gate
     - Same-parameter DSPS flux recomputation and posterior calibration, not
       raw photometric fit quality alone.
     - ``validation_report.json`` and inference evaluation outputs.

Workflow Diagram
----------------

.. code-block:: text

   Diffsky/FENIKS calibration
             |
             v
   independent weighted proposal pools
     train / validation / test
             |
             v
   proposal selection
     logsm_true >= 8
     unclipped SSP metallicity support
             |
             v
   weighted resampling with manifest ESS/duplication counters
             |
             v
   local euclid_dsps forward model
     18 truth parameters -> flux_true_<band>
             |
             v
   configured m5-depth error model
     fluxerr_<band>, noisy flux_<band>, masks
             |
             v
   closure catalogues + schema + manifest + diagnostics
             |
             v
   validation gates
     split sizes, disjoint ids, 18 truths, noise, metallicity, S/N, closure
             |
             v
   supervised prior -> amortized inference -> held-out evaluation

Data Semantics
--------------

.. code-block:: text

   proposal columns
       redshift_true, logsm_true, Diffstar, Diffmah, dust, weights
       source: Diffsky/FENIKS proposal generator
       use: provenance, resampling, diagnostics

   closure truth columns
       one *_true column per DIFFSKY_BASIC_PARAMETER_NAMES parameter
       source: generated truth scalars and FENIKS metallicity relation
       use: supervised prior targets and inference evaluation

   true photometry
       flux_true_<band>
       source: local euclid_dsps model evaluated from closure truths
       use: exact forward-closure validation

   observed photometry
       flux_<band>, fluxerr_<band>, mask_<band>
       source: configured m5-depth error model applied to true photometry
       use: photometric likelihood and amortized encoder inputs

   provenance
       manifest.yaml, schema.json, validation_report.json
       source: generator and validator
       use: reproducibility, acceptance gates, production audits

Do not train the closure model on ``phot_info.obs_mags`` from the Diffsky
proposal generator. Those values are proposal-side photometry, while the
production closure target is photometry recomputed by this repository's DSPS
boundary from the recorded 18D truth vector.

Preflight
---------

On Jean Zay, the production launcher expects the repository, the ``shine``
environment, Diffsky/Diffstar/Diffmah/Diffhalos, and the HLTDS SSP/filter
assets to already be present. It does not run online package installation.

.. code-block:: bash

   cd "$WORK/euclid-dsps-shine"
   conda activate shine

   export JAX_PLATFORMS=cuda
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   export TF_GPU_ALLOCATOR=cuda_malloc_async

   python -c "import jax; print(jax.default_backend()); print(jax.devices())"
   python -m compileall euclid_dsps scripts
   python -m euclid_dsps.cli --config configs/diffsky_synthetic_feniks_260617_50k.yaml \
     diffsky-generate-dsps-closure --help

Required local assets:

.. code-block:: text

   Data/diffsky/raw/hltds_cosmos_260215_04_14_2026/
       diffsky_hltds_cosmos_260215_04_14_2026_ssp_data.hdf5
       diffsky_hltds_cosmos_260215_04_14_2026_transmission_curves.hdf5
       diffsky_hltds_cosmos_260215_04_14_2026_ssp_basis_k64_coeff16.hdf5

Dataset Generation
------------------

Run a small smoke generation first when changing code, config, dependencies, or
assets:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_synthetic_feniks_260617_50k.yaml \
     diffsky-generate-dsps-closure \
     --smoke \
     --overwrite

For the 50k production dataset on H100, submit generation and validation as
separate dependent jobs:

.. code-block:: bash

   GEN_JOB=$(sbatch --parsable --export=ALL,STAGE=generate,OVERWRITE=1,RESUME=0 \
     scripts/diffsky_synthetic_feniks_50k_h100.slurm)

   sbatch --dependency=afterok:${GEN_JOB} --export=ALL,STAGE=validate \
     scripts/diffsky_synthetic_feniks_50k_h100.slurm

To resume an interrupted compatible full run:

.. code-block:: bash

   sbatch --export=ALL,STAGE=generate,OVERWRITE=0,RESUME=1 \
     scripts/diffsky_synthetic_feniks_50k_h100.slurm

Do not set ``OVERWRITE=1`` and ``RESUME=1`` together; the launcher refuses that
combination.

Validation Gates
----------------

Validation must pass before using a dataset for prior learning or amortized
inference:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_synthetic_feniks_260617_50k.yaml \
     diffsky-validate-dsps-closure \
     --dataset-dir Data/diffsky/synthetic/feniks_260617_dsps_closure \
     --sample-size 256 \
     --batch-size 256 \
     --runtime gpu

Inspect:

.. code-block:: text

   Data/diffsky/synthetic/feniks_260617_dsps_closure/manifest.yaml
   Data/diffsky/synthetic/feniks_260617_dsps_closure/schema.json
   Data/diffsky/synthetic/feniks_260617_dsps_closure/validation_report.json
   Data/diffsky/synthetic/feniks_260617_dsps_closure/diagnostics/population/report.md

Required acceptance checks:

* every split has the configured row count;
* ``object_id`` and ``source_proposal_id`` are disjoint across splits;
* every row has finite values for all 18 closure truth columns;
* ``fluxerr_*`` is finite and positive wherever ``mask_*`` is true;
* noisy flux residuals are centered and have approximately unit standard
  deviation after division by ``fluxerr_*``;
* metallicity convention is consistent between absolute ``log10(Z)`` and
  ``log10(Z/Z_sun)``;
* production rows satisfy the configured true-S/N gate;
* recomputed DSPS fluxes match ``flux_true_*`` within validator tolerance.

Prior and Inference
-------------------

Before launching the learning jobs, write the machine-checked workflow plan:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_synthetic_feniks_260617_50k.yaml \
     diffsky-plan-prior-workflow \
     --out outputs/reports/feniks_prior_workflow

Inspect ``outputs/reports/feniks_prior_workflow/workflow_plan.md``. It records
the train/validation/test parquet row counts, the 18D truth schema check,
missing upstream checkpoints, and copy-paste CLI/Jean-Zay commands for each
stage.

Train the supervised prior on the generated closure truth table:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/prior_diffsky_synthetic_feniks_full_realnvp.yaml \
     diffsky-train-supervised-prior \
     --out outputs/runs/prior_diffsky_synthetic_feniks_full_realnvp

Then train amortized inference with that prior frozen:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml \
     amortized-train-diffsky \
     --out outputs/runs/amortized_diffsky_synthetic_feniks_full

Run inference on the held-out test set, then evaluate the posterior outputs:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml \
     amortized-infer-diffsky \
     --dataset Data/diffsky/synthetic/feniks_260617_dsps_closure/test.parquet \
     --checkpoint outputs/runs/amortized_diffsky_synthetic_feniks_full/checkpoints/best.eqx \
     --out outputs/runs/amortized_diffsky_synthetic_feniks_full_infer

   python -m euclid_dsps.cli \
     --config configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml \
     diffsky-evaluate-dsps-closure-inference \
     --run outputs/runs/amortized_diffsky_synthetic_feniks_full_infer \
     --dataset Data/diffsky/synthetic/feniks_260617_dsps_closure/test.parquet \
     --out outputs/runs/amortized_diffsky_synthetic_feniks_full_eval

Production Readiness Checklist
------------------------------

Before promoting a run:

* record the repository SHA, config hash, Diffsky/Diffstar/Diffmah versions,
  DSPS version, SSP hash, filter hashes, and calibration hash from
  ``manifest.yaml``;
* keep the raw proposal shards for weighted proposal diagnostics;
* keep negative noisy fluxes; they are valid likelihood inputs, not corrupt
  rows;
* report ESS and duplication fraction with any science use of the catalogue;
  ``max_duplication_fraction`` is a warning threshold after ``max_shards`` is
  exhausted when ``duplication_gate: warn_after_max_shards`` is configured;
* compare the overlapping low-redshift range against the HLTDS reference only
  as a reference diagnostic, not as proof that the broad FENIKS production
  catalogue matches a volume-complete survey;
* treat exact latent recovery as too strong a target for 18 latents and 14
  broad bands. Use calibration, posterior predictive checks, and weak-direction
  diagnostics as the science acceptance criteria.

Historical Paths
----------------

The HLTDS low-z PopCosmos fit, MCLMC reconstruction dashboards, robust-prior
debug plans, and neural-network experiment matrix remain useful for diagnosis
and regression testing. They are intentionally outside the production runbook
because they rely on projected or diagnostic truth contracts rather than the
strict 18D closure truth contract documented here.
