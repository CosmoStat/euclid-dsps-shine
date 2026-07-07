Run the Active Workflow
=======================

Scope
-----

The active source tree supports one science ladder:

1. Generate the synthetic Diffsky/FENIKS DSPS-closure dataset.
2. Validate same-parameter closure on held-out FENIKS rows.
3. Learn a supervised RealNVP prior on the full 18D closure truth vector.
4. Train and run NN+DSPS+NF amortized inference with that prior.
5. Run MAP under the learned prior and small MCLMC baselines when needed.

HLTDS remains only a download/preparation/debug reference. FS2 remains the
Euclid comparison path.

Active Configs
--------------

.. list-table::
   :header-rows: 1

   * - Config
     - Role
   * - ``configs/diffsky_synthetic_feniks_260617_50k.yaml``
     - Generate and validate the 14-band synthetic FENIKS DSPS-closure train/validation/test splits.
   * - ``configs/diffsky_synthetic_feniks_260617_50k_survey_like_18band.yaml``
     - Generate and validate the LSST+Euclid+Roman 18-band FENIKS comparison sample.
   * - ``configs/prior_diffsky_synthetic_feniks_full_realnvp.yaml``
     - Train the supervised 18D FENIKS RealNVP prior.
   * - ``configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml``
     - Train and infer with the 18D NN+DSPS+NF model.
   * - ``configs/diffsky_dataset_hltds_04_14.yaml``
     - Rebuild and validate the low-z HLTDS debug/reference parquet.
   * - ``configs/diffsky_dataset_hltds_03_31_zmax335_m5depth.yaml``
     - Rebuild the higher-redshift HLTDS truth-rich debug/reference parquet.
   * - ``configs/fs2_gpu.yaml``
     - Euclid FS2 MAP/posterior comparison baseline.
   * - ``configs/amortized_fs2_realnvp.yaml``
     - FS2 amortized comparison baseline.

Preflight
---------

.. code-block:: bash

   conda activate shine
   export JAX_PLATFORMS=cuda
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   export TF_GPU_ALLOCATOR=cuda_malloc_async

   python -m euclid_dsps.cli \
     --config configs/diffsky_synthetic_feniks_260617_50k.yaml \
     diffsky-plan-prior-workflow \
     --out outputs/reports/feniks_prior_workflow

Generate And Validate FENIKS
----------------------------

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_synthetic_feniks_260617_50k.yaml \
     diffsky-generate-dsps-closure \
     --smoke \
     --overwrite

   python -m euclid_dsps.cli \
     --config configs/diffsky_synthetic_feniks_260617_50k.yaml \
     diffsky-validate-dsps-closure \
     --dataset-dir Data/diffsky/synthetic/feniks_260617_dsps_closure \
     --runtime cpu \
     --sample-size 64

Jean-Zay launchers:

.. code-block:: bash

   GEN_JOB=$(sbatch --parsable --export=ALL,STAGE=generate,OVERWRITE=1,RESUME=0 \
     scripts/diffsky_synthetic_feniks_50k_h100.slurm)

   sbatch --dependency=afterok:${GEN_JOB} --export=ALL,STAGE=validate \
     scripts/diffsky_synthetic_feniks_50k_h100.slurm

Learn The Prior
---------------

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/prior_diffsky_synthetic_feniks_full_realnvp.yaml \
     diffsky-train-supervised-prior \
     --out outputs/runs/prior_diffsky_synthetic_feniks_full_realnvp

The config uses ``missing_policy: fail`` and the
``diffsky_dsps_closure_full`` schema. Missing truth columns are blockers, not
silent reductions.

Train NN+DSPS+NF
----------------

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml \
     amortized-train-diffsky \
     --out outputs/runs/amortized_diffsky_synthetic_feniks_full

   python -m euclid_dsps.cli \
     --config configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml \
     amortized-infer-diffsky \
     --checkpoint outputs/runs/amortized_diffsky_synthetic_feniks_full/checkpoints/best.eqx \
     --out outputs/runs/amortized_diffsky_synthetic_feniks_full_test_infer \
     --dataset Data/diffsky/synthetic/feniks_260617_dsps_closure/test.parquet

MAP And MCLMC
-------------

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml \
     diffsky-map-adam-prior \
     --checkpoint outputs/runs/amortized_diffsky_synthetic_feniks_full/checkpoints/best.eqx \
     --dataset Data/diffsky/synthetic/feniks_260617_dsps_closure/test.parquet \
     --out outputs/runs/map_diffsky_synthetic_feniks_under_prior

   python -m euclid_dsps.cli \
     --config configs/diffsky_synthetic_feniks_260617_50k.yaml \
     posterior \
     --dataset Data/diffsky/synthetic/feniks_260617_dsps_closure/test.parquet \
     --sampler mclmc \
     --limit 4 \
     --out outputs/runs/mclmc_diffsky_synthetic_feniks_flat

HLTDS And FS2 References
------------------------

HLTDS data commands live in ``euclid_dsps.diffsky_data`` and are exposed by
``euclid-dsps`` as ``diffsky-list-remote``, ``diffsky-download-subset``,
``diffsky-prepare-dataset``, ``diffsky-validate-dataset``, and
``diffsky-dataset-diagnostics``. Use the two ``diffsky_dataset_hltds_*``
configs for that path.

Use ``configs/fs2_gpu.yaml`` and ``configs/amortized_fs2_realnvp.yaml`` only
for Euclid comparison runs.
