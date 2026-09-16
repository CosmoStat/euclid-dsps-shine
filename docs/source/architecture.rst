Architecture
============

Active Layout
-------------

.. code-block:: text

   euclid_dsps/
     cli.py                 Active CLI surface.
     config.py              YAML loading, inheritance, defaults, validation.
     io.py                  Parquet rows, photometry units, truth transforms.
     filters.py             Filter loading and smoke-test approximations.
     model.py               Production DSPS forward-model boundary.
     parameter_vectors.py   Theta-vector to DSPS array interface.
     fit.py                 MAP optimization.
     mcmc.py                NUTS/HMC/MCLMC posterior sampling.
     posterior_target.py    Pure-JAX log-density for BlackJAX.
     diffsky_data/          HLTDS listing, download, preparation, validation.
     synthetic_diffsky/     FENIKS proposal, DSPS closure, manifests, validation.
     prior_learning/        Supervised and inferred RealNVP prior learning.
     amortized/             NN+DSPS+NF training, inference, MAP-under-prior tools.
     workflows/core.py      Shared fit/check/posterior orchestration.
     reporting/core.py      Shared tables, plots, Markdown/JSON/CSV writers.
   configs/
     diffsky_synthetic_feniks_260617_50k.yaml
     diffsky_synthetic_feniks_260617_50k_survey_like_18band.yaml
     prior_diffsky_synthetic_feniks_full_realnvp.yaml
     amortized_diffsky_synthetic_feniks_full_gpu.yaml
     diffsky_dataset_hltds_04_14.yaml
     diffsky_dataset_hltds_03_31_zmax335_m5depth.yaml
     fs2_gpu.yaml
     amortized_fs2_realnvp.yaml
   scripts/
     diffsky_synthetic_feniks_50k_h100.slurm
     diffsky_synthetic_feniks_18band_h100.slurm
     diffsky_amortized_train_h100.slurm
     diffsky_amortized_infer_h100.slurm
     diffsky_map_adam_prior_h100.slurm
     diffsky_flat_mclmc_calibration_h100.slurm
     diffsky_inferred_prior_h100.slurm
     build_diffsky_lowz_projected_truth_dataset.py
     merge_mclmc_runs.py
   legacy/
     Superseded HLTDS experiments, old PopCosmos-parameterization attempts,
     reconstruction dashboards, incident launchers, and old docs/tests.

Data Flow
---------

.. code-block:: text

   FENIKS proposals
        |
        v
   synthetic_diffsky/
        |
        +--> closure train/validation/test parquet
        +--> manifest, schema, population diagnostics
        +--> validation report
        |
        v
   prior_learning/
        |
        +--> supervised RealNVP prior checkpoint
        +--> truth-vs-prior diagnostics
        |
        v
   amortized/
        |
        +--> NN posterior checkpoint
        +--> posterior samples and predictive residuals
        +--> learned-prior MAP estimates
        |
        v
   fit.py / mcmc.py
        |
        +--> flat-prior MAP and MCLMC baselines

Layer Rules
-----------

``model.py``
  Owns the production DSPS forward model. ``filters.py``, the synthetic-data
  backend, and numerical audit modules may call narrow DSPS utilities or
  reference kernels, but they must not define a competing production model.

``parameter_vectors.py``
  The public JAX theta contract for MAP, MCLMC, and amortized DSPS decoding.

``synthetic_diffsky/``
  Owns FENIKS proposal generation, selection, resampling, truth preservation,
  local DSPS closure photometry, noise injection, manifests, and validation.

``prior_learning/``
  Owns learned population priors from truth or inferred theta tables. It does
  not own photometric encoders.

``amortized/``
  Owns encoder features, NN posterior training/inference, RealNVP priors, and
  MAP under learned prior. It keeps the DSPS decoder fixed behind
  ``parameter_vectors.py``.

``diffsky_data/``
  Owns HLTDS download/preparation/debug contracts. It must keep direct truth,
  generated truth, projected truth, and missing truth flags separate.

``legacy/``
  Stores historical code and docs that are not part of the current FENIKS
  ladder. The active package must not import it at runtime. A small number of
  contract tests may inspect archived paths to keep reproducibility receipts
  readable, but archived launchers are not supported execution entry points.
