Active Workflows
================

This page is the maintained map of supported experiment entry points. It is
based on the current AVI, selection-aware EM, factorial, and geometry work. A
tracked script or config that is not listed here may still be useful for
reproduction, but it is not part of the current launch chain.

Controlled FENIKS Data and Priors
---------------------------------

The public data path creates controlled Diffsky/FENIKS closure splits, projects
them into the spline-15D contract, and trains the supervised prior. Its primary
configs are:

* ``configs/diffsky_synthetic_feniks_260617_50k.yaml``;
* ``configs/diffsky_synthetic_feniks_260617_50k_survey_like_18band.yaml``;
* ``configs/feniks_spline15d_postprocess.yaml``;
* ``configs/prior_feniks_spline15d_realnvp.yaml``;
* ``configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml``.

AVI and Selection-Aware Co-adaptation
-------------------------------------

The maintained execution chain is:

#. ``scripts/feniks_avi_experiments.py`` and
   ``scripts/feniks_avi_experiments.slurm`` for controlled AVI arms;
#. ``scripts/feniks_avi_inference.py`` for frozen-checkpoint inference;
#. ``scripts/feniks_avi_next_experiments.py`` and
   ``scripts/feniks_avi_overnight.py`` for guarded capacity/prior refreshes;
#. ``scripts/feniks_avi_next_validation.py`` for independent validation;
#. ``scripts/feniks_avi_em.py`` for selection-aware EM;
#. ``scripts/feniks_avi_em_factorial.py`` for the controlled factorial audit.

Use their matching ``submit_*.sh``, ``watch_*.sh``, and ``*.slurm`` files. Keep
the object-level importance ratio as ``log p(x,z) - log q(z|x)``. Selection
corrections and ``beta`` belong to the population update, not to ordinary
object weights. Truth is reserved for post-inference qualification and must not
select observed-data checkpoints.

Exact-Posterior Geometry Diagnostics
------------------------------------

``scripts/feniks_geometry_nuts.py`` plus the geometry and observed-NUTS submit
scripts form the active exact-posterior diagnostic support. Allocation,
heartbeats, or GPU utilization are liveness evidence only. Scientific use
requires readable saved chains, convergence diagnostics, integration failure
counts, support checks, and posterior predictive checks.

COSMOS2020 External Benchmark
-----------------------------

The active COSMOS2020 path downloads and prepares the observed catalogue and
uses the published transmission curves as an external benchmark dataset. The
inference and proposal machinery is this repository's FENIKS spline-15D
RWS/AVI stack. It does **not** fit or reproduce the PopCosmos
parameterization, even where historical filenames retain ``popcosmos`` for
dataset provenance. Start from ``euclid_dsps/cosmos2020.py``,
``scripts/download_cosmos2020_assets.py``,
``scripts/prepare_cosmos2020_farmer.py``, and the current
``cosmos2020_*``/``popcosmos_native15d_*`` launchers.

Historical Surface
------------------

``legacy/`` contains superseded model reproductions, HLTDS campaigns,
one-off recovery launchers, and old documentation retained for provenance.
They are not maintained launch entry points and the active package must not
import them at runtime. ``PLAN.md`` remains the chronological record of work
performed; it is not the source of truth for deciding which workflow to run.
