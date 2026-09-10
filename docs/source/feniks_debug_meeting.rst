FENIKS: From Numerical Debugging to Reliable Adaptation
=======================================================================

Meeting dossier | 10 September 2026

.. important::

   **Latest result:** the guarded pilot has completed. All accepted updates
   reduced their training loss, but only 3/9 changed trajectories improved ESS;
   23/32 trajectories did not change. Step safety is verified on this run,
   not reliable posterior recovery. Test 23 adds independent before/after
   batches without using them to choose updates. See :doc:`feniks_debug_experiments`
   and :download:`new run instructions <../feniks_wake_holdout_runbook.md>`.

.. note::

   **New to this project? Start with** :doc:`feniks_debug_experiments`.
   It explains each test in everyday language: what we wanted to know, what
   we did, what happened and what it means. This page keeps the figures,
   numbers and file locations for reference.

**The goal is to infer which galaxies could have produced the measured light,
and how uncertain we are about their properties.** Several combinations of
distance, mass, dust and star-formation history can explain similar observations.
We therefore need a distribution of plausible answers, not just one good fit.

**Main conclusion:** we found and corrected specific numerical and structural
problems. We then reproduced an optimizer step-size problem and implemented
a safeguard. Reliable posterior uncertainties are still a result to demonstrate,
not something guaranteed by those corrections.

.. contents:: On this page
   :local:
   :depth: 2

.. toctree::
   :maxdepth: 1

   feniks_debug_metrics
   feniks_debug_experiments

The Story in Two Minutes
------------------------

1. **The network could produce plausible fits, but its uncertainty estimates
   were not reliable enough.** Among 1,024 proposed solutions, importance
   weights often left only about six effective samples at the historical
   epoch-160 checkpoint. Good-looking flux predictions hid this weakness.
2. **We checked the machinery before asking it to learn more.** We found
   uneven coverage of the parameters in the network, numerical issues in
   parts of the physical calculation, and a precision assumption in an audit.
   Correcting these made the relevant numerical checks pass at tested points.
3. **We tried several ways to improve each galaxy's distribution.** Longer
   optimization, smaller steps, more samples and wider distributions sometimes
   improved the fit, but did not consistently make the weighted samples reliable.
4. **We isolated a concrete problem in wake adaptation.** The update pointed
   downhill locally, but the full step often went too far: seven of nine
   inspected first updates increased the very loss they were meant to reduce.
5. **The guarded update passed its step-safety check.** All accepted steps
   reduced their batch loss, but the distributions did not consistently improve.
   The next test checks each change on two new sets of examples that cannot
   influence training.

How the Pieces Fit Together
---------------------------

* **Physical model / decoder:** turns a proposed galaxy into predicted light.
* **Prior:** describes the population before considering this object's data.
* **Amortized network:** quickly proposes many possible galaxies from measured
  light. The same trained network is used across objects.
* **Local adaptation:** adjusts the proposal for one object; it does not
  retrain the shared network or update the population prior.
* **Importance weights:** compare the proposals with the prior and physical
  likelihood. If one proposal receives almost all the weight, thousands of
  generated samples can still contain very little useful information.

The **posterior** is the distribution implied by the data and model. The
network's **proposal** is our approximation to it. They should not be treated
as identical simply because the network returns samples.

What Were the Problems, and What Did We Fix?
------------------------------------------------------------

**1. Some parameters were not being transformed by the conditional flow.**
The topology audit found seven coordinates with zero transformation counts.
The rebuilt topology transforms every coordinate. This fixes a structural
limitation; it does not establish that the learned distribution is accurate.

**2. Parts of the numerical calculation were not sufficiently qualified.**
We isolated photometric integration, metallicity-related calculations and
redshift/transport precision. Reference quadrature and targeted precision
changes progressively resolved the tested discrepancies. The versioned
transport64 objective passed all 32 starting-point audits. This is evidence
for those tests, not a proof of accuracy everywhere in parameter space.

**3. One numerical test used an inappropriate precision assumption.**
The resolution screen assumed float32 output where the loss was float64.
Correcting that screen removed false obstacles without simply loosening every
tolerance. The checking code needed scrutiny as well as the model code.

**4. An eligible wake batch could still produce a harmful full update.**
The old checks screened weight quality but did not guarantee that the proposed
step decreased its own batch loss. Exact replay confirmed the behavior.
Backtracking with rollback passed the completed pilot's descent checks.
Independent posterior quality did not improve consistently.

**5. Concentrated weights and unreliable joint uncertainties remain open.**
These are measured inference failures, not yet resolved by the fixes above.
They explain why successful numerical audits have not automatically led to
a validated large-scale AVI/RWS run.

What We Learned From the Experiments
-------------------------------------

* **A better fit is not necessarily a better distribution.** Several VI runs
  reduced flux residuals while importance weights remained concentrated.
* **More samples reveal problems; they do not necessarily fix them.** Replaying
  unchanged checkpoints at K4096 exposed fragile apparent successes at K1024.
* **Wider is not automatically safer.** Broadening all directions can add many
  implausible galaxies without recovering the important missing regions.
* **A correct gradient is not a safe finite step.** Local derivative checks
  passed while full wake updates overshot. This directly motivated backtracking.
* **A decreasing training loss is still not enough.** Two inspected updates
  reduced batch loss but worsened independent fit diagnostics. Validation must
  use fresh draws, not only the samples used to construct the update.
* **A good redshift or physical-subset score can hide joint problems.** At
  epoch 160, physical-5D MIRA was near its reference while full-15D MIRA was not.

Where We Stand
---------------

**Verified in completed tests:** corrected topology coverage, numerical
qualification at the tested transport64 starting points, and exact replay of
the problematic wake trajectories.

**Verified on the completed pilot:** guarded wake descent protects the update's
fixed-batch objective. It did not establish reliable importance weights or
calibrated uncertainties. Independent per-step batch measurements are now
implemented as the next diagnostic.

**Still to demonstrate:** stable improvement on fresh samples across objects
and starts, then generalization on an independent cohort. Only then would a
larger training run test scaling of a convincing method rather than repeat
the same unresolved failure at greater cost.

Reading Guide for the Meeting
------------------------------

Start with the recap above. Use the baseline plots below to show why debugging
was necessary. Then use :doc:`feniks_debug_experiments` for the sequence of
tests, and :doc:`feniks_debug_metrics` when an equation or diagnostic needs
explaining. The figures and tables below retain their original evidence and
limitations; historical results are not relabeled as current results.

Where We Started
----------------

The historical large-run reference is **SC-DRWS r29, epoch 160**, with raw and
exponential-moving-average (EMA) network checkpoints and a learned parent prior.
It is frozen, not the model currently optimized in the local pilot. Its receipt
excludes truth from training and checkpoint selection; synthetic truth below
is used only for post-freeze evaluation.

Recent cluster roots share this prefix::

   /lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111

The historical checkpoint directory relative to that prefix is::

   full/current_residual_6x256/seed_260826/train/checkpoints/epoch_0160

The raw-model SHA256 starts ``c6b190b28ba292d2``; the EMA hash starts
``be7a80e473c526db``. The :download:`frozen-checkpoint receipt
<_static/feniks_debug/epoch160_checkpoint.json>` records full hashes, feature
statistics and latent-transform identity.

Three evaluation populations must not be conflated:

* **512 held-out synthetic objects:** support at K=1024; MIRA below uses
  128 samples/object, 100 regions and 1,000 bootstrap draws.
* **4,706 independent observed-selected test objects:** population figures
  use an object-equal mixture of 32 joint draws/object.
* **16 development cases, two starts:** later debugging uses eight observed
  and eight simulated cases. This is not catalogue-wide calibration; simulated
  truth does not drive these local updates.

.. list-table:: Historical epoch-160 importance support
   :header-rows: 1

   * - Model
     - Median ESS / 1024
     - Fraction k > 0.7
     - 90th percentile maximum weight
   * - Raw
     - 5.95 (0.58%)
     - 82.4%
     - 0.983
   * - EMA
     - 5.88 (0.57%)
     - 83.6%
     - 0.988

**Implication:** 32 resampled particles do not provide 32 effective posterior
samples when the underlying bank has ESS near six. Finite weights are not
necessarily useful weights. The debug sequence therefore separates the
learned proposal from the physical target.

.. figure:: _static/feniks_debug/epoch160_heldout_importance_support.png
   :width: 100%

   Historical support distribution. Inspect the low-ESS tail and near-unit
   maximum weights, not only the median. This is not the running pilot.

Corner and Distribution Views
-----------------------------

.. figure:: _static/feniks_debug/epoch160_corner.png
   :width: 100%

   Genuine single-object corner: epoch-160 raw proposal, smallest archived
   row ID, 256 direct joint draws. Diagonals show marginal densities;
   off-diagonals retain paired coordinates. No medians replace distributions.
   Five physical coordinates are displayed, not the remaining ten SFH
   coordinates. The object was not selected by truth, fit or ESS. This is q,
   not an importance-corrected or independently qualified posterior.

.. figure:: _static/feniks_debug/epoch160_individual_posteriors_physical5d.png
   :width: 100%

   Six historical examples spanning archived observed r-band flux ranks.
   Direct q: 256 draws/object; IW: 32 diagnostic resamples/object. Spiky IW
   curves may represent particle collapse, not precise physical inference.

.. figure:: _static/feniks_debug/epoch160_population_selected_marginals.png
   :width: 100%

   Historical selected-catalogue marginals: 4,706 objects, 32 joint draws/object.
   This is an object-equal distribution mixture, not a histogram of medians.
   Compare matching selection predicates: selected population and parent prior
   are different. Marginal agreement does not validate individual conditionals.

Exports: :download:`corner PDF <_static/feniks_debug/epoch160_corner.pdf>`,
:download:`individual distributions PDF <_static/feniks_debug/epoch160_individual_posteriors_physical5d.pdf>`,
:download:`population PDF <_static/feniks_debug/epoch160_population_selected_marginals.pdf>`.
The :download:`figure manifest <_static/feniks_debug/epoch160_figure_manifest.json>`
documents cohorts, selection, draw counts, robust axis ranges and weights.

MIRA: The Joint Distribution Was Not Validated
----------------------------------------------

.. figure:: _static/feniks_debug/epoch160_mira.png
   :width: 100%

   Historical epoch-160 MIRA on 512 synthetic held-out objects. Dashed line:
   reference 2/3. Intervals: object + random-region bootstrap. Raw/EMA and
   direct-q/IW banks are distinct evaluations.

.. list-table:: Full-15D historical MIRA
   :header-rows: 1

   * - Bank
     - Score
     - Bootstrap 95% interval
   * - Raw q
     - 0.5080
     - [0.4853, 0.5326]
   * - Raw IW
     - 0.4451
     - [0.4204, 0.4696]
   * - EMA q
     - 0.5029
     - [0.4790, 0.5287]
   * - EMA IW
     - 0.4446
     - [0.4197, 0.4715]

Raw-q physical-5D MIRA is 0.6606, close to 2/3, while full-15D MIRA is 0.5080.
**A reassuring physical subset can hide a joint/SFH problem.** IW does not fix
this evaluation; resampling concentrated weights cannot manufacture support.

These scores do not measure the later anchor C or transport64 pilots. MIRA
requires truth paired with dense conditional draws; it cannot evaluate every
physical coordinate of real observed galaxies without corresponding truth.
See :doc:`feniks_debug_metrics` for the statistic and its limitations.

Download :download:`all MIRA groups (CSV) <_static/feniks_debug/epoch160_mira_scores.csv>`,
:download:`MIRA manifest <_static/feniks_debug/epoch160_mira_manifest.json>` and
:download:`MIRA PDF <_static/feniks_debug/epoch160_mira.pdf>`.

Which Run Is Current?
---------------------

"Latest" is a mutable monitor pointer, not a scientific model identifier.
This ledger uses supplied cluster readbacks, not a live scheduler query.

.. list-table:: Ancestry, completed evidence and active work
   :header-rows: 1
   :widths: 24 46 30

   * - Role
     - Root relative to the shared prefix
     - Evidence/status
   * - Frozen numerical NPE anchor C
     - ``frozen_parent_precision_night_v1``
     - 1923347: target qualification passed, support failed
   * - Local initial checkpoints
     - ``frozen_parent_long_local_vi_v1``
     - 1957394: final 512-step checkpoints, two starts
   * - Completed adaptation comparison
     - ``frozen_parent_objective_transport64_pilot_v1``
     - 1962310: 16 cases complete, wake unstable
   * - Rejected long extension
     - ``frozen_parent_objective_transport64_night_v1``
     - 1962505: no optimization started
   * - Completed causal diagnostic
     - ``frozen_parent_wake_forensics_v1``
     - 1965476: 32 exact replays, nine first updates inspected
   * - Current corrected adaptation
     - ``frozen_parent_wake_descent_v1``
     - Complete: 16 cases; descent passes, support remains unreliable
   * - Prepared independent-batch test
     - ``frozen_parent_wake_holdout_v1``
     - Implemented; not submitted here

**No current-pilot corner or MIRA has been imported.** Relabeling historical
plots as current would misrepresent the evidence. The completed pilot saves
joint draws and receipts; it does not itself run truth-based MIRA calibration.
The next readback must establish descent and independent support before any
larger training claim.

Read :doc:`feniks_debug_experiments` for every experiment's hypothesis,
controlled change, result, implication and remaining limitation.

Updating the Evidence
----------------------

On Jean-Zay, inspect an explicit root instead of whichever job last overwrote
the monitor environment:

.. code-block:: bash

   cd "$WORK/dsps-popcosmos"
   source "$WORK/miniconda3/etc/profile.d/conda.sh"
   conda activate shine
   BASE="/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111"
   ROOT="$BASE/frozen_parent_wake_descent_v1"
   python scripts/summarize_feniks_sc_drws_objective_pilot.py "$ROOT"
   python scripts/summarize_feniks_wake_descent.py "$ROOT"

For current distribution plots, retain ``RUN_MANIFEST.json``, ``FINAL.json``,
``OBJECTIVE_AUDIT.json``, per-case ``SUMMARY.json``, ``TRANSPORT_CONTRACT.json``,
``optimization.csv`` and ``direct_draws.npz``. Compare a declared list of
cases/starts/checkpoints; do not choose attractive plots after reading ESS.
A new MIRA assessment needs a frozen, truth-paired cohort and its own manifest.

Reproduce documentation figures locally:

.. code-block:: bash

   .venv/bin/python scripts/build_feniks_meeting_assets.py
   .venv/bin/python scripts/plot_feniks_wake_meeting.py
   .venv/bin/sphinx-build -W -b html docs/source docs/_build/html

This uses archived inputs, not cluster access. Copied figures are checked
against existing hashes. New figures retain :download:`source provenance
<_static/feniks_debug/meeting_asset_provenance.json>`. Later diagnostic figures
are labeled terminal transcriptions, not independently downloaded measurements.
