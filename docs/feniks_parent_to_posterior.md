# Fresh-parent posterior: explicit exploratory continuation

This continues completed blind parent `avi_reference_to_parent_20260927_184817`.
It does **not** wait for, cancel, or modify the frozen-recovery diagnostic
(jobs 253496-253498 at handoff). User authorization to proceed despite dust
failure is an execution decision, not a retrospective scientific PASS.

## What advances

We now test the missing final conditional-inference block under the actual
learned parent. This is not another population sweep or truth-trained oracle.
The independent CPU recovery diagnostic can explain parent failure while this
chain tests whether a fresh simulation-trained posterior is usable.

The source parent retains its 256 overlapping components over native joint
15D anchors, including ten stochastic SFH coordinates. Components, coordinates,
feature preprocessing and learned parent weights `u` are frozen byte-for-byte.
No reference redesign, dust removal, classifier fitting or population update.
The strict CDF qualification, joint parent closure, dust and ratio failures are
copied to `PARENT_DECISION.json` and retained in the final decision.

## Training measure

Generate `j ~ Categorical(u)`, `x_latent ~ g_j`, `theta = T^-1(x_latent)`, then
use the saved coherent DSPS decoder, error model and noisy `r < 29` selection.
This samples the **parent** before selection, not the selected weights `v`.
All 15 coordinates vary through the existing joint-anchor/kernel distribution.
Its SFH conditional is still an unvalidated modelling assumption, not a fixed
SFH vector and not a claim of learning SFH from photometry.

Since selection is a function of the observed flux included in the context,
`p(theta | flux, selected) = p(theta | flux)`. Minimize unit-weight
`-log q_phi(x_latent_true | flux, errors, masks)` over fresh selected pairs.
The frozen transform Jacobian does not depend on phi, so this is the same
conditional forward-KL optimum in physical coordinates. No `1/beta(theta)`,
no reference-bank importance weights, no q-generated targets, no RWS or MCMC.

The bank is **1,048,576 parent simulations**, four shards of 262,144, checkpointed
every 8,192. Independent train/validation/evaluation roles are assigned before
selection with probabilities 85/10/5 percent. At the old alpha estimate 0.60235,
expect roughly 537k/63k/32k selected rows; these are expectations, not quotas.
Unique simulation row IDs and distinct RNG streams prevent role overlap.
`training_measure.json` records actual counts; unit weights mean effective
training rows equal unique selected training rows. Repeated native anchors are
possible as part of the continuous generative model, not duplicated simulation
pairs. The old weighted-reference-bank posterior is never resumed.

Architecture: existing independent two-expert conditional rational-quadratic
spline mixture, full 15D output, 12 layers/expert, 256 hidden width, 16 bins,
tail bound 12; residual photometry trunk 512 x 3, context 128. Saved features
remain 18-band flux/error/mask (54 inputs). AdamW with clipping, LR 1e-4,
halved every 50 epochs to floor 1.25e-5. Fixed validation subset up to 32768.
At most 200 epochs; implemented plateau stop checked every 20 epochs after 80,
three gains below 0.001. Neither reaching the cap nor plateau certifies calibration.
The best validation checkpoint receives the existing inverse/Jacobian audit.

## Two distinct evaluations

Each uses 2,048 distinct galaxies, 512 joint 15D posterior draws per galaxy:

1. **In-model**: held-out fresh simulations from the learned parent, excluded
   from training and checkpoint selection. Failure means the posterior has not
   learned its own simulation target sufficiently, even before catalogue shift.
2. **Coherent target**: the existing coherent catalogue's test split, with its
   different true parent. Failure here can persist even if in-model calibration
   succeeds, because the frozen parent/SFH reference is imperfect.

This test split has been inspected during project development. It is not a new,
untouched paper-level test set. Neither cohort establishes real-sky performance.
Report 68/95 coverage, PIT KS/histograms, bias, widths, individual 15D marginals
and physical corners, and dense-aggregate 1D/joint selected-distribution closure.
Keep separate core z/mass/metallicity, dust and SFH flags. Diagnostic coverage
tolerance 0.04 and PIT KS 0.05 do not override inherited population failures.
Broad SFH can be correct; width alone is not a failure criterion.
An aggregate over selected galaxies should match the **selected distribution**,
not the unselected parent. Plot limits never change the full-support metrics.

## Jean-Zay launch

Run on a login node, from the updated repository. The subshell prevents a
failed check from exiting the interactive SSH session.

```bash
(
set -euo pipefail
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
PARENT_RUN="$BASE/avi_reference_to_parent_20260927_184817"
POSTERIOR_RUN="$BASE/avi_parent_to_posterior_$(date +%Y%m%d_%H%M%S)"
bash scripts/submit_feniks_parent_to_posterior.sh --exploratory \
  "$PARENT_RUN" "$POSTERIOR_RUN"
)
```

DAG: four-task bank array (max four H100s) -> one H100 training -> one H100
evaluation -> CPU report, including a blocked report on upstream failure.
Time limits: 60 minutes/bank task, 360 training, 60 evaluation, 15 report.
Total default allocation ceiling 11 H100-hours, **not an expected runtime**.
No `--mem` flags. No dependency on or cancellation of the recovery diagnostic.
Frozen code snapshot, config/source hashes, block receipts and full optimizer
state allow missing-only resubmission. An active run cannot be double-submitted.

Reconnect and monitor:

```bash
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
source "$BASE/avi_parent_to_posterior_latest.env"
bash scripts/watch_feniks_parent_to_posterior.sh "$POSTERIOR_RUN"
```

If jobs failed/timed out, first inspect `logs/`, `sacct` and saved progress.
After no jobs from this run remain active, activate shine and run:

```bash
bash scripts/submit_feniks_parent_to_posterior.sh --resume "$POSTERIOR_RUN"
```

This uses the original frozen code/config, reuses completed blocks and resumes
optimizer state. It does not silently change the epoch cap or a failed transport
contract. For a code bug, a separately documented code recovery is needed.

Local small-file download (no banks, NPZ or EQX checkpoints; max 25 MiB/file):

```bash
bash scripts/rsync_feniks_coherent_results.sh parent_to_posterior
```

Look first at `report/calibration_comparison.png`, `report/training.png`, then
the two `*_physical_pit_truth.png`, `*_sfh_pit_truth.png`, individual corners and
`selected_aggregate_15d.png`. Numerical evidence: `calibration.csv`,
`selected_aggregate_joint.csv`, `bank_measure.json`, `DECISION.json` and root
`ROADMAP_STATUS.md`. New simulation/target calibration is pending until this
remote run completes; local mocked-DSPS smoke tests are not scientific evidence.
