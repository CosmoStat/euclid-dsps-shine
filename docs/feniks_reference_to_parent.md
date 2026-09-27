# Fixed reference to blind parent benchmark

One controlled change and one guarded DAG, not another basis sweep. Read the
[synchronized results](feniks_reference_redesign_results_20260927.md) first.
No jobs are launched by local tests or by generating the analysis figures.

## Recovery after completed classifier, 2026-09-27

Run `avi_reference_to_parent_20260927_184817` completed all four banks and
stopped classifier training on validation plateau at epoch 460 (best NLL
1.59957185). Job 252118 then failed in the regularized weight solver with KKT
gap 1.18467e-5 versus the unchanged 2e-6 tolerance. This is not the CDF admission
gate, nor evidence of failed population recovery: parent fitting was unfinished.

The numerical fix keeps the same KL-regularized objective, constraints and
tolerance. It uses pairwise feasible mass transfers on the simplex and solves
the monotone line derivative instead of comparing tiny objective differences.
With a weak-parent-mass constraint, it retains feasible Frank-Wolfe segments
and uses the same derivative root search. A deterministic 256-component test
that exhausted the old 2000 polishing iterations now meets the original KKT
criterion; constrained and premature-success regression tests also pass.
That fixture is not the remote catalogue; remote completion still needs checking.

Use a **new code snapshot in the same run**, not ordinary resume with the old
snapshot, and do not launch a new bank:

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
bash scripts/recover_feniks_reference_to_parent.sh "$PARENT_RUN"
)
```

Recovery refuses active jobs (including pending ones), scheduler query errors,
incomplete/corrupt banks, an invalid optimizer checkpoint or an already written
parent. It records preserved hashes and old code/job pointers under `recovery/`,
freezes patched code and submits only population -> report. `STOP.json` causes
the existing best classifier to load without another optimizer step. No change
to manifest, experiment configuration, qualification, banks or training history.
The watcher additionally shows calibration/weight-solving stage and penalty.
After partial submission, ordinary `--resume` uses the newly recorded code;
wait until recorded jobs have stopped. No automatic job cancellation is performed.

## Explicit exploratory continuation, 2026-09-27

The completed `avi_reference_to_parent_20260927_173822` run passes physical SW,
marginals and finite-mass criteria. Its only failed capacity gate is CDF:
0.033833 versus 0.032400, an excess of 0.001433. The user explicitly chose to
proceed to photometry-only parent recovery rather than spend another campaign
on this small excess. This does **not** prove statistical equivalence or justify
rewriting the qualification as PASS.

`--explore-cdf FAILED_RUN NEW_ROOT` imports the completed reference and capacity
receipts into a new run, pins their hashes and freezes current code. The source
is unchanged; `passed: false` is retained. No LP, capacity Monte Carlo, target
generation, standalone pilot or posterior training is repeated. The three-job
DAG is reference bank array -> classifier/blind parent -> report. The unchanged
configuration requests 2097152 forward simulations and at most four H100s.

Admission is intentionally narrow and disclosed as post-hoc: only the
`physical_cdf` candidate, all other original capacity gates passing, finite
metrics, CDF excess at most 0.002 and absolute CDF at most 0.04. These bounds
authorize exploratory compute, not scientific qualification. The launcher
recomputes eligibility from saved metrics and unchanged contracts, verifies
source receipts, and rejects incomplete, corrupt or other failed qualifications.
Default strict submissions still stop on any failed qualification.

The report compares **learned parent** and **truth-assisted capacity control**
using the same reserved bank (role 4). Both retain all 15 dimensions. Each
selected row is weighted by its component's parent weight; parent selection
probability is the weighted sum of component efficiencies. No new DSPS calls
are needed for the comparison. Control weights are read only in the report,
never in classifier/population fitting, bank sampling or initialization.

Look at `report/parent_physical.png`, `selected_physical.png`,
`observable_predictive.png`, `observable_predictive_comparison.csv` and
`DECISION.json`. The latter retains `strict_qualification_passed: false`,
the admission policy and control effective sample size. SFH metrics remain
in `marginals.csv` and `joint.csv`; capacity controls have `capacity_` labels.

- Good control photometry but bad learned photometry: investigate inference,
  ratios or identifiability; this alone does not prove classifier error.
- Both fail photometry/selection: the current reference fit may not reproduce
  the joint population/observation distribution; more classifier epochs alone
  are not a demonstrated remedy. The physical-fit control is not an optimal
  photometric oracle, so its failure is not proof of impossibility for the family.
- Learned parent passes physical/selected/observable development checks: proceed
  to a fresh-simulation 15D posterior benchmark, still not production approval.

This replaces the preceding recommendation for two additional preliminary
checks. Development truth has already influenced the reference; reserve an
independent final evaluation for publication claims.

Launch this continuation from the completed **failed parent run**, not the
older redesign root:

```bash
(
set -euo pipefail
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
SOURCE="$BASE/avi_reference_to_parent_20260927_173822"
PARENT_RUN="$BASE/avi_reference_to_parent_$(date +%Y%m%d_%H%M%S)"
bash scripts/submit_feniks_reference_to_parent.sh --explore-cdf "$SOURCE" "$PARENT_RUN"
)
```

Watcher and missing-only resume commands below are unchanged. An exploratory
watcher displays capacity FAIL and explicit exploratory admission side by side.
No remote submission is performed by preparing this code.

## Method and boundaries

Keep `local_256` exactly as saved: 255 overlapping local physical groups plus
one broad component, normalized mixtures over the same 32768 native joint
15D anchors, adaptive physical kernel widths, SFH widths 0.15. Log stellar
mass uses the saved affine coordinate. All 15 coordinates vary normally;
no nuisance coordinates are fixed or removed. No new reference anchors,
population weights, clipping, target catalogue or noise law are introduced.

Qualification fits weights to TRAIN truth as a **capacity diagnostic only**:

```text
F_mj = Monte Carlo component CDF along physical direction m
min_u sum_m delta_t_m * |sum_j F_mj u_j - F_train,m|
u >= 0, sum(u) = 1
|F_old u - F_old,train| <= old TRAIN minimax optimum + 0.003
```

Physical coordinates are scaled by TRAIN IQR. Five axes and 32 random directions
use merged reference/target quantile grids including the finite draw extrema;
no percentile truncation. This is a finite-grid Monte Carlo approximation to
integrated projected CDF error, not an exact continuous SW minimizer. SciPy HiGHS
solves the convex LP with a 600-second solver budget and explicit primal checks.
Evaluation draws and projection seeds are independent of fitting draws/features.

Recompute the old minimax control with the same new evaluation draws. The new
objective is the sole declared candidate, not selected from a hyperparameter
sweep. Unchanged qualification gates are inherited from the source manifest:
CDF <= max(0.03, 2 x empirical comparator), physical SW <= max(0.035,
2 x comparator), maximum marginal W1/IQR <= 0.1, finite mean linear mass.
TRUTH diagnostic weights stay in `qualification/diagnostic_weights.json`.
Only basis geometry, coordinates and observational feature statistics enter
`reference/`. Those diagnostic weights never initialize or train the parent.

Reference design has been informed by development truth; call the downstream
fit *photometry-only*, not an independent blind test of the whole development
process. Do not claim fresh-test or real-sky validation from these results.

## What the one launch runs

1. CPU qualification: 256 x 4096 = **1048576 cheap latent draws**, zero DSPS.
2. On PASS (or the explicit bounded exploration above): four H100 bank tasks, **2097152 new DSPS simulations** total,
   524288 per task. These are necessary because old components/kernels changed.
   The coherent target and oracle are reused, not regenerated/retrained.
3. One H100 classifier/population task: existing MLP (three 256-wide hidden
   layers, actual input 54 for 18-band flux/error/mask features), at most 600
   epochs, fixed validation and existing bounded plateau stopping.
4. CPU parent/selected/observable report, even after failed dependencies.

Bank rows are assigned before selection to independent training/validation/
calibration/ratio-audit/reserved-predictive roles (70/10/5/5/10%). Binomial
alpha_j counts include *all* simulated parents, including rejected objects.
Every component must appear in each selected role: fail explicitly if support
is insufficient; never silently floor alpha or drop components.

```text
ell(v) = mean_i log(sum_j v_j C_j(x_i)/c_j) - selected-weight KL penalty
u_j = (v_j/alpha_j) / sum_l(v_l/alpha_l)
```

The existing simplex/KKT solver, held-out-observation penalty selection and
independent classifier calibration are reused. Selected `v`, parent `u` and
component `alpha` are all saved. Reserved predictive rows are weighted by `u_j`
because the parent reference simulation mixture is balanced; no individual
inverse-selection correction is applied. No q samples or target theta enter
the population fit. Population evaluation is on development validation truth;
a new independent final benchmark is still needed for publication.

No posterior is trained in this run. That is deliberate: the previous NPE
overfit a finite importance-weighted bank under a poor parent. If the parent
passes the declared development checks, the next step is fresh simulations
from its frozen u and supervised full-15D posterior training. SFH conditional
errors are still reported, not declared irrelevant or validated.

Parent checks are distinct from the *capacity* gates: parent and selected
physical SW <= 0.05, physical marginal W1/IQR <= 0.1, absolute alpha error <=
0.03, per-band weighted CDF KS <= 0.05, predicted mass above target q99.9 <=
0.005, classifier NLL gain >= 0.05 and independent ratio-moment median error
<= 0.03. These are explicit engineering development criteria, not statistical
proof of an unbiased prior or calibrated posterior. Classifier stopping reason
is reported separately; maximum epoch is never called convergence.

## Resource and output summary

- CPU: qualification 30 minutes, report 20 minutes, four threads each.
- H100: bank concurrency 4, two-hour limit per task; classifier three-hour limit.
- Full allocation ceiling **11 H100-hours**, not expected runtime or wall time.
  Queue delay is additional. Qualification failure prevents GPU work unless
  the explicit bounded exploration above is requested in a new root.
- Frozen config/source hashes and code archive; immutable old runs.
- Bank receipts every 8192 rows; missing-only bank and optimizer-state resume.
- `qualification/`: old/new metrics, plot, solver and diagnostic weights.
- `population/`: best classifier, training history, STOP, weights, ratio audit.
- `report/`: parent/selected physical plots, all-15D marginal errors, group SW,
  observable CDF/tail CSV and plot, `DECISION.json`, `REPORT.md`.
- `ROADMAP_STATUS.md`: run-specific checklist and next gate, no production approval.

## Strict launch on Jean-Zay

Use a subshell so a failed check does not close the interactive SSH session.
Do not compare full `git rev-parse HEAD` to a short hash.

```bash
(
set -euo pipefail
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
SOURCE="$BASE/avi_reference_redesign_20260927_163901"
PARENT_RUN="$BASE/avi_reference_to_parent_$(date +%Y%m%d_%H%M%S)"
bash scripts/submit_feniks_reference_to_parent.sh "$SOURCE" "$PARENT_RUN"
)
```

Reconnect-safe watcher:

```bash
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
source "$BASE/avi_reference_to_parent_latest.env"
bash scripts/watch_feniks_reference_to_parent.sh "$PARENT_RUN"
```

Resume after a timeout, **only once all prior attempt jobs have stopped**:

```bash
bash scripts/submit_feniks_reference_to_parent.sh --resume "$PARENT_RUN"
```

Completed failed capacity qualification is not resubmitted unchanged. Inspect
`qualification/FINAL.json` and its plot; only the explicit mode above can admit
the narrowly defined CDF-only case. Submission refuses active
jobs and records every ID immediately; the report uses `afterany` and GPU
dependencies use `afterok` with invalid-dependency cancellation.

Small artifacts only, from the **local** checkout:

```bash
bash scripts/rsync_feniks_coherent_results.sh reference_to_parent
```

This excludes banks, NPZs, checkpoints and files above 25 MiB. Failed population
support is described in `population/selected_support.json` and `BLOCKED.json`.

## Local verification

94 focused tests pass across the objective/pipeline, regularized and unregularized
solvers, reference redesign, reference audit, coherent inference and support-repair contracts. The
end-to-end smoke uses mock photometry but real component sampling, transforms,
classifier optimization, calibrated ratios, selection correction and reports.
It checks immutable sources, observation-only target readers, no diagnostic
weight handoff, strict failed qualification blocking all banks, narrow explicit
exploratory admission retaining FAIL, unchanged imported sources, report-only
capacity controls, idempotence and missing-only Slurm resume. The exact downloaded
failed certificate also passes the new execution admission, not qualification.
Recovery tests force a failure after classifier training, then forbid any new
optimizer call and verify unchanged training artifacts. Sharp 256-component
solver fixtures pass the original KKT and weak-parent-mass constraints.
Ruff, compileall and Bash syntax checks pass.
This is not native GPU DSPS validation; that is the remote bank work. No
per-galaxy MCMC/NUTS, unrelated legacy fit runs or new posterior tests were added.
