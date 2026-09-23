# Clean parent tests

Local verification: seven focused tests (including contract pass/fail separation,
real tiny flow training and end-to-end report IO) and 21 existing
population/capacity tests pass. The closure
smoke uses a toy forward law and exact analytic classifier ratios; it does not
certify DSPS/classifier scientific recovery. Full DSPS one-row and two-row fit
attempts are blocked by the local CPU-only JAX environment and a GPU-forcing
configuration. Jean-Zay execution has not been performed.

## What "contracts" means

These are consistency checks, not additional optimization objectives:

1. Specify which probability measure is the target. `population_weight`-weighted
   parent objects are not automatically the same target as unweighted catalogue
   rows. The representation oracle explicitly uses the former; matched closure
   creates a new unweighted iid catalogue from a specified parent.
2. Check theta -> latent x -> theta. Bounds/clipping must not silently change
   the target. `transform_contract.csv` reports errors scaled by truth IQR and
   the fraction at/outside bounds; the configured tolerance is a numerical
   guard, not a scientific acceptance threshold.
3. Check trained flow inversion in both directions, logdet cancellation,
   forward-sampled versus inverse-evaluated density, and spline Jacobians against
   autodiff. Mixture log_prob compared with its own re-evaluation is not a test.
   `transport_contract.json` retains per-expert errors. Failure blocks closure.
   These finite checks do not prove global numerical normalization.
4. A historical catalogue needs a separate observation contract: decoder outputs,
   filters, units, errors, masks and selection must match its generation law.
   The matched experiment below deliberately uses the SAME observation law on
   both sides. It does not resolve or hide historical decoder/noise failures.

## Experiment 2: representation

Four fits: two seeds of a fresh production-style joint 15D density, and two
seeds of a structured density

    p(x) = p(x_a) p(x_b | x_a).

Both factors reuse the existing independent-expert RQ spline implementation,
with 5 outputs / constant input and 10 outputs / 5 conditioning coordinates.
The factors have disjoint parameters and optimizers. The SFH conditional can
learn correlations with a but cannot modify p(a). Their normalized product
is a 15D density, not a five-dimensional simulator or posterior. Physical-space
density additionally requires the existing latent-transform Jacobian.

The joint arm retains production encoder settings. Both arms have four experts
per factor, use 200 epochs, exactly weighted equal identity visits (8 repetitions
per epoch), and the SAME saved train/validation/test identities from the original
weighted-truth capacity run. No old checkpoint is extended. The structured arm
has two networks and is NOT a parameter/FLOP-matched comparison: parameter counts
are reported. Validation selects each factor by its own weighted NLL.

Held-out NLL, 15D marginals, 5D corners in theta and x, and common-direction joint
distances are reported. An object-level weighted bootstrap of the held-out truth
estimates finite-object variation; resampling 16k draws is not treated as 16k
independent galaxies. No arbitrary universal SW threshold promotes the run.
No artificial jitter is added to the empirical truth.

## Experiment 3: matched forward closure

This first recovery test uses the existing normalized component family, NOT
the new structured flow. This isolates the classifier/selection estimator from
representation errors. It does not yet implement learning continuous structured
parent parameters from photometry.

A fixed seeded physical tilt defines known parent weights independently of the
catalogue. Components outside declared reconstruction support receive zero mass;
the omitted mass is reported. Generate exactly 131072 fresh parent simulations,
with all 15 latent coordinates, saved Gaussian m5 observation law, all-observed
masks and observed r selection. Invalid draws fail the run instead of silently
changing the parent. Banks are restartable 8192-row shards.

Use selected objects from the first half to fit v using the existing frozen
classifier and its actual training frequencies c. Recover u proportional to
v/alpha and normalize. The independent second half supplies parent/selected
truth and observable closure. No galaxy posterior is loaded or called. The
true generating weights are not supplied to the optimizer.

`true_selected_using_reference_alpha` is explicitly based on the estimated
reference efficiencies, not an exact analytic selection probability. Compare
`alpha_true_reference` against independent `alpha_true_fresh` as well as
`alpha_fitted`. Parent metrics use fresh samples from the fitted parent. Selected
predictive metrics use diagnostic p_fit/p_true importance reweighting of the
independent forward holdout, with ESS reported; low ESS invalidates that
predictive diagnostic. This is not q-derived IS population learning.

Sixteen catalogue bootstraps retain parent weights, physical means and held-out
observable log-ratio scores. Tangent likelihood curvature reports weak weight
directions. Neither sparse weights nor large weight L1 alone proves a wrong
physical density. This uncertainty is conditional on the frozen classifier and
alpha estimates, not a full population hyperposterior. Checkpoints and the
classifier reference banks are reused, not retrained or regenerated.

## Launch

After these source changes are transferred to the Jean-Zay checkout, the launcher
submits `contracts -> four representation fits -> matched closure -> report`:

```bash
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
CAPACITY="$BASE/avi_weighted_truth_flow_capacity_20260922_112310"
CLEAN="$BASE/avi_clean_parent_$(date +%Y%m%d_%H%M%S)"
bash scripts/submit_feniks_clean_parent.sh "$CAPACITY" "$CLEAN"
printf 'export CLEAN=%q\n' "$CLEAN" > "$BASE/avi_clean_parent_latest.env"
bash scripts/watch_feniks_clean_parent.sh "$CLEAN"
```

The immutable code snapshot follows existing SCRATCH conventions. One 2h
contract job, four single-H100 fits with concurrency 2 and 8h ceilings, followed
by one 4h closure job and one 1h report job: ceiling 39 H100-hours, NOT an
expected runtime. Representation starts only after the numerical contract job
succeeds. Scientific contract failures are recorded, not converted into a job
failure, so all three failure sources remain observable. Closure starts only
after all four fits and their numerical transport gates succeed. A large scientific
distance does not itself block closure: representation and estimator failures
are separate questions. Submitted job IDs are recorded immediately, including
partial-submission failures. All outputs are under CLEAN; no original run changes.

## Read the result

- `report/representation_theta_15d.png`: truth vs the two arms and replicas.
- `report/representation_{latent_x,physical_theta}_corner.png`: physical joint shape.
- `report/representation_joint.csv` and `truth_object_bootstrap_reference.csv`:
  residual error versus finite truth-object variation.
- `*/replica_*/{transform_contract.csv,transport_contract.json,FINAL.json}`:
  numerical tests, held-out density and parameter counts.
- `report/matched_parent_selected.png`: parent and selected must remain distinct.
- `closure/{weights.csv,parent_joint.csv,selected_joint.csv,observable_closure.csv}`:
  density and observable closure, not just weight recovery.
- `closure/{bootstrap.csv,likelihood_geometry.json,FINAL.json}`: degeneracy,
  uncertainty limitations, independent alpha and predictive ESS.

Do not promote a production run merely because both scripts finish. If numerical
tests fail, fix transport first. If only joint representation fails, investigate
the physical-marginal/SFH interaction. If matched recovery fails despite a
representable parent, investigate ratios and selection. Passing these tests still
requires resolving the historical catalogue observation contract before reuse.
