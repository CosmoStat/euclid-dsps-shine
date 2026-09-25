# After coherent dataset preparation

## Verified versus reported

The user supplied a Jean-Zay watcher showing `COHERENT_PARENT_DATASET_COMPLETE`
on 2026-09-25: train 100000/61188, validation 20000/12199, test 20000/12127
parent/selected rows. These artifacts are not yet mirrored in the local checkout.
The new qualification command rechecks the actual completed artifacts on site.

This is the first self-consistent projected-15D target without the old
multi-band photometric preselection and double weighting. It is not yet a
recovered parent, calibrated posterior, independent numerical-physics test,
or validation of the astrophysical construction of the original Diffsky weights.

## Run now: one CPU qualification, no new simulation

This is a direct CPU command, not an SLURM training array. It reads the six
catalogues, verifies report hashes and exact selected views, and checks unit
weights and effective source separation. It then uses **parent train only**
to diagnose old transform bounds/clipping, round trips, SFH zero pileups and
physical/SFH dependencies. It loads no encoder or prior checkpoint.

The old latent spec is a comparison baseline, not the adopted future transform.
The workflow cannot declare a conditional SFH prior correct from marginal
moments or a correlation matrix. Undefined correlations are explicitly counted.
Repeated empirical values are not automatically evidence of a physical atom;
prominent exact-zero pileups require checking the projection/target definition.

```bash
(
set -euo pipefail
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
export JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu
export EUCLID_DSPS_REQUIRE_GPU=0 EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=1
export JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
source "$BASE/avi_coherent_parent_latest.env"
SPEC="$BASE/avi_population_precision_20260925_105102/decoder/runtime/effective_latent_spec.json"
QUAL="${COHERENT}_qualification_$(date +%Y%m%d_%H%M%S)"
test -f "$SPEC"
printf 'export QUAL=%q\nexport COHERENT=%q\n' "$QUAL" "$COHERENT" \
  > "$BASE/avi_coherent_qualification_latest.env"
python -m scripts.audit_feniks_coherent_parent \
  --root "$COHERENT" --latent-spec "$SPEC" --out "$QUAL" |
  tee "$QUAL.log"
)
```

Read the printed table and `FINAL.json`, `support_and_transform.csv`,
`train_spearman.csv`. `COHERENT_QUALIFICATION_COMPLETE` means the audit finished,
not all scientific assumptions passed. Inspect `blockers` and `review_required`.
The source dataset is never modified. No watcher is necessary for this foreground
check. After a disconnection, check for `FINAL.json`; otherwise rerun into a new
qualification directory. No large simulation needs to be restarted.

## Next jobs, conditional on qualification

These are the next implementation/run stages, **not existing ready-to-submit
commands for the coherent dataset**. Do not launch the old forward run against
this new root. `load_forward_runtime` still loads the historical catalogue,
checkpoint, train/validation indices and feature statistics. The old `PhysicalBasis`
also fixes ten independent standard-normal SFH latent coordinates. Learning only
the physical component weights cannot change that conditional distribution.

1. **Adopt a compatible target and reference, CPU configuration/preflight.**
   Resolve any support/clipping issue explicitly, without dropping troublesome
   rows, silently dequantizing SFHs or using held-out truth to tune the model.
   Decide whether the reference's SFH conditional is scientifically adequate;
   retaining 15 coordinates alone is not sufficient. Use new observed-only
   feature preprocessing and new split identities. Reuse old banks only if
   all generating distributions, transforms and observation contracts match.

2. **In parallel: representation oracle and population bank preparation.**
   One direct supervised parent-density capacity job on the new training truth
   can isolate representation errors from photometric inference errors. It is
   a labelled diagnostic, not population recovery. In parallel, once the
   reference is fixed independently, a bounded array generates the reference
   15D bank with the frozen decoder, Gaussian errors and r selection. Neither
   branch needs the individual posterior. A truth-trained conditional/reference
   may only be used as an explicitly labelled oracle, never as blind recovery.
   This is not another joint-vs-structured hyperparameter sweep.

3. **Population learning, after the bank.** Train/calibrate the ratio classifier,
   fit selected weights and reconstruct parent weights with measured efficiencies.
   Use a regularized estimator selected by held-out observables, with KKT and
   classifier convergence checks. Evaluate independent selected observables and
   parent marginals/joints; the latter truth is for reporting only. Earlier
   in-family bootstrap results do not certify this different target. Save
   measured rejection counts and uncertainty, not only normalized coefficients.

4. **Posterior learning, after parent freeze.** Generate/weight a new supervised
   simulation bank from the learned parent; train the 15D amortized posterior.
   Evaluate 68/95 coverage, ranks, bias, widths and predictive agreement on
   independent simulations and the untouched confirmation catalogue. No q draws
   are used as truth or fed back to update the parent. Include population-prior
   uncertainty before interpreting conditional intervals as globally calibrated.

The immediate action is qualification, not regeneration of the completed 140k
catalogue. Exact resource requests for stages 2-4 must be fixed after the reference
decision and runtime adapter; presenting old launch commands now would silently
restart the old experiment. No NUTS, SMC, extra-band campaign or bootstrap sweep
is needed for this qualification.

## Local verification

28 focused qualification/preparation/provenance/precision tests pass, including
four new tests for clipped transforms, descriptive parent dependencies, immutable
input integrity, and an exact selected view. An additional smoke using the saved
bounded-mixed latent specification roundtrips simulated interior points with
maximum error 2.3e-14 in marginal-IQR units. This does not measure the new cluster
catalogue, certify a parent distribution, or test posterior calibration.
