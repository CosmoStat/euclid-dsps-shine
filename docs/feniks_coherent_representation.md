# Coherent parent representation: bounded next job

## Why this job

The 2026-09-25 qualification reported an intact coherent dataset, but the old
bounded transform clips 0.423% of stellar masses and rare SFH contrasts. Exact
zeros reach 5.788% in the final contrast; physical/SFH rank correlation reaches
0.360. These are user-supplied cluster measurements, not a local readback.

Do not regenerate the 140000 galaxies. Do not launch the old 128-component
reference with independent standard-normal SFH coordinates on this new target.
This job isolates **density representation**, using known parent training truth.
It is deliberately not observed-only parent recovery or a production posterior.

## Coordinate and density contracts

- Use parent train only for robust coordinate centers/scales. Apply the frozen
  map to validation and test; fail rather than adapt to held-out support.
- Keep the declared open bounds of z, metallicity, Av and dust slope. Map them
  with logit. Any endpoint or out-of-support row fails preparation.
- Stellar mass is already log10. Map it and all ten SFH contrasts with an
  invertible shifted asinh on the whole real line, not another expanded box.
- No clipping, removal, resampling, weighting a second time or dequantization.
  All SFH zeros and photometry remain unchanged; the old pipeline is untouched.
- The model is `p_x(a,b) = p_x(a) p_x(b|a)`. Independent parameter sets protect
  the physical marginal from a loss dominated by narrow SFH spikes. The
  conditional still learns correlations, and sampling returns all 15 dimensions.
- Each factor is a normalized mixture of two independent conditional flows:
  12 rational-quadratic spline layers, 16 bins, width 256, residual context trunk,
  float64 transport. No new neural architecture is introduced.
- `log p_theta = log p_x - log |d theta/d x|`. The new coordinate Jacobian is
  tested against autodiff; trained flows undergo inverse/logdet tests. This is
  a normalization-by-construction contract, not 15D numerical quadrature.

This standalone coordinate specification is not a drop-in old `LatentSpec` or
old encoder checkpoint. Old reference/posterior banks are incompatible. Train
truth makes it a **capacity oracle**: it cannot silently become a blind reference
prior or a successful photometric population estimate.

## SFH zeros are not silently fixed

In parallel, a CPU task replays the original projection on 1024 parent-train
objects with the saved time-bin count. It counts equal adjacent log-SFR knots
at the floor and away from the floor, and verifies stored contrasts are reproduced.
Projection presently uses float32 internally. Equal nonfloor knots alone do not
separate a real plateau from finite precision. No new DSPS photometry is computed.

A continuous density cannot reproduce genuine point masses exactly. Its NLL can
improve by making arbitrarily narrow peaks. This run preserves the target and
reports exact zeros plus probabilities within 0.01 IQR of zero. It does **not**
claim to resolve the physical-versus-numerical atom question, choose a smoothing
scale, or validate the full SFH density solely from NLL. Physical marginal
representation can be assessed independently of this conditional limitation.

## Launch on Jean-Zay

Use a subshell: a failed command must not disconnect your interactive shell.
Activate the existing `shine` environment. Preparation verifies dataset receipts,
the exact selected views and split independence, then writes a new immutable
coordinate/cache contract. No old checkpoint is loaded.

```bash
(
set -euo pipefail
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
COHERENT="$BASE/avi_coherent_parent_20260925_224312"
SPEC="$BASE/avi_population_precision_20260925_105102/decoder/runtime/effective_latent_spec.json"
TAG=$(date +%Y%m%d_%H%M%S)
REP="$BASE/avi_coherent_representation_$TAG"
printf 'export REP=%q\n' "$REP" > "$BASE/avi_coherent_representation_latest.env"
bash scripts/submit_feniks_coherent_representation.sh "$COHERENT" "$SPEC" "$REP" |
  tee "$BASE/avi_coherent_representation_submission_$TAG.txt"
)
```

Monitor, including after reconnecting:

```bash
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
source "$BASE/avi_coherent_representation_latest.env"
bash scripts/watch_feniks_coherent_representation.sh "$REP"
```

Only **two H100 tasks**, each with a 90-minute ceiling, fit the two independent
factors in parallel, each using the 100k parent training rows. Maximum 120 epochs
per factor, fixed validation, learning-rate decay every 30 epochs; this is not an
automatic convergence certificate. GPU allocation ceiling is 3 H100-hours, not
a runtime prediction. Projection and report each request four CPU cores for
30 minutes. No `--mem` option. Reports use `afterany`, recording missing stages
instead of waiting forever on `DependencyNeverSatisfied`.

If a task times out, after other submitted tasks have finished:

```bash
bash scripts/submit_feniks_coherent_representation.sh --resume "$REP"
```

This retains the immutable code/config, skips completed factors, and restores
model plus optimizer at the last complete epoch for the missing factor. It does
not restart from epoch zero. A disconnect alone requires only the watcher, not
resubmission. The submitter refuses to resume while recorded jobs are active.

## Read results in this order

1. `transform_checks.csv`: no silent clipping; tiny roundtrip errors on all splits.
2. `report/training.png` and each factor's `FINAL.json`: validation versus train,
   best checkpoint and remaining improvement over the last 20 epochs. New NLLs
   have different coordinates/targets and are not comparable to old run values.
3. `report/representation_physical.png` and `physical_corner_theta.png`: parent
   marginals and correlations; `physical_corner_x.png` gives latent counterparts.
4. `report/joint.csv`, `marginals.csv`: flow versus held-out parent, with the
   validation/test distribution difference as context (not an acceptance floor).
5. `report/representation_sfh.png`, `truth_spearman.csv`, `flow_spearman.csv`,
   `sfh_zero_neighborhoods.csv`, and `sfh_zeros/projection_zeros.csv`.

Full dense joint samples are saved in `report/draws.npz`. Checkpoints are
`physical/best.eqx` and `sfh_conditional/best.eqx`; recoverable optimizer states
and histories remain per factor. All completed stages have hashes/receipts.
Small rsync: retain JSON/CSV/PNG/MD/YAML/JSONL/logs, exclude `cache/`, `*.eqx` and
`*.npz`; do not accidentally use an empty remote-root variable.

## Roadmap boundary

This addresses the support bug and tests correlated 15D representation on the
new, once-weighted coherent target. Local invariants and a tiny optimizer smoke
are not remote scientific validation. It does not establish the independent
reference conditional needed by photometric population learning. That choice,
followed by new compatible simulation banks and classifier/selection closure,
is the next gate before the final supervised 15D posterior run. No bootstrap
sweep, extra bands, NUTS/SMC, or q-to-parent feedback is added.

## Local verification and limits

Focused tests exercise coordinate tails/zeros, analytic versus autodiff
Jacobians, fixed train-only fitting, two-factor optimization, strict receipts,
reporting, and resume without repeating completed epochs. The regression set
also covers coherent preparation, qualification, precision and structured priors.
A plot smoke was generated and inspected. A real projection smoke on 32 local
historical galaxies reproduces contrasts exactly across two batch sizes; the new
coordinate roundtrip has maximum absolute error 1.8e-15. These are not the new
Jean-Zay dataset's scientific results.

The local `.venv` runs tests with CPU/x64 JAX. For the optional actual projection
smoke only, pure-Python Diffstar/Diffmah were imported from the local `shine`
site-packages because the `.venv` lacks them; no dependency files were changed.
The legacy one-row and small-batch CLI smoke commands stop at their missing
`configs/diffsky_hltds_04_14_simple_gpu.yaml` config. Actual new-factor optimizer
smokes passed instead. No H100 or remote run has been executed by this patch.
