# Where does the large weight come from?

## The question

The network proposes possible galaxy parameters. Photometry and the frozen
population prior assign a density to each proposal. The importance weight is
their ratio: `log w = log likelihood + log prior - log proposal`.
A large weight can mean a narrow good region, an underrepresented region, or
both. Neither a single good fit nor a large weight measures uncertainty.

We do not know that the learned population prior is correct. We freeze it to
make a controlled comparison, not to certify it. No model is trained here.

## Job 1: inspect the same points in two ways

Four prespecified cases are reused from the completed local pilot:
`observed_000`, `observed_005`, `simulated_003`, `simulated_004`. They contrast
typical, difficult and unstable behaviour; they are not a representative survey.
Simulation truth is never read. Their saved noisy photometry is reused exactly.

For each case we draw two independent banks of 4096 network proposals using the
qualified float64 conditional transport. We check forward/inverse proposal
density agreement, target decomposition, and stable normalized weights.
`ESS = 1 / sum(normalized_weight**2)` counts effective importance draws, not
MCMC draws. It does not prove that all posterior regions were found.

For each bank we inspect the dominant draw and one ordinary draw. Around each
center we evaluate 15 coordinate directions and four oblique directions, at
21 signed distances (zero and magnitudes 0.0001 through 2). Distance units are the network sample standard deviations
in latent x coordinates. The plots separate likelihood, learned prior, target,
proposal, and the alternative initial-prior target. Curves are relative to the
center; plotting limits are stated and full values remain in `slices.csv`.
`target_gradient_checks.csv` compares automatic derivatives with four finite
difference step sizes around each center for both priors. Inspect agreement
across scales; these descriptive checks are not an automatic convergence gate.
These are density slices, not region probabilities. Neighbours are NOT treated
as draws from the old proposal and no neighbour ESS is calculated.

The initial prior is reconstructed from the source identity-initialized RealNVP
configuration. Its standard-normal density in x is checked numerically. This
is NOT uniform in physical parameters: the same bounded physical transforms
are kept. The learned prior and calibration are never reset for A or B.

Inspect `GEOMETRY_COMPLETE.json`, each `AUDIT.json`, `bank_*.npz` and
`slices_*.png` before explicitly submitting job 2. A completed geometry job
does not certify a posterior or the learned prior.

## Job 2: separate initialization from the prior

| Group | Target prior | Starting points | What the comparison tells us |
|---|---|---|---|
| A | Frozen learned prior | Eight initial-prior draws | Can dispersed chains reach the same regions? |
| B | Same frozen learned prior | Four encoder regions with small perturbations | Does encoder initialization change the sampled result? |
| C | Initial identity prior | Exactly the A starts | How much does changing the prior change the answer? |

A versus B changes only initialization. C changes the target; differences are
prior sensitivity, not necessarily a sampling bug. None is called the true
posterior. These are references conditional on the decoder, noise model,
photometry, coordinate transform and chosen prior.

Each group has eight BlackJAX NUTS chains, vectorized on one H100. Each chain
uses 1000 warmup steps, then eight saved blocks of 512 draws (4096 retained).
Target acceptance is 0.9. The cancelled v2 attempt used maximum doubling depth
10: four tasks remained in the opaque warmup for more than three hours and
produced no restart state. The recovery profile therefore uses the previously
tractable depth cap 4. Saturation counts in the final receipt explicitly tell
us if that cap is too short; increasing it requires evidence from those counts.

There are 12 array tasks and the recovery launcher defaults to all 12
simultaneous: one node/one H100 per task, 12 H100 and 96 chains at peak. Set
`NUTS_ARRAY_CONCURRENCY` lower only for an explicit allocation constraint.
Each task has an eight-hour ceiling. The five-minute heartbeat proves only that
the process is alive; BlackJAX adaptation remains one opaque compiled block and
does not expose an iteration count. Warmup and saved-block timings appear in
task logs.

The existing batched sampler saves restart state after blocks. Re-submitting
`nuts` resumes the same bounded calculation; do not edit the manifest, targets,
settings or starts. Avoid duplicate concurrent submissions on the same root.

All chains are retained. `diagnostics.csv` contains rank-normalized split Rhat,
bulk/tail ESS and mean MCSE in x. `FINAL.json` checks Rhat <= 1.01, both ESS >=400,
zero divergences and zero integration-limit hits. Passing does not establish
absence of unseen modes. Compare A/B distributions and inspect `traces.png`,
`corner_first5.png`, `marginals.png`, and `photometric_residuals.png` as well.
Marginal overlays use the learned-prior AVI bank even in C, deliberately showing
a changed target. The residual plot uses fixed retained draws across all chains;
it is a fit diagnostic, not calibrated predictive coverage.

## Launch

```bash
cd "$WORK/dsps-popcosmos"
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
BASE="/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111"
ROOT="$BASE/frozen_geometry_nuts_v1"
bash scripts/submit_feniks_geometry_nuts.sh geometry "$ROOT" "$BASE/frozen_parent_wake_holdout_v1"
```

Job 1 reserves one H100 for at most three hours. The launcher freezes the code
in a worktree and hashes the inputs. After inspecting its outputs, separately:

```bash
GEOMETRY_ROOT="$ROOT"
ROOT="$BASE/frozen_geometry_nuts_float64_depth4_v3"
bash scripts/submit_feniks_geometry_nuts.sh nuts-new "$ROOT" "$GEOMETRY_ROOT"
sacct -j "$(cat "$ROOT/nuts_job.txt")" \
  --format=JobID%24,State,Elapsed,Timelimit,ExitCode
tail -n 40 -F "$ROOT"/logs/*.out "$ROOT"/logs/*.err
```

Ctrl-C stops only the log display. NUTS completion receipts live under
`nuts/<case>/<A|B|C>/FINAL.json`. Missing receipts or failed diagnostics mean
incomplete evidence; never discard problematic chains to obtain a pass.

The `nuts-new` command imports and verifies all completed geometry artifacts
on Jean-Zay, retaining the original directory unchanged. It records the new
sampler commit and explicit float64 target-coordinate contract. The old NUTS
wrapper cast coordinates to float32; do not use the old frozen checkout.
No geometry computation is repeated. Use `nuts "$ROOT"` only to resume the new
root after its previous tasks have stopped, not for concurrent submissions.
The block executor is now built once per draw count and reused across blocks;
adapted step sizes and mass matrices are dynamic arguments. First-block timing
can include compilation. Logs remain per block, not per decoder evaluation.

See [the geometry v1 analysis](feniks_geometry_v1_results.md) for the measured
contrasts motivating this comparison. CPU tests validate precision and block
reuse, not H100 throughput or convergence on these galaxies.

## Limits

This does not fix AVI or learn a new population prior. It tells us whether the
network misses photometrically plausible regions, whether starting points
matter, and whether the learned prior strongly changes the inferred answer.
We then use that evidence to choose an AVI training change. Weight arithmetic
checks here concern direct network draws, not every historical MIS code path.

## Dense-mass depth follow-up

The completed depth-4 run reached its 15-integration-step ceiling for 97.74%
of retained transitions. High acceptance therefore did not establish useful
travel: the configured trajectory was almost always cut short. The next stage
keeps the learned target and encoder starts (group B) fixed while changing only
the NUTS geometry.

The short probe compares dense adapted mass matrices at depths 5 and 6 on the
same four cases. Each task has eight vectorized chains, 500 warmup transitions
and 512 retained draws per chain. Its purpose is configuration selection, not
a reference posterior. The separate long profile uses dense depth 6, eight
chains, 1500 warmup transitions and 4096 retained draws per chain. It is a
reference candidate only if R-hat, bulk/tail ESS, divergences, trajectory-limit
hits, traces and agreement across initial regions are acceptable.

Simulation truth is loaded only after sampling and drawn as a dashed line/star
in `marginals.png` and `corner_first5.png`. It cannot alter starts, targets,
adaptation, convergence diagnostics or selection. Observed galaxies have no
truth overlay.

Prepare two new immutable roots from the completed geometry root:

```bash
cd "$WORK/dsps-popcosmos"
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
git pull --ff-only origin feature/feniks-exact-posterior-benchmark

BASE="/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111"
GEOMETRY_ROOT="$BASE/frozen_geometry_nuts_v1"
PROBE_ROOT="$BASE/frozen_geometry_nuts_dense_depth56_probe_v1"
LONG_ROOT="$BASE/frozen_geometry_nuts_dense_depth6_long_v1"

bash scripts/submit_feniks_geometry_nuts.sh \
  nuts-probe-new "$PROBE_ROOT" "$GEOMETRY_ROOT"
bash scripts/submit_feniks_geometry_nuts.sh \
  nuts-long-new "$LONG_ROOT" "$GEOMETRY_ROOT"
```

This requests eight H100s for the probe and four H100s for the long run, with
eight chains vectorized per GPU: 12 H100s and 96 chains at peak if both arrays
run simultaneously. The time limits are four and twelve hours respectively.
To reduce peak allocation, set `NUTS_ARRAY_CONCURRENCY` separately before each
submission.

After completion:

```bash
python scripts/summarize_feniks_nuts_followup.py "$PROBE_ROOT" | \
  tee "$PROBE_ROOT/summary.txt"
python scripts/summarize_feniks_nuts_followup.py "$LONG_ROOT" | \
  tee "$LONG_ROOT/summary.txt"
```

The probe writes `depth_comparison.png`. Do not accept depth 6 merely because
it has fewer limit hits than depth 5: the absolute limit-hit rate should become
rare, divergences should remain rare, and all chain diagnostics still apply.
