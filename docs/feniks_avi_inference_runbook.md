# Frozen AVI inference and MIRA

This compares the source and seven final v6 encoders, not a newly trained prior.
Use the identical 512 development-validation identities, mask and photometry.
This is not an untouched final test or a population-prior qualification.

Each arm uses four H100 on one node. Eight concurrent arms use 32 H100 on eight
nodes. Concurrency 4 uses at most 16 H100. A dependent one-H100 job compares MIRA.
The six-hour allocations are ceilings, not measured duration estimates.

For each galaxy and arm, draw two fresh independent K4096 banks from the learned
q (including the gate for mixtures). Compute log likelihood + log frozen prior
- log q; do not use the defensive training proposal r as the denominator here.
Selection correction remains enabled in the runtime. Its fixed normalizer
cancels from normalized individual weights; beta/alpha are not inserted again.
Save every physical joint draw and weight as Parquet, plus 256 raw and 256
multinomial IS draws per galaxy and replica. Never deduplicate or smooth an
IS bank, silently replace invalid weights or exclude low-ESS objects.

The existing MIRA implementation runs on raw and IS samples with shared regions
across models and the same seed across replicas. Canonical spline15D truth
columns must be present and finite for every fixed validation identity. Truth
is read only by preparation/reporting and never passed to inference kernels.
Object IDs in these outputs are row positions scoped to the frozen catalogue.
The MIRA reference is 2/3, not a score to maximize. Resampling 256 times does not
create 256 effective samples. Compare both replicas and inspect low-support tails.

## Launch on Jean-Zay

```bash
cd "$WORK/dsps-popcosmos"
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
BASE="/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111"
TRAINING_ROOT="$BASE/avi_encoder_experiments_v6"
INFERENCE_ROOT="$BASE/avi_encoder_inference_v1"
bash scripts/submit_feniks_avi_inference.sh "$TRAINING_ROOT" "$INFERENCE_ROOT" 8
bash scripts/watch_feniks_avi_inference.sh "$INFERENCE_ROOT"
```

Preparation checks the source manifest inputs, seven completion receipts, encoder
hashes and truth columns before submission. Runtime validates frozen-prior hashes,
checkpoint shapes, row ordering and float64 GPU support. Code and filters are
snapshotted, as for AVI training. Do not use the legacy single-flow inference
launcher for these mixture checkpoints.

## Inspect

- `comparison.csv` and `comparison.png`: ESS, largest weight, raw/IS photometry.
- `arms/*/metrics.csv`: every galaxy and replica, including resampling diversity
  and log-evidence estimates for checking replica disagreement.
- `mira_0/mira_scores.csv`, `mira_1/mira_scores.csv` and their PNGs: raw vs IS.
- `marginals/*.png`: eight fixed row positions, source/B/E, truth overlays.
- `arms/*/bank_*/*.parquet`: all joint particles and their normalized weights.
- `arms/*/{raw,is}_*.parquet`: joint samples actually supplied to MIRA.
- `FINAL.json`: workflow completion only, no scientific promotion.

The prior remains the learned frozen source. Disagreement with catalogue truth
may reflect encoder approximation, finite-K IS, or a misspecified frozen target.
MIRA alone does not distinguish these explanations. All parameter correlations
are retained in the sample tables even though the quick plots are marginals.
