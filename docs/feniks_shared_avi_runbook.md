# Shared AVI, no posterior teacher

This is actual shared-network training, not per-galaxy local adaptation.
The photometric likelihood and the current frozen prior define the stochastic
ELBO objective. Catalogue truths and posterior teachers are not used.

Two seeds run as a Slurm array, each using one H100 on one node and 16 CPUs.
Peak concurrency is two nodes / two H100s; each task has a ten-hour limit.
The bounded first run uses up to 512 training galaxies, up to 128 disjoint
tracking galaxies, four epochs and eight Monte Carlo samples. Tracking galaxies
are development data, not a new independent confirmation cohort.

The source is certified precision-night arm C. The existing global training
path preserves its configuration/checkpoint precision. It does NOT use the
local diagnostic transport64 wrapper, and the local 32/32 audit must not be
claimed as certification of this global optimizer. This distinction remains an
open numerical validation item. The run is exploratory, not production.

On Jean-Zay, activate shine and pull the branch first. From the repository:

```bash
BASE="/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111"
bash scripts/submit_feniks_shared_avi.sh \
  "$BASE/frozen_parent_precision_night_v1" \
  "$BASE/shared_avi_v1"
```

Submission prints the array job ID. Use `sacct -j JOBID` and inspect logs under
`$SCRATCH/feniks_sc_drws_runtime/slurm_logs/shared-avi-JOBID_TASK.out` (and `.err`).
Each successful task writes `seed_TASK/AVI_TRAINING_COMPLETE.json`. The receipt
identifies the final checkpoint, not a selected winner. It explicitly records
that posterior evaluation has not run. Retain both seeds and all checkpoints.

No population prior update is implemented by this launcher. That needs a
selection-corrected population stage and refreshed weights under the changed
prior, plus independent posterior evaluation. Do not run the historical
population launcher against these new checkpoints without the appropriate
source and precision contracts. Training completion is not posterior quality.
