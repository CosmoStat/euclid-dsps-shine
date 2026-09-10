# Global guarded RWS

## Corrected smoke and cadence (v2)

The first remote smoke completed eight sleep-only epochs: it did not exercise
wake or prior updates because the inherited bootstrap lasted sixteen epochs.
It must not authorize training. Use a fresh `global_guarded_rws_v2` output root.
The corrected smoke uses one bootstrap epoch and a 2-sleep/1-wake cycle, reaching
warmup wake at epoch 4 and joint wake at epoch 7. Its flow is trainable from the
first epoch. Actual wake descent in both phases and at least one actual prior
update are required; otherwise the job fails and the dependent array stays blocked.
Full training rechecks this gate before starting.

Full training now uses 2 sleep / 1 wake after its unchanged sixteen-epoch
bootstrap. Its flow thaw and 60+120 phase lengths remain unchanged. This raises
the frequency of observed-data updates from one quarter to one third after
bootstrap, not a guaranteed runtime or convergence improvement. The trainer
does not currently support a 3-sleep/2-wake cycle.

This launches actual shared RWS training, not ELBO-only local adaptation.
New opt-in contracts: float64 conditional coupling layers in both transport
directions, and first acceptable decreasing wake objective step among twelve
halvings. Rejection preserves encoder and optimizer state. The guarded objective
includes the existing entropy penalty and temperature; it is not the exact
posterior KL. Sleep and prior optimizers retain their historical safeguards.

Important: this is a fresh run of the historical global trainer, NOT a
continuation of arm C. That trainer requires an identity-initialized prior and
encoder. It retains the original bootstrap, sleep/wake and population update
schedule, proposal mixtures and selection correction. It does not implement the
previously proposed frozen-prior arm-C continuation. Never resume a legacy
optimizer state under this new numerical contract.

The inherited schedule is 60 prior-frozen epochs followed by 120 joint epochs,
including the existing 16-epoch sleep bootstrap. This is a 180-epoch fresh run,
not a short continuation of the epoch-160 model.

The launcher first runs the same trainer with `--smoke`, then two independent
full-cohort tasks with an `afterok` dependency. Tasks are serialized: each uses
one node, four H100s and 48 CPUs; peak allocation is four H100s. The smoke limit
is one hour, each full task twenty hours. These are limits, not predictions.
The original full schedule may exceed one allocation. Same-contract training
states are saved; do not call the submission script again on an existing root.
The smoke checks execution, not scientific quality; failure blocks the array.

```bash
cd "$WORK/dsps-popcosmos"
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
BASE="/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111"
bash scripts/submit_feniks_global_guarded.sh \
  "$BASE/manifests" "$BASE/global_guarded_rws_v1"
```

Use the actual directory containing `manifest.json`, `full_train_indices.npy`
and `confirmation_indices.npy` if the manifest location differs. Missing files
stop submission; do not substitute diagnostic 16-object cohorts.

The launcher prints both job IDs. Inspect `sacct -j SMOKE,ARRAY` and the `.out`
and `.err` files in the new root's `logs` directory. Scientific success requires
separate posterior readback; decreasing training loss is not sufficient. Do not
promote checkpoints or assume coverage is correct from process completion.

CPU tests cover the global transport against the diagnostic reference and
four-device wake acceptance/rollback. Real H100 execution and end-to-end
decoder qualification remain to be observed. Catalogue truth is not introduced
by this patch. Original training data/selection contracts remain enforced.
