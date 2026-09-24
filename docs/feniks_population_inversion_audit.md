# Frozen population-inversion audit

This diagnostic starts from the completed epoch-600 noiseless and noisy ratio
classifiers. It does not simulate galaxies, train a classifier, train an
individual posterior, or modify a production prior.

The selected-population objective is evaluated along a fixed path

`negative log likelihood(v) + lambda KL(v || c)`,

where `c` is the selected reference-component frequency. Consequently the
regularizer shrinks the selection-corrected parent toward the broad reference
parent without using the known target truth.

For each arm, the workflow:

1. recomputes calibrated logits from the frozen best checkpoint;
2. measures direct density distance between the last saved parent checkpoints;
3. solves the complete regularization path with KKT certificates;
4. bootstraps target objects and measures parent-density rather than weight
   instability;
5. keeps candidates within one paired standard error of the best held-out
   likelihood, then selects the smallest bootstrap density instability;
6. uses the known parent only for final closure evaluation.

This distinguishes redundant component weights from an unstable learned
density. If the selected regularized density is stable and closes, the next
block is the still-failing decoder/observation contract. Otherwise the fixed
128-component basis must be reduced or rebuilt before any end-to-end run.

## Extended path after the first audit

The first frozen audit selected the largest tested penalty (`0.01`) for both
photometry arms while stability and parent closure were still improving. That
does not bracket the likelihood/stability compromise, so it is not evidence by
itself that the 128-component basis must be replaced.

`feniks_population_inversion_extension.yaml` preserves the original penalties
and extends the path through `1.0`. It increases the target bootstrap count
from 16 to 32 while retaining the same metric seed. A report now requires the
selected penalty to be strictly below the largest tested value. An unbracketed
path requests another extension; only a bracketed path that still fails the
stability contract recommends reducing or rebuilding the basis.
