# Adaptive FENIKS ratio-classifier continuation

This workflow resumes only the independent seed-1 noiseless and noisy
photometric classifiers from a completed ratio-followup run. It reuses the
same simulation banks, train/validation/calibration/audit split and target
fit/held-out split. It performs no DSPS simulation and does not train a
posterior.

Each arm advances in 20-epoch blocks. At every checkpoint it:

1. selects the best validation-NLL classifier seen so far;
2. refits convex component-logit offsets on the fixed reference calibration
   split;
3. measures ratio normalization on the independent reference audit split;
4. solves the selected and parent mixture weights on the fixed target split;
5. recomputes physical parent and selected sliced-Wasserstein distances.

An arm stops after the configured minimum epoch only when two consecutive NLL
windows improve by less than the preregistered tolerance, the calibrated ratio
identity passes, and the parent physical SW changes by less than its tolerance.
The maximum epoch is a fail-closed resource bound, not a declaration of
convergence.

If converged classifiers retain boundary solutions and poor parent closure,
the next experiment must regularize or reduce the 127 free mixture-weight
degrees of freedom. More classifier epochs are then scientifically unjustified.

## In-place recovery

The initial epoch has no preceding parent-SW checkpoint, so its change is
serialized as JSON `null`. Unexpected non-finite scientific metrics still fail
strict serialization. If an older code snapshot failed while writing that
initial progress receipt, resume the existing root with:

```bash
bash scripts/resume_feniks_ratio_convergence.sh "$RATIO_CONVERGENCE"
```

The recovery validates the saved trajectories and checkpoints, preserves the
original job and code provenance under `recovery/<timestamp>/`, and replaces
only the failed two-cell classifier array and its blocked report job.
