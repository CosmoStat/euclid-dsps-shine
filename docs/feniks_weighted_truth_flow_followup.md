# Weighted-truth flow continuation

The first direct weighted-parent capacity run ended while validation was still
improving and before its configured minimum learning rate was used. This
follow-up starts from each immutable best checkpoint and tests two explanations
without generating any new DSPS simulations:

1. `resampled` continues the original fixed weighted-resample objective;
2. `exact_weighted` gives every unique training identity the same number of
   visits and multiplies its NLL by its exact normalized population weight.

Both arms use the same architecture, split, starting checkpoints, batch size,
low-rate schedule and common evaluation target. Metrics are saved after 50,
100, 150 and 200 additional epochs with shared sliced-Wasserstein directions.

Interpretation:

- both arms improve: the first run stopped too early;
- exact weighting clearly wins: weighted resampling was a material error source;
- both arms plateau far above the `0.05` physical target: the current flow
  family or density objective remains inadequate for the weighted parent;
- neither conclusion validates the population classifier or selection
  correction, which are deliberately absent here.
