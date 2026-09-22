# Weighted true-parent flow capacity

This diagnostic answers one narrow question independently of photometry,
selection, the 128-component basis, and the density-ratio classifier:

> Can the production 15D flow family represent the known weighted parent
> distribution after the exact latent normalization used for inference?

Two independently initialized copies of the production posterior architecture
are fitted at constant context. Training targets are direct truth rows sampled
with probability proportional to `population_weight`, transformed through the
configured invertible `theta -> latent_x` mapping. No approximate posterior
sample, component label, classifier output, or selection weight enters the
fit. Truth is used only for this diagnostic and the production prior is never
modified.

The immutable object split is shared between replicas. Training, validation,
and held-out NLL samples are weighted resamples inside disjoint identity
partitions. The displayed distribution target is one common 65,536-draw
weighted resample of the complete parent, so both flow replicas are compared
to exactly the same empirical density.

The report writes:

- `weighted_truth_flow_capacity_summary.png`: the single audit figure, with
  weighted truth in physical coordinates, the exact normalized truth, both
  flow replicas in normalized coordinates, both replicas transformed back to
  physical coordinates, and joint metrics;
- `weighted_truth_theta_15d.png`: weighted target and both learned flows in
  physical coordinates;
- `weighted_truth_latent_x_15d.png`: the same comparison in normalized latent
  coordinates;
- physical 5D corner plots in both coordinate systems;
- training and fixed-validation loss curves;
- marginal W1/IQR and joint sliced-Wasserstein metrics in both spaces;
- flow-to-flow stability metrics;
- dense joint target and learned draws for reproducible analysis.

This is a capacity diagnostic, not a valid way to learn a prior from real
observations: it deliberately uses simulation truth. A good result means the
flow family is expressive enough for this target. It does not validate the
classifier, selection correction, or observation model.
