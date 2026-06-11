Scientific Validation Plan
==========================

Goal
----

The Diffsky HLTDS workflow is meant to validate a differentiable generative
model of a galaxy population. The goal is not only to fit photometry. The goal
is to test whether the model learns a physically realistic population.

Separate Objectives
-------------------

The workflow separates three objectives:

* supervised prior learning on ``theta_true``;
* same-parameter forward closure ``theta_true -> photometry``;
* photometric posterior inference ``q(theta | flux)``.

These objectives should not be mixed in one interpretation.

Required Warning
----------------

A good photometric fit is not evidence of physical recovery.

Physical claims require:

* same-param forward closure;
* supervised prior vs truth diagnostics;
* posterior calibration;
* comparison of derived quantities, not only raw latent parameters.

Validation Order
----------------

1. Build a clean Diffsky dataset with explicit truth semantics and error model.
2. Learn a supervised prior from available truth parameters.
3. Run true-parameter forward closure.
4. Use standard-normal, supervised-frozen, and joint RealNVP priors in
   photometric amortized inference.
5. Run redshift ablation and posterior calibration checks.
6. Compare truth, prior, and aggregate posterior population diagnostics.
7. Clean public configs and documentation once the science path is stable.
