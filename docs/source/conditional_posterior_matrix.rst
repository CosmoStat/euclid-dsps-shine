Conditional posterior matrix
============================

Purpose
-------

This experiment compares amortized variational inference (AVI) and supervised
neural posterior estimation (NPE) without changing the JAX-COSMO spline data,
the frozen population prior, the photometric likelihood, or the held-out rows.
For each objective it evaluates the existing diagonal Gaussian, a Gaussian in
the frozen prior's base coordinates, a conditional RealNVP, and a conditional
rational-quadratic spline flow.

The conditional models use one shared photometric MLP. Its mean and log-scale
heads define the base density and provide the context for the coupling layers.
The residual conditional flow is followed by the frozen prior transport, so all
models decode through the same physical latent contract and DSPS model.

Execution contract
------------------

Run the guarded submitter from the repository root on Jean-Zay::

   bash scripts/submit_feniks_conditional_posterior_matrix.sh

It submits a ten-minute smoke array and one full array with an ``afterok``
dependency. Every full task trains, performs 5,000-object held-out inference,
finalizes the standard diagnostics, and writes a task metric bundle. The last
successful task aggregates the matrix under::

   outputs/runs/feniks_conditional_posterior_jaxcosmo_v1/comparison/

Scientific outputs
------------------

The comparison report links the complete truth/prior/posterior corner, the
normalized photometric residual distribution, residuals by band, and the worst
posterior-predictive photometric fits for every task. Selection uses posterior
coverage first, then parameter RMSE, subject to a 20 percent per-epoch slowdown
limit relative to the AVI Gaussian baseline.

Runtime safeguards
------------------

The jobs require the exact JAX-COSMO amortized catalog contract and frozen prior
checkpoint, refuse to overwrite outputs, use four H100 devices with ``pmap``,
and force Matplotlib's non-interactive ``Agg`` backend with a per-job cache.
NPE uses a larger global batch because it does not evaluate DSPS during each
training update. AVI retains the smaller batch validated for the DSPS decoder.
