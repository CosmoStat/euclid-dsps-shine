Independent posterior and learned-prior matrix
==============================================

This experiment separates the two invertible transformations:

.. math::

   q_\phi(x\mid y) = T_q(\epsilon; y),\quad \epsilon\sim\mathcal N(0,I),

   p_\psi(x) = T_p(u),\quad u\sim\mathcal N(0,I).

``T_q`` is a conditional RealNVP fed by the flux-and-error MLP. It outputs
``latent_x`` directly. ``T_p`` is an independent unconditional population
flow. Consequently, the prior Jacobian cannot cancel between ``log q`` and
``log p`` as it can when both densities share the same prior transport.

Matrix
------

The six tasks are:

* ``ind_frozen_rqspline``: AVI with an independent posterior and the existing
  pretrained RQ-spline prior frozen;
* ``ind_joint``: AVI with independent RealNVP posterior and prior, updated from
  the same ELBO step;
* ``ind_vem1``: one encoder epoch followed by one prior-only M-step;
* ``ind_vem4``: four encoder epochs followed by one prior-only M-step;
* ``ind_vem4_hybrid``: VEM 4:1 plus ``50 * NPE_NLL`` on the posterior. The
  weight approximately matches the AVI and NPE encoder-gradient norms measured
  in the previous matrix (about 1025 versus 17.6);
* ``ind_vem4_oracle``: the hybrid job plus ``1.0 * prior_truth_NLL``. This last
  term is a synthetic closure oracle and is unavailable on real photometry.

All VEM jobs execute 120 encoder epochs. Their extra prior epochs maximize the
density of stopped-gradient posterior samples and skip the DSPS decoder. The
hybrid posterior term is usable for synthetic pretraining, but must be disabled
for unsupervised fine-tuning on real data. The prior M-step itself remains
unsupervised and can be used on real data, subject to selection-function and
model-misspecification caveats.

Jean-Zay launch
---------------

Run from the repository root in the Jean-Zay ``shine`` environment:

.. code-block:: bash

   git pull --ff-only
   conda activate shine
   bash scripts/submit_feniks_joint_prior_realnvp.sh

This submits a six-task, ten-minute smoke array and one six-task full array with
an ``afterok`` dependency. Each task requests four H100 GPUs, so either array
can use at most 24 H100 GPUs concurrently. The full output root is
``outputs/runs/feniks_joint_prior_realnvp_jaxcosmo_v1``.

The smoke forces VEM jobs to a 1:1 schedule for two epochs so that it exercises
both the encoder and prior-only compiled steps. Production validation and plot
generation run with Matplotlib ``Agg`` and a private cache. JAX GPU
preallocation is disabled to avoid the pinned-host-memory failure observed in
the earlier AVI jobs around epoch 91.

Results
-------

Every task performs held-out inference on 5,000 objects with 128 posterior
samples. It requires the full latent corner, normalized photometric residuals,
residuals by band, and worst-fit photometry plots before writing ``DONE``. Once
all tasks finish, ``comparison/README.md`` and the comparison CSV/plots are
created under the full output root. No model is selected automatically because
the oracle task is not a deployable real-data method.
