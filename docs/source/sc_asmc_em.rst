Selection-Corrected Amortized SMC-EM
====================================

Scientific scope
----------------

We infer the parent distribution within the predefined FENIKS refinement and
catalogue-support domain, while explicitly correcting the additional observed
r<25 selection.

The existing parquet defines the upstream refinement and catalogue-support
domain :math:`C_0`. It may already contain upstream true-space cuts. The
workflow audits and hashes available upstream provenance, but never regenerates
the dataset. The population target and the only additional corrected selection
are

.. math::

   p_\eta(\theta \mid C_0), \qquad
   A = \mathbf{1}[m_{r,\mathrm{observed}} < 25].

No inference claim is made outside :math:`C_0`.

No-truth boundary
-----------------

``configs/experiments/feniks_sc_asmc_em_r25.yaml`` contains no truth-column
mapping. Training, sleep, both E-steps, both prior M-steps, q distillation,
preflight, checkpoint selection, reporting, and final receipt validation read
only object identifiers, observed fluxes, observed flux errors, and optional
passband masks. Previous q and prior checkpoints are evaluation artifacts only;
the final q trunk, conditional flow, and population prior start from fresh
initializations.

Truth is enabled only by the separate
``feniks_sc_asmc_em_r25_truth_closure.yaml`` configuration. The closure script
refuses to start unless ``FINAL_PASS`` hashes the current no-truth
``FINAL_RECEIPT.json``. It cannot update a checkpoint or posterior bank.

Probabilistic model
-------------------

The production observation family is Gaussian throughout object targets,
sleep generation, and Gaussian-PhotoErr selection completeness:

.. math::

   x_i &\sim p_\eta(x), \\
   \theta_i &= T(x_i), \\
   \widehat f_i &= \operatorname{DSPS}(\theta_i), \\
   f_i \mid x_i,\sigma_i
     &\sim \mathcal N(\widehat f_i,\operatorname{diag}(\sigma_i^2)).

Thus the individual posterior target is

.. math::

   \log \pi_i(x) =
   \log p(f_i\mid x,\sigma_i) + \log p_\eta(x) + \mathrm{constant}.

Neither :math:`\beta(x)` nor :math:`\alpha_\eta` enters normalized object
weights. Student-t2 is restricted to a separately configured robustness
ablation and is not the main FENIKS closure likelihood.

Selection correction
--------------------

With noisy observed r-band selection,

.. math::

   \beta(x) &= P(A=1\mid x), \\
   \alpha_\eta &= E_{x\sim p_\eta}[\beta(x)].

For a mean catalogue loss, the population update minimizes

.. math::

   J_{\mathrm{prior}} =
   -\frac{1}{N}\sum_i\sum_k
      \operatorname{stop}(w_{ik})\log p_\eta(x_{ik})
   + \log\alpha_\eta
   + \lambda_{\mathrm{trust}}
      \operatorname{KL}(p_{\eta_\mathrm{old}}\Vert p_\eta).

The Monte Carlo objective value remains ``log(mean(beta))``. Its exact
score-function gradient uses the centered control variate

.. math::

   \sum_k\left(
      \frac{\beta_k}{\sum_j\beta_j} - \frac{1}{M}
   \right)\nabla_\eta\log p_\eta(x_k).

The receipt records alpha, its Monte Carlo relative error, score-weight ESS,
maximum score weight, score-gradient norm, and finite flags. The pmap M-step
normalizes score weights globally across all local devices.

Latent coordinates
------------------

For each bounded physical parameter :math:`\theta_j\in[l_j,u_j]`, the
``bounded_mixed_warp`` first applies a configured monotonic warp

.. math::

   h_j(\theta) =
   \operatorname{asinh}((\theta-c_j)/\lambda_j)

or, for a positive asymmetric parameter,

.. math::

   h_j(\theta) = \log(1+\theta/\lambda_j).

It then uses

.. math::

   v_j &= \frac{h_j(\theta)-h_j(l_j)}{h_j(u_j)-h_j(l_j)}, \\
   r_j &= \operatorname{logit}(v_j), \\
   x_j &= \frac{r_j-r_j(\theta_{j,0})}{s_j}.

The inverse exactly reverses these operations and guarantees the configured
physical bounds. Centers, lambdas, raw scales, bounds, and initials come only
from configuration. ``latent_transform_provenance.json`` stores their complete
contract and semantic hash.

Photometry and q architecture
-----------------------------

For all 18 bands the encoder receives

.. math::

   \operatorname{asinh}(f_b/f_{b,\mathrm{scale}}), \qquad
   \log(\sigma_b/\sigma_{b,\mathrm{scale}}+\epsilon).

Scales are computed only from observed training-split rows. The production
configuration also appends 18 masks, giving 54 inputs. ``feature_stats.json``
records and hashes the scales and their observed-only provenance.

The conditional normalizing-flow posterior is:

* one 512-wide photometry projection;
* three LayerNorm, Linear, GELU, Linear residual blocks;
* a 256-wide final representation;
* independent 15D mean, 15D log-standard-deviation, and direct 128D context
  heads;
* one conditional Gaussian base, with log standard deviation clipped to
  ``[-4.0, 2.5]``;
* six conditional RealNVP layers, 256-wide coupling networks, alternating masks,
  roll permutations, scale clamp 0.45, and shift clamp 3.0.

The coupling layers consume the direct 128D photometric context, not a
concatenation of mean and log standard deviation. Final coupling layers are
zero initialized, so the flow starts at identity. The 54-input q has 2,441,298
trainable parameters.

The population prior is an eight-layer 256-wide RealNVP with roll
permutations, scale clamp 0.25, shift clamp 3.0, a standard-normal base, and
identity initialization. It has 1,179,888 parameters. The combined final model
has 3,621,186 parameters. Base entropy, flow residual log determinant, full
posterior entropy, and mean/minimum/maximum log standard deviation are logged
separately.

Generalized EM
--------------

The workflow executes exactly two outer iterations. Snapshots and banks are
immutable within each phase:

.. code-block:: text

   initialize q0 and p0 from scratch
   freeze p0; train q_sleep on selected Gaussian sleep pairs; retain EMA q0
   run stratified 512-object cost preflight
   if gate fails:
       extended-SMC-teach 192 hardest objects
       distill q with 3 bank updates : 1 sleep update
       rerun the same preflight once; abort if it still fails

   E1: freeze q0,p0; infer every selected object into bank B1
   M1: freeze q,B1; update p0 -> p1 with selection correction and KL gate
   reweight B1 by p1/p0; refresh only low-ESS objects -> B1,p1
   Q1: freeze p1,B1,p1; distill q0 -> q1 with 3:1 bank/sleep replay

   E2: freeze q1,p1; rerun IS on every object and SMC only where weak -> B2
   M2: freeze q1,B2; update p1 -> p2
   reweight B2 by p2/p1; refresh only low-ESS objects -> B2,p2
   stop; freeze the combined q1-EMA + p2 checkpoint

q and p are never optimized simultaneously from changing particles. Static RWS
and NUTS are absent from this workflow.

E-step hierarchy and preflight
------------------------------

The defensive proposal is ``0.70 q_T1 + 0.20 q_T1.5 + 0.10 p_eta``. Ordinary
importance sampling draws K=64 particles and accepts only finite objects with
``ESS/K >= 0.10`` and maximum weight at most 0.80. Otherwise the same particles
continue into primary adaptive bridge SMC: K=64, at most 16 stages, conditional
ESS target 0.75, resampling below 0.50, RW scale 0.30, three post-resampling
moves, and two final moves. Only failures receive fallback K=128/32 stages at
RW scale 0.15; only the remaining hard objects receive extended K=128/48
stages. Extended SMC is never dispatched uniformly.

The 512-object preflight cohort is stratified by distance to the observed r=25
cut, r-band SNR, error quantiles, and observed colours. The wall-time projection
covers two measured E-steps plus an explicit 20% allowance for selective
refresh, cached-bank training, and reporting. Full-catalogue work proceeds only
when resolved fraction is at least 0.95, unresolved fraction is at most 0.05,
extended fraction is at most 0.15, and measured projected wall time fits the
configured budget. One bounded active-bootstrap retry is permitted.

Posterior bank
--------------

Each object-major shard stores row index, object ID, feature vector, method,
latent particles, normalized weights, source log prior, particle count, ESS,
maximum weight, final beta, log evidence when available, stage count, mutation
acceptance, ancestry ESS, unique-ancestor fraction, movement diagnostics, DSPS
evaluation count, and resolution status. Padded arrays have fixed capacity 128.

Metadata binds dataset, q, q-EMA, prior, latent-transform, feature-stat, and code
hashes together with the canonical workflow-config hash and the likelihood,
selection, C0, and upstream-provenance contracts. Shards use atomic completion
markers, reject changed provenance before resume, and merge through a streaming
index without concatenating all particles in host memory.

Launch and resume
-----------------

Do not submit a final job until the local checks and the immutable four-H100
smoke pass. A local asset-backed smoke is:

.. code-block:: bash

   CONFIG=configs/experiments/feniks_sc_asmc_em_r25.yaml
   CATALOG=/absolute/path/to/existing/feniks.parquet
   RUN_ROOT=outputs/runs/sc_asmc_local_smoke
   python scripts/run_feniks_sc_asmc_em.py --config "$CONFIG" --catalog "$CATALOG" --out "$RUN_ROOT" prepare --estep-shards 1
   python scripts/run_feniks_sc_asmc_em.py --config "$CONFIG" --catalog "$CATALOG" --out "$RUN_ROOT" sleep --smoke
   python scripts/run_feniks_sc_asmc_em.py --config "$CONFIG" --catalog "$CATALOG" --out "$RUN_ROOT" smoke --objects 8

The four-H100 smoke launcher is:

.. code-block:: bash

   sbatch --export=ALL,CATALOG="$CATALOG",RUN_ROOT="$SCRATCH/sc_asmc_smoke_$(date +%Y%m%d_%H%M%S)" scripts/feniks_sc_asmc_em_4gpu_smoke.slurm

The final 16-H100 workflow is gated by both the four-H100 scientific smoke and
a 16-H100 scaling smoke. Submit the three jobs with strict Slurm dependencies:

.. code-block:: bash

   STAMP=$(date +%Y%m%d_%H%M%S)
   SMOKE4_ROOT="$SCRATCH/sc_asmc_smoke4_$STAMP"
   SMOKE16_ROOT="$SCRATCH/sc_asmc_smoke16_$STAMP"
   RUN_ROOT="$SCRATCH/sc_asmc_final_$STAMP"
   SMOKE4_JOB=$(sbatch --parsable --export=ALL,CATALOG="$CATALOG",RUN_ROOT="$SMOKE4_ROOT" scripts/feniks_sc_asmc_em_4gpu_smoke.slurm)
   SMOKE16_JOB=$(sbatch --parsable --dependency="afterok:$SMOKE4_JOB" --export=ALL,CATALOG="$CATALOG",SMOKE4_ROOT="$SMOKE4_ROOT",SMOKE16_ROOT="$SMOKE16_ROOT" scripts/feniks_sc_asmc_em_16gpu_smoke.slurm)
   PROD_JOB=$(sbatch --parsable --dependency="afterok:$SMOKE16_JOB" --export=ALL,CATALOG="$CATALOG",SMOKE_ROOT="$SMOKE4_ROOT",SMOKE16_ROOT="$SMOKE16_ROOT",RUN_ROOT="$RUN_ROOT" scripts/feniks_sc_asmc_em_16gpu.slurm)
   printf 'smoke4=%s smoke16=%s production=%s\n' "$SMOKE4_JOB" "$SMOKE16_JOB" "$PROD_JOB"

The scaling smoke processes four disjoint eight-object cohorts on four
independent four-H100 shards. The production worker verifies all five smoke
receipts against the same catalogue hash, canonical workflow-config hash, and
git commit before doing any production work.

Use ``feniks_sc_asmc_em_4gpu.slurm`` or ``feniks_sc_asmc_em_8gpu.slurm`` for
smaller allocations. A failed Slurm run is resumed by submitting the same
launcher with exactly the same immutable ``RUN_ROOT``; completed bank shards,
component checkpoints, and phase receipts are validated and reused. Inspect
progress with:

.. code-block:: bash

   python scripts/run_feniks_sc_asmc_em.py --config "$CONFIG" --catalog "$CATALOG" --out "$RUN_ROOT" status

The launchers preallocate 88% of device memory, keep K64 and K128 compilation
caches separate, vectorize particles and objects, prefetch host work, and write
bank shards asynchronously. At 16 GPUs all four array workers run disjoint
four-H100 E-step shards; no multi-host JAX collective is required. The measured
512-object preflight replaces the provisional estimate of 24--32 hours on four
H100s, 12--16 hours on eight, or 6--8 hours on sixteen.

Post-training validation
------------------------

NUTS is a separate 8-object validation job and requires ``FINAL_PASS``:

.. code-block:: bash

   sbatch --export=ALL,CATALOG="$CATALOG",RUN_ROOT="$RUN_ROOT",NUTS_OUT="$SCRATCH/sc_asmc_nuts_$(date +%Y%m%d_%H%M%S)" scripts/feniks_sc_asmc_postfreeze_nuts_8gpu.slurm

Truth closure retains 128 dense joint draws per resolved object for q0, SMC
after EM1, distilled q1, and final SMC after EM2. It keeps four statistical
objects separate: individual posteriors, equal-object selected-catalog
posterior mixtures, parent p0/p1/p2 distributions, and beta-weighted selected
p0/p1/p2 distributions. It evaluates 15D coverage, PIT, bias and pulls,
conditional calibration, photo-z bias/NMAD/outliers/CRPS, population recovery
inside C0, MIRA, and TARP. It always consumes the final bank named by the
hash-bound final receipt, including a repaired final bank when present:

.. code-block:: bash

   sbatch \
     --export=ALL,REPO_DIR="$PWD",RUN_ROOT="$RUN_ROOT",CLOSURE_ROOT="$SCRATCH/sc_asmc_truth_closure_$(date +%Y%m%d_%H%M%S)" \
     scripts/feniks_sc_asmc_truth_closure_4gpu.slurm

After the truth closure, the diagnostic-only full-catalogue photometric audit
evaluates DSPS at truth for every selected object and evaluates q0, SMC EM1,
q1, and SMC EM2 posterior-predictive residuals for every resolved object:

.. code-block:: bash

   sbatch \
     --export=ALL,REPO_DIR="$PWD",RUN_ROOT="$RUN_ROOT",CLOSURE_ROOT="$CLOSURE_ROOT",AUDIT_ROOT="$SCRATCH/sc_asmc_predictive_audit_$(date +%Y%m%d_%H%M%S)" \
     scripts/feniks_sc_asmc_predictive_audit_4gpu.slurm

The photo-z scatter uses the posterior median only as a displayed point
estimator. PIT, coverage, CRPS, MIRA, TARP, posterior mixtures, and population
comparisons use the full dense distributions and never collapse a posterior to
a point estimate.

Final artifacts
---------------

The no-truth report contains parent and beta-weighted selected samples for p0,
p1, and p2; population marginals and correlations; raw-q, corrected-E-step,
and final-EM individual comparisons; posterior-predictive photometry; method
fractions; runtime and DSPS cost; alpha and score-gradient diagnostics; the C0
statement; the frozen q1-EMA+p2 checkpoint; and a hash-bound final receipt.

Scientific limitations
----------------------

The method targets only the predefined C0 domain and corrects only the
additional noisy observed r<25 event. Identifiability remains limited by the 18
fluxes, reported errors, the frozen DSPS/calibration model, prior flexibility,
and the assumed Gaussian PhotoErr model. A passing local synthetic test is not
an H100 performance result; a passing preflight is not truth closure; and
post-freeze MIRA/TARP or NUTS cannot retroactively select a training checkpoint.
