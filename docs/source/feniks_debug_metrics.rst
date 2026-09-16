Metrics: Definitions and Interpretation
=======================================

Return to :doc:`feniks_debug_meeting` or :doc:`feniks_debug_experiments`.
These definitions follow the repository's diagnostics.

First Read: What Each Diagnostic Asks
-------------------------------------

* **Residual RMS:** do proposed galaxies reproduce the measured light?
* **ESS:** after weighting, how many samples meaningfully contribute?
* **Maximum weight / Pareto k:** is the answer dominated by a few rare samples?
* **Evidence delta:** do two independent sample batches give similar estimates?
* **PIT / coverage / MIRA:** when truth is available, do the inferred
  distributions behave as expected across independent examples?
* **Gradient audit:** does the calculated local direction agree with direct
  perturbations of the objective?
* **Descent check:** did the actual update reduce its fixed-batch loss?

No one diagnostic answers all these questions. In particular, low residuals
do not imply high ESS, and a safe optimizer step does not imply calibration.
The equations below specify precisely what the reported numbers mean.

Target and Training Objectives
-------------------------------

x is the unconstrained joint latent vector, y the photometry, p(x) the frozen
prior and f_b(x) the decoder. Densities in an importance ratio must use the
same coordinates and include the appropriate transform Jacobians.

.. math::

   p(x\mid y)=p(x)p(y\mid x)/Z(y),\qquad
   \log p(y\mid x)=-\frac12\sum_{b\in\mathcal B}
   [(y_b-f_b(x))^2/\sigma_b^2+\log(2\pi\sigma_b^2)].

This is the fixed independent-Gaussian diagnostic likelihood; masked bands
are excluded. Units and noise conventions must remain fixed. Positive
log-likelihood is possible for continuous densities; it is not a probability.

Sleep/NPE fits the shared network using simulated joint samples:

.. math::

   L_{sleep}=-\mathbb E_{p(x)p(y\mid x)}\log q_\phi(x\mid y).

Reverse VI minimizes the negative ELBO:

.. math::

   L_{VI}=\mathbb E_q[\log q-\log p(x)-\log p(y\mid x)]
   =D_{KL}(q\Vert p(x\mid y))-\log Z(y).

Lower is better in expectation at fixed target. Finite-sample estimates need
not improve monotonically. ``negative_elbo`` is this loss, not its negative;
a lower value does not guarantee reliable importance tails.

Wake draws from the exact mixture of local q and frozen anchor a:

.. math::

   m(x)=\tfrac12q(x)+\tfrac12a(x),\quad
   \ell_s=\log p(x_s)+\log p(y\mid x_s)-\log m(x_s),\quad
   L_{wake}=-\sum_s\bar w_s\log q_\phi(x_s).

Samples and weights are held fixed during differentiation. Using only the
sampled component's density instead of m changes the estimator. This is an
importance approximation to a forward-KL objective, sensitive to poor support.

ESS and Concentration
----------------------

For direct-q evaluation replace m by q. Normalize stably with log-sum-exp:

.. math::

   \bar w_s=e^{\ell_s-\operatorname{LSE}(\ell)},\quad
   ESS=1/\sum_{s=1}^K\bar w_s^2,\quad ESS_{fraction}=ESS/K.

Uniform weights give ESS=K; one dominant particle gives ESS approximately one.
At K=4096, fraction 0.001 means about 4.1 effective draws, not 0.1% error.
This is raw importance ESS, not MCMC autocorrelation ESS, and does not prove
all target modes were discovered.

.. math::

   w_{max}=\max_s\bar w_s,\qquad W_{10}=\sum_{s\in\text{ten largest}}\bar w_s.

Maximum weight 0.99 means one draw carries 99% of empirical mass. Top-ten mass
near one means almost all other particles contribute negligibly. The reported
log-weight range is max(log w)-min(log w); it alone cannot distinguish a bad
proposal from a bad target implementation.

The local evaluator generates **two independent replicates, pooled for
ESS/k/maximum-weight summaries**. K is pooled K. Replicate diagnostics are also
saved; their average ESS is not pooled ESS. Fresh draws can yield very different
ESS at identical parameters when rare high weights dominate.

Pareto k and Evidence Stability
-------------------------------

The fitted generalized-Pareto shape diagnoses the upper tail of importance
ratios. The pipeline's flag is

.. math::

   bad_k=\mathbf1[\hat k>0.7\ \text{or nonfinite }\hat k].

For multiple objects this is a fraction, **not k itself**. Lower finite k is
more reassuring; this pipeline flags values above 0.7 and separately reports
nonfinite values. A flag passing once does not certify the posterior.

For independent replicate r with n draws:

.. math::

   \widehat{\log Z}_r=\operatorname{LSE}(\ell_{r,1:n})-\log n,\quad
   \Delta_Z=|\widehat{\log Z}_1-\widehat{\log Z}_2|.

``evidence_delta`` measures disagreement, not bias against truth. Two
replicates can agree while missing the same mode. Increasing K tests sampling
sensitivity but does not change the proposal distribution.

Residuals and Geometry
-----------------------

The local evaluator computes unweighted residuals under direct proposal draws:

.. math::

   r_{sb}=(f_b(x_s)-y_b)/\sigma_b,\quad
   RMS=\sqrt{\frac{1}{K|\mathcal B|}\sum_s\sum_{b\in\mathcal B}r_{sb}^2}.

This is not a posterior-mean SED residual or best-fit reduced chi-square.
It combines spread under q and fit; an accurate posterior need not give exactly
one. Read per-band residuals and predictive tails too. Smaller RMS alone may
reward an overconcentrated proposal.

Latent standard deviations/covariances refer to latent, not physical units.
The concentration audit's weighted coordinate shift is

.. math::

   d_j=(\sum_s\bar w_s x_{sj}-\bar x_j)/s_j.

At ESS near one, this mostly describes one particle, not a reliable posterior
mean displacement. Corner plots must retain paired joint coordinates. Population
aggregation is an object-equal mixture, never a distribution of medians:

.. math::

   \hat p_{agg}(x)=N^{-1}\sum_i\sum_s\bar w_{is}\delta(x-x_{is}).

For direct q use uniform weights. Selection-weighted prior and parent prior
are distinct populations; compare each against matching selection conditions.

PIT, Coverage and MIRA
----------------------

Independent truth pairs (x_i*, y_i) are evaluation-only, never used for observed
training or checkpoint selection. For a marginal coordinate:

.. math::

   u_i=F_{q_i}(x_i^*),\quad
   D_{KS}=\sup_u|\widehat F_U(u)-u|,\quad
   \widehat C(c)=N^{-1}\sum_i\mathbf1[x_i^*\in I_i(c)].

Calibrated PIT is uniform; interval coverage follows nominal mass c. Marginal
calibration does not imply joint calibration. Finite samples, ties and cohort
selection matter; weighted CDFs are only meaningful with adequate support.

**Implemented MIRA:** normalize coordinates using fixed truth min/max, choose
uniform random centers in that unit hypercube, then a random posterior draw
defines a ball radius. For S posterior draws, let n count draws strictly inside
(excluding the radius-defining draw) and I indicate whether truth is inside.
``mira_region_contributions`` in ``euclid_dsps/amortized/mira.py`` computes

.. math::

   A_{ir}=\frac{I_{ir}(n_{ir}+1)+(1-I_{ir})(S-n_{ir})}{S},\quad
   \widehat M=(NR)^{-1}\sum_{i,r}A_{ir}.

This includes the code's S/(S+1) finite-sample normalization. Inputs are
truth [objects, dimensions] and posterior [models, objects, samples, dimensions].
Quantiles/medians cannot reconstruct these inputs. Posterior coordinates are
not clipped to the truth range. Comparisons use fixed normalization and shared
region seeds. The matched continuous-distribution reference used here is

.. math::

   M_0=2/3,\qquad \sigma_0=\sqrt{1/(18N)}.

For N=512, sigma is 0.01042. This scale does not replace the archived bootstrap
interval, which resamples held-out objects plus a region draw. Region-only
Monte Carlo scatter is not galaxy-level uncertainty. More regions do not create
more independent galaxies. A score near 2/3 is not a universal certificate;
neither interpret it as a percentage of correct galaxies nor assume larger
always means better.

Reference: `Sharief et al., MIRA: A Score for Conditional Distribution Accuracy
and Model Comparison <https://arxiv.org/abs/2605.02014>`_. Exact finite-sample
definitions above follow the local code and archived manifest.

Gradient Coherence and Step Safety
-----------------------------------

For Adam displacement delta, compare derivatives on the same fixed batch:

.. math::

   d_{AD}=\nabla L(\phi)^T\delta,\quad
   d_{FD}(h)=[L(\phi+h\delta)-L(\phi-h\delta)]/(2h).

Agreement across stable step sizes supports local coherence. Tiny h may expose
rounding, large h may cross nonlinear regions. INCONCLUSIVE is not proof of
wrong AD. Float64 output does not establish float64 intermediate transport.

The corrected contract accepts a finite strictly decreasing step satisfying

.. math::

   L(\phi+\alpha\delta)\le L(\phi)+10^{-4}\alpha d_{AD},\quad d_{AD}<0,
   \quad\alpha\in\{1,1/2,\ldots,1/2048\}.

Rejection rolls back parameters and Adam state. This protects the fixed-batch
loss, not independent support/calibration. Report attempts, applied updates,
accepted scales and loss changes. Rejecting every update is not adaptation.
