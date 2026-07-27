# FENIKS exact-posterior benchmark

This benchmark keeps the validated `rws_k8_t2_seed2` encoder and learned prior
fixed. For the same seven synthetic galaxies, it compares:

1. the amortized conditional-flow encoder;
2. the same encoder corrected by self-normalized importance sampling and a
   Pareto-smoothed diagnostic weight;
3. multi-start posterior MAP;
4. four-chain NUTS;
5. four-chain adjusted MCLMC;
6. the catalog truth as a diagnostic point only.

Every method uses the same 15D `spline15d_checkpoint` normalization, the same
learned RealNVP prior in `latent_x`, the same DSPS decoder, and the same
Student-t likelihood with two degrees of freedom and no added error floor.
Truth is never used to initialize or tune a sampler.

## Fixed production quantities

The smoke test uses two galaxies:

- 256 encoder draws and 16 MAP starts with 20 Adam iterations;
- NUTS: 4 chains per galaxy, 10 warmup steps and 10 stored draws;
- adjusted MCLMC: 2 chains per galaxy, 10 tuning steps and 10 stored draws.

The production workflow first runs a bounded pilot:

- full preparation of two galaxies: 32,768 encoder draws and 16 MAP starts;
- NUTS reference: 4 chains per galaxy, target acceptance 0.65, 500 warmup
  steps and 600 stored draws split into resumable chunks of 100 and 500;
- one-galaxy MCLMC grid: seven settings, two chains per setting, 500 tuning
  steps and 600 stored draws;
- the grid varies the three BlackJAX tuning fractions, actual kernel thinning
  (`1, 4, 8, 16`), and two unadjusted energy targets used for diagnostics only;
- adjusted MCLMC is selected by R-hat, agreement with NUTS, then ESS per
  integrator step;
- the selected setting must pass R-hat <= 1.10 on the second galaxy.

After the pilot gate, production runs seven galaxies:

- 32,768 encoder draws and at most 8,192 resampled IS draws per galaxy;
- 16 posterior-MAP starts with 2,000 Adam iterations;
- NUTS: 4 chains, 500 warmup steps, 1,600 stored draws per chain;
- adjusted MCLMC: 4 chains, 500 tuning steps, 1,600 stored draws per chain,
  with the selected thinning applied inside the transition kernel.

The pilot NUTS chains and the two second-galaxy MCLMC chains are prefixes of
the production chains. Production resumes them and adds the final 1,000-draw
chunk rather than discarding pilot compute.

## Outputs

Each galaxy directory contains all numerical inputs needed to restyle plots:

- `encoder_samples.parquet`;
- `importance_weighted_samples.parquet` and
  `importance_resampled_samples.parquet`;
- `map_solutions.parquet` and `map_trace.parquet`;
- resumable NUTS/MCLMC chain chunks, tuning parameters, states and manifests;
- physical `samples.parquet` plus R-hat, bulk ESS and tail ESS tables;
- `truth.parquet`, `observation.parquet`, `photometric_predictions.parquet`;
- `sed_draws.npz`;
- 5D and 15D corner plots in PNG/PDF;
- SED/photometry comparison and sampler convergence plots.

The run root additionally contains the posterior-agreement table and heatmap,
photometric-fit table and plot, sampler scoreboard, and a completion marker.

## Jean-Zay execution

Run from the repository checkout with the `shine` environment active. The
canonical model must be present at:

```text
outputs/runs/feniks_selfsup_paper_v1/rws_k8_t2_seed2
```

Submit the smoke:

```bash
bash scripts/submit_feniks_exact_posterior_smoke.sh
```

After its aggregate job completes and its root contains `DONE`, submit the
full dependency graph:

```bash
export SMOKE_ROOT="outputs/runs/feniks_exact_posterior_smoke_YYYYMMDD_HHMMSS"
bash scripts/submit_feniks_exact_posterior_full.sh
```

The scripts refuse an existing full output root. Every sampling chain is
chunked and resumable, but a recovery submission should keep the same
checkpoint, normalization, pilot selection and existing chunk sizes.

## Compute accounting

The smoke requests at most 8.33 H100-hours and peaks at 12 concurrent H100s.
The complete graph has a deliberately conservative Slurm allocation upper
bound of 825 H100-hours and peaks at 16 concurrent H100s. This is not the
expected consumption: completed pilot chunks are reused, and most jobs should
finish before their time limit.

After a run, compute the actual allocation from Slurm elapsed times:

```bash
sacct -X -j "$JOB_IDS" --noheader --parsable2 \
  --format=ElapsedRaw,AllocTRES,State
```

The smoke and pilot wall times are the evidence used to replace the upper bound
with a defensible projected and actual H100-hour figure.
