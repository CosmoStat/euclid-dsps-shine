# MDF precision qualification, 2026-09-09

## Evidence and decision

Full decoder job 1920226 completed, but all six merged points remain NOT_PASSED.
Nine failed per-band checks are in metallicity. Redshift has one unresolved
check; several weak dust/SFH checks are unresolved. Reverse/JVP and canonical
likelihood gradient identities pass. This localizes the remaining investigation;
it does not establish that autodiff is wrong or that float32 is the sole cause.

This experiment holds merged quadrature fixed and compares the historical MDF
arithmetic with `model.mdf_weight_precision: float64_v1`. The latter evaluates
the same DSPS compact triweight MDF, including its normalization and edge rules,
in double precision. It also promotes the induced SSP/survival contractions.
Stored assets, physical equations, scatter, physical bounds, input transforms,
downstream casts and likelihood are not replaced. This is NOT a full-float64
decoder and NOT a Gaussian-CDF substitution. The historical default is unchanged.

Small MDF probes export weights, analytic polynomial-reference derivatives and
20 finite-difference scales. The full target repeats all 15 coordinates at the
same six saved inputs and observation contexts, with the existing ten scales
and unchanged tolerances. The comparison rechecks every coordinate, not only
the nine failures. It does not evaluate either expensive NPE bank again.

One node, one H100, 16 CPUs, no array, maximum concurrency one GPU. Slurm ceiling
90 minutes (1.5 GPU-hours); internal ceiling 80 minutes and 6000 full-target
forward/gradient evaluations. Small MDF-only probes are separate from that
decoder count. Compilation is included in walltime. No optimization is started.

## Launch on Jean-Zay

Run after pulling the commit containing this file. No previous job needs to
be cancelled: 1920226 is complete. Do not edit its frozen worktree or artifacts.

```bash
cd "$WORK/dsps-popcosmos"
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark

source outputs/logs/feniks_sc_drws_balanced_npe_latest.env
export REPO_DIR="$PWD"
export CACHE_ROOT="$SCRATCH/feniks_sc_drws_runtime"
export LOCAL_VI_MDF_PRECISION_REFERENCE="$(dirname "$BALANCED_ROOT")/frozen_parent_full_decoder_qualification_v1"
export DIAGNOSTIC_ROOT="$(dirname "$BALANCED_ROOT")/frozen_parent_mdf_precision_qualification_v1"
unset DIAGNOSTIC_LOG_ROOT LOCAL_VI_GRADIENT_ISOLATION LOCAL_VI_REDSHIFT_DECOMPOSITION
unset LOCAL_VI_PHOTOMETRY_REFERENCE LOCAL_VI_FULL_DECODER_REFERENCE
unset LOCAL_VI_OBJECTS LOCAL_VI_STEPS LOCAL_VI_DRAWS

bash scripts/submit_feniks_sc_drws_local_vi_diagnostic.sh \
  outputs/logs/feniks_sc_drws_balanced_npe_latest.env
bash scripts/monitor_feniks_sc_drws_local_vi_diagnostic.sh
```

Preparation verifies the previous final receipt, its artifact hashes, original
source manifest and observed-row identity. Runtime checks exact saved point
identity, unchanged model parameters, and legacy cache/live flux parity. Old
flux banks are used only for this legacy check, never as candidate predictions.
Changed MDF numerics invalidate training-cache reuse even with the same prior.

After completion:

```bash
source outputs/logs/feniks_sc_drws_local_vi_diagnostic_latest.env
python scripts/summarize_feniks_sc_drws_decoder_qualification.py "$DIAGNOSTIC_ROOT"
```

`merged_mdf32` is the control; `merged_mdf64` is the candidate. Inspect
`MDF_WEIGHT_PROBES.json`, `mdf_weight_probes.csv`,
`FULL_DECODER_QUALIFICATION.json` and `decoder_qualification.csv`. Final receipt
hashes cover these files. Finite weight derivatives alone do not qualify SEDs.

## Decision after the run

- If weights match the polynomial reference but full-target checks remain
  unresolved, examine downstream casts, survival/mass normalization and the
  signed multiscale curves. Do not enlarge tolerances to force acceptance.
- If full numerical checks pass, review catalogue-simulator compatibility and
  the per-band forward change before any new experiment. A small value change
  does not imply small likelihood or gradient change.
- Only after that review: fresh versioned simulation banks, bounded local VI,
  then matched frozen-parent NPE with simulated and observed validation.
  Population learning remains blocked until posterior support also qualifies.

No posterior or population model is promoted by this diagnostic, even on PASS.
Real cluster assets and GPU behavior must be verified by the submitted job;
local synthetic and mock tests are software checks, not this science result.
