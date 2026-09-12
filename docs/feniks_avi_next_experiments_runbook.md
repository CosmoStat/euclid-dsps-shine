# FENIKS AVI expert-capacity and population-prior follow-up

## Questions and fixed contracts

This four-task array answers two independent questions using the completed
`avi_encoder_experiments_v6` run.

| Task | Arm | Trained component | Fixed component | Question |
| --- | --- | --- | --- | --- |
| 0 | `H2_experts` | two-expert encoder | learned source prior | are four experts already redundant? |
| 1 | `H8_experts` | eight-expert encoder | learned source prior | does additional mixture capacity improve support? |
| 2 | `P_latest_prior` | learned source prior | completed B encoder | can the population target improve from its current state? |
| 3 | `P_scratch_prior` | identity-initialized prior | completed B encoder | is the learned source prior a bad basin? |

The existing completed `B_experts` arm is the four-expert control and is not
retrained. Tasks 0/1 use exactly its 60-epoch sleep/sleep/wake curriculum,
catalogue, batch size, optimizer, defensive balance MIS and frozen physical
model. Tasks 2/3 each perform five full population sweeps. Every macro-update
uses 1,024 galaxies and K=256 defensive mixture draws per galaxy.

All posterior weights remain the exact full-15D ratio
`softmax(log p(y,x) - log r(x|y))`. The ten SFH coordinates remain in the
joint target. They are treated as nuisance coordinates only when ranking
scientific diagnostics: physical-5D MIRA and physical population closure are
primary, while 15D support remains a safety requirement.

Prior updates minimize the stopped weighted full-joint negative log prior plus
the differentiable `+log_alpha_eta` observed-r<29 selection normalization.
Selection never enters object-level normalized weights. No galaxy is removed
for low ESS; finite support, proposed KL and alpha Monte Carlo error are logged,
and a rejected trust-region update fails the task.

After all four tasks, a one-H100 job audits B/H2/H8. It records learned gate
probabilities, gate entropy, active/winning experts, exact mixture
responsibilities, pairwise expert-mean separation separately in physical 5D and
SFH 10D, K512 ESS, and raw physical-5D MIRA from 256 dense joint draws. Truth is
used only for this final synthetic MIRA diagnostic.

This is deliberately a prior-only first population test. Moving the prior also
moves the posterior target, so a successful arm still requires a subsequent
joint B/prior alternation and a fresh K4096 inference audit.

## Resources

- Four array tasks, one node and four H100s per task.
- Default concurrency four: peak 16 H100s on four nodes.
- Successive preflight and training waves; they do not overlap.
- One dependent audit job using one H100.
- Task time ceiling 20 hours; preflight ceiling 2 hours; audit ceiling 2 hours.

## Launch on Jean-Zay

```bash
cd "$WORK/dsps-popcosmos"
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine

BASE="/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111"
TRAINING="$BASE/avi_encoder_experiments_v6"
ROOT="$BASE/avi_expert_prior_followup_v1"

bash scripts/submit_feniks_avi_next_experiments.sh "$TRAINING" "$ROOT" 4
```

The launcher snapshots code and filter contents, validates all upstream hashes,
runs a real preflight for each arm, and holds training behind `afterok`.

## Monitor and resume

```bash
bash scripts/watch_feniks_avi_next_experiments.sh "$ROOT"
```

After reconnecting, redefine `ROOT` and run the same watcher. For raw logs:

```bash
tail -n 50 -F "$ROOT"/logs/*.out "$ROOT"/logs/*.err
```

Prior tasks checkpoint every 1,024-galaxy macro-update. Encoder tasks retain the
existing deterministic optimizer resume. After an interrupted allocation has
stopped, resume only unfinished tasks, for example:

```bash
bash scripts/submit_feniks_avi_next_experiments.sh resume "$ROOT" '2-3%2'
```

The original dependent audit will not run if the first training array fails.
After a successful resume, submit it explicitly from the frozen code snapshot:

```bash
CODE=$(cat "$ROOT/CODE_DIR")
export AVI_NEXT_ROOT="$ROOT" AVI_NEXT_CODE="$CODE"
sbatch --export=ALL \
  --output="$ROOT/logs/audit-%j.out" --error="$ROOT/logs/audit-%j.err" \
  "$CODE/scripts/feniks_avi_expert_audit.slurm"
```

Completion requires all per-arm `FINAL.json` files and
`EXPERT_AUDIT_COMPLETE.json`, not only empty `squeue` output.

Summarize the matched capacity and prior-training diagnostics with:

```bash
CODE=$(cat "$ROOT/CODE_DIR")
(cd "$CODE" && python -m scripts.feniks_avi_next_experiments summarize --root "$ROOT")
```
