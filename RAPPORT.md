# Rapport d'implementation collapse-fix

Date: 2026-06-17

## Objectif

Le but reste l'inference photometrie-only avec redshift libre:

- entrainer encodeur + decodeur DSPS + prior NF sans supervision par truth;
- apprendre un prior de population degenere entre les parametres;
- utiliser ensuite MAP-Adam sous le prior appris pour estimer les parametres,
  notamment `z_obs`;
- utiliser les colonnes truth du dataset Diffsky seulement comme diagnostics de
  closure, jamais comme loss supervisee ni comme redshift fixe.

## Plan suivi

Le plan complet est dans `PLAN_FIX.md`. Les phases A a E ont ete implementees:

- A: diagnostics truth/proxy et plots etendus.
- B: closure du vrai decoder amortized `popcosmos_bins`.
- C: MAP-Adam multistart independant de l'encoder.
- D: configs lowdim et full-bounds pour tester la degenerescence.
- E: gates automatiques de collapse pour train/inference/MAP.

## Changements implementes

### Diagnostics truth/proxy et plots

Nouveau module: `euclid_dsps/amortized/truth_diagnostics.py`.

Les runs d'inference et MAP peuvent maintenant produire:

- `posterior_vs_truth_extended.csv`
- `posterior_vs_truth_extended_objects.parquet`
- `map_vs_truth_extended.csv`
- `map_vs_truth_extended_objects.parquet`
- `prior_vs_truth_population.csv`
- `extended_truth_diagnostics_summary.json`
- plots truth vs posterior/MAP pour `z_obs`, masse, `tau2` proxy,
  `dust_index_n` proxy;
- plots de biais vs truth;
- overlays de populations truth/posterior/MAP/prior;
- `corner_truth_prior_posterior_map.png`.

Les snapshots truth incluent maintenant plus de colonnes Diffsky/Diffstar/Diffmah
quand elles existent.

### Closure du decoder amortized

Nouveau CLI:

```bash
python -m euclid_dsps.cli --config configs/experiments/diffsky_hltds_popcosmos_proxy_truth_closure_h100.yaml \
  diffsky-popcosmos-proxy-closure ...
```

Nouveau Slurm:

```bash
scripts/diffsky_popcosmos_proxy_closure_h100.slurm
```

Cette closure utilise le meme decoder que l'amortized (`popcosmos_bins`) et mappe
les truth disponibles vers des proxies:

- `redshift_true -> z_obs`
- `logsm_true -> log10_stellar_mass`
- `logssfr_true` ou `logsfr_true - logsm_true -> dlog10_sfr_*`
- `dust_av / 1.086 -> tau2`
- `dust_delta -> dust_index_n`

Elle ecrit les residus par bande, un rapport markdown, un `closure_gate.json` et
des plots residus vs `redshift_true`, `logsm_true`, `dust_av`, `dust_delta`.

### MAP-Adam multistart

`diffsky-map-adam-prior` supporte maintenant:

- `--start-mode encoder`
- `--start-mode prior`
- `--start-mode z_grid`
- `--start-mode lowz_grid`
- `--start-mode latin_hypercube`
- `--start-mode mixed`

Le Slurm `scripts/diffsky_map_adam_prior_h100.slurm` expose:

- `START_MODE`
- `PRIOR_WEIGHT`
- `N_STARTS`
- `START_CHUNK_SIZE`
- `MAXITER`

Nouveaux outputs MAP:

- `map_estimates_by_start.parquet`
- `map_estimates_by_start.csv`
- `map_start_family_summary.csv`
- `map_best_by_start_family.csv`
- `map_chi2_vs_start_z.png`
- `map_selected_z_by_start_family.png`

Cela permet de separer:

- likelihood-only (`PRIOR_WEIGHT=0.0`);
- prior learned (`PRIOR_WEIGHT>0`);
- depart encoder vs depart grille redshift vs depart prior.

Le MAP optimise maintenant les starts par chunks (`START_CHUNK_SIZE`, defaut 1)
pour eviter les OOM JAX du type `[N_STARTS, BATCH_SIZE, n_wave]`.

### Configs de validation

Nouvelles configs:

- `configs/experiments/diffsky_hltds_popcosmos_proxy_truth_closure_h100.yaml`
- `configs/experiments/diffsky_hltds_popcosmos_full_bounds_h100.yaml`
- `configs/experiments/diffsky_hltds_popcosmos_lowdim_z_mass_dust_h100.yaml`
- `configs/experiments/diffsky_hltds_popcosmos_lowdim_z_mass_dust_lowz_h100.yaml`

La config lowdim remplace explicitement les parametres libres par:

- `z_obs`
- `log10_stellar_mass`
- `dlog10_sfr_1`
- `tau2`
- `dust_index_n`

et met `amortized.encoder.latent_dim: 5`.

### Gates de collapse

Nouveau module: `euclid_dsps/amortized/collapse_gates.py`.

Les runs ecrivent maintenant:

- `training_collapse_gate.json` apres training;
- `collapse_gate.json` apres inference/MAP.

Le gate inference/MAP lit les metriques photo-z, les diagnostics truth/proxy et
les overlays prior-vs-truth quand disponibles. Sur vraies donnees sans truth,
ce gate reste un diagnostic de runtime/self-consistency, pas une validation
science.

## Validations effectuees

Commandes passees:

```bash
uv run ruff format euclid_dsps/amortized/catalog_identity.py euclid_dsps/amortized/collapse_gates.py euclid_dsps/amortized/infer.py euclid_dsps/amortized/map_adam.py euclid_dsps/amortized/train.py euclid_dsps/amortized/truth_diagnostics.py euclid_dsps/cli.py euclid_dsps/diffsky_forward_closure.py
uv run python -m compileall euclid_dsps scripts
uv run ruff check euclid_dsps tests
uv run pytest tests/test_cli.py tests/test_diffsky_redshift_ablation.py tests/test_full_validation_alpha_report.py
bash -n scripts/diffsky_amortized_train_h100.slurm scripts/diffsky_amortized_infer_h100.slurm scripts/diffsky_map_adam_prior_h100.slurm scripts/diffsky_popcosmos_proxy_closure_h100.slurm
git diff --check
```

Resultats:

- compileall: OK
- Ruff: OK
- pytest cible: 11 passed
- Slurm syntax: OK
- config-load smoke: OK, lowdim charge bien 5 parametres libres
- smoke local proxy-closure CPU sur 2 objets: OK

Smoke local execute:

```bash
JAX_PLATFORMS=cpu uv run python - <<'PY'
from euclid_dsps.config import load_config
from euclid_dsps.diffsky_forward_closure import run_popcosmos_proxy_truth_closure

cfg = load_config('configs/experiments/diffsky_hltds_popcosmos_proxy_truth_closure_h100.yaml')
report = run_popcosmos_proxy_truth_closure(
    cfg,
    dataset_path='Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_noerr.parquet',
    out_dir='outputs/runs/dev_popcosmos_proxy_closure_smoke',
    limit=2,
    batch_size=2,
)
print(report)
PY
```

## Commandes Jean-Zay recommandees

Depuis Jean-Zay:

```bash
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
git pull
```

### 1. Closure du decoder PopCosmos proxy

```bash
RUN_NAME=diffsky_popcosmos_proxy_truth_closure_1k \
LIMIT=1000 \
BATCH_SIZE=128 \
sbatch scripts/diffsky_popcosmos_proxy_closure_h100.slurm
```

A inspecter:

- `outputs/runs/diffsky_popcosmos_proxy_truth_closure_1k/popcosmos_proxy_closure_report.md`
- `popcosmos_proxy_closure_residuals_by_band.csv`
- `popcosmos_proxy_residual_by_band.png`
- `popcosmos_proxy_residual_vs_redshift_true.png`
- `closure_gate.json`

Si cette closure est mauvaise par couleur/bande, ne pas scaler le NF: il faut
d'abord corriger la parameterisation/decoder.

### 2. MAP likelihood-only sur un checkpoint deja entraine

Remplacer `TRAIN_RUN` par le run existant a diagnostiquer.

```bash
TRAIN_RUN=outputs/runs/diffsky_realnvp_unsup_stable_smoke_5k_e5 \
RUN_NAME=diffsky_map_no_prior_zgrid_1k \
LIMIT=1000 \
BATCH_SIZE=64 \
START_MODE=z_grid \
N_STARTS=16 \
START_CHUNK_SIZE=1 \
MAXITER=220 \
PRIOR_WEIGHT=0.0 \
sbatch scripts/diffsky_map_adam_prior_h100.slurm
```

A inspecter:

- `map_closure_photoz_metrics.csv`
- `map_vs_truth_extended.csv`
- `map_start_family_summary.csv`
- `map_best_by_start_family.csv`
- `map_chi2_vs_start_z.png`
- `closure_photoz_truth_vs_map.png`
- `collapse_gate.json`

Interpretation:

- si `PRIOR_WEIGHT=0.0` ne retrouve pas `z`, le probleme est decoder,
  likelihood, bounds ou degenerescence physique;
- si likelihood-only marche mais le prior-weighted echoue, le prior appris est
  trop restrictif ou deja collapse.

### 3. MAP avec prior appris, meme checkpoint

```bash
TRAIN_RUN=outputs/runs/diffsky_realnvp_unsup_stable_smoke_5k_e5 \
RUN_NAME=diffsky_map_with_prior_zgrid_1k \
LIMIT=1000 \
BATCH_SIZE=64 \
START_MODE=z_grid \
N_STARTS=16 \
START_CHUNK_SIZE=1 \
MAXITER=220 \
PRIOR_WEIGHT=0.02 \
sbatch scripts/diffsky_map_adam_prior_h100.slurm
```

Comparer au run `PRIOR_WEIGHT=0.0`.

### 4. Training lowdim controle

```bash
CONFIG=configs/experiments/diffsky_hltds_popcosmos_lowdim_z_mass_dust_h100.yaml \
RUN_NAME=diffsky_lowdim_z_mass_dust_5k_e10 \
LIMIT=5000 \
EPOCHS=10 \
BATCH_SIZE=128 \
JAX_BATCH_SIZE=128 \
N_SAMPLES=4 \
sbatch scripts/diffsky_amortized_train_h100.slurm
```

Puis inference lowdim:

```bash
CONFIG=configs/experiments/diffsky_hltds_popcosmos_lowdim_z_mass_dust_h100.yaml \
TRAIN_RUN=outputs/runs/diffsky_lowdim_z_mass_dust_5k_e10 \
RUN_NAME=diffsky_lowdim_z_mass_dust_5k_e10_infer1k \
LIMIT=1000 \
POSTERIOR_SAMPLES=128 \
sbatch scripts/diffsky_amortized_infer_h100.slurm
```

Puis MAP lowdim likelihood-only:

```bash
CONFIG=configs/experiments/diffsky_hltds_popcosmos_lowdim_z_mass_dust_h100.yaml \
TRAIN_RUN=outputs/runs/diffsky_lowdim_z_mass_dust_5k_e10 \
RUN_NAME=diffsky_lowdim_map_no_prior_zgrid_1k \
LIMIT=1000 \
BATCH_SIZE=64 \
START_MODE=z_grid \
N_STARTS=16 \
START_CHUNK_SIZE=1 \
MAXITER=220 \
PRIOR_WEIGHT=0.0 \
sbatch scripts/diffsky_map_adam_prior_h100.slurm
```

### 5. Full 9D avec bounds redshift elargis

A lancer seulement si les etapes 1-4 sont interpretables.

```bash
CONFIG=configs/experiments/diffsky_hltds_popcosmos_full_bounds_h100.yaml \
RUN_NAME=diffsky_full_bounds_realnvp_10k_e15 \
LIMIT=10000 \
EPOCHS=15 \
BATCH_SIZE=128 \
JAX_BATCH_SIZE=128 \
N_SAMPLES=4 \
sbatch scripts/diffsky_amortized_train_h100.slurm
```

Inference diagnostic:

```bash
CONFIG=configs/experiments/diffsky_hltds_popcosmos_full_bounds_h100.yaml \
TRAIN_RUN=outputs/runs/diffsky_full_bounds_realnvp_10k_e15 \
RUN_NAME=diffsky_full_bounds_realnvp_10k_e15_infer5k \
LIMIT=5000 \
POSTERIOR_SAMPLES=128 \
sbatch scripts/diffsky_amortized_infer_h100.slurm
```

MAP final diagnostic:

```bash
CONFIG=configs/experiments/diffsky_hltds_popcosmos_full_bounds_h100.yaml \
TRAIN_RUN=outputs/runs/diffsky_full_bounds_realnvp_10k_e15 \
RUN_NAME=diffsky_full_bounds_map_mixed_prior_2k \
LIMIT=2000 \
BATCH_SIZE=64 \
START_MODE=mixed \
N_STARTS=16 \
START_CHUNK_SIZE=1 \
MAXITER=220 \
PRIOR_WEIGHT=0.02 \
sbatch scripts/diffsky_map_adam_prior_h100.slurm
```

## Rapatrier les resultats

Exemple pour un run:

```bash
RUN=diffsky_popcosmos_proxy_truth_closure_1k
rsync -avh --progress \
  -e "ssh -J mr287471@hubble.extra.cea.fr" \
  urx63nr@jean-zay.idris.fr:/lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine/outputs/runs/${RUN}/ \
  /home/maxime/src/DSPS/outputs/runs/${RUN}/
```

Rapatrier aussi les logs:

```bash
rsync -avh --progress \
  -e "ssh -J mr287471@hubble.extra.cea.fr" \
  urx63nr@jean-zay.idris.fr:/lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine/outputs/logs/ \
  /home/maxime/src/DSPS/outputs/logs/
```

## Commandes locales utiles

Finaliser une inference shardee rapatriee:

```bash
uv run python -m euclid_dsps.cli \
  --config configs/experiments/diffsky_hltds_popcosmos_full_bounds_h100.yaml \
  amortized-finalize-inference \
  --out outputs/runs/diffsky_full_bounds_realnvp_10k_e15_infer5k \
  --limit 5000
```

Relancer une closure proxy locale petite sur CPU:

```bash
JAX_PLATFORMS=cpu uv run python - <<'PY'
from euclid_dsps.config import load_config
from euclid_dsps.diffsky_forward_closure import run_popcosmos_proxy_truth_closure

cfg = load_config('configs/experiments/diffsky_hltds_popcosmos_proxy_truth_closure_h100.yaml')
run_popcosmos_proxy_truth_closure(
    cfg,
    dataset_path='Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_noerr.parquet',
    out_dir='outputs/runs/dev_popcosmos_proxy_closure_local',
    limit=32,
    batch_size=8,
)
PY
```

Verifier rapidement un run:

```bash
python - <<'PY'
import json
from pathlib import Path
run = Path('outputs/runs/diffsky_map_no_prior_zgrid_1k')
for name in ['map_summary.json', 'collapse_gate.json']:
    path = run / name
    if path.exists():
        print(name, json.loads(path.read_text()).get('status') or json.loads(path.read_text()).get('collapse_gate'))
print('plots:', sorted(p.name for p in run.glob('*.png'))[:20])
PY
```

## Decision apres les prochains runs

1. Si proxy closure echoue: corriger decoder/parameterisation avant tout NF.
2. Si proxy closure passe mais MAP likelihood-only echoue: reduire dimensions,
   revoir bounds/likelihood/error model et degenerescences.
3. Si MAP likelihood-only passe mais prior-weighted echoue: le NF apprend une
   population collapsee; augmenter freeze prior, entropy floor, ou entrainer le
   prior apres posterior agg plus sain.
4. Si lowdim passe mais full 9D echoue: le probleme est la degenerescence des
   nuisances; garder la strategie lowdim puis reintroduire les parametres un par
   un.
5. Si full 9D passe sur 5k/10k: scaler training puis inference/MAP sharde.
