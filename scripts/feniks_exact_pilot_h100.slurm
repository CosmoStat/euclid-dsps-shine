#!/bin/bash
#SBATCH --job-name=feniks_xpilot
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit as array 0-6}"
LABELS=(
  adj_official_t1 adj_jaxfli_t1 unadj_jaxfli_e1e3
  unadj_jaxfli_e5e4 adj_jaxfli_t4 adj_jaxfli_t8 adj_jaxfli_t16
)
SAMPLERS=(mclmc mclmc mclmc_unadjusted mclmc_unadjusted mclmc mclmc mclmc)
FRACTIONS=("0.1,0.1,0.1" "0.4,0.4,0.2" "0.4,0.4,0.2" "0.4,0.4,0.2" "0.4,0.4,0.2" "0.4,0.4,0.2" "0.4,0.4,0.2")
THINNINGS=(1 1 1 1 4 8 16)
ENERGIES=(0.0005 0.0005 0.001 0.0005 0.0005 0.0005 0.0005)
(( TASK_ID < 7 )) || exit 2

export SAMPLER="${SAMPLERS[$TASK_ID]}"
export SAMPLER_LABEL="${LABELS[$TASK_ID]}"
export FRAC_TUNE="${FRACTIONS[$TASK_ID]}"
export THINNING="${THINNINGS[$TASK_ID]}"
export DESIRED_ENERGY_VAR="${ENERGIES[$TASK_ID]}"
export MODE=pilot
export CHAINS=2
export N_GALAXIES=1
export MCLMC_TUNE="${MCLMC_TUNE:-500}"
export SAMPLE_CHUNKS="${SAMPLE_CHUNKS:-100,500}"

for chain in 0 1; do
  export SLURM_ARRAY_TASK_ID="$chain"
  source scripts/feniks_exact_chain_h100.slurm
done
