#!/bin/bash
#SBATCH --job-name=feniks_sc_em16
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-3%4
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail
: "${SMOKE16_ROOT:?Set SMOKE16_ROOT to the completed 16-H100 smoke}"
export REQUIRE_SMOKE16=1
export SHARD_COUNT=4
exec bash scripts/feniks_sc_asmc_em_array_worker.sh
