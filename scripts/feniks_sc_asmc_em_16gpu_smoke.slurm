#!/bin/bash
#SBATCH --job-name=feniks_sc_smoke16
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=03:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-3%4
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail

SMOKE4_ROOT="${SMOKE4_ROOT:?Set SMOKE4_ROOT to the completed four-H100 smoke}"
SMOKE16_ROOT="${SMOKE16_ROOT:?Set SMOKE16_ROOT to a new immutable directory}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?This launcher must run as a Slurm array}"

export RUN_ROOT="$SMOKE16_ROOT/shard_$TASK_ID"
export SMOKE_ROW_OFFSET="$((TASK_ID * 8))"
export SMOKE4_ROOT

exec bash scripts/feniks_sc_asmc_em_4gpu_smoke.slurm
