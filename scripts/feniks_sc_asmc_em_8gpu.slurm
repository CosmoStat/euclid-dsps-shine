#!/bin/bash
#SBATCH --job-name=feniks_sc_em8
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-1%2
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail
export SHARD_COUNT=2
exec bash scripts/feniks_sc_asmc_em_array_worker.sh
