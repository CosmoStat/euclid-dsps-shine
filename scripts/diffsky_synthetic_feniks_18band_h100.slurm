#!/bin/bash
#SBATCH --job-name=feniks18
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail

# LSST+Euclid+Roman FENIKS/DSPS closure generation. This wrapper reuses the
# standard FENIKS H100 runner but points it at the 18-band survey-like config.

export CONFIG="${CONFIG:-configs/diffsky_synthetic_feniks_260617_50k_survey_like_18band.yaml}"
export VALIDATION_CONFIG="${VALIDATION_CONFIG:-configs/diffsky_synthetic_feniks_260617_trueparam_closure_18band.yaml}"
export DATASET_DIR="${DATASET_DIR:-Data/diffsky/synthetic/feniks_260617_dsps_closure_18band}"
export LOG_DIR="${LOG_DIR:-outputs/logs}"
export LOG_FILE="${LOG_FILE:-${LOG_DIR}/diffsky_synthetic_feniks_18band_${STAGE:-generate}_${SLURM_JOB_ID:-manual}.log}"

exec bash scripts/diffsky_synthetic_feniks_50k_h100.slurm
