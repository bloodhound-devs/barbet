#!/bin/bash
 
# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=gambit-lightning
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
 
# set your minimum acceptable walltime=days-hours:minutes:seconds
# SBATCH -t 7-0:00:00
#SBATCH -t 4-0:00:00
 
#SBATCH --mem-per-cpu=12G
 
# SBATCH -p cascade
#SBATCH -p gpu-a100-short,gpu-h100,gpu-a100
# SBATCH -p gpu-a100-short,gpu-a100,gpu-a100-preempt
# SBATCH -p gpu-a100-short,gpu-a100
# SBATCH -p gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --account=punim2199
 
# Specify your email address to be notified of progress.
#SBATCH --mail-user=robert.turnbull@unimelb.edu.au
#SBATCH --mail-type=ALL
 
# Load the environment variables
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
export PATH=/data/gpfs/projects/punim2199/poetry-env/bin:$PATH
 
export HF_HOME=/home/rturnbull/wytamma/huggingface

source /home/rturnbull/wytamma/rob/gambit-lightning/.venv/bin/activate


DOMAIN=bac120
#DOMAIN=ar53
ESM_LAYERS=6
FEATURES=2048
GROWTH=2
LAYERS=2
EMBEDDING=4
LR=0.0001

MEMMAP=/tmp/${DOMAIN}-esm${ESM_LAYERS}.np
if [[ ! -s "$MEMMAP" ]]; then
        cp /data/projects/punim2199/rob/release220/${DOMAIN}/esm${ESM_LAYERS}/esm${ESM_LAYERS}.np $MEMMAP
fi

SEQTREE_NAME=${DOMAIN}-esm${ESM_LAYERS}-reps-d4.st
#SEQTREE=/data/projects/punim2199/rob/release220/${DOMAIN}/esm${ESM_LAYERS}/esm${ESM_LAYERS}.st
SEQTREE=/data/projects/punim2199/rob/release220/${DOMAIN}/esm${ESM_LAYERS}/esm${ESM_LAYERS}-reps.st
#SEQTREE=/data/projects/punim2199/rob/release220/${DOMAIN}/esm${ESM_LAYERS}/${SEQTREE_NAME}


gambit-tools train  \
        --memmap  $MEMMAP \
        --memmap-index  /data/projects/punim2199/rob/release220/${DOMAIN}/esm${ESM_LAYERS}/esm${ESM_LAYERS}.txt \
        --seqtree  $SEQTREE \
        --features $FEATURES \
        --intermediate-layers $LAYERS \
        --growth-factor $GROWTH \
        --family-embedding-size $EMBEDDING \
        --max-epochs 20 \
        --max-learning-rate $LR \
        --run-name  ${SEQTREE_NAME}-b16f${FEATURES}l${LAYERS}g${GROWTH}e${EMBEDDING}lr${LR} \
        --num-workers 4 \
        --wandb-project   "Gambit" \
        --wandb-entity   "mariadelmarq-The University of Melbourne" \
        --wandb 

#        --profiler-path lightning-${DOMAIN}-esm${ESM_LAYERS}-b16f${FEATURES}l${LAYERS}g${GROWTH}e${EMBEDDING}lr${LR}-memmap-reps-d4-profile.txt \
#        --profile-memory \
