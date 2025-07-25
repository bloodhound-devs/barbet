#!/bin/bash
 
# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=barbet
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
 
# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH -t 7-0:00:00
# SBATCH -t 4-0:00:00
 
#SBATCH --mem-per-cpu=12G
 
# SBATCH -p cascade
#SBATCH -p gpu-h100
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

source /data/gpfs/projects/punim2199/rob/barbet/.venv/bin/activate

export PATH=/data/gpfs/projects/punim2199/poetry-env/bin:$PATH
export HF_HOME=/home/rturnbull/wytamma/huggingface

DOMAIN=bac120
#DOMAIN=ar53
ESM_LAYERS=12
#FEATURES=2048
FEATURES=768
GROWTH=2
LAYERS=2
EMBEDDING=4
LR=0.0001
EPOCHS=20
BATCH=16

PREPROCESSED_DIR=/data/gpfs/projects/punim2199/preprocessed

MEMMAP=/tmp/${DOMAIN}-esm${ESM_LAYERS}.npy
if [[ ! -s "$MEMMAP" ]]; then
        cp ${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/esm${ESM_LAYERS}.npy $MEMMAP
fi

TREEDICT_NAME=esm${ESM_LAYERS}-reps.st
#TREEDICT_NAME=esm${ESM_LAYERS}-reps-d4.st
TREEDICT=${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/${TREEDICT_NAME}


barbet-tools train  \
        --memmap  $MEMMAP \
        --memmap-index  ${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/esm${ESM_LAYERS}.txt \
        --treedict  $TREEDICT \
        --features $FEATURES \
        --intermediate-layers $LAYERS \
        --growth-factor $GROWTH \
        --family-embedding-size $EMBEDDING \
        --max-epochs $EPOCHS \
        --max-learning-rate $LR \
        --run-name  ${DOMAIN}-${TREEDICT_NAME}-b${BATCH}f${FEATURES}l${LAYERS}g${GROWTH}e${EMBEDDING}lr${LR}ep${EPOCHS} \
        --num-workers 4 \
        --wandb-entity   "mariadelmarq-The University of Melbourne" \
        --wandb 

#        --profiler-path lightning-${DOMAIN}-esm${ESM_LAYERS}-b16f${FEATURES}l${LAYERS}g${GROWTH}e${EMBEDDING}lr${LR}-memmap-reps-d4-profile.txt \
#        --profile-memory \
