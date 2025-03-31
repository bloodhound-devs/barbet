#!/bin/bash
 
# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=bh-bac120
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
 
# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH -t 7-0:00:00
# SBATCH -t 4-0:00:00
 
#SBATCH --mem-per-cpu=64G
 
# SBATCH -p cascade
#SBATCH -p gpu-h100,gpu-a100,gpu-a100-short
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

source /data/gpfs/projects/punim2199/rob/bloodhound-stack/.venv/bin/activate

export PATH=/data/gpfs/projects/punim2199/poetry-env/bin:$PATH
export HF_HOME=/home/rturnbull/wytamma/huggingface

DOMAIN=bac120
#DOMAIN=ar53
ESM_LAYERS=33
#ESM_LAYERS=6
#FEATURES=2048
#FEATURES=768
FEATURES=1536
#FEATURES=256
ATTENTION=512
SEQ_COUNT=32

GROWTH=2
LAYERS=2
EMBEDDING=4
LR=0.0002
EPOCHS=70
BATCH=4


PREPROCESSED_DIR=/data/gpfs/projects/punim2199/preprocessed
MEMMAP=${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/esm${ESM_LAYERS}.npy

MEMMAP=/tmp/${DOMAIN}-esm${ESM_LAYERS}.npy
if [[ ! -s "$MEMMAP" ]]; then
        cp ${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/esm${ESM_LAYERS}.npy $MEMMAP
fi

#SEQTREE_NAME=esm${ESM_LAYERS}-reps.st
SEQTREE_NAME=esm${ESM_LAYERS}.st
#SEQTREE_NAME=esm${ESM_LAYERS}-reps.st
SEQTREE=${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/${SEQTREE_NAME}


bloodhound-tools train  \
        --memmap  $MEMMAP \
        --memmap-index  ${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/esm${ESM_LAYERS}.txt \
        --seqtree  $SEQTREE \
        --features $FEATURES \
        --intermediate-layers $LAYERS \
        --growth-factor $GROWTH \
        --max-epochs $EPOCHS \
        --attention-size $ATTENTION \
        --seq-count $SEQ_COUNT \
        --batch-size $BATCH \
        --max-learning-rate $LR \
        --embedding-model ESM${ESM_LAYERS} \
        --run-name  ${DOMAIN}-stack-${SEQTREE_NAME}-b${BATCH}f${FEATURES}l${LAYERS}g${GROWTH}lr${LR}ep${EPOCHS}a${ATTENTION}s${SEQ_COUNT} \
        --num-workers 4 \
        --train-all \
        --wandb-entity   "mariadelmarq-The University of Melbourne" \
        --wandb 
