#!/bin/bash
 
# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=bh-ar53
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
 
# set your minimum acceptable walltime=days-hours:minutes:seconds
# SBATCH -t 7-0:00:00
#SBATCH -t 1-0:00:00
 
#SBATCH --mem-per-cpu=12G
 
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
 
module purge
module load GCCcore/11.3.0
module load Python/3.10.4

source /data/gpfs/projects/punim2199/rob/barbet/.venv/bin/activate

export PATH=/data/gpfs/projects/punim2199/poetry-env/bin:$PATH
export HF_HOME=/home/rturnbull/wytamma/huggingface

#DOMAIN=bac120
DOMAIN=ar53
ESM_LAYERS=6
#ESM_LAYERS=12
#ESM_LAYERS=30

FEATURES=768
#FEATURES=1536
#FEATURES=256
ATTENTION=512
STACK_SIZE=32

GROWTH=2
LAYERS=2
EMBEDDING=4
LR=0.0002
EPOCHS=70
BATCH=4


PREPROCESSED_DIR=/data/gpfs/projects/punim2199/preprocessed-r226
MEMMAP=${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/esm${ESM_LAYERS}.npy

MEMMAP=/tmp/${DOMAIN}-esm${ESM_LAYERS}.npy
if [[ ! -s "$MEMMAP" ]]; then
        cp ${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/esm${ESM_LAYERS}.npy $MEMMAP
fi

TREEDICT_NAME=esm${ESM_LAYERS}.td
#TREEDICT_NAME=esm${ESM_LAYERS}-reps.td
TREEDICT=${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/${TREEDICT_NAME}

RUN_NAME=${DOMAIN}-stack-${TREEDICT_NAME}-b${BATCH}f${FEATURES}l${LAYERS}g${GROWTH}lr${LR}ep${EPOCHS}a${ATTENTION}s${STACK_SIZE}val0-r226sp

barbet-tools train  \
        --memmap  $MEMMAP \
        --memmap-index  ${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/esm${ESM_LAYERS}.txt \
        --treedict  $TREEDICT \
        --features $FEATURES \
        --intermediate-layers $LAYERS \
        --growth-factor $GROWTH \
        --max-epochs $EPOCHS \
        --attention-size $ATTENTION \
        --stack-size $STACK_SIZE \
        --batch-size $BATCH \
        --max-learning-rate $LR \
        --embedding-model ESM${ESM_LAYERS} \
        --output-dir outputs/$RUN_NAME \
        --run-name  $RUN_NAME \
        --num-workers 4 \
        --project-name Barbet-${DOMAIN} \
        --wandb-entity   "mariadelmarq-The University of Melbourne" \
        --wandb 
