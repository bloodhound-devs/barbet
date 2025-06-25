#!/bin/bash
 
# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=predict
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

 
# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH -t 4:00:00
 
#SBATCH --mem-per-cpu=12G
 
# SBATCH -p cascade
#SBATCH -p gpu-a100,gpu-h100,gpu-a100-short
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


CHECKPOINT_BASE=/data/gpfs/projects/punim2199/rob/barbet/logs/bac120-stack-esm6.td-b4f768l2g2lr0.0002ep70a512s32prod-r226/version_0/checkpoints/weights-epoch\=69-valid_loss\=0.29.ckpt
CHECKPOINT_LARGE=/data/gpfs/projects/punim2199/rob/barbet/logs/bac120-stack-esm6.td-b4f1536l2g2lr0.0002ep70a512s32prod-r226/version_0/checkpoints/weights-epoch\=69-valid_loss\=0.17.ckpt
CHECKPOINT=$CHECKPOINT_BASE


DOMAIN=bac120
#DOMAIN=ar53

ESM_LAYERS=6
PREPROCESSED_DIR=/data/gpfs/projects/punim2199/preprocessed-r226
MEMMAP=${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/esm${ESM_LAYERS}.npy
TREEDICT_NAME=esm${ESM_LAYERS}.td
TREEDICT=${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/${TREEDICT_NAME}

barbet-tools predict-memmap \
        --checkpoint $CHECKPOINT \
        --memmap  $MEMMAP \
        --memmap-index  ${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/esm${ESM_LAYERS}.txt \
        --treedict $TREEDICT \
        --treedict-partition 0 \
        --no-greedy-only \
        --output-csv validation-base-${DOMAIN}-esm${ESM_LAYERS}.csv



DOMAIN=ar53
ESM_LAYERS=6
PREPROCESSED_DIR=/data/gpfs/projects/punim2199/preprocessed-r226
MEMMAP=${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/esm${ESM_LAYERS}.npy
TREEDICT_NAME=esm${ESM_LAYERS}.td
TREEDICT=${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/${TREEDICT_NAME}

CHECKPOINT=/data/gpfs/projects/punim2199/rob/bloodhound-stack/logs/ar53-stack-esm6.st-b4f32l2g2lr0.0002ep70a512s32-r226/version_0/checkpoints/weights-epoch=43-valid_loss=5.8.ckpt
barbet-tools predict-memmap \
        --checkpoint $CHECKPOINT \
        --memmap  $MEMMAP \
        --memmap-index  ${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/esm${ESM_LAYERS}.txt \
        --treedict $TREEDICT \
        --treedict-partition 0 \
        --output-csv validation-base-${DOMAIN}-esm${ESM_LAYERS}.csv

ESM_LAYERS=12
PREPROCESSED_DIR=/data/gpfs/projects/punim2199/preprocessed-r226
MEMMAP=${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/esm${ESM_LAYERS}.npy
TREEDICT_NAME=esm${ESM_LAYERS}.td
TREEDICT=${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/${TREEDICT_NAME}

CHECKPOINT=/data/gpfs/projects/punim2199/rob/bloodhound-stack/logs/ar53-stack-esm12.st-b4f768l2g2lr0.0002ep70a512s32-r226/version_0/checkpoints/weights-epoch\=39-valid_loss\=3.6.ckpt
barbet-tools predict-memmap \
        --checkpoint $CHECKPOINT \
        --memmap  $MEMMAP \
        --memmap-index  ${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/esm${ESM_LAYERS}.txt \
        --treedict $TREEDICT \
        --treedict-partition 0 \
        --output-csv validation-base-${DOMAIN}-esm${ESM_LAYERS}.csv


