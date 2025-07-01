#!/bin/bash
 
# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=barbet-validate
 
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


MAX_REPEATS=4
PREPROCESSED_DIR=/data/gpfs/projects/punim2199/preprocessed-r226
DOMAIN=bac120



ESM_LAYERS=6
FEATURES=768
MEMMAP=${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/esm${ESM_LAYERS}.npy
TREEDICT_NAME=esm${ESM_LAYERS}.td
TREEDICT=${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/${TREEDICT_NAME}

RUN_NAME=${DOMAIN}-stack-esm${ESM_LAYERS}.td-b4f${FEATURES}l2g2lr0.0002ep70a512s32val0-r226sp
VERSION=$(ls -r1 outputs/$RUN_NAME/lightning_logs)
CHECKPOINT=$(ls -1 outputs/$RUN_NAME/lightning_logs/$VERSION/checkpoints/weights* )

for REPEATS in $(seq 1 $MAX_REPEATS); do    
    OUTPUT=validation/$DOMAIN/validation-${DOMAIN}-esm${ESM_LAYERS}-f${FEATURES}-rep${REPEATS}.csv
    if [ ! -s "$OUTPUT" ]; then
        barbet-tools predict-memmap \
                --checkpoint $CHECKPOINT \
                --memmap  $MEMMAP \
                --memmap-index  ${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/esm${ESM_LAYERS}.txt \
                --treedict $TREEDICT \
                --treedict-partition 0 \
                --repeats $REPEATS \
                --num-workers 4 \
                --output-csv $OUTPUT
    fi
done



ESM_LAYERS=12
FEATURES=768
MEMMAP=${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/esm${ESM_LAYERS}.npy
TREEDICT_NAME=esm${ESM_LAYERS}.td
TREEDICT=${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/${TREEDICT_NAME}

RUN_NAME=${DOMAIN}-stack-esm${ESM_LAYERS}.td-b4f${FEATURES}l2g2lr0.0002ep70a512s32val0-r226sp
VERSION=$(ls -r1 outputs/$RUN_NAME/lightning_logs)
CHECKPOINT=$(ls -1 outputs/$RUN_NAME/lightning_logs/$VERSION/checkpoints/weights* )

for REPEATS in $(seq 1 $MAX_REPEATS); do    
    OUTPUT=validation/$DOMAIN/validation-${DOMAIN}-esm${ESM_LAYERS}-f${FEATURES}-rep${REPEATS}.csv
    if [ ! -s "$OUTPUT" ]; then
        barbet-tools predict-memmap \
                --checkpoint $CHECKPOINT \
                --memmap  $MEMMAP \
                --memmap-index  ${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/esm${ESM_LAYERS}.txt \
                --treedict $TREEDICT \
                --treedict-partition 0 \
                --repeats $REPEATS \
                --num-workers 4 \
                --output-csv $OUTPUT
    fi
done

