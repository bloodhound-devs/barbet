#!/bin/bash
 
# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=barbet-lightning
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
 
# set your minimum acceptable walltime=days-hours:minutes:seconds
# SBATCH -t 7-0:00:00
#SBATCH -t 4:00:00
 
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

source /home/rturnbull/wytamma/rob/barbet-lightning/.venv/bin/activate

DOMAIN=bac120
#DOMAIN=ar53
ESM_LAYERS=6

SEQTREE=/data/projects/punim2199/rob/release220/${DOMAIN}/esm${ESM_LAYERS}/esm${ESM_LAYERS}.st
REPRESENTATIVES=/home/rturnbull/wytamma/barbet_data/release220-partial/${DOMAIN}_marker_genes_reps_r220.tar.gz
SEQTREE_REPS=/data/projects/punim2199/rob/release220/${DOMAIN}/esm${ESM_LAYERS}/esm${ESM_LAYERS}-reps.st

python seqtree-representatives.py $SEQTREE $REPRESENTATIVES $SEQTREE_REPS

