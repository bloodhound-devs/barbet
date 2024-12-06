#!/bin/bash
 
# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=bh-ar53
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=34,41%10
# SBATCH --array=0-52%10
 
# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH -t 4:00:00
 
#SBATCH --mem-per-cpu=12G
 
# SBATCH -p cascade
#SBATCH -p gpu-a100,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --account=punim2199
 
# Specify your email address to be notified of progress.
#SBATCH --mail-user=robert.turnbull@unimelb.edu.au
#SBATCH --mail-type=ALL
 
# Load the environment variables
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
 
source /data/gpfs/projects/punim2199/rob/bloodhound/.venv/bin/activate

export HF_HOME=/data/gpfs/projects/punim2199/huggingface
export TORCH_HOME=/data/gpfs/projects/punim2199/torch-hub


DOMAIN=ar53


#LAYERS=6
#MAX_LENGTH=0

#LAYERS=12
#MAX_LENGTH=3049

#LAYERS=30
#MAX_LENGTH=2019

LAYERS=33
MAX_LENGTH=2019

#LAYERS=36
#LAYERS=48

FAMILY_INDEX=${SLURM_ARRAY_TASK_ID:-0}

bloodhound-esm build-gene-array \
    --layers $LAYERS \
    --marker-genes /data/gpfs/projects/punim2199/gambit_data/release220-partial/${DOMAIN}_marker_genes_all_r220.tar.gz \
    --output-dir /data/gpfs/projects/punim2199/preprocessed/${DOMAIN}/esm${LAYERS} \
    --hub-dir $TORCH_HOME \
    --family-index $FAMILY_INDEX \
    --max-length $MAX_LENGTH




