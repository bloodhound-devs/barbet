#!/bin/bash
 
# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=preprocess
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-119%10
 
# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH -t 12:00:00
 
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
export PATH="/data/gpfs/projects/punim2199/poetry-env/bin:$PATH"
source .venv/bin/activate

# export HF_HOME=/data/gpfs/projects/punim2199/huggingface
# export TORCH_HOME=/data/gpfs/projects/punim2199/torch-hub

DOMAIN=bac120
# DOMAIN=ar53
 AF_MODEL=1 # 1,2,3,4,5

bloodhound-af build-gene-array \
    --layers $AF_MODEL \
    --marker-genes /data/gpfs/projects/punim2199/gambit_data/release220-partial/${DOMAIN}_marker_genes_all_r220.tar.gz \
    --output-dir /data/gpfs/projects/punim2199/preprocessed/${DOMAIN}/af${AF_MODEL} \
    --family-index $SLURM_ARRAY_TASK_ID
