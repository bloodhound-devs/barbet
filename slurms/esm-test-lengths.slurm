#!/bin/bash
 
# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=bh-ar53
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
 
# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH -t 4:00:00
 
#SBATCH --mem-per-cpu=12G
 
# SBATCH -p cascade
#SBATCH -p gpu-a100,gpu-a100-short
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


#LAYERS=30
#MAX_LENGTH=2019

#LAYERS=33

LAYERS=36
#LAYERS=48

bloodhound-esm test-lengths --layers $LAYERS  --end 2019 --start 800




