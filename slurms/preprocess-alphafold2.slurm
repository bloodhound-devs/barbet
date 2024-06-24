#!/bin/bash
# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name="preprocess-alphafold2"
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
 
# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH --time=4:00:00
 
#SBATCH --mem-per-cpu=12G
 
# SBATCH -p cascade
#SBATCH -p gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --account="punim2199"
 
# Specify your email address to be notified of progress.
#SBATCH --mail-user="g.marini@unimelb.edu.au"
#SBATCH --mail-type=ALL

#SBATCH --chdir="/data/gpfs/projects/punim2199/gabry/gambit"
 
# Load the environment variables
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
export PATH=/data/gpfs/projects/punim2199/poetry-env/bin:$PATH
 
poetry run gambit-preprocess-alphafold2-test