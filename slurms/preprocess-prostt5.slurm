#!/bin/bash
cd ..
 
# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=preprocess
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
 
# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH -t 4:00:00
 
#SBATCH --mem-per-cpu=8G
 
# SBATCH -p cascade
#SBATCH -p gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --account=punim2199
 
# Specify your email address to be notified of progress.
# #SBATCH --mail-user=robert.turnbull@unimelb.edu.au
#SBATCH --mail-type=ALL
 
# Load the environment variables
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
export PATH=/data/gpfs/projects/punim2199/poetry-env/bin:$PATH
 
poetry run gambit-preprocess-prostt5  \
    /data/gpfs/projects/punim2199/gambit_data/release220-partial/ar53_taxonomy_r220.tsv     \
    /data/gpfs/projects/punim2199/gambit_data/release220-partial/ar53_marker_genes_all_r220.tar.gz     \
    /data/gpfs/projects/punim2199/preprocessed/ar53/prostt5/prostt5.st     \
    /data/gpfs/projects/punim2199/preprocessed/ar53/prostt5/prostt5.sb     